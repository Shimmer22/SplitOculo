"""
SplitOculo Cloud Server

云端服务器：接收端侧压缩特征，完成推理并返回 LLM 响应。

Usage:
    python scripts/cloud_server.py \
        --checkpoint ./checkpoints/gan_bottleneck/gan_best.pth \
        --port 8080

API:
    POST /infer
        Request: {"features": base64, "scale": float, "zero_point": float, "prompt": str}
        Response: {"response": str, "latency_ms": float}

    POST /infer_qwen
        Request: {"frames": [base64 JPEG, ...], "prompt": str}
        Response: native full-Qwen video inference and timing metrics

    POST /infer_stream, POST /infer_qwen_stream
        Same requests as /infer and /infer_qwen. The response is NDJSON:
        zero or more {"type": "delta", "text": "..."} events followed by
        one {"type": "result", "result": {...}} event.

    POST /load_checkpoint (alias: /load_ckpt)
        Request: {"checkpoint_path": "/path/on/the/cloud"}
        The path is resolved on the cloud server. An absolute http(s) URL is
        also accepted and downloaded to a temporary cloud-local file.
    
    GET /health
        Response includes the active checkpoint metadata.
"""
import argparse
import sys
from pathlib import Path
import base64
from io import BytesIO
import time
import json
import queue
import threading
import gc
import os
import tempfile
import traceback
from functools import wraps
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from flask import Flask, Response, request, jsonify

# Transformers 5.x imports SciPy/Scikit-learn while initializing
# GenerationMixin. Keep the cloud entry point compatible with the NumPy 1.x
# deployment image before any Transformers import happens.
from core.runtime_compat import patch_numpy_legacy_aliases

patch_numpy_legacy_aliases()

from transformers import TextIteratorStreamer

from models.cloud_upsampler import CloudUpsampler, TransformerUpsampler
from models.bottleneck import DimensionBottleneck
from models.multilevel import pad_dim, resize_tokens


MAX_REMOTE_CHECKPOINT_BYTES = 4 * 1024 * 1024 * 1024
REMOTE_CHECKPOINT_CHUNK_BYTES = 8 * 1024 * 1024
REMOTE_CHECKPOINT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_VIDEO_FRAMES = 16


def _is_remote_checkpoint_reference(reference):
    return urlparse(str(reference)).scheme.lower() in {'http', 'https'}


def _is_windows_drive_path(reference):
    reference = str(reference)
    return (
        len(reference) >= 3
        and reference[1] == ':'
        and reference[2] in {'/', '\\'}
    )


class CloudInferenceEngine:
    """云端推理引擎"""
    
    def __init__(
        self,
        checkpoint_path,
        device='cuda',
        split_layer=4,
        max_video_frames=None,
    ):
        self.device = device
        self.split_layer = split_layer
        self.max_video_frames = (
            int(max_video_frames) if max_video_frames is not None else None
        )
        checkpoint_reference = str(checkpoint_path)
        temporary_path = None
        if _is_remote_checkpoint_reference(checkpoint_reference):
            local_path, source_key, temporary_path = _materialize_checkpoint_reference(
                checkpoint_reference
            )
        else:
            local_path = Path(checkpoint_reference).expanduser()
            if local_path.is_dir():
                local_path = local_path / 'cloud_weights.pth'
            local_path = local_path.resolve()
            source_key = str(local_path)
        self.checkpoint_path = str(local_path)
        self.checkpoint_source = source_key
        self._checkpoint_temp_path = (
            str(temporary_path) if temporary_path else None
        )
        
        print(f"Loading cloud components from {self.checkpoint_path}")
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        except Exception:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
            raise
        args = ckpt.get('args', {})
        
        self.transmission_tokens = args.get('transmission_tokens', 49)
        self.target_tokens = args.get('target_tokens', 256)
        hidden_size = args.get('target_hidden_size', 1280)
        self.hidden_size = hidden_size
        
        # 瓶颈层 Decoder
        bottleneck_dim = args.get('bottleneck_dim', 0)
        self.bottleneck_dim = bottleneck_dim
        
        if bottleneck_dim > 0:
            bottleneck_method = args.get('bottleneck_method', 'linear')
            self.bottleneck = DimensionBottleneck(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                method=bottleneck_method
            ).to(device)
            
            # 支持拆分权重和 AIO 权重
            if 'bottleneck_decoder_state_dict' in ckpt:
                # 拆分权重: 只有 decoder 部分，需要去掉 'decoder.' 前缀
                decoder_sd = {k.replace('decoder.', ''): v for k, v in ckpt['bottleneck_decoder_state_dict'].items()}
                self.bottleneck.decoder.load_state_dict(decoder_sd)
                print(f"   Bottleneck decoder (split): {bottleneck_dim} -> {hidden_size}")
            elif 'bottleneck_state_dict' in ckpt:
                # AIO 权重: 完整 bottleneck
                self.bottleneck.load_state_dict(ckpt['bottleneck_state_dict'])
                print(f"   Bottleneck decoder (AIO): {bottleneck_dim} -> {hidden_size}")
            else:
                print(f"   Warning: no bottleneck weights found, using random init")
            
            self.bottleneck.eval()
        else:
            self.bottleneck = None
            print(f"   No bottleneck (full dimension)")
        
        # Upsampler
        upsampler_type = args.get('upsampler_type', 'transformer')
        transformer_layers = args.get('transformer_layers', 4)
        
        if upsampler_type == 'transformer':
            initial_upsample = args.get('initial_upsample', 'bilinear')
            self.upsampler = TransformerUpsampler(
                hidden_size=hidden_size,
                input_tokens=self.transmission_tokens,
                target_tokens=self.target_tokens,
                num_layers=transformer_layers,
                initial_upsample=initial_upsample
            ).to(device)
            print(f"   TransformerUpsampler: {transformer_layers} layers")
        else:
            self.upsampler = CloudUpsampler(
                hidden_size=hidden_size,
                input_tokens=self.transmission_tokens,
                target_tokens=self.target_tokens,
                method=upsampler_type,
            ).to(device)
            print(f"   CloudUpsampler: {upsampler_type}")
        
        self.upsampler.load_state_dict(ckpt['upsampler_state_dict'])
        self.upsampler.eval()
        self.cloud_compute_dtype = (
            torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        )
        self.upsampler.to(dtype=self.cloud_compute_dtype)
        if self.bottleneck is not None:
            self.bottleneck.to(dtype=self.cloud_compute_dtype)
        
        print("Cloud components loaded")
        
        # Qwen (延迟加载)
        self.qwen_model = None
        self.processor = None
        self.qwen_model_name = None
        self._qwen_lock = threading.RLock()

    def _synchronize(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _limit_video_tensor(self, compressed_frame_features):
        """Uniformly subsample video features before expensive Qwen work."""
        input_frame_count = int(compressed_frame_features.shape[0])
        max_frames = self.max_video_frames
        if (
            max_frames is None
            or max_frames <= 0
            or input_frame_count <= max_frames
        ):
            return compressed_frame_features, input_frame_count

        indices = torch.linspace(
            0,
            input_frame_count - 1,
            steps=max_frames,
            device=compressed_frame_features.device,
        ).round().to(dtype=torch.long)
        print(
            f"Video payload has {input_frame_count} frames; "
            f"uniformly sampling {max_frames} before Qwen inference."
        )
        return compressed_frame_features.index_select(0, indices), input_frame_count

    def _limit_video_frames(self, frames):
        """Uniformly subsample RGB frames before native Qwen processing."""
        input_frame_count = len(frames)
        max_frames = self.max_video_frames
        if (
            max_frames is None
            or max_frames <= 0
            or input_frame_count <= max_frames
        ):
            return frames, input_frame_count

        indices = np.linspace(
            0,
            input_frame_count - 1,
            num=max_frames,
        ).round().astype(np.int64)
        print(
            f"Native Qwen request has {input_frame_count} frames; "
            f"uniformly sampling {max_frames} before processing."
        )
        return [frames[int(index)] for index in indices], input_frame_count

    def _unload_qwen_locked(self):
        """Release the cached Qwen model; caller must hold _qwen_lock."""
        self.qwen_model = None
        self.processor = None
        self.qwen_model_name = None
        gc.collect()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_qwen(self):
        """Unload the cached Qwen model."""
        with self._qwen_lock:
            self._unload_qwen_locked()

    def load_qwen(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        local_only=False,
        force_reload=False,
    ):
        """Load or switch the cached Qwen model.

        Previously this method returned whenever any Qwen model was already
        cached.  Consequently changing qwen_path in a client had no effect
        until the whole cloud process was restarted.
        """
        model_name = str(model_name)
        with self._qwen_lock:
            if (
                self.qwen_model is not None
                and self.qwen_model_name == model_name
                and not force_reload
            ):
                return False

            if self.qwen_model is not None:
                print(f"Switching Qwen model: {self.qwen_model_name} -> {model_name}")
                self._unload_qwen_locked()

            print(f"Loading Qwen from {model_name}...")
            if local_only:
                print("   offline mode; not connecting to HuggingFace")

            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            device_map = "cpu" if self.device == "cpu" else "auto"

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=local_only,
            )
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_only,
                use_fast=True,
                min_pixels=224 * 224,
                max_pixels=224 * 224,
            )

            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            self.qwen_model = model
            self.processor = processor
            self.qwen_model_name = model_name

            print("Qwen loaded")
            return True
    
    def decode_features(
        self,
        features_b64: str,
        scale: float,
        zero_point: float,
        payload_tokens=None,
        payload_dim=None,
        feature_shape=None,
    ):
        """
        反序列化并反量化特征
        
        Args:
            features_b64: base64 编码的 int8 特征
            scale: 量化缩放因子
            zero_point: 量化零点
        
        Returns:
            解压后的特征 tensor
        """
        # base64 解码
        features_bytes = base64.b64decode(features_b64)
        
        # 转为 numpy int8
        features_int8 = np.frombuffer(features_bytes, dtype=np.uint8)
        
        # 确定形状
        expected_tokens = int(payload_tokens or self.transmission_tokens)
        expected_dim = int(payload_dim or (self.bottleneck_dim if self.bottleneck else self.hidden_size))
        if feature_shape:
            shape = tuple(int(value) for value in feature_shape)
            if len(shape) != 3 or shape[1] <= 0 or shape[2] <= 0:
                raise ValueError(f"Invalid feature_shape: {feature_shape}")
            expected_tokens, expected_dim = shape[1], shape[2]
            features_int8 = features_int8.reshape(shape)
        else:
            features_int8 = features_int8.reshape(1, expected_tokens, expected_dim)
        
        # 转为 tensor 并反量化
        features = torch.from_numpy(features_int8.astype(np.float32)).to(self.device)
        features = (features - zero_point) * scale
        
        return features

    def decode_payload_to_edge_tokens(self, compressed_features):
        compressed_features = compressed_features.to(
            device=self.device,
            dtype=self.cloud_compute_dtype,
        )
        if self.bottleneck is not None:
            compressed_features = pad_dim(compressed_features, self.bottleneck_dim)
            edge_tokens = self.bottleneck.decode(compressed_features)
        else:
            edge_tokens = pad_dim(compressed_features, self.hidden_size)
        return resize_tokens(edge_tokens, self.transmission_tokens, mode="bilinear")

    def is_multilevel_payload(self, features, declared=False):
        """Determine whether a payload needs bottleneck-dimension padding.

        Older clients did not always send payload_dim for temporal/codec
        requests.  The tensor shape is still authoritative: a checkpoint
        with bottleneck_dim=128 cannot send a 64-channel tensor directly to
        the decoder.
        """
        if declared:
            return True
        if self.bottleneck is None or features.shape[-1] == self.bottleneck_dim:
            return False
        if features.shape[-1] < self.bottleneck_dim:
            return True
        raise ValueError(
            "Payload channel dimension "
            f"{features.shape[-1]} exceeds checkpoint bottleneck_dim "
            f"{self.bottleneck_dim}"
        )
    
    @torch.no_grad()
    def infer(self, compressed_features, prompt="Describe this image."):
        """
        完成云端推理

        Args:
            compressed_features: [1, 49, bottleneck_dim] 压缩特征
            prompt: 用户提示

        Returns:
            LLM 响应文本
        """
        # 1. Bottleneck 解码
        if self.bottleneck is not None:
            edge_tokens = self.bottleneck.decode(compressed_features)
        else:
            edge_tokens = compressed_features

        # 2. Upsampler
        upsampled = self.upsampler(edge_tokens)

        # 3. 特征缩放匹配 Qwen 分布 (按 split_layer 区分)
        if self.split_layer == 4:
            # Layer 4: mean=-0.022, std=0.847 (COCO 100 样本实测)
            target_std, target_mean = 0.847, -0.022
            current_std = upsampled.std()
            if current_std > 0:
                upsampled = (upsampled - upsampled.mean()) / current_std * target_std + target_mean
        elif self.split_layer == 8:
            # Layer 8: mean=-0.021, std=1.066 (COCO 100 样本实测)
            target_std, target_mean = 1.066, -0.021
            current_std = upsampled.std()
            if current_std > 0:
                upsampled = (upsampled - upsampled.mean()) / current_std * target_std + target_mean
        elif self.split_layer == 0:
            # Layer 0 (patch_embed): mean=-0.000, std=0.362
            target_std, target_mean = 0.362, -0.0001
            current_std = upsampled.std()
            if current_std > 0:
                upsampled = (upsampled - upsampled.mean()) / current_std * target_std + target_mean
        elif self.split_layer in (-1,):
            # Layer -1 (pixel patches): 像素空间，不强制归一化
            pass

        # 4. Qwen 后续处理
        if self.qwen_model is None:
            model_path = getattr(self, 'qwen_path', "Qwen/Qwen2.5-VL-3B-Instruct")
            offline = getattr(self, 'offline_mode', False)
            self.load_qwen(model_name=model_path, local_only=offline)
        
        visual = self.qwen_model.model.visual
        B = upsampled.shape[0]
        target_h = target_w = int(self.target_tokens ** 0.5)
        
        grid_thw = torch.tensor([[1, target_h, target_w]] * B, dtype=torch.long).to(self.device)
        hidden_states = upsampled.view(-1, upsampled.shape[-1])
        hidden_states = hidden_states.to(visual.blocks[0].attn.qkv.weight.dtype)
        
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=self.device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        for layer_num, blk in enumerate(visual.blocks):
            if layer_num < self.split_layer:
                continue
            if layer_num in visual.fullatt_block_indexes:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_window_seqlens, position_embeddings=position_embeddings)
        
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        visual_tokens = visual.merger(hidden_states)
        visual_tokens = visual_tokens.unsqueeze(0) if visual_tokens.dim() == 2 else visual_tokens
        
        # 5. 生成文本
        num_visual_tokens = visual_tokens.shape[1]
        image_placeholder = "<|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>"
        messages = [{'role': 'user', 'content': image_placeholder + prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        text_inputs = self.processor.tokenizer(text, return_tensors='pt', padding=True)
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        
        embed_layer = self.qwen_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        
        image_token_id = self.qwen_model.config.image_token_id
        image_mask = (input_ids == image_token_id)
        num_placeholders = image_mask.sum().item()
        
        visual_tokens_flat = visual_tokens.view(-1, visual_tokens.shape[-1])
        if visual_tokens_flat.shape[0] != num_placeholders:
            if visual_tokens_flat.shape[0] < num_placeholders:
                pad = visual_tokens_flat[-1:].repeat(num_placeholders - visual_tokens_flat.shape[0], 1)
                visual_tokens_flat = torch.cat([visual_tokens_flat, pad], dim=0)
            else:
                visual_tokens_flat = visual_tokens_flat[:num_placeholders]
        
        visual_tokens_flat = visual_tokens_flat.to(inputs_embeds.dtype)
        batch_indices, token_indices = torch.where(image_mask)
        for i, (b, t) in enumerate(zip(batch_indices, token_indices)):
            inputs_embeds[b, t] = visual_tokens_flat[i]
        
        outputs = self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        
        response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if 'assistant' in response.lower():
            response = response.split('assistant')[-1].strip()
        
        return response

    @torch.no_grad()
    def reconstruct_tokens(self, compressed_features):
        """Decode edge features and upsample them to split-layer visual tokens."""
        compressed_features = compressed_features.to(
            device=self.device,
            dtype=self.cloud_compute_dtype,
        )
        if self.bottleneck is not None:
            edge_tokens = self.bottleneck.decode(compressed_features)
        else:
            edge_tokens = compressed_features

        upsampled = self.upsampler(edge_tokens)

        if self.split_layer == 4:
            target_std, target_mean = 0.847, -0.022
            current_mean = upsampled.float().mean(dim=(1, 2), keepdim=True)
            current_std = upsampled.float().std(dim=(1, 2), keepdim=True)
            upsampled = (
                (upsampled - current_mean.to(upsampled.dtype))
                / current_std.clamp_min(1e-6).to(upsampled.dtype)
                * target_std
                + target_mean
            )
        elif self.split_layer == 8:
            target_std, target_mean = 1.066, -0.021
            current_mean = upsampled.float().mean(dim=(1, 2), keepdim=True)
            current_std = upsampled.float().std(dim=(1, 2), keepdim=True)
            upsampled = (
                (upsampled - current_mean.to(upsampled.dtype))
                / current_std.clamp_min(1e-6).to(upsampled.dtype)
                * target_std
                + target_mean
            )
        elif self.split_layer == 0:
            target_std, target_mean = 0.362, -0.0001
            current_mean = upsampled.float().mean(dim=(1, 2), keepdim=True)
            current_std = upsampled.float().std(dim=(1, 2), keepdim=True)
            upsampled = (
                (upsampled - current_mean.to(upsampled.dtype))
                / current_std.clamp_min(1e-6).to(upsampled.dtype)
                * target_std
                + target_mean
            )

        return upsampled

    @torch.no_grad()
    def reconstruct_payload_tokens(self, compressed_features):
        """Decode a possibly truncated multi-level payload to split-layer tokens."""
        edge_tokens = self.decode_payload_to_edge_tokens(compressed_features)
        upsampled = self.upsampler(edge_tokens)

        if self.split_layer == 4:
            target_std, target_mean = 0.847, -0.022
            current_mean = upsampled.float().mean(dim=(1, 2), keepdim=True)
            current_std = upsampled.float().std(dim=(1, 2), keepdim=True)
            upsampled = (
                (upsampled - current_mean.to(upsampled.dtype))
                / current_std.clamp_min(1e-6).to(upsampled.dtype)
                * target_std
                + target_mean
            )
        elif self.split_layer == 8:
            target_std, target_mean = 1.066, -0.021
            current_mean = upsampled.float().mean(dim=(1, 2), keepdim=True)
            current_std = upsampled.float().std(dim=(1, 2), keepdim=True)
            upsampled = (
                (upsampled - current_mean.to(upsampled.dtype))
                / current_std.clamp_min(1e-6).to(upsampled.dtype)
                * target_std
                + target_mean
            )
        elif self.split_layer == 0:
            target_std, target_mean = 0.362, -0.0001
            current_mean = upsampled.float().mean(dim=(1, 2), keepdim=True)
            current_std = upsampled.float().std(dim=(1, 2), keepdim=True)
            upsampled = (
                (upsampled - current_mean.to(upsampled.dtype))
                / current_std.clamp_min(1e-6).to(upsampled.dtype)
                * target_std
                + target_mean
            )

        return upsampled

    @torch.no_grad()
    def infer_payload(self, compressed_features, prompt="Describe this image.", max_new_tokens=256):
        """Run image inference from a standard or multi-level payload tensor."""
        upsampled = self.reconstruct_payload_tokens(compressed_features)
        target_h = target_w = int(self.target_tokens ** 0.5)
        grid_thw = torch.tensor([[1, target_h, target_w]], dtype=torch.long, device=self.device)
        visual_tokens = self.run_visual_tail(upsampled, grid_thw=grid_thw, modality="image")
        return self.generate_from_visual_tokens(
            visual_tokens,
            prompt=prompt,
            modality="image",
            max_new_tokens=max_new_tokens,
        )

    @torch.no_grad()
    def infer_payload_with_timing(
        self,
        compressed_features,
        prompt="Describe this image.",
        max_new_tokens=256,
        on_text=None,
    ):
        """Timed image path with TTFT measured at the first streamed token."""
        self._synchronize()
        reconstruct_start = time.perf_counter()
        upsampled = self.reconstruct_payload_tokens(compressed_features)
        self._synchronize()
        reconstruct_seconds = time.perf_counter() - reconstruct_start

        target_h = target_w = int(self.target_tokens ** 0.5)
        grid_thw = torch.tensor([[1, target_h, target_w]], dtype=torch.long, device=self.device)
        visual_tail_start = time.perf_counter()
        visual_tokens = self.run_visual_tail(upsampled, grid_thw=grid_thw, modality="image")
        self._synchronize()
        visual_tail_seconds = time.perf_counter() - visual_tail_start

        answer, generation_metrics = self.generate_from_visual_tokens_with_timing(
            visual_tokens,
            prompt=prompt,
            modality="image",
            max_new_tokens=max_new_tokens,
            on_text=on_text,
        )
        return answer, {
            **generation_metrics,
            "reconstruct_seconds": reconstruct_seconds,
            "visual_tail_seconds": visual_tail_seconds,
            "ttft_seconds": reconstruct_seconds + visual_tail_seconds + generation_metrics.get("ttft_seconds", 0.0),
        }

    @torch.no_grad()
    def run_visual_tail(self, upsampled, grid_thw=None, modality="image"):
        """Continue Qwen visual blocks from the configured split layer."""
        if self.qwen_model is None:
            model_path = getattr(self, 'qwen_path', "Qwen/Qwen2.5-VL-3B-Instruct")
            offline = getattr(self, 'offline_mode', False)
            self.load_qwen(model_name=model_path, local_only=offline)

        visual = self.qwen_model.model.visual
        batch_size = upsampled.shape[0]
        if grid_thw is None:
            target_h = target_w = int(self.target_tokens ** 0.5)
            grid_thw = torch.tensor([[1, target_h, target_w]] * batch_size, dtype=torch.long, device=self.device)
        else:
            grid_thw = torch.as_tensor(grid_thw, dtype=torch.long, device=self.device)
            if grid_thw.dim() == 1:
                grid_thw = grid_thw.unsqueeze(0)

        expected_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        if upsampled.shape[1] != expected_tokens:
            raise ValueError(f"{modality} grid_thw expects {expected_tokens} tokens, got {upsampled.shape[1]}")

        hidden_states = upsampled.view(-1, upsampled.shape[-1])
        hidden_states = hidden_states.to(visual.blocks[0].attn.qkv.weight.dtype)

        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=self.device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(visual.blocks):
            if layer_num < self.split_layer:
                continue
            if layer_num in visual.fullatt_block_indexes:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_window_seqlens, position_embeddings=position_embeddings)

        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        visual_tokens = visual.merger(hidden_states)
        return visual_tokens.unsqueeze(0) if visual_tokens.dim() == 2 else visual_tokens

    @torch.no_grad()
    def generate_from_visual_tokens(self, visual_tokens, prompt, modality="image", max_new_tokens=256):
        """Inject reconstructed image/video visual tokens into Qwen's language model."""
        if self.qwen_model is None:
            model_path = getattr(self, 'qwen_path', "Qwen/Qwen2.5-VL-3B-Instruct")
            offline = getattr(self, 'offline_mode', False)
            self.load_qwen(model_name=model_path, local_only=offline)

        if modality == "video":
            pad_token = "<|video_pad|>"
            visual_token_id = self.qwen_model.config.video_token_id
        else:
            pad_token = "<|image_pad|>"
            visual_token_id = self.qwen_model.config.image_token_id

        generation_prepare_start = time.perf_counter()
        num_visual_tokens = visual_tokens.shape[1]
        placeholder = "<|vision_start|>" + pad_token * num_visual_tokens + "<|vision_end|>"
        messages = [{'role': 'user', 'content': placeholder + prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        text_inputs = self.processor.tokenizer(text, return_tensors='pt', padding=True)
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)

        embed_layer = self.qwen_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        visual_mask = (input_ids == visual_token_id)
        num_placeholders = visual_mask.sum().item()
        visual_tokens_flat = visual_tokens.view(-1, visual_tokens.shape[-1])
        if visual_tokens_flat.shape[0] != num_placeholders:
            if visual_tokens_flat.shape[0] < num_placeholders:
                pad = visual_tokens_flat[-1:].repeat(num_placeholders - visual_tokens_flat.shape[0], 1)
                visual_tokens_flat = torch.cat([visual_tokens_flat, pad], dim=0)
            else:
                visual_tokens_flat = visual_tokens_flat[:num_placeholders]

        visual_tokens_flat = visual_tokens_flat.to(inputs_embeds.dtype)
        batch_indices, token_indices = torch.where(visual_mask)
        for i, (b, t) in enumerate(zip(batch_indices, token_indices)):
            inputs_embeds[b, t] = visual_tokens_flat[i]

        outputs = self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if 'assistant' in response.lower():
            response = response.split('assistant')[-1].strip()
        return response

    @torch.no_grad()
    def generate_from_visual_tokens_with_timing(
        self,
        visual_tokens,
        prompt,
        modality="image",
        max_new_tokens=256,
        on_text=None,
    ):
        """Inject visual tokens and stream text to measure first-token latency and TPS."""
        if self.qwen_model is None:
            model_path = getattr(self, 'qwen_path', "Qwen/Qwen2.5-VL-3B-Instruct")
            offline = getattr(self, 'offline_mode', False)
            self.load_qwen(model_name=model_path, local_only=offline)

        if modality == "video":
            pad_token = "<|video_pad|>"
            visual_token_id = self.qwen_model.config.video_token_id
        else:
            pad_token = "<|image_pad|>"
            visual_token_id = self.qwen_model.config.image_token_id

        generation_prepare_start = time.perf_counter()
        num_visual_tokens = visual_tokens.shape[1]
        placeholder = "<|vision_start|>" + pad_token * num_visual_tokens + "<|vision_end|>"
        messages = [{'role': 'user', 'content': placeholder + prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        text_inputs = self.processor.tokenizer(text, return_tensors='pt', padding=True)
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)

        embed_layer = self.qwen_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        visual_mask = (input_ids == visual_token_id)
        num_placeholders = visual_mask.sum().item()
        visual_tokens_flat = visual_tokens.view(-1, visual_tokens.shape[-1])
        if visual_tokens_flat.shape[0] != num_placeholders:
            if visual_tokens_flat.shape[0] < num_placeholders:
                pad = visual_tokens_flat[-1:].repeat(num_placeholders - visual_tokens_flat.shape[0], 1)
                visual_tokens_flat = torch.cat([visual_tokens_flat, pad], dim=0)
            else:
                visual_tokens_flat = visual_tokens_flat[:num_placeholders]

        visual_tokens_flat = visual_tokens_flat.to(inputs_embeds.dtype)
        batch_indices, token_indices = torch.where(visual_mask)
        for i, (b, t) in enumerate(zip(batch_indices, token_indices)):
            inputs_embeds[b, t] = visual_tokens_flat[i]

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        output_holder = {}
        error_holder = {}

        def _run_generate():
            try:
                output_holder["outputs"] = self.qwen_model.generate(**generation_kwargs)
            except Exception as exc:
                error_holder["error"] = exc
                print("Qwen generation failed:", file=sys.stderr)
                traceback.print_exc()
                # TextIteratorStreamer otherwise waits forever because the
                # consumer is blocked in ``for chunk in streamer``.
                streamer.end()

        generation_start = time.perf_counter()
        generation_prepare_seconds = generation_start - generation_prepare_start
        thread = threading.Thread(target=_run_generate)
        thread.start()

        chunks = []
        first_token_seconds = None
        for chunk in streamer:
            if chunk and first_token_seconds is None:
                first_token_seconds = time.perf_counter() - generation_start
            chunks.append(chunk)
            if chunk and on_text is not None:
                on_text(chunk)

        thread.join()
        generation_seconds = time.perf_counter() - generation_start
        if "error" in error_holder:
            raise error_holder["error"]

        outputs = output_holder["outputs"]
        generated_tokens = int(outputs[0].numel()) if outputs is not None else 0
        answer = "".join(chunks).strip()
        if not answer and outputs is not None:
            answer = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if 'assistant' in answer.lower():
                answer = answer.split('assistant')[-1].strip()

        average_tps = None
        if generated_tokens > 0 and generation_seconds > 0:
            average_tps = generated_tokens / generation_seconds

        return answer, {
            "generation_prepare_seconds": generation_prepare_seconds,
            "first_token_seconds": first_token_seconds,
            "ttft_seconds": generation_prepare_seconds + (first_token_seconds or generation_seconds),
            "generation_seconds": generation_seconds,
            "generated_tokens": generated_tokens,
            "average_tps": average_tps,
        }

    @torch.no_grad()
    def infer_video_from_frame_features(self, compressed_frame_features, prompt="Describe this video.", max_new_tokens=256):
        """
        First-pass SplitOculo video inference: reconstruct each frame with the
        existing image split model, then preserve frame order as T x H x W tokens.
        """
        if compressed_frame_features.dim() == 4:
            compressed_frame_features = compressed_frame_features.squeeze(1)
        if compressed_frame_features.dim() != 3:
            raise ValueError("compressed_frame_features must be [T, tokens, channels]")

        compressed_frame_features, _ = self._limit_video_tensor(
            compressed_frame_features
        )

        frame_tokens = []
        for frame_features in compressed_frame_features:
            if self.is_multilevel_payload(frame_features):
                upsampled = self.reconstruct_payload_tokens(frame_features.unsqueeze(0))
            else:
                upsampled = self.reconstruct_tokens(frame_features.unsqueeze(0))
            frame_tokens.append(upsampled.squeeze(0))

        video_tokens = torch.cat(frame_tokens, dim=0).unsqueeze(0)
        target_h = target_w = int(self.target_tokens ** 0.5)
        grid_thw = torch.tensor([[len(frame_tokens), target_h, target_w]], dtype=torch.long, device=self.device)
        visual_tokens = self.run_visual_tail(video_tokens, grid_thw=grid_thw, modality="video")
        return self.generate_from_visual_tokens(
            visual_tokens,
            prompt=prompt,
            modality="video",
            max_new_tokens=max_new_tokens,
        )

    @torch.no_grad()
    def infer_video_from_frame_features_with_timing(
        self,
        compressed_frame_features,
        prompt="Describe this video.",
        max_new_tokens=256,
        multilevel_payload=False,
        on_text=None,
    ):
        """Timed SplitOculo video path with reconstruction, visual tail, and generation metrics."""
        if compressed_frame_features.dim() == 4:
            compressed_frame_features = compressed_frame_features.squeeze(1)
        if compressed_frame_features.dim() != 3:
            raise ValueError("compressed_frame_features must be [T, tokens, channels]")

        compressed_frame_features, input_frame_count = self._limit_video_tensor(
            compressed_frame_features
        )

        self._synchronize()
        reconstruct_start = time.perf_counter()
        multilevel_payload = self.is_multilevel_payload(
            compressed_frame_features,
            declared=multilevel_payload,
        )
        if multilevel_payload:
            upsampled = self.reconstruct_payload_tokens(
                compressed_frame_features
            )
        else:
            upsampled = self.reconstruct_tokens(compressed_frame_features)
        self._synchronize()
        reconstruct_seconds = time.perf_counter() - reconstruct_start

        frame_count = int(upsampled.shape[0])
        video_tokens = upsampled.reshape(
            1, frame_count * upsampled.shape[1], upsampled.shape[2]
        )
        target_h = target_w = int(self.target_tokens ** 0.5)
        grid_thw = torch.tensor(
            [[frame_count, target_h, target_w]],
            dtype=torch.long,
            device=self.device,
        )

        self._synchronize()
        visual_tail_start = time.perf_counter()
        visual_tokens = self.run_visual_tail(video_tokens, grid_thw=grid_thw, modality="video")
        self._synchronize()
        visual_tail_seconds = time.perf_counter() - visual_tail_start

        answer, generation_metrics = self.generate_from_visual_tokens_with_timing(
            visual_tokens,
            prompt=prompt,
            modality="video",
            max_new_tokens=max_new_tokens,
            on_text=on_text,
        )
        metrics = {
            **generation_metrics,
            "reconstruct_seconds": reconstruct_seconds,
            "visual_tail_seconds": visual_tail_seconds,
            "ttft_seconds": reconstruct_seconds + visual_tail_seconds + generation_metrics.get("ttft_seconds", 0.0),
            "cloud_compute_dtype": str(self.cloud_compute_dtype),
            "input_frame_count": input_frame_count,
            "frame_count": frame_count,
            "frame_sampling": (
                "uniform"
                if input_frame_count != frame_count
                else "none"
            ),
            "visual_grid_thw": grid_thw.detach().cpu().tolist(),
            "visual_tokens_after_merge": int(
                frame_count * target_h * target_w
                // (self.qwen_model.model.visual.spatial_merge_size**2)
            ),
        }
        return answer, metrics

    @torch.no_grad()
    def infer_qwen_frames_with_timing(
        self,
        frames,
        prompt="Describe this video.",
        max_new_tokens=256,
        video_pixel_budget=224 * 224,
        video_fps=2.0,
        on_text=None,
    ):
        """Run the complete native Qwen vision encoder and language model."""
        if not frames:
            raise ValueError("Pure Qwen inference requires at least one RGB frame")
        frames, input_frame_count = self._limit_video_frames(frames)
        if self.qwen_model is None:
            model_path = getattr(
                self, 'qwen_path', "Qwen/Qwen2.5-VL-3B-Instruct"
            )
            offline = getattr(self, 'offline_mode', False)
            self.load_qwen(model_name=model_path, local_only=offline)

        modality = "image" if len(frames) == 1 else "video"
        media_content = (
            {"type": "image", "image": "sampled_frame"}
            if modality == "image"
            else {"type": "video", "video": "sampled_frames"}
        )
        messages = [{
            "role": "user",
            "content": [
                media_content,
                {"type": "text", "text": prompt},
            ],
        }]

        processor_start = time.perf_counter()
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if modality == "image":
            processor_kwargs["images"] = [frames[0]]
        else:
            processor_kwargs["videos"] = [frames]
            processor_kwargs["videos_kwargs"] = {
                "size": {
                    "shortest_edge": int(video_pixel_budget),
                    "longest_edge": int(video_pixel_budget),
                },
                "fps": float(video_fps),
            }
        inputs = self.processor(**processor_kwargs)
        grid_key = "image_grid_thw" if modality == "image" else "video_grid_thw"
        native_grid_thw = inputs[grid_key].detach().cpu().tolist()
        merge_size = (
            self.processor.image_processor.merge_size
            if modality == "image"
            else self.processor.video_processor.merge_size
        )
        native_visual_tokens = sum(
            int(grid[0] * grid[1] * grid[2]) // (merge_size**2)
            for grid in native_grid_thw
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        processor_seconds = time.perf_counter() - processor_start

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "streamer": streamer,
        }
        output_holder = {}
        error_holder = {}

        def _run_generate():
            try:
                output_holder["outputs"] = self.qwen_model.generate(
                    **generation_kwargs
                )
            except Exception as exc:
                error_holder["error"] = exc
                print("Qwen generation failed:", file=sys.stderr)
                traceback.print_exc()
                # Wake the consumer so the original exception can be
                # returned by the Flask handler instead of deadlocking it.
                streamer.end()

        generation_start = time.perf_counter()
        thread = threading.Thread(target=_run_generate)
        thread.start()
        chunks = []
        first_token_seconds = None
        for chunk in streamer:
            if chunk and first_token_seconds is None:
                first_token_seconds = time.perf_counter() - generation_start
            chunks.append(chunk)
            if chunk and on_text is not None:
                on_text(chunk)
        thread.join()
        generation_seconds = time.perf_counter() - generation_start
        if "error" in error_holder:
            raise error_holder["error"]

        outputs = output_holder["outputs"]
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], outputs)
        ]
        generated_tokens = int(generated_ids[0].numel()) if generated_ids else 0
        answer = "".join(chunks).strip()
        if not answer and generated_ids:
            answer = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        effective_first_token = first_token_seconds or generation_seconds
        return answer, {
            "processor_seconds": processor_seconds,
            "first_token_seconds": first_token_seconds,
            "generation_seconds": generation_seconds,
            "generated_tokens": generated_tokens,
            "average_tps": (
                generated_tokens / generation_seconds
                if generated_tokens > 0 and generation_seconds > 0
                else None
            ),
            "ttft_seconds": processor_seconds + effective_first_token,
            "modality": modality,
            "input_frame_count": input_frame_count,
            "frames": len(frames),
            "frame_sampling": (
                "uniform"
                if input_frame_count != len(frames)
                else "none"
            ),
            "native_grid_thw": native_grid_thw,
            "native_visual_tokens": native_visual_tokens,
        }


# 全局引擎实例。动态切换 checkpoint 时，所有模型访问都通过这把锁串行化，
# 避免在一次推理使用旧模型的过程中替换模块或释放 CUDA 内存。
engine = None
_engine_lock = threading.RLock()
app = Flask(__name__)


def _engine_access(func):
    """Serialize model access for Flask's threaded request handler."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _engine_lock:
            return func(*args, **kwargs)

    return wrapped


def _as_bool(value, default=False):
    """Parse JSON booleans without treating the string ``"false"`` as true."""
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off', ''}:
            return False
    return bool(value)


def _checkpoint_reference_from_request(data):
    """Read the supported checkpoint field aliases from a JSON request."""
    for key in (
        'checkpoint_path',
        'ckpt_path',
        'checkpoint',
        'ckpt',
        'cloud_checkpoint',
        'cloud_ckpt',
        'checkpoint_url',
    ):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _local_checkpoint_path(reference):
    """Normalize a checkpoint path visible to the cloud server."""
    path = Path(reference).expanduser()
    if path.is_dir():
        path = path / 'cloud_weights.pth'
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Cloud checkpoint does not exist on the server: {path}"
        )
    return path


def _download_checkpoint(reference):
    """Download an HTTP(S) checkpoint to a temporary cloud-local file."""
    parsed = urlparse(reference)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError(
            'checkpoint URL must be an absolute http(s) URL, or provide a '
            'filesystem path visible to the cloud server'
        )

    fd, temporary_name = tempfile.mkstemp(
        prefix='splitoculo-cloud-checkpoint-',
        suffix='.pth',
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    downloaded_bytes = 0
    try:
        request = Request(
            reference,
            headers={'User-Agent': 'SplitOculo-cloud-server/1.0'},
        )
        with urlopen(request, timeout=REMOTE_CHECKPOINT_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > MAX_REMOTE_CHECKPOINT_BYTES:
                raise ValueError(
                    'remote checkpoint is larger than the server limit '
                    f'({MAX_REMOTE_CHECKPOINT_BYTES} bytes)'
                )
            with temporary_path.open('wb') as output:
                while True:
                    chunk = response.read(REMOTE_CHECKPOINT_CHUNK_BYTES)
                    if not chunk:
                        break
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_REMOTE_CHECKPOINT_BYTES:
                        raise ValueError(
                            'remote checkpoint is larger than the server limit '
                            f'({MAX_REMOTE_CHECKPOINT_BYTES} bytes)'
                        )
                    output.write(chunk)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if downloaded_bytes <= 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError('remote checkpoint response was empty')
    return temporary_path


def _materialize_checkpoint_reference(reference):
    """Return ``(local_path, source_key, temporary_path)`` for a request ref."""
    parsed = urlparse(reference)
    if _is_remote_checkpoint_reference(reference):
        temporary_path = _download_checkpoint(reference)
        return temporary_path, reference, temporary_path
    if parsed.scheme and not _is_windows_drive_path(reference):
        raise ValueError(
            'checkpoint_path must be a cloud-local filesystem path or an '
            'absolute http(s) URL'
        )
    local_path = _local_checkpoint_path(reference)
    return local_path, str(local_path), None


def _checkpoint_info(current_engine):
    if current_engine is None:
        return {
            'checkpoint_path': None,
            'checkpoint_source': None,
            'checkpoint_hidden_size': None,
            'checkpoint_bottleneck_dim': None,
            'checkpoint_transmission_tokens': None,
            'checkpoint_target_tokens': None,
        }
    return {
        'checkpoint_path': current_engine.checkpoint_path,
        'checkpoint_source': current_engine.checkpoint_source,
        'checkpoint_hidden_size': current_engine.hidden_size,
        'checkpoint_bottleneck_dim': current_engine.bottleneck_dim,
        'checkpoint_transmission_tokens': current_engine.transmission_tokens,
        'checkpoint_target_tokens': current_engine.target_tokens,
    }


def _remove_temporary_checkpoint(path):
    if path:
        Path(path).unlink(missing_ok=True)


def _switch_checkpoint_locked(reference, force_reload=False, preload_qwen=None):
    """Switch the global cloud engine; caller must hold ``_engine_lock``."""
    global engine

    if engine is None:
        raise RuntimeError('cloud engine is not loaded')

    current_engine = engine
    parsed = urlparse(reference)
    if _is_remote_checkpoint_reference(reference):
        source_key = reference
    elif parsed.scheme and not _is_windows_drive_path(reference):
        raise ValueError(
            'checkpoint_path must be a cloud-local filesystem path or an '
            'absolute http(s) URL'
        )
    else:
        source_key = str(_local_checkpoint_path(reference))

    if not force_reload and current_engine.checkpoint_source == source_key:
        if preload_qwen and getattr(current_engine, 'qwen_model', None) is None:
            current_engine.load_qwen(
                model_name=getattr(
                    current_engine,
                    'qwen_path',
                    'Qwen/Qwen2.5-VL-3B-Instruct',
                ),
                local_only=getattr(current_engine, 'offline_mode', False),
            )
        return current_engine, False

    local_path = None
    temporary_path = None
    replacement = None
    old_temporary_path = getattr(current_engine, '_checkpoint_temp_path', None)
    old_qwen_loaded = getattr(current_engine, 'qwen_model', None) is not None
    qwen_path = getattr(
        current_engine,
        'qwen_path',
        'Qwen/Qwen2.5-VL-3B-Instruct',
    )
    offline_mode = getattr(current_engine, 'offline_mode', False)
    if preload_qwen is None:
        preload_qwen = old_qwen_loaded

    try:
        if _is_remote_checkpoint_reference(reference):
            local_path, _, temporary_path = _materialize_checkpoint_reference(reference)
        else:
            local_path = _local_checkpoint_path(reference)

        # The old Qwen model is the largest allocation in the usual deployment.
        # Release it before constructing the replacement to leave room for the
        # temporary overlap of the old and new cloud components.
        current_engine.unload_qwen()
        replacement = CloudInferenceEngine(
            checkpoint_path=local_path,
            device=current_engine.device,
            split_layer=current_engine.split_layer,
            max_video_frames=current_engine.max_video_frames,
        )
        replacement.checkpoint_source = source_key
        replacement._checkpoint_temp_path = (
            str(temporary_path) if temporary_path else None
        )
        replacement.qwen_path = qwen_path
        replacement.offline_mode = offline_mode
        if preload_qwen:
            replacement.load_qwen(
                model_name=replacement.qwen_path,
                local_only=replacement.offline_mode,
            )
    except Exception:
        if replacement is not None:
            replacement.unload_qwen()
        _remove_temporary_checkpoint(temporary_path)
        gc.collect()
        if str(current_engine.device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    engine = replacement
    _remove_temporary_checkpoint(old_temporary_path)
    del current_engine
    gc.collect()
    if str(replacement.device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replacement, True


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    # Health is a read-only snapshot.  It must remain responsive while an
    # inference request is using the serialized model lock.
    current_engine = engine
    result = {
        'status': 'ok',
        'model_loaded': current_engine is not None,
        'qwen_loaded': current_engine.qwen_model is not None if current_engine else False,
        'qwen_model_name': current_engine.qwen_model_name if current_engine else None,
        'qwen_path': getattr(current_engine, 'qwen_path', None) if current_engine else None,
        'max_video_frames': (
            current_engine.max_video_frames if current_engine else None
        ),
    }
    result.update(_checkpoint_info(current_engine))
    return jsonify(result)


@app.route('/warmup', methods=['POST'])
@_engine_access
def warmup():
    """Load Qwen before the first measured request."""
    started = time.perf_counter()
    try:
        if engine is None:
            return jsonify({'error': 'cloud engine is not loaded'}), 503
        if engine.qwen_model is None:
            model_path = getattr(engine, 'qwen_path', 'Qwen/Qwen2.5-VL-3B-Instruct')
            offline = getattr(engine, 'offline_mode', False)
            engine.load_qwen(model_name=model_path, local_only=offline)
        return jsonify({
            'status': 'ok',
            'model_loaded': True,
            'qwen_loaded': True,
            'qwen_model_name': engine.qwen_model_name,
            'warmup_ms': (time.perf_counter() - started) * 1000,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/load_qwen', methods=['POST'])
@_engine_access
def load_qwen():
    """Load, switch, or reload the cached Qwen model.

    Request JSON:
        {"model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
         "offline": true, "force_reload": false}
    """
    started = time.perf_counter()
    try:
        if engine is None:
            return jsonify({'error': 'cloud engine is not loaded'}), 503
        data = request.get_json(silent=True) or {}
        model_name = data.get('model_name') or data.get('qwen_path')
        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400

        offline = bool(data.get('offline', getattr(engine, 'offline_mode', False)))
        force_reload = bool(data.get('force_reload', False))
        engine.qwen_path = str(model_name)
        engine.offline_mode = offline
        reloaded = engine.load_qwen(
            model_name=engine.qwen_path,
            local_only=offline,
            force_reload=force_reload,
        )
        return jsonify({
            'status': 'ok',
            'qwen_loaded': True,
            'qwen_model_name': engine.qwen_model_name,
            'reloaded': bool(reloaded),
            'load_ms': (time.perf_counter() - started) * 1000,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/load_checkpoint', methods=['POST'])
@app.route('/load_ckpt', methods=['POST'])
@_engine_access
def load_checkpoint():
    """Load or switch the cloud-side SplitOculo checkpoint.

    The path is resolved on the cloud machine, not on the client machine.
    Request JSON accepts a local path or an HTTP(S) URL:

        {"checkpoint_path": "/models/exp-a/cloud_weights.pth"}
        {"checkpoint_path": "https://host.example/exp-a/cloud_weights.pth",
         "preload_qwen": false, "force_reload": false}

    ``ckpt_path``, ``checkpoint``, ``ckpt``, ``cloud_checkpoint``, ``cloud_ckpt`` and
    ``checkpoint_url`` are accepted as aliases for clients that already use
    those names.  The old engine remains active if the replacement fails.
    """
    started = time.perf_counter()
    try:
        if engine is None:
            return jsonify({'error': 'cloud engine is not loaded'}), 503
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'error': 'request body must be a JSON object'}), 400
        reference = _checkpoint_reference_from_request(data)
        if not reference:
            return jsonify({
                'error': 'checkpoint_path is required and must be visible to '
                         'the cloud server',
            }), 400

        replacement, reloaded = _switch_checkpoint_locked(
            reference,
            force_reload=_as_bool(data.get('force_reload'), False),
            preload_qwen=(
                _as_bool(data.get('preload_qwen'))
                if 'preload_qwen' in data
                else None
            ),
        )
        result = {
            'status': 'ok',
            'reloaded': bool(reloaded),
            'load_ms': (time.perf_counter() - started) * 1000,
            'qwen_loaded': replacement.qwen_model is not None,
            'qwen_model_name': replacement.qwen_model_name,
        }
        result.update(_checkpoint_info(replacement))
        return jsonify(result)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


def _infer_payload_data(data, on_text=None):
    """Run a SplitOculo request and return the normal response dictionary."""
    start_time = time.perf_counter()
    features_b64 = data['features']
    scale = data['scale']
    zero_point = data['zero_point']
    prompt = data.get('prompt', 'Describe this image.')
    modality = data.get('modality', 'image')
    decode_start = time.perf_counter()

    features = engine.decode_features(
        features_b64,
        scale,
        zero_point,
        payload_tokens=data.get('payload_tokens'),
        payload_dim=data.get('payload_dim'),
        feature_shape=data.get('feature_shape'),
    )

    cloud_decode_ms = (time.perf_counter() - decode_start) * 1000
    infer_start = time.perf_counter()
    if modality == 'video':
        multilevel_payload = engine.is_multilevel_payload(
            features,
            declared=bool(
                data.get('payload_tokens') is not None
                or data.get('payload_dim') is not None
            ),
        )
        response, inference_metrics = engine.infer_video_from_frame_features_with_timing(
            features,
            prompt=prompt,
            max_new_tokens=int(data.get('max_new_tokens', 256)),
            multilevel_payload=multilevel_payload,
            on_text=on_text,
        )
    else:
        response, inference_metrics = engine.infer_payload_with_timing(
            features,
            prompt,
            max_new_tokens=int(data.get('max_new_tokens', 256)),
            on_text=on_text,
        )
    cloud_inference_ms = (time.perf_counter() - infer_start) * 1000
    cloud_process_ms = (time.perf_counter() - start_time) * 1000
    return {
        'response': response,
        'latency_ms': cloud_process_ms,
        'cloud_process_ms': cloud_process_ms,
        'cloud_decode_ms': cloud_decode_ms,
        'cloud_inference_ms': cloud_inference_ms,
        'cloud_ttft_ms': float(inference_metrics.get('ttft_seconds', 0.0)) * 1000,
        'inference_metrics': inference_metrics,
        'modality': modality,
    }


def _infer_qwen_data(data, on_text=None):
    """Run a native-Qwen request and return the normal response dictionary."""
    start_time = time.perf_counter()
    encoded_frames = data.get("frames") or []
    if not isinstance(encoded_frames, list) or not encoded_frames:
        raise ValueError('frames must be a non-empty list')

    decode_start = time.perf_counter()
    frames = []
    for encoded in encoded_frames:
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            frames.append(image.convert("RGB").copy())
    cloud_decode_ms = (time.perf_counter() - decode_start) * 1000

    inference_start = time.perf_counter()
    response, inference_metrics = engine.infer_qwen_frames_with_timing(
        frames,
        prompt=data.get("prompt", "Describe this video."),
        max_new_tokens=int(data.get("max_new_tokens", 256)),
        video_pixel_budget=int(data.get("video_pixel_budget", 224 * 224)),
        video_fps=float(data.get("video_fps", 2.0)),
        on_text=on_text,
    )
    cloud_inference_ms = (time.perf_counter() - inference_start) * 1000
    cloud_process_ms = (time.perf_counter() - start_time) * 1000
    return {
        'response': response,
        'latency_ms': cloud_process_ms,
        'cloud_process_ms': cloud_process_ms,
        'cloud_decode_ms': cloud_decode_ms,
        'cloud_inference_ms': cloud_inference_ms,
        'cloud_ttft_ms': float(inference_metrics.get('ttft_seconds', 0.0)) * 1000,
        'inference_metrics': inference_metrics,
        'modality': inference_metrics.get('modality', 'video'),
        'pure_qwen': True,
    }


def _ndjson_response(task):
    """Run an inference task in a worker and emit delta/result NDJSON events."""
    events = queue.Queue()
    finished = object()

    def emit_text(chunk):
        for character in chunk:
            events.put({'type': 'delta', 'text': character})

    def worker():
        try:
            with _engine_lock:
                result = task(emit_text)
            events.put({'type': 'result', 'result': result})
        except Exception as exc:
            events.put({'type': 'error', 'error': str(exc)})
        finally:
            events.put(finished)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            event = events.get()
            if event is finished:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return Response(
        generate(),
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/infer', methods=['POST'])
@_engine_access
def infer():
    """推理端点"""
    try:
        return jsonify(_infer_payload_data(request.get_json() or {}))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/infer_stream', methods=['POST'])
def infer_stream():
    data = request.get_json(silent=True) or {}
    return _ndjson_response(lambda on_text: _infer_payload_data(data, on_text))


@app.route('/infer_qwen', methods=['POST'])
@_engine_access
def infer_qwen():
    """Run native Qwen on uploaded JPEG RGB frames."""
    try:
        return jsonify(_infer_qwen_data(request.get_json(silent=True) or {}))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/infer_qwen_stream', methods=['POST'])
def infer_qwen_stream():
    data = request.get_json(silent=True) or {}
    return _ndjson_response(lambda on_text: _infer_qwen_data(data, on_text))


def main():
    global engine
    
    parser = argparse.ArgumentParser(description='SplitOculo Cloud Server')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--port', type=int, default=8080,
                        help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--preload_qwen', action='store_true',
                        help='Preload Qwen model on startup')
    parser.add_argument('--offline', action='store_true',
                        help='Use cached Qwen model, do not connect to HuggingFace')
    parser.add_argument('--qwen_path', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help='Qwen model path (local path or HuggingFace ID)')
    parser.add_argument(
        '--max_video_frames',
        type=int,
        default=DEFAULT_MAX_VIDEO_FRAMES,
        help=(
            'Maximum frames sent through the cloud Qwen path; non-positive '
            'disables server-side sampling'
        ),
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SplitOculo Cloud Server")
    print("=" * 60)
    
    engine = CloudInferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_video_frames=args.max_video_frames,
    )
    
    # 存储配置供后续使用
    engine.qwen_path = args.qwen_path
    engine.offline_mode = args.offline
    
    if args.preload_qwen:
        engine.load_qwen(model_name=args.qwen_path, local_only=args.offline)
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"   POST /infer - Run inference")
    print(f"   POST /load_checkpoint - Load or switch cloud checkpoint")
    print(f"   POST /load_qwen - Load or switch Qwen model")
    print(f"   GET /health - Health check")
    print(f"   Max video frames: {args.max_video_frames}")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
