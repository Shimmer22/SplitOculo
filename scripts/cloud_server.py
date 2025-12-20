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
    
    GET /health
        Response: {"status": "ok", "model_loaded": bool}
"""
import argparse
import sys
from pathlib import Path
import base64
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify

from models.cloud_upsampler import CloudUpsampler, TransformerUpsampler
from models.bottleneck import DimensionBottleneck


class CloudInferenceEngine:
    """云端推理引擎"""
    
    def __init__(self, checkpoint_path, device='cuda', split_layer=4):
        self.device = device
        self.split_layer = split_layer
        
        print(f"☁️ Loading cloud components from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
                # 拆分权重: 只有 decoder 部分
                self.bottleneck.decoder.load_state_dict(ckpt['bottleneck_decoder_state_dict'])
                print(f"   Bottleneck decoder (split): {bottleneck_dim} → {hidden_size}")
            elif 'bottleneck_state_dict' in ckpt:
                # AIO 权重: 完整 bottleneck
                self.bottleneck.load_state_dict(ckpt['bottleneck_state_dict'])
                print(f"   Bottleneck decoder (AIO): {bottleneck_dim} → {hidden_size}")
            else:
                print(f"   ⚠️ No bottleneck weights found, using random init")
            
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
        
        print(f"✅ Cloud components loaded")
        
        # Qwen (延迟加载)
        self.qwen_model = None
        self.processor = None
    
    def load_qwen(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        """加载 Qwen 模型"""
        if self.qwen_model is not None:
            return
        
        print(f"📥 Loading Qwen from {model_name}...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        self.qwen_model.eval()
        
        print(f"✅ Qwen loaded")
    
    def decode_features(self, features_b64: str, scale: float, zero_point: float):
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
        expected_dim = self.bottleneck_dim if self.bottleneck else self.hidden_size
        features_int8 = features_int8.reshape(1, self.transmission_tokens, expected_dim)
        
        # 转为 tensor 并反量化
        features = torch.from_numpy(features_int8.astype(np.float32)).to(self.device)
        features = (features - zero_point) * scale
        
        return features
    
    @torch.no_grad()
    def infer(self, compressed_features, prompt="这张图里有什么?"):
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
        
        # 3. 特征缩放匹配 Qwen 分布
        target_std = 0.83
        target_mean = -0.017
        current_std = upsampled.std()
        if current_std > 0:
            upsampled = (upsampled - upsampled.mean()) / current_std * target_std + target_mean
        
        # 4. Qwen 后续处理
        if self.qwen_model is None:
            self.load_qwen()
        
        visual = self.qwen_model.visual
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


# 全局引擎实例
engine = None
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': engine is not None,
        'qwen_loaded': engine.qwen_model is not None if engine else False
    })


@app.route('/infer', methods=['POST'])
def infer():
    """推理端点"""
    start_time = time.time()
    
    try:
        data = request.json
        features_b64 = data['features']
        scale = data['scale']
        zero_point = data['zero_point']
        prompt = data.get('prompt', '这张图里有什么?')
        
        # 反序列化特征
        features = engine.decode_features(features_b64, scale, zero_point)
        
        # 推理
        response = engine.infer(features, prompt)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            'response': response,
            'latency_ms': latency_ms
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SplitOculo Cloud Server")
    print("=" * 60)
    
    engine = CloudInferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    if args.preload_qwen:
        engine.load_qwen()
    
    print(f"\n🚀 Starting server on {args.host}:{args.port}")
    print(f"   POST /infer - Run inference")
    print(f"   GET /health - Health check")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
