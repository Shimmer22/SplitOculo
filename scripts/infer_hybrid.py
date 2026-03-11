"""
使用可学习上采样器的混合推理脚本

层级语义 (split_layer):
  -1 : 与 JPEG 像素 patch 对齐 (pixel-space reconstruction)
       推理: upsampled tokens → patch_embed → blocks[0:] → merger
   0 : 与 patch_embed 输出对齐
       推理: upsampled tokens → blocks[0:] → merger
   4 : 与 block 4 输出对齐 (默认)
       推理: upsampled tokens → blocks[4:] → merger
   8 : 与 block 8 输出对齐
       推理: upsampled tokens → blocks[8:] → merger

Usage:
    python scripts/infer_hybrid.py \\
        --checkpoint checkpoints/upsampler/best_model.pth \\
        --image photo.jpg --split_layer 4 --full_inference
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
import math

from models.cloud_upsampler import CloudUpsampler
from models.projector_v3 import StridedProjector
from models.bottleneck import DimensionBottleneck


class EdgeProjector(nn.Module):
    """端侧 Projector: CNN 特征 → 传输 tokens"""
    def __init__(self, in_channels, hidden_size=1280,
                 hidden_channels=512, transmission_tokens=49):
        super().__init__()
        
        self.transmission_size = int(math.sqrt(transmission_tokens))
        
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((self.transmission_size, self.transmission_size))
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_channels, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class HybridUpsamplerVLM:
    """
    使用可学习上采样器的混合 VLM
    
    加载 train_with_upsampler.py 训练的检查点
    """
    def __init__(self, checkpoint_path, 
                 qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 split_layer=4, device='cuda'):
        self.device = device
        self.qwen_model_name = qwen_model_name
        self.split_layer = split_layer
        
        # 加载检查点
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        args = ckpt.get('args', {})
        
        self.transmission_tokens = args.get('transmission_tokens', 49)
        self.target_tokens = args.get('target_tokens', 256)
        hidden_size = args.get('target_hidden_size', 1280)
        upsampler_method = args.get('upsampler_method', 'deconv')
        upsampler_layers = args.get('upsampler_layers', 2)
        
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        print(f"   传输 tokens: {self.transmission_tokens}")
        print(f"   目标 tokens: {self.target_tokens}")
        print(f"   上采样方法: {upsampler_method}")
        
        # CNN backbone
        student_model = args.get('student_model', 'mobilenetv2_100')
        student_layer = args.get('student_layer', 3)
        
        self.student = timm.create_model(
            student_model, pretrained=False, features_only=True,
            out_indices=[student_layer]
        ).to(device)
        self.student.load_state_dict(ckpt['student_state_dict'])
        self.student.eval()
        
        # 获取 student 通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            student_channels = self.student(dummy)[-1].shape[1]
        
        # 端侧 Projector
        # 端侧 Projector
        projector_type = args.get('projector_type', 'pooling')
        if projector_type == 'strided':
            self.projector = StridedProjector(
                in_channels=student_channels,
                hidden_size=hidden_size,
                hidden_channels=args.get('projector_hidden', 512),
                transmission_tokens=self.transmission_tokens
            ).to(device)
            print(f"   Using StridedProjector (v3)")
        else:
            self.projector = EdgeProjector(
                in_channels=student_channels,
                hidden_size=hidden_size,
                hidden_channels=args.get('projector_hidden', 512),
                transmission_tokens=self.transmission_tokens
            ).to(device)
            print(f"   Using EdgeProjector (pooling)")
            
        self.projector.load_state_dict(ckpt['projector_state_dict'])
        self.projector.eval()
        
        # 云端 Upsampler (支持 TransformerUpsampler)
        upsampler_type = args.get('upsampler_type', upsampler_method)
        transformer_layers = args.get('transformer_layers', 4)
        
        if upsampler_type == 'transformer':
            from models.cloud_upsampler import TransformerUpsampler
            self.upsampler = TransformerUpsampler(
                hidden_size=hidden_size,
                input_tokens=self.transmission_tokens,
                target_tokens=self.target_tokens,
                num_layers=transformer_layers
            ).to(device)
            print(f"   使用 TransformerUpsampler ({transformer_layers} layers)")
            
            # 重新实例化以包含 initial_upsample 参数
            initial_upsample = args.get('initial_upsample', 'bilinear')
            self.upsampler = TransformerUpsampler(
                hidden_size=hidden_size,
                input_tokens=self.transmission_tokens,
                target_tokens=self.target_tokens,
                num_layers=transformer_layers,
                initial_upsample=initial_upsample
            ).to(device)
        else:
            self.upsampler = CloudUpsampler(
                hidden_size=hidden_size,
                input_tokens=self.transmission_tokens,
                target_tokens=self.target_tokens,
                method=upsampler_type,
                num_refine_layers=upsampler_layers
            ).to(device)
        self.upsampler.load_state_dict(ckpt['upsampler_state_dict'])
        self.upsampler.eval()
        
        # 瓶颈层 (可选)
        bottleneck_dim = args.get('bottleneck_dim', 0)
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        
        if bottleneck_dim > 0 and 'bottleneck_state_dict' in ckpt:
            bottleneck_method = args.get('bottleneck_method', 'linear')
            self.bottleneck = DimensionBottleneck(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                method=bottleneck_method
            ).to(device)
            self.bottleneck.load_state_dict(ckpt['bottleneck_state_dict'])
            self.bottleneck.eval()
            print(f"   瓶颈层: {hidden_size} → {bottleneck_dim} → {hidden_size} ({bottleneck_method})")
            print(f"   传输大小 (int8): {self.bottleneck.get_transmission_size_kb(self.transmission_tokens):.2f} KB")
        else:
            self.bottleneck = None
            print(f"   无瓶颈层 (全维度传输)")
        
        print(f"✅ 模型加载完成")
        
        self.qwen_model = None
        self.processor = None
    
    def load_qwen(self):
        """加载 Qwen 模型"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print(f"📥 Loading Qwen from {self.qwen_model_name}...")
        
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.qwen_model_name,
            trust_remote_code=True
        )
        
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        self.qwen_model.eval()
        
        print(f"✅ Qwen loaded")
        print(f"📍 Will use Qwen blocks from layer {self.split_layer} onwards")
        
        return self
    
    @torch.no_grad()
    def encode_edge(self, image_tensor):
        """端侧编码: Image → 49 tokens (可选压缩)
        
        Returns:
            tokens: 原始 tokens [B, 49, 1280] 或压缩后 [B, 49, bottleneck_dim]
            compressed: 是否压缩
        """
        feat = self.student(image_tensor)[-1]
        tokens = self.projector(feat)
        
        # 如果有瓶颈层，进行压缩
        if self.bottleneck is not None:
            compressed_tokens = self.bottleneck.encode(tokens)
            return compressed_tokens, True
        return tokens, False
    
    @torch.no_grad()
    def upsample_cloud(self, edge_tokens, is_compressed=False):
        """云端上采样: tokens → 256 tokens，并按 split_layer 匹配分布

        分布 magic numbers (COCO val2017, 100 样本实测):
          layer -1 (pixel patches): mean=-0.041, std=1.015  dim=1176 (3×2×14×14)
          layer  0 (patch_embed)  : mean=-0.000, std=0.362  dim=1280
          layer  4                 : mean=-0.022, std=0.847  dim=1280
          layer  8                 : mean=-0.021, std=1.066  dim=1280
          layer 16                 : mean=-0.030, std=2.255  dim=1280
        """
        # 如果是压缩的，先解压
        if is_compressed and self.bottleneck is not None:
            edge_tokens = self.bottleneck.decode(edge_tokens)

        upsampled = self.upsampler(edge_tokens)

        # 特征缩放以匹配目标 Qwen 层的分布
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
        elif self.split_layer == 16:
            # Layer 16: mean=-0.030, std=2.255 (COCO 1000 样本实测)
            target_std, target_mean = 2.255, -0.030
            current_std = upsampled.std()
            if current_std > 0:
                upsampled = (upsampled - upsampled.mean()) / current_std * target_std + target_mean
        elif self.split_layer == 0:
            # Layer 0 (patch_embed): mean=-0.000, std=0.362
            target_std, target_mean = 0.362, -0.0001
            current_std = upsampled.std()
            if current_std > 0:
                upsampled = (upsampled - upsampled.mean()) / current_std * target_std + target_mean
        elif self.split_layer == -1:
            # Layer -1 (pixel patches): mean=-0.041, std=1.015
            # 像素空间分布不平稳，不强制归一化
            pass

        return upsampled
    
    @torch.no_grad()
    def complete_visual_encoding(self, upsampled_tokens):
        """继续 Qwen blocks → merger

        split_layer 语义:
          -1: upsampled 代表 pixel patches → 先过 patch_embed → 再过所有 blocks
           0: upsampled 代表 patch_embed 输出 → 直接从 block 0 开始
           N: upsampled 代表 block N 输出 → 从 block N 开始
        """
        if self.qwen_model is None:
            raise RuntimeError("请先调用 load_qwen()")

        visual = self.qwen_model.visual
        B = upsampled_tokens.shape[0]
        target_h = target_w = int(self.target_tokens ** 0.5)

        # 设置 grid
        grid_thw = torch.tensor([[1, target_h, target_w]] * B, dtype=torch.long).to(self.device)

        # --- layer -1: pixel patches → patch_embed ---
        if self.split_layer == -1:
            # patch_embed.proj 是 Conv3d: weight.shape = (out=1280, in_ch=3, T=2, H=14, W=14)
            proj = visual.patch_embed.proj
            in_ch = proj.weight.shape[1]   # 3
            kT    = proj.weight.shape[2]   # 2 (temporal)
            kH    = proj.weight.shape[3]   # 14
            kW    = proj.weight.shape[4]   # 14
            N_patches = upsampled_tokens.shape[1]
            # upsampled_tokens: [B, N, 1176] → [B*N, 3, 2, 14, 14]
            patches = upsampled_tokens.view(B * N_patches, in_ch, kT, kH, kW)
            patches = patches.to(proj.weight.dtype)
            hidden_states = proj(patches).squeeze(-1).squeeze(-1).squeeze(-1)  # [B*N, 1280]
            if hasattr(visual.patch_embed, 'norm') and visual.patch_embed.norm is not None:
                hidden_states = visual.patch_embed.norm(hidden_states)
            start_layer = 0
        else:
            # 转为 Qwen 需要的格式
            hidden_states = upsampled_tokens.view(-1, upsampled_tokens.shape[-1])
            hidden_states = hidden_states.to(visual.blocks[0].attn.qkv.weight.dtype)
            start_layer = self.split_layer

        # Rotary position embedding
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

        # 执行 blocks[start_layer:]
        for layer_num, blk in enumerate(visual.blocks):
            if layer_num < start_layer:
                continue
            if layer_num in visual.fullatt_block_indexes:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_window_seqlens, position_embeddings=position_embeddings)

        # 反转 window indexing
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        # Merger
        visual_tokens = visual.merger(hidden_states)
        visual_tokens = visual_tokens.unsqueeze(0) if visual_tokens.dim() == 2 else visual_tokens

        return visual_tokens
    
    @torch.no_grad()
    def generate(self, visual_tokens, prompt):
        """生成文本回复"""
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
    def generate_original(self, image_path, prompt):
        """标准 Qwen 完整推理"""
        if self.qwen_model is None:
            self.load_qwen()
            
        from qwen_vl_utils import process_vision_info
        
        # 准备消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # 生成
        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response


def get_image_transform(size=224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description='Hybrid Inference with Learned Upsampler')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/upsampler/best_model.pth',
                        help='Path to upsampler checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--split_layer', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--full_inference', action='store_true',
                        help='Run complete inference including Qwen')
    parser.add_argument('--original', action='store_true',
                        help='Run standard Qwen inference for comparison')
    parser.add_argument('--prompt', type=str, default='这张图里有什么?')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Hybrid Inference with Learned Upsampler")
    print("=" * 60)
    
    # 加载模型
    model = HybridUpsamplerVLM(
        checkpoint_path=args.checkpoint,
        split_layer=args.split_layer,
        device=device
    )
    
    # 加载图像
    transform = get_image_transform(224)
    img = Image.open(args.image).convert('RGB')
    image = transform(img).unsqueeze(0).to(device)
    print(f"\nLoading image: {args.image}")
    
    # 端侧编码
    print("\n📡 Edge Encoding...")
    edge_tokens, is_compressed = model.encode_edge(image)
    
    if is_compressed:
        print(f"   输出 (压缩): {edge_tokens.shape} tokens")
        edge_size_kb = edge_tokens.numel() * 1 / 1024  # int8
    else:
        print(f"   输出 (未压缩): {edge_tokens.shape} tokens")
        edge_size_kb = edge_tokens.numel() * 1 / 1024  # int8
    
    print(f"   传输大小 (int8): {edge_size_kb:.2f} KB")
    
    # 云端上采样
    print("\n☁️  云端上采样...")
    upsampled_tokens = model.upsample_cloud(edge_tokens, is_compressed=is_compressed)
    print(f"   上采样: {edge_tokens.shape[1]} → {upsampled_tokens.shape[1]} tokens")
    
    if args.full_inference:
        print("\n☁️  云端完成推理 (Hybrid)...")
        if model.qwen_model is None:
            model.load_qwen()
        
        visual_tokens = model.complete_visual_encoding(upsampled_tokens)
        print(f"   视觉 tokens: {visual_tokens.shape}")
        
        print(f"\n💬 生成回复 (Hybrid, prompt: {args.prompt})")
        response = model.generate(visual_tokens, args.prompt)
        
        print(f"\n🤖 Hybrid 回复:")
        print(response)

    if args.original:
        print("\n🚀 运行标准 Qwen 推理 (Original)...")
        response_ori = model.generate_original(args.image, args.prompt)
        print(f"\n🤖 Original Qwen 回复:")
        print(response_ori)
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
