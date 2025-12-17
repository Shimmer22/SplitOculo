"""
使用可学习上采样器的混合推理脚本

架构:
- 端侧: Image → CNN → Projector → 49 tokens (7×7)
- 传输: 49 tokens (~61 KB int8)
- 云端: Upsampler → 256 tokens → Qwen[4:] → Merger → LLM

Usage:
    python scripts/infer_with_upsampler.py \
        --checkpoint checkpoints/upsampler/best_model.pth \
        --image photo.jpg \
        --full_inference
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
        self.projector = EdgeProjector(
            in_channels=student_channels,
            hidden_size=hidden_size,
            hidden_channels=args.get('projector_hidden', 512),
            transmission_tokens=self.transmission_tokens
        ).to(device)
        self.projector.load_state_dict(ckpt['projector_state_dict'])
        self.projector.eval()
        
        # 云端 Upsampler
        self.upsampler = CloudUpsampler(
            hidden_size=hidden_size,
            input_tokens=self.transmission_tokens,
            target_tokens=self.target_tokens,
            method=upsampler_method,
            num_refine_layers=upsampler_layers
        ).to(device)
        self.upsampler.load_state_dict(ckpt['upsampler_state_dict'])
        self.upsampler.eval()
        
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
        """端侧编码: Image → 49 tokens"""
        feat = self.student(image_tensor)[-1]
        tokens = self.projector(feat)
        return tokens
    
    @torch.no_grad()
    def upsample_cloud(self, edge_tokens):
        """云端上采样: 49 tokens → 256 tokens"""
        upsampled = self.upsampler(edge_tokens)
        
        # 特征缩放以匹配 Qwen Layer 4 分布
        # Qwen Layer 4: mean≈-0.017, std≈0.83
        target_std = 0.83
        target_mean = -0.017
        
        current_std = upsampled.std()
        if current_std > 0:
            upsampled = (upsampled - upsampled.mean()) / current_std * target_std + target_mean
        
        return upsampled
    
    @torch.no_grad()
    def complete_visual_encoding(self, upsampled_tokens):
        """继续 Qwen blocks → merger"""
        if self.qwen_model is None:
            raise RuntimeError("请先调用 load_qwen()")
        
        visual = self.qwen_model.visual
        B = upsampled_tokens.shape[0]
        target_h = target_w = int(self.target_tokens ** 0.5)
        
        # 设置 grid
        grid_thw = torch.tensor([[1, target_h, target_w]] * B, dtype=torch.long).to(self.device)
        
        # 转为 Qwen 需要的格式
        hidden_states = upsampled_tokens.view(-1, upsampled_tokens.shape[-1])
        hidden_states = hidden_states.to(visual.blocks[0].attn.qkv.weight.dtype)
        
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
        
        # 执行 blocks[split_layer:]
        for layer_num, blk in enumerate(visual.blocks):
            if layer_num < self.split_layer:
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
    parser.add_argument('--prompt', type=str, default='What is in this image?')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print("🔧 Hybrid Inference with Learned Upsampler")
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
    print(f"\n📷 加载图像: {args.image}")
    
    # 端侧编码
    print("\n🖥️  端侧 (Edge) 编码...")
    edge_tokens = model.encode_edge(image)
    print(f"   输出: {edge_tokens.shape} tokens")
    
    # 云端上采样
    print("\n☁️  云端上采样...")
    upsampled_tokens = model.upsample_cloud(edge_tokens)
    print(f"   上采样: {edge_tokens.shape[1]} → {upsampled_tokens.shape[1]} tokens")
    
    # 计算大小
    edge_size_kb = edge_tokens.numel() * 1 / 1024  # int8
    upsampled_size_kb = upsampled_tokens.numel() * 1 / 1024
    print(f"   传输大小 (int8): {edge_size_kb:.2f} KB")
    
    if args.full_inference:
        print("\n☁️  云端完成推理...")
        model.load_qwen()
        
        visual_tokens = model.complete_visual_encoding(upsampled_tokens)
        print(f"   视觉 tokens: {visual_tokens.shape}")
        
        print(f"\n💬 生成回复 (prompt: {args.prompt})")
        response = model.generate(visual_tokens, args.prompt)
        
        print(f"\n🤖 回复:")
        print(response)
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
