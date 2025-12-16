"""
混合推理脚本：CNN 替换 Qwen ViT 浅层，剩余深层仍用 Qwen

端云协同场景:
- 端侧: Image → CNN → Projector → features (1280 dim, 可量化)
- 云端: features → Remaining Qwen Blocks → Merger → LLM → response

Usage:
    python scripts/infer_hybrid.py --checkpoint checkpoints/best_model.pth --image photo.jpg
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms


class LLMProjector(nn.Module):
    """将 CNN 特征投影到 ViT 隐藏空间 (1280 dim)"""
    def __init__(self, in_channels, hidden_size=1280, 
                 hidden_channels=512, downsample_ratio=2):
        super().__init__()
        
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_ratio, stride=downsample_ratio)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_channels, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.avg_pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_tokens, hidden_size)
        return x


class EdgeVisualEncoder(nn.Module):
    """
    端侧视觉编码器
    
    CNN (MobileNetV2) + Projector → features (1280 dim)
    输出与 Qwen ViT 中间层兼容
    """
    def __init__(self, student_model='mobilenetv2_100', student_layer=3,
                 student_channels=96, hidden_size=1280,
                 projector_hidden=512, downsample_ratio=2):
        super().__init__()
        
        self.student = timm.create_model(
            student_model,
            pretrained=False,
            features_only=True,
            out_indices=[student_layer]
        )
        
        self.projector = LLMProjector(
            in_channels=student_channels,
            hidden_size=hidden_size,
            hidden_channels=projector_hidden,
            downsample_ratio=downsample_ratio
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            (B, num_tokens, 1280) features compatible with Qwen ViT
        """
        feat = self.student(x)[-1]
        tokens = self.projector(feat)
        return tokens
    
    def load_checkpoint(self, checkpoint_path, device='cpu'):
        """加载训练好的权重"""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.student.load_state_dict(ckpt['student_state_dict'])
        self.projector.load_state_dict(ckpt['projector_state_dict'])
        print(f"✅ 已加载检查点: {checkpoint_path}")
        return self


class HybridQwenVLM:
    """
    混合视觉语言模型
    
    CNN + Projector 替换 Qwen ViT 前 N 层
    剩余层 + Merger 仍使用 Qwen
    
    架构:
        Image → CNN → Projector (1280) → Qwen Blocks[N:] → Merger (2048) → LLM
    """
    def __init__(self, 
                 edge_checkpoint=None,
                 qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 split_layer=8,
                 device='cuda'):
        self.device = device
        self.qwen_model_name = qwen_model_name
        self.split_layer = split_layer
        
        # 端侧编码器
        self.edge_encoder = EdgeVisualEncoder(
            student_model='mobilenetv2_100',
            student_layer=3,
            student_channels=96,
            hidden_size=1280
        )
        
        if edge_checkpoint:
            self.edge_encoder.load_checkpoint(edge_checkpoint, device)
        
        self.edge_encoder = self.edge_encoder.to(device)
        self.edge_encoder.eval()
        
        # Qwen 模型 (延迟加载)
        self.qwen_model = None
        self.processor = None
    
    def load_qwen(self):
        """加载 Qwen 模型 (云端)"""
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
        
        # 冻结参数
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        self.qwen_model.eval()
        
        print(f"✅ Qwen loaded")
        print(f"📍 Will use Qwen blocks from layer {self.split_layer} onwards")
        
        return self
    
    @torch.no_grad()
    def encode_image_edge(self, image_tensor):
        """
        端侧编码
        
        Args:
            image_tensor: (B, 3, H, W) normalized tensor
        Returns:
            features: (B, num_tokens, 1280)
        """
        return self.edge_encoder(image_tensor)
    
    @torch.no_grad() 
    def complete_visual_encoding(self, edge_features, grid_thw=None):
        """
        云端完成剩余视觉编码
        
        Args:
            edge_features: (B, num_tokens, 1280) 端侧输出
            grid_thw: grid 信息 (可选)
        Returns:
            visual_tokens: (B, merged_tokens, 2048) 可送入 LLM
        """
        if self.qwen_model is None:
            raise RuntimeError("请先调用 load_qwen()")
        
        visual = self.qwen_model.visual
        
        # 如果没有 grid_thw，从 224x224 图像推断
        if grid_thw is None:
            # 224/14 = 16 patches per dim, t=1 for static image
            B = edge_features.shape[0]
            grid_thw = torch.tensor([[1, 16, 16]] * B, dtype=torch.long).to(self.device)
        
        # Rotary position embedding
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        
        # cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        
        # 继续执行剩余的 blocks
        hidden_states = edge_features.to(visual.blocks[0].attn.qkv.weight.dtype)
        
        for i, block in enumerate(visual.blocks):
            if i < self.split_layer:
                continue  # 跳过前 N 层 (已由 CNN 替代)
            
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb
            )
        
        # 通过 merger
        visual_tokens = visual.merger(hidden_states)
        
        return visual_tokens  # (B, merged_tokens, 2048)
    
    def quantize_features(self, features, method='int8'):
        """
        量化特征以减少传输大小
        
        Args:
            features: (B, num_tokens, hidden_size) float tensor
            method: 'int8', 'fp16'
        Returns:
            quantized_data, metadata for dequantization
        """
        if method == 'int8':
            # 简单的 min-max int8 量化
            min_val = features.min()
            max_val = features.max()
            scale = (max_val - min_val) / 255
            quantized = ((features - min_val) / scale).round().to(torch.uint8)
            return quantized, {'min': min_val.item(), 'scale': scale.item()}
        elif method == 'fp16':
            return features.half(), {}
        else:
            return features, {}
    
    def dequantize_features(self, quantized, metadata, method='int8'):
        """反量化"""
        if method == 'int8':
            return quantized.float() * metadata['scale'] + metadata['min']
        elif method == 'fp16':
            return quantized.float()
        else:
            return quantized


def get_image_transform(image_size=224):
    """图像预处理"""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description='Hybrid CNN-Qwen Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/qwen_precomputed/best_model.pth')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--split_layer', type=int, default=8,
                        help='Which layer CNN replaces (1-32)')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--quantize', type=str, default='int8',
                        choices=['none', 'fp16', 'int8'],
                        help='Quantization method for transmission')
    parser.add_argument('--full_inference', action='store_true',
                        help='Run complete inference including Qwen deep layers')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"🔧 Device: {device}")
    print(f"📍 Split layer: {args.split_layer}")
    
    # 加载混合模型
    hybrid = HybridQwenVLM(
        edge_checkpoint=args.checkpoint if Path(args.checkpoint).exists() else None,
        split_layer=args.split_layer,
        device=device
    )
    
    # 准备输入
    transform = get_image_transform(224)
    
    if args.dummy:
        print("📷 使用假数据测试...")
        image = torch.randn(1, 3, 224, 224).to(device)
    elif args.image:
        print(f"📷 加载图像: {args.image}")
        img = Image.open(args.image).convert('RGB')
        image = transform(img).unsqueeze(0).to(device)
    else:
        print("❌ 请指定 --image 或 --dummy")
        return
    
    # 端侧编码
    print("\n🖥️  端侧 (Edge) 编码...")
    edge_features = hybrid.encode_image_edge(image)
    print(f"   输出形状: {edge_features.shape}")
    print(f"   维度: {edge_features.shape[-1]} (应为 1280)")
    
    # 量化分析
    print(f"\n📡 传输大小分析 ({args.quantize} 量化):")
    quantized, metadata = hybrid.quantize_features(edge_features, method=args.quantize)
    
    if args.quantize == 'int8':
        bytes_per_val = 1
    elif args.quantize == 'fp16':
        bytes_per_val = 2
    else:
        bytes_per_val = 4
    
    transmission_bytes = quantized.numel() * bytes_per_val
    print(f"   原始 JPEG 224x224: ~30 KB")
    print(f"   量化特征: {transmission_bytes / 1024:.2f} KB")
    print(f"   压缩比: {30 * 1024 / transmission_bytes:.1f}x" if transmission_bytes < 30 * 1024 else f"   大于原图 {transmission_bytes / 1024 / 30:.1f}x")
    
    # 完整推理 (可选)
    if args.full_inference:
        print("\n☁️  云端 (Cloud) 完成推理...")
        hybrid.load_qwen()
        
        # 反量化
        dequantized = hybrid.dequantize_features(quantized, metadata, method=args.quantize)
        
        # 完成视觉编码
        visual_tokens = hybrid.complete_visual_encoding(dequantized)
        print(f"   视觉 tokens: {visual_tokens.shape}")
        print(f"   维度: {visual_tokens.shape[-1]} (应为 2048)")
        
        return visual_tokens
    
    return edge_features


if __name__ == '__main__':
    main()
