"""
SplitOculo Edge Client

端侧客户端：编码图像并发送压缩特征到云端服务器。

Usage:
    python scripts/edge_client.py \
        --checkpoint ./checkpoints/gan_bottleneck/gan_best.pth \
        --image ./test.jpg \
        --server http://localhost:8080

特点:
- 仅加载端侧模型 (CNN + Projector + Bottleneck.encoder)
- 不需要 Qwen 模型
- 发送 ~3KB 压缩特征到云端
"""
import argparse
import sys
from pathlib import Path
import base64
import time
import requests
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import timm
from PIL import Image
from torchvision import transforms
import math

from models.strided_projector import StridedTokenProjector
from models.bottleneck import DimensionBottleneck
from models.importance_scorer import TokenImportanceScorer
from models.budgeted_transmission import SoftBudgetedTransmission


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


class EdgeEncoder:
    """端侧编码器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        print(f"📱 Loading edge components from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        args = ckpt.get('args', {})
        
        self.transmission_tokens = args.get('transmission_tokens', 49)
        hidden_size = args.get('target_hidden_size', 1280)
        self.hidden_size = hidden_size
        
        # CNN backbone
        student_model = args.get('student_model', 'mobilenetv2_100')
        student_layer = args.get('student_layer', 3)
        
        self.student = timm.create_model(
            student_model, pretrained=False, features_only=True,
            out_indices=[student_layer]
        ).to(device)
        self.student.load_state_dict(ckpt['student_state_dict'])
        self.student.eval()
        
        # 获取通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            student_channels = self.student(dummy)[-1].shape[1]
        
        print(f"   CNN: {student_model} → {student_channels} channels")
        
        # Projector
        projector_type = args.get('projector_type', 'pooling')
        if projector_type == 'strided':
            self.projector = StridedTokenProjector(
                in_channels=student_channels,
                hidden_size=hidden_size,
                hidden_channels=args.get('projector_hidden', 512),
                transmission_tokens=self.transmission_tokens
            ).to(device)
            print("   Strided token projector")
        else:
            self.projector = EdgeProjector(
                in_channels=student_channels,
                hidden_size=hidden_size,
                hidden_channels=args.get('projector_hidden', 512),
                transmission_tokens=self.transmission_tokens
            ).to(device)
            print("   Pooling token projector")
        
        self.projector.load_state_dict(ckpt['projector_state_dict'])
        self.projector.eval()
        
        # Bottleneck encoder (只需要 encoder 部分)
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
            if 'bottleneck_encoder_state_dict' in ckpt:
                # 拆分权重: 只有 encoder 部分，需要去掉 'encoder.' 前缀
                encoder_sd = {k.replace('encoder.', ''): v for k, v in ckpt['bottleneck_encoder_state_dict'].items()}
                self.bottleneck.encoder.load_state_dict(encoder_sd)
                print(f"   Bottleneck encoder (split): {hidden_size} → {bottleneck_dim}")
            elif 'bottleneck_state_dict' in ckpt:
                # AIO 权重: 完整 bottleneck
                self.bottleneck.load_state_dict(ckpt['bottleneck_state_dict'])
                print(f"   Bottleneck encoder (AIO): {hidden_size} → {bottleneck_dim}")
            else:
                print(f"   ⚠️ No bottleneck weights found, using random init")
            
            self.bottleneck.eval()
        else:
            self.bottleneck = None
            print(f"   No bottleneck (full dimension)")
        
        # Importance-aware sparse transmission (optional)
        self.importance_aware = args.get('importance_aware', False)
        self.importance_scorer = None
        self.budgeted_transmission = None
        
        if self.importance_aware:
            scorer_method = args.get('scorer_method', 'mlp')
            self.importance_scorer = TokenImportanceScorer(
                hidden_size=hidden_size,
                method=scorer_method
            ).to(device)
            
            token_budget = args.get('token_budget', 24)
            min_tokens = args.get('min_tokens', 8)
            self.budgeted_transmission = SoftBudgetedTransmission(
                max_tokens=self.transmission_tokens,
                target_budget=token_budget,
                min_tokens=min_tokens
            ).to(device)
            
            # Load state dicts
            if 'importance_scorer_state_dict' in ckpt:
                self.importance_scorer.load_state_dict(ckpt['importance_scorer_state_dict'])
                print(f"   Importance scorer ({scorer_method}): loaded")
            else:
                print(f"   Importance scorer ({scorer_method}): no weights found, using random init")
            
            if 'budgeted_transmission_state_dict' in ckpt:
                self.budgeted_transmission.load_state_dict(ckpt['budgeted_transmission_state_dict'])
                print(f"   Budgeted transmission: budget={token_budget}, min={min_tokens}, loaded")
            else:
                print(f"   Budgeted transmission: budget={token_budget}, min={min_tokens}, no weights found")
            
            self.importance_scorer.eval()
            self.budgeted_transmission.eval()
            print(f"   Importance-aware sparse encoding: ENABLED")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"✅ Edge encoder loaded")
    
    @torch.no_grad()
    def encode(self, image_path):
        """
        编码图像为压缩特征
        
        Args:
            image_path: 图像路径
        
        Returns:
            features: 压缩特征 tensor
            is_compressed: 是否经过瓶颈层压缩
            indices: 选中的 token 索引 (importance-aware 时), 否则 None
        """
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # CNN + Projector
        feat = self.student(image_tensor)[-1]
        tokens = self.projector(feat)
        
        # Importance-aware sparse token selection (before bottleneck)
        if self.importance_aware:
            importance_logits = self.importance_scorer(tokens)
            selected_tokens, indices, _, _ = self.budgeted_transmission(tokens, importance_logits)
            
            # Bottleneck encode the selected tokens
            if self.bottleneck is not None:
                compressed = self.bottleneck.encode(selected_tokens)
                return compressed, True, indices
            
            return selected_tokens, False, indices
        
        # Standard path (no importance scoring)
        if self.bottleneck is not None:
            compressed = self.bottleneck.encode(tokens)
            return compressed, True, None
        
        return tokens, False, None
    
    def quantize_int8(self, features):
        """
        int8 量化
        
        Args:
            features: [B, N, C] 浮点特征
        
        Returns:
            quantized: int8 数组
            scale: 缩放因子
            zero_point: 零点
        """
        features_np = features.cpu().numpy()
        
        # 计算量化参数
        f_min, f_max = features_np.min(), features_np.max()
        scale = (f_max - f_min) / 255.0
        zero_point = -f_min / scale
        
        # 量化
        quantized = np.clip(np.round(features_np / scale + zero_point), 0, 255).astype(np.uint8)
        
        return quantized, float(scale), float(zero_point)
    
    def encode_to_payload(self, image_path):
        """
        完整编码流水线：图像 → base64 编码的压缩特征
        
        Returns:
            payload: dict 包含 features, scale, zero_point (及 indices, num_selected if importance-aware)
            stats: dict 包含统计信息
        """
        start_time = time.time()
        
        # 编码
        features, is_compressed, indices = self.encode(image_path)
        encode_time = time.time() - start_time
        
        # 量化
        quantized, scale, zero_point = self.quantize_int8(features)
        
        # base64 编码
        features_b64 = base64.b64encode(quantized.tobytes()).decode('ascii')
        
        payload = {
            'features': features_b64,
            'scale': scale,
            'zero_point': zero_point
        }
        
        stats = {
            'encode_time_ms': encode_time * 1000,
            'feature_shape': list(features.shape),
            'payload_bytes': len(payload['features']),
            'is_compressed': is_compressed
        }
        
        # Add sparse token indices if importance-aware
        if indices is not None:
            payload['indices'] = indices.squeeze(0).cpu().tolist()
            payload['num_selected'] = len(payload['indices'])
            stats['num_selected'] = payload['num_selected']
            stats['total_tokens'] = self.transmission_tokens
        
        return payload, stats


def main():
    parser = argparse.ArgumentParser(description='SplitOculo Edge Client')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--server', type=str, default='http://localhost:8080',
                        help='Cloud server URL')
    parser.add_argument('--prompt', type=str, default='这张图里有什么?',
                        help='Prompt for the model')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Request timeout in seconds (default: 300)')
    parser.add_argument('--importance_aware', action='store_true',
                        help='Display importance-aware info (actual behavior is checkpoint-driven)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SplitOculo Edge Client")
    print("=" * 60)
    
    # 加载端侧编码器
    encoder = EdgeEncoder(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 编码图像
    print(f"\n📷 Encoding image: {args.image}")
    payload, stats = encoder.encode_to_payload(args.image)
    payload['prompt'] = args.prompt
    
    print(f"   Feature shape: {stats['feature_shape']}")
    print(f"   Compressed: {stats['is_compressed']}")
    if 'num_selected' in stats:
        print(f"   Importance-aware: {stats['num_selected']}/{stats['total_tokens']} tokens selected "
              f"({stats['num_selected']/stats['total_tokens']*100:.0f}%)")
        bandwidth_saving = (1 - stats['num_selected'] / stats['total_tokens']) * 100
        print(f"   Bandwidth saving: {bandwidth_saving:.0f}%")
    print(f"   Encode time: {stats['encode_time_ms']:.2f} ms")
    print(f"   Payload size: {stats['payload_bytes']} bytes ({stats['payload_bytes']/1024:.2f} KB)")
    
    # 发送到云端
    print(f"\n🌐 Sending to cloud server: {args.server}")
    print(f"   Timeout: {args.timeout}s")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{args.server}/infer",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=args.timeout
        )
        network_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Network round-trip: {network_time*1000:.2f} ms")
            print(f"   Cloud inference: {result.get('latency_ms', 0):.2f} ms")
            
            print(f"\n💬 Response:")
            print("-" * 40)
            print(result['response'])
            print("-" * 40)
            
            # 总结
            total_time = network_time * 1000
            print(f"\n📊 Summary:")
            print(f"   Edge encode: {stats['encode_time_ms']:.2f} ms")
            print(f"   Transmission: {stats['payload_bytes']/1024:.2f} KB")
            if 'num_selected' in stats:
                print(f"   Sparse tokens: {stats['num_selected']}/{stats['total_tokens']}")
            print(f"   Total latency: {total_time:.2f} ms")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {args.server}")
        print("   Make sure the cloud server is running.")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
