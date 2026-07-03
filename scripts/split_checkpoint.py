"""
拆分 AIO 检查点为端侧和云端权重

Usage:
    python scripts/split_checkpoint.py \
        --input ./checkpoints/gan_bottleneck/gan_best.pth \
        --output_dir ./checkpoints/split/

输出:
    - edge_weights.pth: CNN + Projector + Bottleneck.encoder
    - cloud_weights.pth: Bottleneck.decoder + Upsampler
"""
import argparse
import sys
from pathlib import Path
from collections import OrderedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def split_bottleneck_weights(bottleneck_state_dict, method='linear'):
    """
    将瓶颈层权重拆分为 encoder 和 decoder
    
    线性瓶颈结构:
        encoder: nn.Linear(hidden_size, bottleneck_dim)
        decoder: nn.Linear(bottleneck_dim, hidden_size)
    
    MLP/Autoencoder 结构:
        encoder: nn.Sequential(...)
        decoder: nn.Sequential(...)
    """
    encoder_weights = OrderedDict()
    decoder_weights = OrderedDict()
    
    for key, value in bottleneck_state_dict.items():
        if key.startswith('encoder.'):
            encoder_weights[key] = value
        elif key.startswith('decoder.'):
            decoder_weights[key] = value
    
    return encoder_weights, decoder_weights


def main():
    parser = argparse.ArgumentParser(description='Split AIO checkpoint into edge and cloud weights')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to AIO checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for split weights')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Split AIO Checkpoint")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output dir: {output_dir}")
    
    # 加载原始检查点
    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    original_args = ckpt.get('args', {})
    
    print(f"\n📦 Original checkpoint:")
    print(f"   Epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"   Bottleneck dim: {original_args.get('bottleneck_dim', 0)}")
    print(f"   Bottleneck method: {original_args.get('bottleneck_method', 'linear')}")
    
    # 拆分瓶颈层权重
    has_bottleneck = 'bottleneck_state_dict' in ckpt
    if has_bottleneck:
        encoder_weights, decoder_weights = split_bottleneck_weights(
            ckpt['bottleneck_state_dict'],
            method=original_args.get('bottleneck_method', 'linear')
        )
        print(f"   Bottleneck encoder keys: {len(encoder_weights)}")
        print(f"   Bottleneck decoder keys: {len(decoder_weights)}")
    
    # 创建端侧权重
    edge_checkpoint = {
        'student_state_dict': ckpt['student_state_dict'],
        'projector_state_dict': ckpt['projector_state_dict'],
        'args': {
            'student_model': original_args.get('student_model', 'mobilenetv2_100'),
            'student_layer': original_args.get('student_layer', 3),
            'projector_type': original_args.get('projector_type', 'strided'),
            'projector_hidden': original_args.get('projector_hidden', 512),
            'transmission_tokens': original_args.get('transmission_tokens', 49),
            'target_hidden_size': original_args.get('target_hidden_size', 1280),
            'bottleneck_dim': original_args.get('bottleneck_dim', 0),
            'bottleneck_method': original_args.get('bottleneck_method', 'linear'),
            'multilevel_payload': original_args.get('multilevel_payload', False),
            'payload_levels': original_args.get('payload_levels'),
            'image_size': original_args.get('image_size', 224),
            'data_dir': original_args.get('data_dir'),
            'features_dir': original_args.get('features_dir'),
        }
    }
    
    if has_bottleneck:
        edge_checkpoint['bottleneck_encoder_state_dict'] = encoder_weights
    
    # 创建云端权重
    cloud_checkpoint = {
        'upsampler_state_dict': ckpt['upsampler_state_dict'],
        'args': {
            'upsampler_type': original_args.get('upsampler_type', 'transformer'),
            'transformer_layers': original_args.get('transformer_layers', 4),
            'initial_upsample': original_args.get('initial_upsample', 'bilinear'),
            'transmission_tokens': original_args.get('transmission_tokens', 49),
            'target_tokens': original_args.get('target_tokens', 256),
            'target_hidden_size': original_args.get('target_hidden_size', 1280),
            'bottleneck_dim': original_args.get('bottleneck_dim', 0),
            'bottleneck_method': original_args.get('bottleneck_method', 'linear'),
            'multilevel_payload': original_args.get('multilevel_payload', False),
            'payload_levels': original_args.get('payload_levels'),
            'image_size': original_args.get('image_size', 224),
            'data_dir': original_args.get('data_dir'),
            'features_dir': original_args.get('features_dir'),
        }
    }
    
    if has_bottleneck:
        cloud_checkpoint['bottleneck_decoder_state_dict'] = decoder_weights
    
    # 可选：保留 discriminator (用于继续训练)
    if 'discriminator_state_dict' in ckpt:
        cloud_checkpoint['discriminator_state_dict'] = ckpt['discriminator_state_dict']
    
    # 保存
    edge_path = output_dir / 'edge_weights.pth'
    cloud_path = output_dir / 'cloud_weights.pth'
    
    torch.save(edge_checkpoint, edge_path)
    torch.save(cloud_checkpoint, cloud_path)
    
    # 统计大小
    edge_size = edge_path.stat().st_size / 1024 / 1024
    cloud_size = cloud_path.stat().st_size / 1024 / 1024
    original_size = Path(args.input).stat().st_size / 1024 / 1024
    
    print(f"\n✅ Split complete!")
    print(f"\n📊 File sizes:")
    print(f"   Original: {original_size:.2f} MB")
    print(f"   Edge:     {edge_size:.2f} MB ({edge_path})")
    print(f"   Cloud:    {cloud_size:.2f} MB ({cloud_path})")
    
    # 参数量统计
    edge_params = sum(v.numel() for v in edge_checkpoint['student_state_dict'].values())
    edge_params += sum(v.numel() for v in edge_checkpoint['projector_state_dict'].values())
    if has_bottleneck:
        edge_params += sum(v.numel() for v in edge_checkpoint['bottleneck_encoder_state_dict'].values())
    
    cloud_params = sum(v.numel() for v in cloud_checkpoint['upsampler_state_dict'].values())
    if has_bottleneck:
        cloud_params += sum(v.numel() for v in cloud_checkpoint['bottleneck_decoder_state_dict'].values())
    
    print(f"\n📈 Parameters:")
    print(f"   Edge:  {edge_params:,} ({edge_params/1e6:.2f}M)")
    print(f"   Cloud: {cloud_params:,} ({cloud_params/1e6:.2f}M)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
