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
    
    # 检测 importance-aware 模式
    importance_aware = original_args.get('importance_aware', False)
    if importance_aware:
        print(f"\n   [Importance-Aware Mode Detected]")
        print(f"   Scorer method: {original_args.get('scorer_method', 'mlp')}")
        print(f"   Token budget: {original_args.get('token_budget', 24)}")
        print(f"   Min tokens: {original_args.get('min_tokens', 8)}")
        print(f"   Completion layers: {original_args.get('completion_layers', 2)}")
    
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
        }
    }
    
    if has_bottleneck:
        edge_checkpoint['bottleneck_encoder_state_dict'] = encoder_weights
    
    # 添加 importance-aware 端侧模块
    if importance_aware:
        if 'importance_scorer_state_dict' in ckpt:
            edge_checkpoint['importance_scorer_state_dict'] = ckpt['importance_scorer_state_dict']
        # Backward/forward compatibility:
        # trainer saves "budgeted_tx_state_dict", older code may use "budgeted_transmission_state_dict".
        if 'budgeted_tx_state_dict' in ckpt:
            edge_checkpoint['budgeted_tx_state_dict'] = ckpt['budgeted_tx_state_dict']
            edge_checkpoint['budgeted_transmission_state_dict'] = ckpt['budgeted_tx_state_dict']
        elif 'budgeted_transmission_state_dict' in ckpt:
            edge_checkpoint['budgeted_tx_state_dict'] = ckpt['budgeted_transmission_state_dict']
            edge_checkpoint['budgeted_transmission_state_dict'] = ckpt['budgeted_transmission_state_dict']
        edge_checkpoint['args']['importance_aware'] = original_args.get('importance_aware', False)
        edge_checkpoint['args']['scorer_method'] = original_args.get('scorer_method', 'mlp')
        edge_checkpoint['args']['token_budget'] = original_args.get('token_budget', 24)
        edge_checkpoint['args']['min_tokens'] = original_args.get('min_tokens', 8)
    
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
        }
    }
    
    if has_bottleneck:
        cloud_checkpoint['bottleneck_decoder_state_dict'] = decoder_weights
    
    # 添加 importance-aware 云端模块
    if importance_aware:
        if 'sparse_upsampler_state_dict' in ckpt:
            cloud_checkpoint['sparse_upsampler_state_dict'] = ckpt['sparse_upsampler_state_dict']
        cloud_checkpoint['args']['importance_aware'] = original_args.get('importance_aware', False)
        cloud_checkpoint['args']['completion_layers'] = original_args.get('completion_layers', 2)
    
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
    if importance_aware:
        if 'importance_scorer_state_dict' in edge_checkpoint:
            edge_params += sum(v.numel() for v in edge_checkpoint['importance_scorer_state_dict'].values())
        if 'budgeted_transmission_state_dict' in edge_checkpoint:
            edge_params += sum(v.numel() for v in edge_checkpoint['budgeted_transmission_state_dict'].values())
    
    cloud_params = sum(v.numel() for v in cloud_checkpoint['upsampler_state_dict'].values())
    if has_bottleneck:
        cloud_params += sum(v.numel() for v in cloud_checkpoint['bottleneck_decoder_state_dict'].values())
    if importance_aware:
        if 'sparse_upsampler_state_dict' in cloud_checkpoint:
            cloud_params += sum(v.numel() for v in cloud_checkpoint['sparse_upsampler_state_dict'].values())
    
    print(f"\n📈 Parameters:")
    print(f"   Edge:  {edge_params:,} ({edge_params/1e6:.2f}M)")
    print(f"   Cloud: {cloud_params:,} ({cloud_params/1e6:.2f}M)")
    
    if importance_aware:
        print(f"\n🎯 Importance-Aware modules:")
        scorer_keys = len(edge_checkpoint.get('importance_scorer_state_dict', {}))
        budget_keys = len(edge_checkpoint.get('budgeted_transmission_state_dict', {}))
        sparse_keys = len(cloud_checkpoint.get('sparse_upsampler_state_dict', {}))
        print(f"   Edge  - importance_scorer keys: {scorer_keys}")
        print(f"   Edge  - budgeted_transmission keys: {budget_keys}")
        print(f"   Cloud - sparse_upsampler keys: {sparse_keys}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
