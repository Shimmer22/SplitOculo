"""
测量预计算 Qwen 特征的统计分布 (mean / std)

用于获取推理时特征归一化所需的 magic number。

Usage:
    # 测量已有特征目录 (layer 4)
    python scripts/measure_feature_stats.py \\
        --features_dir ./data/coco_features_layer4 --split train --max_files 200

    # 实时提取并测量 (layer 8 / layer 0 / layer -1)
    python scripts/measure_feature_stats.py \\
        --data_dir ./data/coco --layer 8 --split train --max_files 100 --realtime

    # 测量所有实验层级 (realtime)
    python scripts/measure_feature_stats.py \\
        --data_dir ./data/coco --split train --max_files 100 --realtime --all_layers
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm


def measure_from_precomputed(features_dir, split='train', max_files=200):
    """从预计算的 .pt 文件测量分布统计"""
    feat_dir = Path(features_dir) / split
    files = sorted(feat_dir.glob("*.pt"))[:max_files]
    
    if not files:
        raise ValueError(f"No .pt files found in {feat_dir}")
    
    print(f"Measuring from {len(files)} files in {feat_dir}")
    
    all_means = []
    all_stds = []
    all_features = []
    
    for f in tqdm(files, desc="Loading"):
        data = torch.load(f, weights_only=False)
        feat = data['features'].float()  # (N_tokens, hidden_size)
        all_means.append(feat.mean().item())
        all_stds.append(feat.std().item())
        all_features.append(feat)
    
    # 全局统计
    all_cat = torch.cat(all_features, dim=0)
    global_mean = all_cat.mean().item()
    global_std = all_cat.std().item()
    per_file_mean = np.mean(all_means)
    per_file_std = np.mean(all_stds)
    
    # 读取元数据
    meta_path = feat_dir / 'metadata.json'
    layer = 'unknown'
    if meta_path.exists():
        import json
        with open(meta_path) as f_:
            meta = json.load(f_)
            layer = meta.get('extract_layer', 'unknown')
    
    print(f"\n{'='*50}")
    print(f"Layer {layer} Distribution Stats ({len(files)} files, {all_cat.shape[0]} tokens total)")
    print(f"  Global mean : {global_mean:.6f}")
    print(f"  Global std  : {global_std:.6f}")
    print(f"  Per-file mean (avg): {per_file_mean:.6f}")
    print(f"  Per-file std  (avg): {per_file_std:.6f}")
    print(f"  Feature shape: {data['features'].shape}")
    print(f"{'='*50}\n")
    
    return global_mean, global_std


def measure_realtime(data_dir, layer, split='train', max_files=100,
                     qwen_model="Qwen/Qwen2.5-VL-3B-Instruct", device='cuda'):
    """实时提取并测量分布统计"""
    from core.qwen_extractor import QwenFeatureExtractor
    from PIL import Image
    from torchvision import transforms

    images_dir = Path(data_dir) / split
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ])[:max_files]
    
    print(f"Extracting layer={layer} from {len(image_files)} images...")
    
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    
    extractor = QwenFeatureExtractor(
        model_name=qwen_model,
        device=device,
        extract_layer=layer
    ).load()
    
    all_features = []
    for img_path in tqdm(image_files, desc=f"Layer {layer}"):
        try:
            pil_img = Image.open(img_path).convert('RGB')
            pil_img = transform(pil_img)
            feat = extractor.extract_features(pil_img).float()
            all_features.append(feat)
        except Exception as e:
            print(f"  Error: {img_path}: {e}")
    
    all_cat = torch.cat(all_features, dim=0)
    global_mean = all_cat.mean().item()
    global_std = all_cat.std().item()
    
    print(f"\n{'='*50}")
    print(f"Layer {layer} Distribution Stats ({len(all_features)} images, {all_cat.shape[0]} tokens)")
    print(f"  Global mean : {global_mean:.6f}")
    print(f"  Global std  : {global_std:.6f}")
    print(f"  Feature dim : {all_cat.shape[1]}")
    print(f"{'='*50}\n")
    
    return global_mean, global_std


def main():
    parser = argparse.ArgumentParser(description='Measure Qwen feature distribution stats')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='Precomputed features dir (for precomputed mode)')
    parser.add_argument('--data_dir', type=str, default='./data/coco',
                        help='Raw image dir (for realtime mode)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--max_files', type=int, default=200,
                        help='Max files/images to sample')
    parser.add_argument('--realtime', action='store_true',
                        help='Extract features in realtime (loads Qwen)')
    parser.add_argument('--all_layers', action='store_true',
                        help='Measure layers -1, 0, 4, 8 in sequence (realtime mode only)')
    parser.add_argument('--layer', type=int, default=4,
                        help='Which layer to measure (for realtime mode)')
    parser.add_argument('--qwen_model', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    if args.all_layers:
        # 逐层测量并汇总
        print("\n" + "="*60)
        print("Measuring all experiment layers: -1, 0, 4, 8")
        print("="*60)
        results = {}
        for layer in [-1, 0, 4, 8]:
            print(f"\n--- Layer {layer} ---")
            mean, std = measure_realtime(
                args.data_dir, layer, args.split, args.max_files,
                args.qwen_model, args.device
            )
            results[layer] = {'mean': mean, 'std': std}
        
        print("\n" + "="*60)
        print("Summary (use these values for feature normalization):")
        print("="*60)
        for layer, stats in results.items():
            label = {
                -1: "pixel patches (JPEG level)",
                0:  "patch_embed output",
                4:  "after 4 blocks",
                8:  "after 8 blocks"
            }.get(layer, f"layer {layer}")
            print(f"  Layer {layer:3d} ({label:30s}): mean={stats['mean']:+.4f}, std={stats['std']:.4f}")
        
    elif args.realtime:
        measure_realtime(args.data_dir, args.layer, args.split,
                         args.max_files, args.qwen_model, args.device)
    elif args.features_dir:
        measure_from_precomputed(args.features_dir, args.split, args.max_files)
    else:
        parser.error("Specify --features_dir (precomputed) or --realtime (live extraction)")


if __name__ == '__main__':
    main()
