"""
Qwen2.5-VL 视觉特征预计算脚本

离线提取所有图片的 Qwen 视觉特征并保存，支持断点续传。
训练时直接加载预计算的特征，无需再运行 Qwen。

Usage:
    # 预计算所有特征
    python precompute_qwen_features.py --data_dir ./data/imagenette2-320 --output_dir ./data/qwen_features
    
    # 限制样本数量（用于测试）
    python precompute_qwen_features.py --data_dir ./data/imagenette2-320 --max_samples 100
    
    # 从断点继续
    python precompute_qwen_features.py --data_dir ./data/imagenette2-320 --resume
"""
import argparse
import torch
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms


# 导入共享的 QwenFeatureExtractor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.qwen_extractor import QwenFeatureExtractor





def get_image_paths(data_dir, split='train'):
    """获取所有图片路径 (支持平铺目录或 ImageFolder 结构)"""
    data_path = Path(data_dir) / split
    if not data_path.exists():
        raise ValueError(f"路径不存在: {data_path}")
    
    print(f"Scanning {data_path}...")
    
    # 检查是否有子目录 (ImageFolder 结构)
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if len(subdirs) > 0:
        # ImageFolder 结构 (如 ImageNet, Imagenette)
        print(f"   检测到 ImageFolder 结构 ({len(subdirs)} 个类别)")
        dataset = ImageFolder(data_path)
        paths = []
        for idx in range(len(dataset)):
            img_path, label = dataset.samples[idx]
            paths.append({
                'path': img_path,
                'label': label,
                'idx': idx
            })
    else:
        # 平铺目录 (如 COCO)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = sorted([
            f for f in data_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        print(f"   检测到平铺目录 ({len(image_files)} 张图片)")
        paths = []
        for idx, img_path in enumerate(image_files):
            paths.append({
                'path': str(img_path),
                'label': 0,  # 无标签
                'idx': idx
            })
    
    return paths


def load_checkpoint(checkpoint_path):
    """加载检查点"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'processed': [], 'last_idx': -1}


def save_checkpoint(checkpoint_path, processed_items, last_idx):
    """保存检查点"""
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'processed': processed_items,
            'last_idx': last_idx
        }, f)


def main():
    parser = argparse.ArgumentParser(description='Precompute Qwen Vision Features')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/imagenette2-320',
                        help='ImageNet-style data directory')
    parser.add_argument('--output_dir', type=str, default='./data/qwen_features',
                        help='Output directory for features')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'], help='Which split to process')
    
    # 控制参数
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    
    # 模型参数
    parser.add_argument('--qwen_model', type=str, default='Qwen/Qwen2.5-VL-32B-Instruct',
                        help='Qwen model name')
    parser.add_argument('--offline', action='store_true',
                        help='load Qwen only from the local Hugging Face cache')
    parser.add_argument('--layer', type=int, default=8,
                        help='Which ViT layer to extract (1-32, default 8 for shallow, -1 for merger output)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查点路径
    checkpoint_path = output_dir / 'checkpoint.json'
    
    # 获取所有图片路径
    print(f"Scanning {args.data_dir}/{args.split}...")
    all_images = get_image_paths(args.data_dir, args.split)
    total_images = len(all_images)
    print(f"   Found {total_images} images")
    
    # 限制样本数量
    if args.max_samples:
        all_images = all_images[:args.max_samples]
        print(f"   Limited to {len(all_images)} samples")
    
    # 加载检查点
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        processed_set = set(checkpoint['processed'])
        print(f"Resuming from checkpoint: {len(processed_set)} already processed")
    else:
        processed_set = set()
    
    # 过滤已处理的
    to_process = [img for img in all_images if img['path'] not in processed_set]
    print(f"To process: {len(to_process)} images")
    
    if len(to_process) == 0:
        print("All images already processed!")
        return
    
    # 加载模型
    extractor = QwenFeatureExtractor(
        model_name=args.qwen_model,
        device=args.device,
        extract_layer=args.layer,
        local_files_only=args.offline,
        min_pixels=224 * 224,
        max_pixels=224 * 224,
        visual_only=True,
    ).load()
    
    # 预处理 transform (只做基本的 resize)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    
    # 开始处理
    processed_list = list(processed_set)
    errors = []
    
    print(f"Starting feature extraction...")
    pbar = tqdm(to_process, desc="Extracting")
    
    for item in pbar:
        img_path = item['path']
        idx = item['idx']
        label = item['label']
        
        try:
            # 加载并预处理图像
            pil_img = Image.open(img_path).convert('RGB')
            pil_img = transform(pil_img)
            
            # 提取特征
            features = extractor.extract_features(pil_img)
            
            # 保存特征
            # 使用 idx 作为文件名，方便后续对应
            feature_path = output_dir / f"{idx:06d}.pt"
            torch.save({
                'features': features,
                'label': label,
                'path': img_path,
                'num_tokens': features.shape[0],
                'hidden_size': features.shape[1]
            }, feature_path)
            
            # 更新进度
            processed_list.append(img_path)
            
            # 定期保存检查点
            if len(processed_list) % 50 == 0:
                save_checkpoint(checkpoint_path, processed_list, idx)
                pbar.set_postfix({'saved': len(processed_list)})
                
        except Exception as e:
            errors.append({'path': img_path, 'error': str(e)})
            pbar.set_postfix({'errors': len(errors)})
    
    # 最终保存检查点
    save_checkpoint(checkpoint_path, processed_list, -1)
    
    # 保存元数据
    hidden_size = 2048 if args.layer == -1 else 1280  # merger output vs intermediate
    metadata = {
        'total_processed': len(processed_list),
        'total_errors': len(errors),
        'hidden_size': hidden_size,
        'extract_layer': args.layer,
        'split': args.split,
        'data_dir': str(args.data_dir),
        'qwen_model': args.qwen_model
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if errors:
        with open(output_dir / 'errors.json', 'w') as f:
            json.dump(errors, f, indent=2)
    
    print(f"\nDone! Processed: {len(processed_list)}, Errors: {len(errors)}, Output: {output_dir}")


if __name__ == '__main__':
    main()
