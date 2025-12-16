#!/usr/bin/env python
"""
训练日志可视化脚本

Usage:
    python scripts/visualize_training.py --log checkpoints/qwen_precomputed/train.log
    python scripts/visualize_training.py --log path/to/train.log --output results/curves.png
"""
import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(log_path):
    """解析训练日志"""
    epochs = []
    train_loss, train_mse, train_cos = [], [], []
    val_mse, val_cos_sim = [], []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_epoch = None
    for line in lines:
        if 'Epoch' in line and '/' in line and 'INFO' in line:
            match = re.search(r'Epoch (\d+)/\d+', line)
            if match:
                current_epoch = int(match.group(1))
        
        if 'Train - Loss:' in line and current_epoch:
            match = re.search(r'Loss: ([\d.]+), MSE: ([\d.]+), Cos: ([\d.]+)', line)
            if match and current_epoch not in epochs:
                epochs.append(current_epoch)
                train_loss.append(float(match.group(1)))
                train_mse.append(float(match.group(2)))
                train_cos.append(float(match.group(3)))
        
        if 'Val - MSE:' in line and current_epoch:
            match = re.search(r'MSE: ([\d.]+), Cos Sim: ([\d.]+)', line)
            if match and len(val_mse) < len(epochs):
                val_mse.append(float(match.group(1)))
                val_cos_sim.append(float(match.group(2)))
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_mse': train_mse,
        'train_cos': train_cos,
        'val_mse': val_mse,
        'val_cos_sim': val_cos_sim
    }


def plot_curves(data, output_path, title_suffix=''):
    """绘制训练曲线"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = data['epochs']
    
    # Train Loss
    axes[0, 0].plot(epochs, data['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training Loss {title_suffix}', fontweight='bold')
    
    # MSE
    axes[0, 1].plot(epochs, data['train_mse'], 'b-', label='Train MSE', linewidth=2)
    if data['val_mse']:
        axes[0, 1].plot(epochs[:len(data['val_mse'])], data['val_mse'], 'r--', label='Val MSE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('MSE Loss', fontweight='bold')
    axes[0, 1].legend()
    
    # Train Cosine Loss
    axes[1, 0].plot(epochs, data['train_cos'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cosine Loss')
    axes[1, 0].set_title('Training Cosine Loss', fontweight='bold')
    
    # Validation Cosine Similarity
    if data['val_cos_sim']:
        axes[1, 1].plot(epochs[:len(data['val_cos_sim'])], data['val_cos_sim'], 'm-', linewidth=2)
        final_cos = data['val_cos_sim'][-1] if data['val_cos_sim'] else 0
        axes[1, 1].axhline(y=final_cos, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_cos:.3f}')
        axes[1, 1].legend()
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Cosine Similarity')
    axes[1, 1].set_title('Validation Cosine Similarity', fontweight='bold')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    
    # 打印总结
    print(f"\n📊 Training Summary:")
    print(f"   Epochs: {len(epochs)}")
    print(f"   Final Train Loss: {data['train_loss'][-1]:.4f}")
    print(f"   Final Train MSE: {data['train_mse'][-1]:.4f}")
    if data['val_mse']:
        print(f"   Final Val MSE: {data['val_mse'][-1]:.4f}")
    if data['val_cos_sim']:
        print(f"   Final Val Cos Sim: {data['val_cos_sim'][-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Training Log')
    parser.add_argument('--log', type=str, default='checkpoints/qwen_precomputed/train.log',
                        help='Path to train.log')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: results/training_curves.png)')
    parser.add_argument('--title', type=str, default='',
                        help='Additional title suffix')
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return
    
    output_path = args.output or 'results/training_curves.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Parsing: {log_path}")
    data = parse_log(log_path)
    
    if not data['epochs']:
        print("❌ No training data found in log")
        return
    
    plot_curves(data, output_path, args.title)


if __name__ == '__main__':
    main()
