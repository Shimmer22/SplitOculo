"""
可视化训练日志

Usage:
    python scripts/plot_training.py --log checkpoints/upsampler/train.log
"""
import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(log_path):
    """解析训练日志"""
    epochs = []
    losses = []
    mses = []
    val_mses = []
    cos_sims = []
    
    pattern = r'Epoch (\d+): loss=([\d.]+), mse=([\d.]+), val_mse=([\d.]+), val_cos_sim=([\d.]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                mses.append(float(match.group(3)))
                val_mses.append(float(match.group(4)))
                cos_sims.append(float(match.group(5)))
    
    return {
        'epochs': epochs,
        'loss': losses,
        'mse': mses,
        'val_mse': val_mses,
        'val_cos_sim': cos_sims
    }


def plot_metrics(data, output_path=None):
    """绘制训练曲线"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = data['epochs']
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, data['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # MSE
    ax = axes[0, 1]
    ax.plot(epochs, data['mse'], 'b-', label='Train MSE', linewidth=2)
    ax.plot(epochs, data['val_mse'], 'r-', label='Val MSE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('MSE (Train vs Val)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cosine Similarity
    ax = axes[1, 0]
    ax.plot(epochs, data['val_cos_sim'], 'g-', linewidth=2)
    ax.axhline(y=0.86, color='red', linestyle='--', alpha=0.7, label='CNN Baseline (0.86)')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='Target (0.95)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Validation Cosine Similarity', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # Best metrics
    ax = axes[1, 1]
    ax.axis('off')
    best_idx = data['val_cos_sim'].index(max(data['val_cos_sim']))
    best_epoch = data['epochs'][best_idx]
    best_cos_sim = data['val_cos_sim'][best_idx]
    best_val_mse = data['val_mse'][best_idx]
    
    text = f"""
    训练结果摘要
    ─────────────────────
    总 Epochs: {len(epochs)}
    
    最佳结果 (Epoch {best_epoch}):
      • Val Cos Sim: {best_cos_sim:.4f}
      • Val MSE: {best_val_mse:.4f}
    
    最终结果 (Epoch {epochs[-1]}):
      • Val Cos Sim: {data['val_cos_sim'][-1]:.4f}
      • Val MSE: {data['val_mse'][-1]:.4f}
      • Loss: {data['loss'][-1]:.4f}
    """
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 图表已保存: {output_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize training log')
    parser.add_argument('--log', type=str, default='checkpoints/upsampler/train.log',
                        help='Path to training log file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (optional)')
    
    args = parser.parse_args()
    
    if not Path(args.log).exists():
        print(f"❌ Log file not found: {args.log}")
        return
    
    print(f"📖 Parsing log: {args.log}")
    data = parse_log(args.log)
    
    if not data['epochs']:
        print("❌ No training data found in log")
        return
    
    print(f"✅ Found {len(data['epochs'])} epochs")
    print(f"   Best cos_sim: {max(data['val_cos_sim']):.4f}")
    
    output_path = args.output or str(Path(args.log).parent / 'training_curve.png')
    plot_metrics(data, output_path)


if __name__ == '__main__':
    main()
