"""
特征蒸馏训练脚本

训练 CNN 模型逼近 ViT (如 CLIP) 的输出特征
"""
import argparse
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import timm
import matplotlib.pyplot as plt

from core.utils import set_seed, get_logger, count_parameters
from data.dataset import get_dummy_loader, get_imagenet_loader
from models.adapters import FeatureAdapter, DistillationHead


def visualize_training(log_path, output_dir='./results'):
    """解析日志并生成训练曲线"""
    epochs = []
    train_loss, train_mse, train_cos = [], [], []
    val_mse, val_cos_sim = [], []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_epoch = None
    for line in lines:
        if 'Epoch' in line and '/' in line:
            match = re.search(r'Epoch (\d+)/\d+', line)
            if match:
                current_epoch = int(match.group(1))
        
        if 'Train - Loss:' in line:
            match = re.search(r'Loss: ([\d.]+), MSE: ([\d.]+), Cos: ([\d.]+)', line)
            if match:
                epochs.append(current_epoch)
                train_loss.append(float(match.group(1)))
                train_mse.append(float(match.group(2)))
                train_cos.append(float(match.group(3)))
        
        if 'Val - MSE:' in line:
            match = re.search(r'MSE: ([\d.]+), Cos Sim: ([\d.]+)', line)
            if match:
                val_mse.append(float(match.group(1)))
                val_cos_sim.append(float(match.group(2)))
    
    if not epochs:
        print("⚠️ 日志中没有找到训练数据")
        return None
    
    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(epochs, train_loss, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    
    axes[0, 1].plot(epochs, train_mse, 'b-', label='Train MSE', linewidth=2)
    if val_mse:
        axes[0, 1].plot(epochs[:len(val_mse)], val_mse, 'r--', label='Val MSE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('MSE Loss', fontweight='bold')
    axes[0, 1].legend()
    
    axes[1, 0].plot(epochs, train_cos, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cosine Loss')
    axes[1, 0].set_title('Training Cosine Loss', fontweight='bold')
    
    if val_cos_sim:
        axes[1, 1].plot(epochs[:len(val_cos_sim)], val_cos_sim, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].set_title('Validation Cosine Similarity', fontweight='bold')
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    fig_path = output_path / 'training_curves.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 训练曲线已保存: {fig_path}")
    return fig_path


class DistillationTrainer:
    """
    特征蒸馏训练器
    
    Teacher: 冻结的 ViT (如 CLIP ViT-L/14)
    Student: CNN backbone (如 MobileNetV2) + FeatureAdapter
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger('distill', log_file=f'{args.output_dir}/train.log')
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self._load_teacher()
        self._load_student()
        
        # 优化器
        self.optimizer = optim.AdamW(
            list(self.student.parameters()) + list(self.adapter.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        
        self.logger.info(f"Teacher params: {count_parameters(self.teacher) / 1e6:.2f}M")
        self.logger.info(f"Student params: {count_parameters(self.student) / 1e6:.2f}M")
        self.logger.info(f"Adapter params: {count_parameters(self.adapter) / 1e6:.2f}M")
    
    def _load_teacher(self):
        """加载 Teacher 模型 (冻结的 ViT)"""
        self.logger.info(f"Loading teacher: {self.args.teacher_model}")
        
        # 使用 timm 加载 ViT
        self.teacher = timm.create_model(
            self.args.teacher_model,
            pretrained=True,
            features_only=True,
            out_indices=[self.args.teacher_layer]
        ).to(self.device)
        
        # 冻结 Teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 获取 Teacher 输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.args.image_size, self.args.image_size).to(self.device)
            teacher_feat = self.teacher(dummy)[-1]
            self.teacher_channels = teacher_feat.shape[1]
            self.teacher_size = teacher_feat.shape[2]
        
        self.logger.info(f"Teacher output: {self.teacher_channels} channels, {self.teacher_size}x{self.teacher_size}")
    
    def _load_student(self):
        """加载 Student 模型 (可训练的 CNN)"""
        self.logger.info(f"Loading student: {self.args.student_model}")
        
        self.student = timm.create_model(
            self.args.student_model,
            pretrained=True,
            features_only=True,
            out_indices=[self.args.student_layer]
        ).to(self.device)
        
        # 获取 Student 输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.args.image_size, self.args.image_size).to(self.device)
            student_feat = self.student(dummy)[-1]
            self.student_channels = student_feat.shape[1]
            self.student_size = student_feat.shape[2]
        
        self.logger.info(f"Student output: {self.student_channels} channels, {self.student_size}x{self.student_size}")
        
        # 创建适配器
        self.adapter = FeatureAdapter(
            self.student_channels, 
            self.teacher_channels
        ).to(self.device)
        
        # 如果尺寸不同，添加上采样/下采样
        if self.student_size != self.teacher_size:
            self.spatial_align = nn.AdaptiveAvgPool2d((self.teacher_size, self.teacher_size)).to(self.device)
        else:
            self.spatial_align = nn.Identity().to(self.device)
    
    def train_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        self.student.train()
        self.adapter.train()
        
        total_loss = 0
        total_mse = 0
        total_cos = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Teacher 前向 (不计算梯度)
            with torch.no_grad():
                teacher_feat = self.teacher(images)[-1]
            
            # Student 前向
            student_feat = self.student(images)[-1]
            adapted_feat = self.adapter(student_feat)
            adapted_feat = self.spatial_align(adapted_feat)
            
            # 计算损失
            mse = self.mse_loss(adapted_feat, teacher_feat)
            
            # Cosine similarity loss
            b, c, h, w = adapted_feat.shape
            adapted_flat = adapted_feat.view(b, -1)
            teacher_flat = teacher_feat.view(b, -1)
            target = torch.ones(b).to(self.device)
            cos = self.cos_loss(adapted_flat, teacher_flat, target)
            
            loss = mse + self.args.cos_weight * cos
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse.item():.4f}',
                'cos': f'{cos.item():.4f}'
            })
        
        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'mse': total_mse / n,
            'cos': total_cos / n
        }
    
    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.student.eval()
        self.adapter.eval()
        
        total_mse = 0
        total_cos_sim = 0
        
        for images, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            
            teacher_feat = self.teacher(images)[-1]
            student_feat = self.student(images)[-1]
            adapted_feat = self.adapter(student_feat)
            adapted_feat = self.spatial_align(adapted_feat)
            
            mse = self.mse_loss(adapted_feat, teacher_feat)
            
            # Cosine similarity
            b = adapted_feat.shape[0]
            adapted_flat = adapted_feat.view(b, -1)
            teacher_flat = teacher_feat.view(b, -1)
            cos_sim = nn.functional.cosine_similarity(adapted_flat, teacher_flat, dim=1).mean()
            
            total_mse += mse.item()
            total_cos_sim += cos_sim.item()
        
        n = len(dataloader)
        return {
            'val_mse': total_mse / n,
            'val_cos_sim': total_cos_sim / n
        }
    
    def train(self, train_loader, val_loader=None):
        """完整训练循环"""
        best_loss = float('inf')
        
        for epoch in range(1, self.args.epochs + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            self.logger.info(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                           f"MSE: {train_metrics['mse']:.4f}, "
                           f"Cos: {train_metrics['cos']:.4f}")
            
            # 验证
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.logger.info(f"Val - MSE: {val_metrics['val_mse']:.4f}, "
                               f"Cos Sim: {val_metrics['val_cos_sim']:.4f}")
            
            # 保存检查点
            if train_metrics['loss'] < best_loss:
                best_loss = train_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(epoch)
            
            self.scheduler.step()
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'adapter_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='CNN-to-ViT Feature Distillation')
    
    # 模型参数
    parser.add_argument('--teacher_model', type=str, default='vit_large_patch14_clip_224',
                        help='Teacher model (ViT)')
    parser.add_argument('--teacher_layer', type=int, default=3,
                        help='Teacher layer to extract features from')
    parser.add_argument('--student_model', type=str, default='mobilenetv2_100',
                        help='Student model (CNN)')
    parser.add_argument('--student_layer', type=int, default=3,
                        help='Student layer to match')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--cos_weight', type=float, default=0.5,
                        help='Weight for cosine similarity loss')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='ImageNet data directory')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy data for testing')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据加载器
    if args.dummy:
        train_loader = get_dummy_loader(
            batch_size=args.batch_size,
            num_samples=1000,
            image_size=args.image_size
        )
        val_loader = get_dummy_loader(
            batch_size=args.batch_size,
            num_samples=200,
            image_size=args.image_size
        )
    elif args.data_dir:
        train_loader = get_imagenet_loader(
            args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            is_train=True,
            num_workers=args.num_workers
        )
        val_loader = get_imagenet_loader(
            args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            is_train=False,
            num_workers=args.num_workers
        )
    else:
        print("⚠️ No data specified. Use --data_dir or --dummy")
        return
    
    # 创建训练器
    trainer = DistillationTrainer(args)
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    
    # 训练结束后生成可视化
    log_path = Path(args.output_dir) / 'train.log'
    if log_path.exists():
        visualize_training(log_path, output_dir='./results')


if __name__ == '__main__':
    main()
