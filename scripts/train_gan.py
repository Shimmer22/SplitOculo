"""
SplitOculo v2.0 GAN Training Script

训练流程:
- Phase 1 (Warmup): 只用 MSE Loss 预热，让 Upsampler 学会基本的上采样
- Phase 2 (GAN): Generator vs Discriminator 对抗训练，让特征更 sharp

Usage:
    # Phase 1: Warmup
    python scripts/train_gan.py \
        --features_dir ./data/qwen_features \
        --data_dir ./data/coco \
        --phase warmup \
        --epochs 20

    # Phase 2: GAN finetuning
    python scripts/train_gan.py \
        --features_dir ./data/qwen_features \
        --data_dir ./data/coco \
        --phase gan \
        --warmup_checkpoint ./checkpoints/gan/warmup_best.pth \
        --epochs 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import timm
import math

from core.utils import set_seed, get_logger, count_parameters
from models.cloud_upsampler import CloudUpsampler, TransformerUpsampler
from models.discriminator import FeatureDiscriminator


class EdgeProjector(nn.Module):
    """端侧 Projector: CNN 特征 → 传输 tokens"""
    def __init__(self, in_channels, hidden_size=1280,
                 hidden_channels=512, transmission_tokens=49):
        super().__init__()
        
        self.transmission_size = int(math.sqrt(transmission_tokens))
        assert self.transmission_size ** 2 == transmission_tokens
        
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


class PrecomputedFeatureDataset(Dataset):
    """加载预计算的 Qwen 特征和对应的原始图像"""
    
    def __init__(self, features_dir, images_dir, split='train', image_size=224):
        self.features_dir = Path(features_dir) / split
        self.images_dir = Path(images_dir) / split
        
        metadata_path = self.features_dir / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError(f"找不到元数据: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_files = sorted(self.features_dir.glob("*.pt"))
        print(f"📂 Found {len(self.feature_files)} precomputed features")
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_data = torch.load(self.feature_files[idx], weights_only=False)
        teacher_features = feature_data['features']
        img_path = feature_data['path']
        
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        return img_tensor, teacher_features


def collate_fn(batch):
    """处理不同长度的特征"""
    images = torch.stack([item[0] for item in batch])
    features = [item[1] for item in batch]
    min_tokens = min(f.shape[0] for f in features)
    aligned_features = torch.stack([f[:min_tokens] for f in features])
    return images, aligned_features


class GANTrainer:
    """SplitOculo v2.0 GAN 训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger('gan_train', log_file=f'{args.output_dir}/train.log')
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_models()
        self._setup_optimizers()
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 记录参数量
        g_params = (count_parameters(self.student) + 
                   count_parameters(self.projector) + 
                   count_parameters(self.upsampler))
        d_params = count_parameters(self.discriminator)
        
        self.logger.info(f"Generator params: {g_params / 1e6:.2f}M")
        self.logger.info(f"  Student: {count_parameters(self.student) / 1e6:.2f}M")
        self.logger.info(f"  Projector: {count_parameters(self.projector) / 1e6:.2f}M")
        self.logger.info(f"  Upsampler: {count_parameters(self.upsampler) / 1e6:.2f}M")
        self.logger.info(f"Discriminator params: {d_params / 1e6:.2f}M")
    
    def _load_models(self):
        """加载模型"""
        args = self.args
        
        # CNN backbone
        self.logger.info(f"Loading student: {args.student_model}")
        self.student = timm.create_model(
            args.student_model,
            pretrained=True,
            features_only=True,
            out_indices=[args.student_layer]
        ).to(self.device)
        
        # 获取输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.image_size, args.image_size).to(self.device)
            student_feat = self.student(dummy)[-1]
            student_channels = student_feat.shape[1]
        
        self.logger.info(f"Student output: {student_channels} channels")
        
        # 端侧 Projector
        self.projector = EdgeProjector(
            in_channels=student_channels,
            hidden_size=args.target_hidden_size,
            hidden_channels=args.projector_hidden,
            transmission_tokens=args.transmission_tokens
        ).to(self.device)
        
        # 云端 Upsampler (使用 TransformerUpsampler)
        if args.upsampler_type == 'transformer':
            self.upsampler = TransformerUpsampler(
                hidden_size=args.target_hidden_size,
                input_tokens=args.transmission_tokens,
                target_tokens=args.target_tokens,
                num_layers=args.transformer_layers
            ).to(self.device)
            self.logger.info(f"Using TransformerUpsampler with {args.transformer_layers} layers")
        else:
            self.upsampler = CloudUpsampler(
                hidden_size=args.target_hidden_size,
                input_tokens=args.transmission_tokens,
                target_tokens=args.target_tokens,
                method=args.upsampler_type,
            ).to(self.device)
            self.logger.info(f"Using CloudUpsampler ({args.upsampler_type})")
        
        # Discriminator
        self.discriminator = FeatureDiscriminator(
            hidden_size=args.target_hidden_size,
            num_tokens=args.target_tokens
        ).to(self.device)
        
        self.logger.info(f"Upsampler: {args.transmission_tokens} → {args.target_tokens} tokens")
    
    def _setup_optimizers(self):
        """设置优化器"""
        args = self.args
        
        # Generator 参数
        g_params = (
            list(self.student.parameters()) + 
            list(self.projector.parameters()) +
            list(self.upsampler.parameters())
        )
        
        # Generator 优化器
        self.opt_G = optim.AdamW(g_params, lr=args.lr_g, weight_decay=args.weight_decay,
                                  betas=(0.5, 0.9))
        
        # Discriminator 优化器 (学习率通常低一些)
        self.opt_D = optim.AdamW(self.discriminator.parameters(), lr=args.lr_d, 
                                  weight_decay=args.weight_decay, betas=(0.5, 0.9))
        
        # 学习率调度器
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.opt_G, T_max=args.epochs)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.opt_D, T_max=args.epochs)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.student.load_state_dict(ckpt['student_state_dict'])
        self.projector.load_state_dict(ckpt['projector_state_dict'])
        self.upsampler.load_state_dict(ckpt['upsampler_state_dict'])
        
        if 'discriminator_state_dict' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        
        self.logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', 'unknown')})")
        return ckpt.get('epoch', 0)
    
    def _align_tokens(self, tokens, target_num):
        """对齐 token 数量"""
        b, n, c = tokens.shape
        h = w = int(n ** 0.5)
        tokens = tokens.view(b, h, w, c).permute(0, 3, 1, 2)
        target_h = target_w = int(target_num ** 0.5)
        tokens = F.adaptive_avg_pool2d(tokens, (target_h, target_w))
        tokens = tokens.permute(0, 2, 3, 1).view(b, -1, c)
        return tokens
    
    def train_epoch_warmup(self, dataloader, epoch):
        """Warmup 训练: 只用 MSE Loss"""
        self.student.train()
        self.projector.train()
        self.upsampler.train()
        
        total_loss = 0
        total_cos_sim = 0
        
        pbar = tqdm(dataloader, desc=f"Warmup Epoch {epoch}")
        for images, teacher_tokens in pbar:
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device).float()
            
            # Forward
            student_feat = self.student(images)[-1]
            edge_tokens = self.projector(student_feat)
            output_tokens = self.upsampler(edge_tokens)
            
            # 对齐 token 数量
            if output_tokens.shape[1] != teacher_tokens.shape[1]:
                teacher_tokens = self._align_tokens(teacher_tokens, output_tokens.shape[1])
            
            # MSE Loss
            loss = self.mse_loss(output_tokens, teacher_tokens)
            
            self.opt_G.zero_grad()
            loss.backward()
            self.opt_G.step()
            
            # 计算 cos_sim
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    output_tokens.reshape(-1, self.args.target_hidden_size),
                    teacher_tokens.reshape(-1, self.args.target_hidden_size),
                    dim=-1
                ).mean()
            
            total_loss += loss.item()
            total_cos_sim += cos_sim.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cos_sim': f'{cos_sim.item():.4f}'})
        
        n = len(dataloader)
        return {'loss': total_loss / n, 'cos_sim': total_cos_sim / n}
    
    def train_epoch_gan(self, dataloader, epoch):
        """GAN 训练: Generator vs Discriminator"""
        self.student.train()
        self.projector.train()
        self.upsampler.train()
        self.discriminator.train()
        
        total_loss_g = 0
        total_loss_d = 0
        total_cos_sim = 0
        total_mse = 0
        
        pbar = tqdm(dataloader, desc=f"GAN Epoch {epoch}")
        for images, teacher_tokens in pbar:
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device).float()
            
            # 对齐 token 数量
            if teacher_tokens.shape[1] != self.args.target_tokens:
                teacher_tokens = self._align_tokens(teacher_tokens, self.args.target_tokens)
            
            # ===================
            # 1. Train Discriminator
            # ===================
            self.opt_D.zero_grad()
            
            # 真实样本
            pred_real = self.discriminator(teacher_tokens)
            label_real = torch.ones_like(pred_real)
            loss_d_real = self.bce_loss(pred_real, label_real)
            
            # 假样本 (detach 防止梯度传回 G)
            with torch.no_grad():
                student_feat = self.student(images)[-1]
                edge_tokens = self.projector(student_feat)
            fake_tokens = self.upsampler(edge_tokens).detach()
            
            pred_fake = self.discriminator(fake_tokens)
            label_fake = torch.zeros_like(pred_fake)
            loss_d_fake = self.bce_loss(pred_fake, label_fake)
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            self.opt_D.step()
            
            # ===================
            # 2. Train Generator
            # ===================
            self.opt_G.zero_grad()
            
            # 重新生成 (带梯度)
            student_feat = self.student(images)[-1]
            edge_tokens = self.projector(student_feat)
            fake_tokens = self.upsampler(edge_tokens)
            
            # 内容损失 (MSE)
            loss_mse = self.mse_loss(fake_tokens, teacher_tokens)
            
            # 对抗损失 (GAN) - 骗过 D
            pred_fake = self.discriminator(fake_tokens)
            loss_adv = self.bce_loss(pred_fake, torch.ones_like(pred_fake))
            
            # 组合损失
            loss_g = loss_mse * self.args.lambda_mse + loss_adv * self.args.lambda_adv
            
            loss_g.backward()
            self.opt_G.step()
            
            # 计算 cos_sim
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    fake_tokens.reshape(-1, self.args.target_hidden_size),
                    teacher_tokens.reshape(-1, self.args.target_hidden_size),
                    dim=-1
                ).mean()
            
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            total_cos_sim += cos_sim.item()
            total_mse += loss_mse.item()
            
            pbar.set_postfix({
                'G': f'{loss_g.item():.3f}',
                'D': f'{loss_d.item():.3f}',
                'cos': f'{cos_sim.item():.3f}'
            })
        
        n = len(dataloader)
        return {
            'loss_g': total_loss_g / n,
            'loss_d': total_loss_d / n,
            'cos_sim': total_cos_sim / n,
            'mse': total_mse / n
        }
    
    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.student.eval()
        self.projector.eval()
        self.upsampler.eval()
        
        total_mse = 0
        total_cos_sim = 0
        total_std = 0
        
        for images, teacher_tokens in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device).float()
            
            student_feat = self.student(images)[-1]
            edge_tokens = self.projector(student_feat)
            output_tokens = self.upsampler(edge_tokens)
            
            if output_tokens.shape[1] != teacher_tokens.shape[1]:
                teacher_tokens = self._align_tokens(teacher_tokens, output_tokens.shape[1])
            
            mse = self.mse_loss(output_tokens, teacher_tokens)
            total_mse += mse.item()
            
            cos_sim = F.cosine_similarity(
                output_tokens.reshape(-1, self.args.target_hidden_size),
                teacher_tokens.reshape(-1, self.args.target_hidden_size),
                dim=-1
            ).mean()
            total_cos_sim += cos_sim.item()
            
            # 特征方差 (越大越好，说明特征越 "sharp")
            total_std += output_tokens.std().item()
        
        n = len(dataloader)
        return {
            'val_mse': total_mse / n,
            'val_cos_sim': total_cos_sim / n,
            'val_std': total_std / n
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False, prefix=''):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'upsampler_state_dict': self.upsampler.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        torch.save(checkpoint, self.output_dir / f'{prefix}latest.pth')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / f'{prefix}best.pth')
            self.logger.info(f"💾 Saved best model (cos_sim: {metrics['val_cos_sim']:.4f})")
    
    def train_warmup(self, train_loader, val_loader):
        """Phase 1: Warmup 训练"""
        best_cos_sim = 0
        
        self.logger.info("=" * 60)
        self.logger.info("Phase 1: Warmup Training (MSE only)")
        self.logger.info("=" * 60)
        
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_epoch_warmup(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            self.scheduler_G.step()
            
            metrics = {**train_metrics, **val_metrics}
            
            self.logger.info(
                f"Epoch {epoch}: "
                f"loss={metrics['loss']:.4f}, cos_sim={metrics['cos_sim']:.4f}, "
                f"val_cos_sim={metrics['val_cos_sim']:.4f}, val_std={metrics['val_std']:.3f}"
            )
            
            is_best = metrics['val_cos_sim'] > best_cos_sim
            if is_best:
                best_cos_sim = metrics['val_cos_sim']
            
            if epoch % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best, prefix='warmup_')
        
        self.logger.info("=" * 60)
        self.logger.info(f"Warmup 完成! Best cos_sim: {best_cos_sim:.4f}")
        self.logger.info("=" * 60)
    
    def train_gan(self, train_loader, val_loader):
        """Phase 2: GAN 训练"""
        best_cos_sim = 0
        
        self.logger.info("=" * 60)
        self.logger.info("Phase 2: GAN Training")
        self.logger.info(f"λ_MSE={self.args.lambda_mse}, λ_ADV={self.args.lambda_adv}")
        self.logger.info("=" * 60)
        
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_epoch_gan(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            metrics = {**train_metrics, **val_metrics}
            
            self.logger.info(
                f"Epoch {epoch}: "
                f"G={metrics['loss_g']:.4f}, D={metrics['loss_d']:.4f}, "
                f"mse={metrics['mse']:.4f}, val_cos_sim={metrics['val_cos_sim']:.4f}, "
                f"val_std={metrics['val_std']:.3f}"
            )
            
            is_best = metrics['val_cos_sim'] > best_cos_sim
            if is_best:
                best_cos_sim = metrics['val_cos_sim']
            
            if epoch % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best, prefix='gan_')
        
        self.logger.info("=" * 60)
        self.logger.info(f"GAN 训练完成! Best cos_sim: {best_cos_sim:.4f}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='SplitOculo v2.0 GAN Training')
    
    # 数据参数
    parser.add_argument('--features_dir', type=str, required=True,
                        help='预计算特征目录')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='原始图像目录')
    parser.add_argument('--image_size', type=int, default=224)
    
    # Student 模型参数
    parser.add_argument('--student_model', type=str, default='mobilenetv2_100')
    parser.add_argument('--student_layer', type=int, default=3)
    
    # Projector 参数
    parser.add_argument('--target_hidden_size', type=int, default=1280)
    parser.add_argument('--projector_hidden', type=int, default=512)
    
    # Token 参数
    parser.add_argument('--transmission_tokens', type=int, default=49)
    parser.add_argument('--target_tokens', type=int, default=256)
    
    # Upsampler 参数
    parser.add_argument('--upsampler_type', type=str, default='transformer',
                        choices=['transformer', 'mlp', 'deconv'])
    parser.add_argument('--transformer_layers', type=int, default=4)
    
    # 训练阶段
    parser.add_argument('--phase', type=str, required=True,
                        choices=['warmup', 'gan'],
                        help='训练阶段: warmup (MSE only) 或 gan (adversarial)')
    parser.add_argument('--warmup_checkpoint', type=str, default=None,
                        help='GAN 阶段加载的 warmup checkpoint')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_g', type=float, default=1e-4,
                        help='Generator 学习率')
    parser.add_argument('--lr_d', type=float, default=4e-5,
                        help='Discriminator 学习率 (通常低于 G)')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # GAN 损失权重
    parser.add_argument('--lambda_mse', type=float, default=10.0,
                        help='MSE 损失权重 (内容)')
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                        help='对抗损失权重 (样式)')
    
    # 其他
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/gan')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 加载数据
    train_dataset = PrecomputedFeatureDataset(
        features_dir=args.features_dir,
        images_dir=args.data_dir,
        split='train',
        image_size=args.image_size
    )
    
    val_dataset = PrecomputedFeatureDataset(
        features_dir=args.features_dir,
        images_dir=args.data_dir,
        split='val',
        image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    # 创建训练器
    trainer = GANTrainer(args)
    
    # 根据阶段训练
    if args.phase == 'warmup':
        trainer.train_warmup(train_loader, val_loader)
    elif args.phase == 'gan':
        if args.warmup_checkpoint:
            trainer.load_checkpoint(args.warmup_checkpoint)
        trainer.train_gan(train_loader, val_loader)


if __name__ == '__main__':
    main()
