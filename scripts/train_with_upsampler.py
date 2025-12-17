"""
使用可学习上采样器训练

支持端侧下采样 + 云端上采样的训练流程:
- CNN → Projector → transmission_tokens (如 49)
- Upsampler → target_tokens (如 256)
- Loss 对齐到预计算的 Qwen 特征

Usage:
    python scripts/train_with_upsampler.py \
        --features_dir ./data/qwen_features \
        --data_dir ./data/imagenette2-320 \
        --transmission_tokens 49 \
        --target_tokens 256
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import timm
import math

from core.utils import set_seed, get_logger, count_parameters
from models.cloud_upsampler import CloudUpsampler


class EdgeProjector(nn.Module):
    """
    端侧 Projector: CNN 特征 → 传输 tokens
    
    支持自定义传输 token 数量
    """
    def __init__(self, in_channels, hidden_size=1280,
                 hidden_channels=512, transmission_tokens=49):
        super().__init__()
        
        self.transmission_size = int(math.sqrt(transmission_tokens))
        assert self.transmission_size ** 2 == transmission_tokens
        
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        
        # 自适应池化到传输尺寸
        self.pool = nn.AdaptiveAvgPool2d((self.transmission_size, self.transmission_size))
        
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        
        self.pw_conv2 = nn.Conv2d(hidden_channels, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) CNN 特征图
        Returns:
            (B, transmission_tokens, hidden_size)
        """
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, tokens, hidden_size)
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
        teacher_features = feature_data['features']  # (num_tokens, hidden_size)
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


class UpsamplerTrainer:
    """支持可学习上采样器的训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger('upsampler_train', log_file=f'{args.output_dir}/train.log')
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_models()
        
        # 优化器: 训练 student + projector + upsampler
        params = (
            list(self.student.parameters()) + 
            list(self.projector.parameters()) +
            list(self.upsampler.parameters())
        )
        
        self.optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        self.mse_loss = nn.MSELoss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        
        total_params = (count_parameters(self.student) + 
                       count_parameters(self.projector) + 
                       count_parameters(self.upsampler))
        self.logger.info(f"Total trainable params: {total_params / 1e6:.2f}M")
        self.logger.info(f"  Student: {count_parameters(self.student) / 1e6:.2f}M")
        self.logger.info(f"  Projector: {count_parameters(self.projector) / 1e6:.2f}M")
        self.logger.info(f"  Upsampler: {count_parameters(self.upsampler) / 1e6:.2f}M")
    
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
        
        self.logger.info(f"Projector output: {args.transmission_tokens} tokens")
        
        # 云端 Upsampler
        self.upsampler = CloudUpsampler(
            hidden_size=args.target_hidden_size,
            input_tokens=args.transmission_tokens,
            target_tokens=args.target_tokens,
            method=args.upsampler_method,
            num_refine_layers=args.upsampler_layers
        ).to(self.device)
        
        self.logger.info(f"Upsampler: {args.transmission_tokens} → {args.target_tokens} tokens ({args.upsampler_method})")
    
    def train_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        self.student.train()
        self.projector.train()
        self.upsampler.train()
        
        total_loss = 0
        total_mse = 0
        total_cos = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for images, teacher_tokens in pbar:
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device).float()
            
            # 端侧: CNN → Projector
            student_feat = self.student(images)[-1]
            edge_tokens = self.projector(student_feat)  # (B, transmission_tokens, hidden_size)
            
            # 云端: Upsampler
            output_tokens = self.upsampler(edge_tokens)  # (B, target_tokens, hidden_size)
            
            # 对齐 token 数量 (如果目标 tokens 与 teacher 不一致)
            if output_tokens.shape[1] != teacher_tokens.shape[1]:
                # 下采样 teacher 到 output 大小
                teacher_tokens = self._align_tokens(teacher_tokens, output_tokens.shape[1])
            
            # 计算损失
            mse = self.mse_loss(output_tokens, teacher_tokens)
            
            b, n, c = output_tokens.shape
            output_flat = output_tokens.contiguous().view(b * n, c)
            teacher_flat = teacher_tokens.contiguous().view(b * n, c)
            target = torch.ones(b * n).to(self.device)
            cos = self.cos_loss(output_flat, teacher_flat, target)
            
            loss = mse + self.args.cos_weight * cos
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse.item():.4f}',
                'cos': f'{cos.item():.4f}'
            })
        
        n = len(dataloader)
        return {'loss': total_loss / n, 'mse': total_mse / n, 'cos': total_cos / n}
    
    def _align_tokens(self, tokens, target_num):
        """对齐 token 数量"""
        b, n, c = tokens.shape
        h = w = int(n ** 0.5)
        tokens = tokens.view(b, h, w, c).permute(0, 3, 1, 2)
        target_h = target_w = int(target_num ** 0.5)
        tokens = nn.functional.adaptive_avg_pool2d(tokens, (target_h, target_w))
        tokens = tokens.permute(0, 2, 3, 1).view(b, -1, c)
        return tokens
    
    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.student.eval()
        self.projector.eval()
        self.upsampler.eval()
        
        total_mse = 0
        total_cos_sim = 0
        
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
            
            # 余弦相似度
            output_flat = output_tokens.reshape(-1, output_tokens.shape[-1])
            teacher_flat = teacher_tokens.reshape(-1, teacher_tokens.shape[-1])
            cos_sim = nn.functional.cosine_similarity(output_flat, teacher_flat, dim=-1).mean()
            total_cos_sim += cos_sim.item()
        
        n = len(dataloader)
        return {'val_mse': total_mse / n, 'val_cos_sim': total_cos_sim / n}
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'upsampler_state_dict': self.upsampler.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        torch.save(checkpoint, self.output_dir / 'latest.pth')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            self.logger.info(f"💾 Saved best model (cos_sim: {metrics['val_cos_sim']:.4f})")
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        best_cos_sim = 0
        
        self.logger.info("=" * 60)
        self.logger.info("开始训练 (端侧下采样 + 云端上采样)")
        self.logger.info(f"传输: {self.args.transmission_tokens} tokens → 目标: {self.args.target_tokens} tokens")
        self.logger.info("=" * 60)
        
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            self.scheduler.step()
            
            metrics = {**train_metrics, **val_metrics}
            
            self.logger.info(
                f"Epoch {epoch}: "
                f"loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}, "
                f"val_mse={metrics['val_mse']:.4f}, val_cos_sim={metrics['val_cos_sim']:.4f}"
            )
            
            is_best = metrics['val_cos_sim'] > best_cos_sim
            if is_best:
                best_cos_sim = metrics['val_cos_sim']
            
            if epoch % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best)
        
        self.logger.info("=" * 60)
        self.logger.info(f"训练完成! Best cos_sim: {best_cos_sim:.4f}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train with learnable upsampler')
    
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
    parser.add_argument('--target_hidden_size', type=int, default=1280,
                        help='目标隐藏维度 (Qwen Layer 4 = 1280)')
    parser.add_argument('--projector_hidden', type=int, default=512)
    
    # 传输和目标 token 数量
    parser.add_argument('--transmission_tokens', type=int, default=49,
                        help='端侧传输的 token 数量 (7x7=49)')
    parser.add_argument('--target_tokens', type=int, default=256,
                        help='目标 token 数量 (16x16=256, 需与 teacher 特征匹配)')
    
    # Upsampler 参数
    parser.add_argument('--upsampler_method', type=str, default='mlp',
                        choices=['mlp', 'deconv', 'pixelshuffle', 'transformer'])
    parser.add_argument('--upsampler_layers', type=int, default=2,
                        help='Upsampler 精炼层数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--cos_weight', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # 其他
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/upsampler')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    # 调试选项
    parser.add_argument('--overfit', type=str, default=None,
                        help='单图过拟合模式: 指定 .pt 特征文件路径进行过拟合调试')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 单图过拟合模式
    if args.overfit:
        print("=" * 60)
        print("🔬 单图过拟合调试模式")
        print("=" * 60)
        run_overfit_debug(args)
        return
    
    # 正常训练模式
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
    
    # 训练
    trainer = UpsamplerTrainer(args)
    trainer.train(train_loader, val_loader)


def run_overfit_debug(args):
    """单图过拟合调试: 训练后用 Qwen 验证语义理解"""
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    
    device = torch.device(args.device)
    
    # 加载数据
    feat_path = args.overfit
    feat_data = torch.load(feat_path, weights_only=False)
    teacher_256 = feat_data['features'].unsqueeze(0).float().to(device)
    img_path = feat_data['path']
    
    print(f"📁 特征文件: {feat_path}")
    print(f"📷 图像路径: {img_path}")
    print(f"🎯 Teacher: {teacher_256.shape}, std={teacher_256.std():.3f}")
    
    # 加载图像
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    
    # 创建模型
    student = timm.create_model(
        args.student_model, pretrained=True, features_only=True, 
        out_indices=[args.student_layer]
    ).to(device)
    
    with torch.no_grad():
        student_feat = student(image)[-1]
        student_channels = student_feat.shape[1]
    
    projector = EdgeProjector(
        in_channels=student_channels,
        hidden_size=args.target_hidden_size,
        hidden_channels=args.projector_hidden,
        transmission_tokens=args.transmission_tokens
    ).to(device)
    
    from models.cloud_upsampler import CloudUpsampler
    upsampler = CloudUpsampler(
        hidden_size=args.target_hidden_size,
        input_tokens=args.transmission_tokens,
        target_tokens=args.target_tokens,
        method=args.upsampler_method,
    ).to(device)
    
    # 优化器
    params = list(student.parameters()) + list(projector.parameters()) + list(upsampler.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    
    print()
    print(f"🚀 开始过拟合训练 ({args.epochs} iterations)...")
    
    for i in range(args.epochs):
        optimizer.zero_grad()
        
        feat = student(image)[-1]
        tokens_49 = projector(feat)
        tokens_256 = upsampler(tokens_49)
        
        loss = F.mse_loss(tokens_256, teacher_256)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0 or i == 0:
            cos_sim = F.cosine_similarity(
                tokens_256.reshape(-1, args.target_hidden_size),
                teacher_256.reshape(-1, args.target_hidden_size), dim=-1
            ).mean()
            print(f"   Iter {i+1}: loss={loss.item():.4f}, cos_sim={cos_sim.item():.4f}")
    
    # 最终验证
    student.eval()
    projector.eval()
    upsampler.eval()
    
    with torch.no_grad():
        feat = student(image)[-1]
        tokens_49 = projector(feat)
        tokens_256 = upsampler(tokens_49)
        
        cos_sim = F.cosine_similarity(
            tokens_256.reshape(-1, args.target_hidden_size),
            teacher_256.reshape(-1, args.target_hidden_size), dim=-1
        ).mean()
    
    print()
    print("=" * 60)
    print(f"✅ 过拟合完成!")
    print(f"   最终 cos_sim: {cos_sim.item():.4f}")
    print(f"   输出 std: {tokens_256.std():.3f}")
    print(f"   目标 std: {teacher_256.std():.3f}")
    print("=" * 60)
    
    # 保存模型用于推理测试
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'student_state_dict': student.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'upsampler_state_dict': upsampler.state_dict(),
        'args': vars(args),
        'overfit_image': img_path,
        'final_cos_sim': cos_sim.item(),
    }
    
    save_path = output_path / 'overfit_model.pth'
    torch.save(checkpoint, save_path)
    print(f"💾 模型已保存: {save_path}")
    print()
    print("📝 用以下命令测试推理:")
    print(f"   python scripts/infer_hybrid.py --checkpoint {save_path} --image {img_path} --full_inference")


if __name__ == '__main__':
    main()
