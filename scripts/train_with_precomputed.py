"""
使用预计算的 Qwen 特征进行训练

先运行 precompute_qwen_features.py 提取特征，然后用本脚本训练。
这样训练速度可以提升 10-50 倍。

Usage:
    python scripts/train_with_precomputed.py --features_dir ./data/qwen_features --data_dir ./data/imagenette2-320
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import timm

from core.utils import set_seed, get_logger, count_parameters


class LLMProjector(nn.Module):
    """将 CNN 特征投影到 LLM 隐藏空间"""
    def __init__(self, in_channels, llm_hidden_size=2048, 
                 hidden_channels=512, downsample_ratio=2):
        super().__init__()
        
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_ratio, stride=downsample_ratio)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_channels, llm_hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(llm_hidden_size)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.avg_pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class PrecomputedFeatureDataset(Dataset):
    """
    加载预计算的 Qwen 特征和对应的原始图像
    """
    def __init__(self, features_dir, images_dir, split='train', image_size=224):
        self.features_dir = Path(features_dir) / split
        self.images_dir = Path(images_dir) / split
        
        # 加载元数据
        metadata_path = self.features_dir / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError(f"找不到元数据: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # 获取所有特征文件
        self.feature_files = sorted(self.features_dir.glob("*.pt"))
        print(f"📂 Found {len(self.feature_files)} precomputed features")
        
        # 图像变换
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
        # 加载预计算特征
        feature_data = torch.load(self.feature_files[idx], weights_only=False)
        teacher_features = feature_data['features']  # (num_tokens, hidden_size)
        img_path = feature_data['path']
        
        # 加载原始图像
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        return img_tensor, teacher_features


def collate_fn(batch):
    """
    自定义 collate 函数，处理不同长度的特征
    """
    images = torch.stack([item[0] for item in batch])
    
    # 对齐特征长度（取最小值）
    features = [item[1] for item in batch]
    min_tokens = min(f.shape[0] for f in features)
    
    # 截断到相同长度
    aligned_features = torch.stack([f[:min_tokens] for f in features])
    
    return images, aligned_features


class PrecomputedTrainer:
    """使用预计算特征的训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger('precomputed_train', log_file=f'{args.output_dir}/train.log')
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self._load_student()
        
        # 优化器
        self.optimizer = optim.AdamW(
            list(self.student.parameters()) + list(self.projector.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        
        self.logger.info(f"Student params: {count_parameters(self.student) / 1e6:.2f}M")
        self.logger.info(f"Projector params: {count_parameters(self.projector) / 1e6:.2f}M")
    
    def _load_student(self):
        """加载 Student 模型"""
        self.logger.info(f"Loading student: {self.args.student_model}")
        
        self.student = timm.create_model(
            self.args.student_model,
            pretrained=True,
            features_only=True,
            out_indices=[self.args.student_layer]
        ).to(self.device)
        
        # 获取输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.args.image_size, self.args.image_size).to(self.device)
            student_feat = self.student(dummy)[-1]
            self.student_channels = student_feat.shape[1]
            self.student_size = student_feat.shape[2]
        
        self.logger.info(f"Student output: {self.student_channels} channels, {self.student_size}x{self.student_size}")
        
        # Projector
        self.projector = LLMProjector(
            in_channels=self.student_channels,
            llm_hidden_size=self.args.llm_hidden_size,
            hidden_channels=self.args.projector_hidden,
            downsample_ratio=self.args.downsample_ratio
        ).to(self.device)
        
        output_size = self.student_size // self.args.downsample_ratio
        self.num_tokens = output_size * output_size
        self.logger.info(f"Output tokens: {self.num_tokens} ({output_size}x{output_size})")
    
    def _align_tokens(self, tokens, target_num):
        """对齐 token 数量"""
        b, n, c = tokens.shape
        h = w = int(n ** 0.5)
        
        tokens = tokens.view(b, h, w, c).permute(0, 3, 1, 2)
        
        target_h = target_w = int(target_num ** 0.5)
        tokens = nn.functional.adaptive_avg_pool2d(tokens, (target_h, target_w))
        
        tokens = tokens.permute(0, 2, 3, 1).view(b, -1, c)
        return tokens
    
    def train_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        self.student.train()
        self.projector.train()
        
        total_loss = 0
        total_mse = 0
        total_cos = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for images, teacher_tokens in pbar:
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device)
            
            # Student forward
            student_feat = self.student(images)[-1]
            student_tokens = self.projector(student_feat)
            
            # 对齐 token 数量
            if student_tokens.shape[1] != teacher_tokens.shape[1]:
                student_tokens = self._align_tokens(student_tokens, teacher_tokens.shape[1])
            
            # 计算损失
            mse = self.mse_loss(student_tokens, teacher_tokens.float())
            
            b, n, c = student_tokens.shape
            student_flat = student_tokens.contiguous().view(b * n, c)
            teacher_flat = teacher_tokens.contiguous().view(b * n, c).float()
            target = torch.ones(b * n).to(self.device)
            cos = self.cos_loss(student_flat, teacher_flat, target)
            
            loss = mse + self.args.cos_weight * cos
            
            # 反向传播
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
        return {
            'loss': total_loss / n,
            'mse': total_mse / n,
            'cos': total_cos / n
        }
    
    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.student.eval()
        self.projector.eval()
        
        total_mse = 0
        total_cos_sim = 0
        
        for images, teacher_tokens in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device)
            
            student_feat = self.student(images)[-1]
            student_tokens = self.projector(student_feat)
            
            if student_tokens.shape[1] != teacher_tokens.shape[1]:
                student_tokens = self._align_tokens(student_tokens, teacher_tokens.shape[1])
            
            mse = self.mse_loss(student_tokens, teacher_tokens.float())
            cos_sim = nn.functional.cosine_similarity(
                student_tokens, teacher_tokens.float(), dim=-1
            ).mean()
            
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
            
            train_metrics = self.train_epoch(train_loader, epoch)
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                           f"MSE: {train_metrics['mse']:.4f}, "
                           f"Cos: {train_metrics['cos']:.4f}")
            
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.logger.info(f"Val - MSE: {val_metrics['val_mse']:.4f}, "
                               f"Cos Sim: {val_metrics['val_cos_sim']:.4f}")
            
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
            'projector_state_dict': self.projector.state_dict(),
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
    parser = argparse.ArgumentParser(description='Train with Precomputed Qwen Features')
    
    # 数据参数
    parser.add_argument('--features_dir', type=str, default='./data/qwen_features',
                        help='Precomputed features directory')
    parser.add_argument('--data_dir', type=str, default='./data/imagenette2-320',
                        help='Original images directory')
    parser.add_argument('--image_size', type=int, default=224)
    
    # Student 参数
    parser.add_argument('--student_model', type=str, default='mobilenetv2_100')
    parser.add_argument('--student_layer', type=int, default=3)
    parser.add_argument('--llm_hidden_size', type=int, default=2048)
    parser.add_argument('--projector_hidden', type=int, default=512)
    parser.add_argument('--downsample_ratio', type=int, default=2)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--cos_weight', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # 其他参数
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/qwen_precomputed')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 创建数据集
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
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 训练
    trainer = PrecomputedTrainer(args)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
