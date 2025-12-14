"""
推理脚本：使用训练好的蒸馏模型提取特征

Usage:
    python infer.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
    python infer.py --checkpoint checkpoints/best_model.pth --dummy  # 使用假数据测试
"""
import argparse
import torch
import torch.nn as nn
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms

from models.adapters import FeatureAdapter


class DistilledFeatureExtractor(nn.Module):
    """
    使用蒸馏训练后的特征提取器
    Student (CNN) + Adapter -> ViT-like features
    """
    def __init__(self, student_model='mobilenetv2_100', student_layer=3,
                 student_channels=96, teacher_channels=1024):
        super().__init__()
        
        # Student backbone
        self.student = timm.create_model(
            student_model,
            pretrained=False,
            features_only=True,
            out_indices=[student_layer]
        )
        
        # Feature adapter
        self.adapter = FeatureAdapter(student_channels, teacher_channels)
        
        # Spatial alignment (if needed)
        self.spatial_align = nn.AdaptiveAvgPool2d((16, 16))
    
    def forward(self, x):
        """
        Extract ViT-like features from image
        
        Args:
            x: (B, 3, H, W) input image tensor
        Returns:
            (B, teacher_channels, 16, 16) ViT-like features
        """
        feat = self.student(x)[-1]
        adapted = self.adapter(feat)
        aligned = self.spatial_align(adapted)
        return aligned
    
    def load_checkpoint(self, checkpoint_path, device='cpu'):
        """加载训练好的权重"""
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.student.load_state_dict(ckpt['student_state_dict'])
        self.adapter.load_state_dict(ckpt['adapter_state_dict'])
        print(f"✅ 已加载检查点: {checkpoint_path}")
        return self


def get_transforms(image_size=224):
    """获取推理用的图像预处理"""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_image(image_path, transform):
    """加载并预处理图像"""
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='Feature Extraction with Distilled Model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--dummy', action='store_true', help='Use dummy image for testing')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None, help='Save features to file')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"🔧 Device: {device}")
    
    # 创建模型
    model = DistilledFeatureExtractor(
        student_model='mobilenetv2_100',
        student_layer=3,
        student_channels=96,
        teacher_channels=1024
    )
    model.load_checkpoint(args.checkpoint, device)
    model = model.to(device)
    model.eval()
    
    # 准备输入
    transform = get_transforms(224)
    
    if args.dummy:
        print("📷 使用假数据测试...")
        image = torch.randn(1, 3, 224, 224).to(device)
    elif args.image:
        print(f"📷 加载图像: {args.image}")
        image = load_image(args.image, transform).to(device)
    else:
        print("❌ 请指定 --image 或 --dummy")
        return
    
    # 推理
    with torch.no_grad():
        features = model(image)
    
    print(f"✅ 特征形状: {features.shape}")  # (1, 1024, 16, 16)
    print(f"   特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"   特征均值: {features.mean():.4f}")
    
    # 保存特征
    if args.output:
        torch.save(features.cpu(), args.output)
        print(f"💾 特征已保存: {args.output}")
    
    return features


if __name__ == '__main__':
    main()
