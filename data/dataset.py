"""
数据集加载工具
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


def get_clip_transforms(image_size=224):
    """
    获取 CLIP 风格的图像预处理
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])


def get_imagenet_transforms(image_size=224, is_train=True):
    """
    ImageNet 预处理
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class DummyDataset(Dataset):
    """
    用于快速测试的假数据集
    """
    def __init__(self, num_samples=1000, image_size=224, num_classes=1000):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


def get_dummy_loader(batch_size=32, num_samples=1000, image_size=224, num_workers=4):
    """
    获取假数据加载器
    """
    dataset = DummyDataset(num_samples=num_samples, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def get_imagenet_loader(data_dir, batch_size=32, image_size=224, 
                        is_train=True, num_workers=8):
    """
    获取 ImageNet 数据加载器
    
    Args:
        data_dir: ImageNet 数据目录 (包含 train/ 和 val/ 子目录)
        batch_size: 批次大小
        image_size: 图像尺寸
        is_train: 是否训练集
        num_workers: 数据加载线程数
    """
    from torchvision.datasets import ImageFolder
    
    data_path = Path(data_dir)
    split = 'train' if is_train else 'val'
    dataset_path = data_path / split
    
    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    transform = get_imagenet_transforms(image_size, is_train)
    dataset = ImageFolder(dataset_path, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
