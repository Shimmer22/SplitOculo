"""
数据模块
"""
from .dataset import get_imagenet_loader, get_dummy_loader, get_clip_transforms

__all__ = ['get_imagenet_loader', 'get_dummy_loader', 'get_clip_transforms']
