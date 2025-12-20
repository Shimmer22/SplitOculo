"""
Projector v3: Strided Convolutions
完全弃用池化层，使用步长卷积进行下采样，以保留更多语义信息。
"""
import torch
import torch.nn as nn
import math

class StridedProjector(nn.Module):
    """
    使用步长卷积实现的 Projector。
    
    结构:
    1. PW Conv (升维/降维)
    2. DW Conv (Stride=2, Downsample) -> 核心下采样层
    3. Residual (若下采样，残差路径需处理)
    4. PW Conv (投影到目标维度)
    """
    def __init__(self, in_channels, hidden_size=1280,
                 hidden_channels=512, transmission_tokens=49,
                 input_resolution=14):
        super().__init__()
        
        self.transmission_size = int(math.sqrt(transmission_tokens)) # 7
        assert self.transmission_size ** 2 == transmission_tokens
        
        # 验证步长
        # 假设输入是 14x14, 目标是 7x7 => Stride 2
        stride = input_resolution // self.transmission_size
        if input_resolution % self.transmission_size != 0:
            print(f"Warning: Input {input_resolution} not divisible by target {self.transmission_size}. Using stride {stride} and adaptive pooling fallback.")
            self.use_fallback = True
        else:
            self.use_fallback = False
        
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        
        # 核心下采样: DW Conv with Stride
        # 如果 stride > 1，则会有下采样效果
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                  stride=stride, padding=1, 
                                  groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        
        # 残差路径的下采样
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),  # 这里的 1x1 不做 stride
                nn.AvgPool2d(kernel_size=stride, stride=stride) # 残差路径可以用简单的池化
            )
        else:
            self.downsample = nn.Identity()
            
        self.pw_conv2 = nn.Conv2d(hidden_channels, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        
        # 最后的保障：如果尺寸还不匹配（例如输入尺寸波动），用 AdaptiveAvgPool 兜底
        self.fallback_pool = nn.AdaptiveAvgPool2d((self.transmission_size, self.transmission_size))

    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        
        residual = self.downsample(x)
        
        x = self.act2(self.bn2(self.dw_conv(x)))
        
        if x.shape == residual.shape:
             x = x + residual
        
        # 兜底保障确保 7x7
        if x.shape[2] != self.transmission_size:
            x = self.fallback_pool(x)
            
        x = self.bn3(self.pw_conv2(x))
        
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x
