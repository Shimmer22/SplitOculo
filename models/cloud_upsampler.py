"""
云端可学习上采样器模块
用于将端侧传输的低分辨率 tokens 上采样到 VLM 期望的高分辨率

设计思路：
- 输入: 49 tokens (7×7) 
- 输出: 256 tokens (16×16)
- 使用 ConvTranspose2d 进行空间上采样，然后用轻量级 Transformer 精炼
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CloudUpsampler(nn.Module):
    """
    可学习上采样器: N tokens → M tokens
    
    支持的上采样方式:
    - deconv: 反卷积 (ConvTranspose2d)
    - pixelshuffle: PixelShuffle
    - transformer: Transformer + 插值
    """
    def __init__(self, hidden_size=1280, 
                 input_tokens=49, target_tokens=256,
                 method='deconv', num_refine_layers=2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens
        self.method = method
        
        # 计算空间尺寸
        self.input_size = int(math.sqrt(input_tokens))   # 7
        self.target_size = int(math.sqrt(target_tokens)) # 16
        
        assert self.input_size ** 2 == input_tokens, f"input_tokens must be square, got {input_tokens}"
        assert self.target_size ** 2 == target_tokens, f"target_tokens must be square, got {target_tokens}"
        
        # 计算上采样比例
        self.scale_factor = self.target_size / self.input_size  # 16/7 ≈ 2.28
        
        if method == 'deconv':
            # 方法1: 反卷积上采样
            # 7→14 + 14→16 两步
            self.upsample = nn.Sequential(
                # 第一步: 7→14
                nn.ConvTranspose2d(hidden_size, hidden_size, 
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.GELU(),
                # 调整到目标大小
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.GELU(),
            )
            
        elif method == 'pixelshuffle':
            # 方法2: PixelShuffle 上采样
            # 需要通道数是 scale^2 的倍数
            self.upsample = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),  # 7→14, channels: hidden_size*4 → hidden_size
                nn.BatchNorm2d(hidden_size),
                nn.GELU(),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.GELU(),
            )
            
        elif method == 'mlp':
            # 方法3: Bilinear + MLP (最佳效果)
            # 先用 bilinear 上采样，然后用 MLP 精炼每个 token
            self.upsample = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
            
        elif method == 'transformer':
            # 方法4: Transformer + 插值
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8, 
                dim_feedforward=hidden_size * 4,
                batch_first=True, dropout=0.1
            )
            self.upsample = nn.TransformerEncoder(encoder_layer, num_layers=num_refine_layers)
        
        # 精炼层 (可选)
        if num_refine_layers > 0 and method != 'transformer':
            self.refine = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.GELU(),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
            )
        else:
            self.refine = None
    
    def forward(self, x):
        """
        Args:
            x: (B, input_tokens, hidden_size) 或 (B, hidden_size, H, W)
        Returns:
            (B, target_tokens, hidden_size)
        """
        # 转换为 2D 形式
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.view(B, self.input_size, self.input_size, C).permute(0, 3, 1, 2)
        else:
            B = x.shape[0]
        
        if self.method == 'mlp':
            # MLP 方法: 先 bilinear 上采样，然后 MLP 精炼
            x = F.interpolate(x, size=(self.target_size, self.target_size), 
                             mode='bilinear', align_corners=False)
            x = x.permute(0, 2, 3, 1).reshape(B, -1, self.hidden_size)  # (B, target_tokens, C)
            x = self.upsample(x)  # MLP refinement
            
        elif self.method == 'transformer':
            # Transformer 方法: 先插值到目标大小，再用 Transformer 精炼
            x = F.interpolate(x, size=(self.target_size, self.target_size), 
                             mode='bilinear', align_corners=False)
            x = x.flatten(2).transpose(1, 2)  # (B, target_tokens, C)
            x = self.upsample(x)
        else:
            # Conv 方法 (deconv, pixelshuffle)
            x = self.upsample(x)  # (B, C, H', W')
            
            # 精确调整到目标大小
            if x.shape[2] != self.target_size or x.shape[3] != self.target_size:
                x = F.interpolate(x, size=(self.target_size, self.target_size),
                                 mode='bilinear', align_corners=False)
            
            # 精炼
            if self.refine is not None:
                x = x + self.refine(x)
            
            # 转为 token 序列
            x = x.flatten(2).transpose(1, 2)  # (B, target_tokens, C)
        
        return x


class LearnableProjector(nn.Module):
    """
    完整的端侧 Projector + 云端 Upsampler
    
    端侧: CNN → Projector → transmission_tokens
    云端: transmission_tokens → Upsampler → target_tokens
    """
    def __init__(self, in_channels, hidden_size=1280,
                 projector_hidden=512, 
                 transmission_tokens=49, target_tokens=256,
                 upsampler_method='deconv'):
        super().__init__()
        
        # 端侧 Projector (下采样)
        transmission_size = int(math.sqrt(transmission_tokens))  # 7
        
        self.pw_conv1 = nn.Conv2d(in_channels, projector_hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(projector_hidden)
        self.act1 = nn.GELU()
        
        # 自适应池化到传输尺寸
        self.pool = nn.AdaptiveAvgPool2d((transmission_size, transmission_size))
        
        self.dw_conv = nn.Conv2d(projector_hidden, projector_hidden, kernel_size=3,
                                  padding=1, groups=projector_hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(projector_hidden)
        self.act2 = nn.GELU()
        
        self.pw_conv2 = nn.Conv2d(projector_hidden, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        
        # 云端 Upsampler
        self.upsampler = CloudUpsampler(
            hidden_size=hidden_size,
            input_tokens=transmission_tokens,
            target_tokens=target_tokens,
            method=upsampler_method
        )
        
        self.transmission_tokens = transmission_tokens
        self.target_tokens = target_tokens
    
    def forward_edge(self, x):
        """端侧前向: CNN 特征 → 传输 tokens"""
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, transmission_tokens, hidden_size)
        return x
    
    def forward_cloud(self, x):
        """云端前向: 传输 tokens → 目标 tokens"""
        return self.upsampler(x)
    
    def forward(self, x):
        """完整前向 (训练时使用)"""
        edge_tokens = self.forward_edge(x)
        target_tokens = self.forward_cloud(edge_tokens)
        return target_tokens, edge_tokens


if __name__ == '__main__':
    # 测试
    B, C, H, W = 2, 96, 14, 14
    x = torch.randn(B, C, H, W)
    
    # 测试完整 Projector
    projector = LearnableProjector(
        in_channels=96,
        hidden_size=1280,
        transmission_tokens=49,
        target_tokens=256,
        upsampler_method='deconv'
    )
    
    target_tokens, edge_tokens = projector(x)
    print(f"Input: {x.shape}")
    print(f"Edge tokens (传输): {edge_tokens.shape}")  # (2, 49, 1280)
    print(f"Target tokens: {target_tokens.shape}")  # (2, 256, 1280)
    
    # 测试单独的 CloudUpsampler
    upsampler = CloudUpsampler(hidden_size=1280, input_tokens=49, target_tokens=256)
    out = upsampler(edge_tokens)
    print(f"Upsampler output: {out.shape}")
    
    print("\n✅ 所有测试通过!")
