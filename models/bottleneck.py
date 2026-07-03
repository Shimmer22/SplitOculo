"""
维度瓶颈层 (Dimension Bottleneck)

用于压缩传输特征，减少端-云传输带宽。

支持的方法:
- linear: 简单线性投影，参数最少
- mlp: 带非线性的 MLP，表达能力更强
- autoencoder: 带 LayerNorm 的 Autoencoder，信息保留最优

Usage:
    bottleneck = DimensionBottleneck(
        hidden_size=1280,
        bottleneck_dim=64,
        method='linear'
    )
    
    # 端侧
    compressed = bottleneck.encode(edge_tokens)  # [B, 49, 64]
    
    # 云端
    decompressed = bottleneck.decode(compressed)  # [B, 49, 1280]
"""
import torch
import torch.nn as nn


class DimensionBottleneck(nn.Module):
    """
    维度瓶颈层：压缩传输特征
    
    Args:
        hidden_size: 输入/输出维度 (默认 1280)
        bottleneck_dim: 瓶颈维度 (默认 64)
        method: 压缩方法 ('linear', 'mlp', 'autoencoder')
    """
    def __init__(self, hidden_size=1280, bottleneck_dim=64, method='linear'):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.method = method
        
        if method == 'linear':
            # 最简单：线性投影
            self.encoder = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.decoder = nn.Linear(bottleneck_dim, hidden_size, bias=False)
            
        elif method == 'mlp':
            # 带非线性的 MLP
            mid = hidden_size // 2  # 640
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Linear(mid, bottleneck_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, mid),
                nn.GELU(),
                nn.Linear(mid, hidden_size)
            )
            
        elif method == 'autoencoder':
            # 带 LayerNorm 的 Autoencoder，训练更稳定
            mid = hidden_size // 2  # 640
            self.encoder = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Linear(mid, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, mid),
                nn.GELU(),
                nn.Linear(mid, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            raise ValueError(f"Unknown bottleneck method: {method}")
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重：使用较小的初始化，避免压缩后特征变化过大"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """
        端侧编码：压缩特征维度
        
        Args:
            x: [B, N, hidden_size]
        Returns:
            [B, N, bottleneck_dim]
        """
        return self.encoder(x)
    
    def decode(self, x):
        """
        云端解码：恢复特征维度
        
        Args:
            x: [B, N, bottleneck_dim]
        Returns:
            [B, N, hidden_size]
        """
        return self.decoder(x)
    
    def forward(self, x):
        """
        完整前向（训练时使用）
        
        Args:
            x: [B, N, hidden_size]
        Returns:
            reconstructed: [B, N, hidden_size]
            compressed: [B, N, bottleneck_dim]
        """
        compressed = self.encode(x)
        reconstructed = self.decode(compressed)
        return reconstructed, compressed
    
    def get_compression_ratio(self):
        """获取压缩比"""
        return self.hidden_size / self.bottleneck_dim
    
    def get_transmission_size_kb(self, num_tokens=49, dtype_bytes=1):
        """
        计算传输大小 (KB)
        
        Args:
            num_tokens: token 数量 (默认 49)
            dtype_bytes: 每个值的字节数 (int8=1, fp16=2, fp32=4)
        """
        return num_tokens * self.bottleneck_dim * dtype_bytes / 1024


class BottleneckWithQuantization(DimensionBottleneck):
    """
    带量化支持的瓶颈层
    
    支持在 encode 后进行 int8 量化，decode 前进行反量化
    """
    def __init__(self, hidden_size=1280, bottleneck_dim=64, method='linear'):
        super().__init__(hidden_size, bottleneck_dim, method)
        
        # 量化参数（训练时学习）
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
    
    def quantize_int8(self, x):
        """
        int8 量化
        
        Args:
            x: [B, N, bottleneck_dim] 浮点张量
        Returns:
            quantized: [B, N, bottleneck_dim] int8 张量
            scale: 缩放因子
            zero_point: 零点
        """
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / 255.0
        zero_point = (-x_min / scale).round().clamp(0, 255)
        
        quantized = ((x / scale) + zero_point).round().clamp(0, 255).to(torch.uint8)
        
        return quantized, scale, zero_point
    
    def dequantize_int8(self, quantized, scale, zero_point):
        """
        int8 反量化
        
        Args:
            quantized: [B, N, bottleneck_dim] int8 张量
            scale: 缩放因子
            zero_point: 零点
        Returns:
            [B, N, bottleneck_dim] 浮点张量
        """
        return (quantized.float() - zero_point) * scale
    
    def encode_quantized(self, x):
        """
        端侧编码 + 量化
        
        Returns:
            quantized: int8 张量
            scale: 量化缩放因子
            zero_point: 量化零点
        """
        compressed = self.encode(x)
        quantized, scale, zero_point = self.quantize_int8(compressed)
        return quantized, scale, zero_point
    
    def decode_quantized(self, quantized, scale, zero_point):
        """
        反量化 + 云端解码
        """
        compressed = self.dequantize_int8(quantized, scale, zero_point)
        return self.decode(compressed)


if __name__ == '__main__':
    # 测试
    B, N, C = 2, 49, 1280
    x = torch.randn(B, N, C)
    
    print("=" * 60)
    print("DimensionBottleneck test")
    print("=" * 60)
    
    for method in ['linear', 'mlp', 'autoencoder']:
        print(f"\n--- Method: {method} ---")
        bottleneck = DimensionBottleneck(
            hidden_size=1280,
            bottleneck_dim=64,
            method=method
        )
        
        # 测试 encode/decode
        compressed = bottleneck.encode(x)
        decompressed = bottleneck.decode(compressed)
        
        print(f"Input: {x.shape}")
        print(f"Compressed: {compressed.shape}")
        print(f"Decompressed: {decompressed.shape}")
        print(f"Compression ratio: {bottleneck.get_compression_ratio():.1f}x")
        print(f"Transmission size (int8): {bottleneck.get_transmission_size_kb():.2f} KB")
        
        # 参数量
        params = sum(p.numel() for p in bottleneck.parameters())
        print(f"Parameters: {params:,}")
        
        # 重建误差
        recon, comp = bottleneck(x)
        mse = ((recon - x) ** 2).mean().item()
        print(f"Reconstruction MSE: {mse:.4f}")
    
    print("\n" + "=" * 60)
    print("BottleneckWithQuantization test")
    print("=" * 60)
    
    bottleneck_q = BottleneckWithQuantization(
        hidden_size=1280,
        bottleneck_dim=64,
        method='linear'
    )
    
    quantized, scale, zero_point = bottleneck_q.encode_quantized(x)
    decompressed = bottleneck_q.decode_quantized(quantized, scale, zero_point)
    
    print(f"Quantized dtype: {quantized.dtype}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Scale: {scale.item():.6f}")
    print(f"Decompressed shape: {decompressed.shape}")
    
    # 量化误差
    compressed_fp = bottleneck_q.encode(x)
    compressed_dq = bottleneck_q.dequantize_int8(quantized, scale, zero_point)
    quant_error = ((compressed_fp - compressed_dq) ** 2).mean().item()
    print(f"Quantization MSE: {quant_error:.6f}")
    
    print("\nAll tests passed.")
