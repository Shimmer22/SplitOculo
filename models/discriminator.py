"""
Feature Discriminator for SplitOculo v2.0 GAN Training

鉴别器用于区分:
- 真实特征: Qwen ViT 提取的原生特征 (sharp, 分布自然)
- 假特征: CNN + Upsampler 生成的特征 (可能模糊, 分布偏移)

通过对抗训练，迫使 Generator 输出的特征分布更接近 Qwen 原生特征。
"""
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class FeatureDiscriminator(nn.Module):
    """
    特征鉴别器: 判断输入的 256 tokens 是否来自 Qwen ViT
    
    设计思路:
    - 使用 1D Conv 扫描 token 序列 (把每个 token 看作一个时间步)
    - Spectral Normalization 防止 GAN 训练不稳定
    - 输出单个分数: 真实特征 → 1, 假特征 → 0
    """
    def __init__(self, hidden_size=1280, num_tokens=256):
        super().__init__()
        
        # Spectral Norm 包装函数
        def SN(module):
            return spectral_norm(module)
        
        # 特征压缩网络
        self.net = nn.Sequential(
            # [B, 1280, 256] → [B, 512, 256]
            SN(nn.Conv1d(hidden_size, 512, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 512, 256] → [B, 256, 256]
            SN(nn.Conv1d(512, 256, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 256, 256] → [B, 128, 256]
            SN(nn.Conv1d(256, 128, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 128, 256] → [B, 1, 256]
            SN(nn.Conv1d(128, 1, kernel_size=1)),
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 256, 1280] - token 序列
        Returns:
            [B, 1] - 真实性分数 (logits, 未经 sigmoid)
        """
        # permute: [B, 256, 1280] → [B, 1280, 256]
        x = x.transpose(1, 2)
        
        # 通过网络: [B, 1280, 256] → [B, 1, 256]
        score = self.net(x)
        
        # Global Average Pooling: [B, 1, 256] → [B, 1]
        score = score.mean(dim=2)
        
        return score


class PatchDiscriminator(nn.Module):
    """
    Patch-based 鉴别器 (可选)
    
    对每个 token 单独打分，提供更精细的梯度信号
    """
    def __init__(self, hidden_size=1280):
        super().__init__()
        
        def SN(module):
            return spectral_norm(module)
        
        # 对每个 token 独立判别
        self.net = nn.Sequential(
            SN(nn.Linear(hidden_size, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Linear(256, 1)),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, 1280]
        Returns:
            [B, N, 1] - 每个 token 的真实性分数
        """
        return self.net(x)


if __name__ == '__main__':
    # 测试
    B, N, C = 2, 256, 1280
    x = torch.randn(B, N, C)
    
    # 测试 FeatureDiscriminator
    disc = FeatureDiscriminator(hidden_size=C, num_tokens=N)
    score = disc(x)
    print(f"FeatureDiscriminator: {x.shape} → {score.shape}")  # [2, 1]
    
    # 测试 PatchDiscriminator
    patch_disc = PatchDiscriminator(hidden_size=C)
    patch_score = patch_disc(x)
    print(f"PatchDiscriminator: {x.shape} → {patch_score.shape}")  # [2, 256, 1]
    
    # 参数量
    from core.utils import count_parameters
    print(f"\nFeatureDiscriminator params: {count_parameters(disc) / 1e6:.2f}M")
    print(f"PatchDiscriminator params: {count_parameters(patch_disc) / 1e6:.2f}M")
    
    print("\n✅ 所有测试通过!")
