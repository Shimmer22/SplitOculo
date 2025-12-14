"""
特征对齐适配器
用于将 Student (CNN) 特征映射到 Teacher (ViT) 维度
"""
import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    将 Student (CNN) 的特征映射到 Teacher (ViT) 的维度。
    使用 1x1 卷积实现，计算量极小。
    """
    def __init__(self, student_channels, teacher_channels, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(teacher_channels))
        # 不加激活函数，因为要直接拟合 Teacher 的分布
        self.adapt = nn.Sequential(*layers)

    def forward(self, x):
        return self.adapt(x)


class MultiScaleAdapter(nn.Module):
    """
    多尺度特征适配器
    处理 Student 和 Teacher 特征图分辨率不同的情况
    """
    def __init__(self, student_channels, teacher_channels, 
                 student_size, teacher_size, use_bn=True):
        super().__init__()
        self.student_size = student_size
        self.teacher_size = teacher_size
        
        # 通道对齐
        self.channel_adapt = FeatureAdapter(student_channels, teacher_channels, use_bn)
        
        # 空间对齐 (如果尺寸不同)
        if student_size != teacher_size:
            self.spatial_adapt = nn.AdaptiveAvgPool2d((teacher_size, teacher_size))
        else:
            self.spatial_adapt = nn.Identity()
    
    def forward(self, x):
        x = self.channel_adapt(x)
        x = self.spatial_adapt(x)
        return x


class ProjectorLDPv2(nn.Module):
    """
    Lightweight Downsample Projector v2 (来自 MobileVLM V2)
    用于将 Vision 特征投影到 LLM 空间
    
    结构: PointWise Conv -> AvgPool -> DepthWise Conv (PEG) -> PointWise Conv
    """
    def __init__(self, in_channels, hidden_channels, out_channels, downsample_ratio=2):
        super().__init__()
        
        # Point-wise conv: 降维
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        
        # Average Pooling: token reduction
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_ratio, stride=downsample_ratio)
        
        # Depth-wise conv + PEG (位置编码增强)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        
        # Point-wise conv: 升维到 LLM hidden size
        self.pw_conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.avg_pool(x)
        
        # PEG with skip connection
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        
        x = self.bn3(self.pw_conv2(x))
        return x


class DistillationHead(nn.Module):
    """
    蒸馏头：包含适配器和损失计算
    """
    def __init__(self, student_channels, teacher_channels, 
                 student_size=None, teacher_size=None):
        super().__init__()
        
        if student_size and teacher_size and student_size != teacher_size:
            self.adapter = MultiScaleAdapter(
                student_channels, teacher_channels,
                student_size, teacher_size
            )
        else:
            self.adapter = FeatureAdapter(student_channels, teacher_channels)
    
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: (B, C_s, H_s, W_s)
            teacher_feat: (B, C_t, H_t, W_t)
        Returns:
            adapted_feat: 对齐后的 student 特征
            loss_dict: 包含各项损失
        """
        adapted = self.adapter(student_feat)
        
        # MSE Loss
        mse_loss = nn.functional.mse_loss(adapted, teacher_feat)
        
        # Cosine Similarity Loss
        # 将特征展平后计算
        adapted_flat = adapted.flatten(2)  # (B, C, H*W)
        teacher_flat = teacher_feat.flatten(2)
        cos_sim = nn.functional.cosine_similarity(adapted_flat, teacher_flat, dim=1)
        cos_loss = 1 - cos_sim.mean()
        
        return adapted, {
            'mse_loss': mse_loss,
            'cos_loss': cos_loss,
            'total_loss': mse_loss + 0.5 * cos_loss
        }
