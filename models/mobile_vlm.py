"""
MobileVLM V2 模型包装器 (模拟实现)
"""
import torch
import torch.nn as nn
from core.framework import BaseSplitModel, register_model


class LDPv2(nn.Module):
    """Lightweight Downsample Projector v2"""
    def __init__(self, in_channels=1024, hidden_channels=512, out_channels=2048, downsample_ratio=2):
        super().__init__()
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_ratio, stride=downsample_ratio)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.avg_pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        return x


class FakeViTEncoder(nn.Module):
    """模拟 CLIP ViT-L/14 视觉编码器"""
    def __init__(self, input_size=336, patch_size=14, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.grid_size = input_size // patch_size
        
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.adaptive_pool(x)
        return x


class FakeMobileLLaMA(nn.Module):
    """模拟 MobileLLaMA 语言模型"""
    def __init__(self, hidden_size=2048, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8, 
                dim_feedforward=hidden_size * 4, batch_first=True
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MobileVLMv2(nn.Module):
    """MobileVLM V2 完整架构"""
    def __init__(self, vision_embed_dim=1024, projector_hidden=512,
                 llm_hidden_size=2048, downsample_ratio=2):
        super().__init__()
        self.vision_encoder = FakeViTEncoder(input_size=336, patch_size=14, embed_dim=vision_embed_dim)
        self.projector = LDPv2(vision_embed_dim, projector_hidden, llm_hidden_size, downsample_ratio)
        self.llm = FakeMobileLLaMA(hidden_size=llm_hidden_size, num_layers=4)
        
    def forward(self, image):
        vision_features = self.vision_encoder(image)
        projected = self.projector(vision_features)
        b, c, h, w = projected.shape
        visual_tokens = projected.flatten(2).permute(0, 2, 1)
        output = self.llm(visual_tokens)
        return output


@register_model
class MobileVLM(BaseSplitModel):
    """MobileVLM V2 模型包装器"""
    def load_model(self):
        self.model = MobileVLMv2(
            vision_embed_dim=1024, projector_hidden=512,
            llm_hidden_size=2048, downsample_ratio=2
        ).to(self.device)
        self.model.eval()
        
        self.split_points = [
            {"name": "ViT Output",       "desc": "Vision encoder output (24x24)"},
            {"name": "LDPv2 Pooled",     "desc": "After avg pooling (12x12)"},
            {"name": "LDPv2 Output",     "desc": "Projected features"},
        ]
        
        self._features = []
        self._hooks = []
        
        def make_hook():
            def hook(module, input, output):
                self._features.append(output)
            return hook
        
        self._hooks.append(self.model.vision_encoder.register_forward_hook(make_hook()))
        self._hooks.append(self.model.projector.avg_pool.register_forward_hook(make_hook()))
        self._hooks.append(self.model.projector.register_forward_hook(make_hook()))
        
    def get_features_at_splits(self, x):
        self._features = []
        if x.shape[-1] != 336:
            x = nn.functional.interpolate(x, size=(336, 336), mode='bilinear', align_corners=False)
        _ = self.model(x)
        
        processed = []
        for feat in self._features:
            if feat.dim() == 3:
                b, n, c = feat.shape
                h = w = int(n ** 0.5)
                feat = feat.permute(0, 2, 1).reshape(b, c, h, w)
            processed.append(feat)
        return processed
