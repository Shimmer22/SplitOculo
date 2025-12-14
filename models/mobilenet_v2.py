"""
MobileNetV2 模型包装器
"""
import timm
from core.framework import BaseSplitModel, register_model


@register_model
class MobileNetV2(BaseSplitModel):
    def load_model(self):
        self.model = timm.create_model(
            'mobilenetv2_100', 
            features_only=True, 
            out_indices=[1, 2, 3, 4],
            pretrained=True
        ).to(self.device)
        self.model.eval()
        
        self.split_points = [
            {"name": "Stride 4 (Local)",  "desc": "Detail-heavy"},
            {"name": "Stride 8 (Mid)",    "desc": "Balanced"},
            {"name": "Stride 16 (Rec)",   "desc": "Bandwidth Optimal"},
            {"name": "Stride 32 (Global)","desc": "Semantic only"},
        ]

    def get_features_at_splits(self, x):
        return self.model(x)
