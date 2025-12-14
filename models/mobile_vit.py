"""
MobileViT 模型包装器
"""
import timm
from core.framework import BaseSplitModel, register_model


@register_model
class MobileViT(BaseSplitModel):
    def load_model(self):
        self.model = timm.create_model(
            'mobilevit_s',
            features_only=True,
            out_indices=[1, 2, 3, 4],
            pretrained=True
        ).to(self.device)
        self.model.eval()
        
        self.split_points = [
            {"name": "Stage 1 (Conv)",    "desc": "Early CNN features"},
            {"name": "Stage 2 (MV2)",     "desc": "MobileNetV2 blocks"},
            {"name": "Stage 3 (MViT)",    "desc": "MobileViT block"},
            {"name": "Stage 4 (Global)",  "desc": "Final features"},
        ]

    def get_features_at_splits(self, x):
        return self.model(x)
