"""
LeViT 模型包装器
"""
import timm
import torch
from core.framework import BaseSplitModel, register_model


@register_model
class LeViT(BaseSplitModel):
    def load_model(self):
        self.model = timm.create_model(
            'levit_256',
            pretrained=True
        ).to(self.device)
        self.model.eval()
        
        self.split_points = [
            {"name": "Stem (Conv)",       "desc": "Initial conv embedding"},
            {"name": "Stage 1 (256d)",    "desc": "First transformer stage"},
            {"name": "Stage 2 (384d)",    "desc": "Second transformer stage"},
            {"name": "Stage 3 (512d)",    "desc": "Final transformer stage"},
        ]
        
        self._features = []
        self._hooks = []
        
        def make_hook():
            def hook(module, input, output):
                self._features.append(output)
            return hook
        
        self._hooks.append(self.model.stem.register_forward_hook(make_hook()))
        for stage in self.model.stages:
            self._hooks.append(stage.register_forward_hook(make_hook()))

    def get_features_at_splits(self, x):
        self._features = []
        _ = self.model(x)
        
        processed = []
        for feat in self._features:
            if feat.dim() == 3:
                b, n, c = feat.shape
                h = w = int(n ** 0.5)
                if h * w == n:
                    feat = feat.permute(0, 2, 1).reshape(b, c, h, w)
                else:
                    import math
                    h = int(math.sqrt(n))
                    while n % h != 0:
                        h -= 1
                    w = n // h
                    feat = feat.permute(0, 2, 1).reshape(b, c, h, w)
            processed.append(feat)
        
        return processed
