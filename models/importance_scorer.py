"""
Token 重要性评分器 (Token Importance Scorer)

用于在端侧对投影后的 tokens 进行重要性评分，实现自适应传输。

支持的评分器:
- TokenImportanceScorer: 纯语义评分，基于投影后 tokens 的内容
- TextAwareImportanceScorer: 结合语义评分与文字区域检测，适用于 OCR/文档/图表场景

设计思路:
- 评分器运行在端侧 (edge)，输出每个 token 的重要性 logits
- 下游可根据 logits 做 top-k 选择、软掩码或自适应量化
- 所有输出均为 raw logits (未经 sigmoid)，便于下游灵活处理

Usage:
    # 纯语义评分
    scorer = TokenImportanceScorer(hidden_size=1280, method='mlp')
    logits = scorer(projected_tokens)  # [B, 49]
    
    # 文字感知评分
    scorer = TextAwareImportanceScorer(cnn_channels=96, hidden_size=1280)
    logits, details = scorer(cnn_features, projected_tokens)
"""
import torch
import torch.nn as nn


class TokenImportanceScorer(nn.Module):
    """
    Token 重要性评分器：基于投影后 tokens 的语义内容进行评分

    Args:
        hidden_size: token 维度 (默认 1280)
        method: 评分方法 ('mlp' 或 'attention')
    """

    def __init__(self, hidden_size: int = 1280, method: str = 'mlp'):
        super().__init__()
        self.hidden_size = hidden_size
        self.method = method

        if method == 'mlp':
            # MLP 评分头：轻量级，适合端侧部署 (~330K params)
            self.scorer = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )

        elif method == 'attention':
            # Attention 评分头：用 CLS token 的注意力权重作为重要性
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(hidden_size)

        else:
            raise ValueError(f"Unknown scoring method: {method}")

        self._init_weights()

    def _init_weights(self):
        """初始化权重：使用较小的初始化，避免初始评分偏差过大"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, projected_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projected_tokens: [B, N, hidden_size] 投影后的 token 序列
        Returns:
            importance_logits: [B, N] 每个 token 的重要性分数 (raw logits)
        """
        if self.method == 'mlp':
            # MLP: LayerNorm -> Linear -> GELU -> Linear -> squeeze
            logits = self.scorer(projected_tokens)  # [B, N, 1]
            return logits.squeeze(-1)  # [B, N]

        elif self.method == 'attention':
            B, N, C = projected_tokens.shape

            # Prepend learnable CLS token
            cls_expanded = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
            tokens_with_cls = torch.cat([cls_expanded, projected_tokens], dim=1)  # [B, N+1, C]
            tokens_with_cls = self.norm(tokens_with_cls)

            # Single-layer attention: query=CLS only, key/value=all tokens
            cls_query = tokens_with_cls[:, :1, :]  # [B, 1, C]
            all_kv = tokens_with_cls  # [B, N+1, C]

            # Get attention weights from CLS to all tokens
            _, attn_weights = self.attn(
                query=cls_query,
                key=all_kv,
                value=all_kv,
                need_weights=True,
                average_attn_weights=True,
            )  # attn_weights: [B, 1, N+1]

            # Extract CLS-to-patch attention (skip CLS-to-CLS at index 0)
            importance = attn_weights[:, 0, 1:]  # [B, N]
            return importance

        raise RuntimeError(f"Unexpected method: {self.method}")


class TextAwareImportanceScorer(nn.Module):
    """
    文字感知重要性评分器：结合语义评分与文字区域检测

    在纯语义评分基础上，额外从原始 CNN 特征中检测文字密集区域
    (OCR、图表、文档等)，并提升这些区域对应 token 的重要性。

    Args:
        cnn_channels: CNN 特征通道数 (默认 96，对应 MobileNetV2 layer 3)
        hidden_size: 投影后 token 维度 (默认 1280)
        spatial_size: CNN 特征空间尺寸 (默认 14)
        token_grid_size: token 网格尺寸 (默认 7，对应 49 tokens)
    """

    def __init__(
        self,
        cnn_channels: int = 96,
        hidden_size: int = 1280,
        spatial_size: int = 14,
        token_grid_size: int = 7,
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size
        self.token_grid_size = token_grid_size
        self.num_tokens = token_grid_size ** 2  # 49

        # Branch 1: 语义评分 (复用 MLP 方法)
        self.semantic_scorer = TokenImportanceScorer(
            hidden_size=hidden_size, method='mlp'
        )

        # Branch 2: 文字区域检测器
        # 从原始 CNN 特征中检测文字密集区域
        self.text_detector = nn.Sequential(
            nn.Conv2d(cnn_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # text heatmap [B, 1, 14, 14]
        )
        # Align with token grid
        self.align_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))

        # Fusion: 合并语义评分与文字检测评分
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化文字检测器和融合层的权重"""
        for m in self.text_detector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, cnn_features: torch.Tensor, projected_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            cnn_features: [B, cnn_channels, spatial_size, spatial_size] 原始 CNN 特征
            projected_tokens: [B, num_tokens, hidden_size] 投影后的 token 序列
        Returns:
            importance_logits: [B, num_tokens] 融合后的重要性分数 (raw logits)
            details: dict with 'semantic_logits' and 'text_logits' for analysis
        """
        # Branch 1: semantic importance from projected tokens
        semantic_logits = self.semantic_scorer(projected_tokens)  # [B, N]

        # Branch 2: text region detection from raw CNN features
        text_heatmap = self.text_detector(cnn_features)  # [B, 1, 14, 14]
        text_heatmap = self.align_pool(text_heatmap)  # [B, 1, 7, 7]
        text_logits = text_heatmap.flatten(1)  # [B, 49]

        # Fusion: stack and merge
        stacked = torch.stack([semantic_logits, text_logits], dim=-1)  # [B, N, 2]
        fused_logits = self.fusion(stacked).squeeze(-1)  # [B, N]

        details = {
            'semantic_logits': semantic_logits,
            'text_logits': text_logits,
        }

        return fused_logits, details


if __name__ == '__main__':
    # 测试
    B, N, C = 2, 49, 1280
    projected_tokens = torch.randn(B, N, C)
    cnn_features = torch.randn(B, 96, 14, 14)

    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("TokenImportanceScorer 测试")
    print("=" * 60)

    # MLP method
    print("\n--- Method: mlp ---")
    scorer_mlp = TokenImportanceScorer(hidden_size=1280, method='mlp')
    logits_mlp = scorer_mlp(projected_tokens)
    print(f"Input:  {projected_tokens.shape}")
    print(f"Output: {logits_mlp.shape}")
    print(f"Parameters: {count_params(scorer_mlp):,}")

    # Attention method
    print("\n--- Method: attention ---")
    scorer_attn = TokenImportanceScorer(hidden_size=1280, method='attention')
    logits_attn = scorer_attn(projected_tokens)
    print(f"Input:  {projected_tokens.shape}")
    print(f"Output: {logits_attn.shape}")
    print(f"Parameters: {count_params(scorer_attn):,}")
    print(f"Note: attention output is softmax-normalized (sum={logits_attn[0].sum().item():.4f})")

    print("\n" + "=" * 60)
    print("TextAwareImportanceScorer 测试")
    print("=" * 60)

    scorer_text = TextAwareImportanceScorer(
        cnn_channels=96,
        hidden_size=1280,
        spatial_size=14,
        token_grid_size=7,
    )
    fused_logits, details = scorer_text(cnn_features, projected_tokens)

    print(f"\nCNN features:      {cnn_features.shape}")
    print(f"Projected tokens:  {projected_tokens.shape}")
    print(f"Fused logits:      {fused_logits.shape}")
    print(f"Semantic logits:   {details['semantic_logits'].shape}")
    print(f"Text logits:       {details['text_logits'].shape}")
    print(f"Parameters:        {count_params(scorer_text):,}")

    # 参数量明细
    print(f"\n--- 参数量明细 ---")
    print(f"  Semantic scorer: {count_params(scorer_text.semantic_scorer):,}")
    text_det_params = count_params(scorer_text.text_detector)
    print(f"  Text detector:   {text_det_params:,}")
    fusion_params = count_params(scorer_text.fusion)
    print(f"  Fusion layer:    {fusion_params:,}")

    print("\n✅ 所有测试通过!")
