"""Lightweight two-frame fusion for native Qwen video temporal patches."""

import torch
import torch.nn as nn


class TemporalPairFusion(nn.Module):
    """Fuse two shared-backbone feature maps into one temporal-grid feature.

    The residual branch is zero-initialized, so a new module starts as the
    arithmetic mean of the two frame features.  For a repeated still image
    pair ``(I, I)`` this exactly preserves the old single-frame feature map.
    """

    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        hidden_channels = min(int(hidden_channels), int(in_channels))
        self.in_channels = int(in_channels)
        self.hidden_channels = hidden_channels
        groups = min(32, hidden_channels)
        while hidden_channels % groups:
            groups -= 1

        self.residual = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.in_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, frame0, frame1):
        if frame0.shape != frame1.shape:
            raise ValueError(
                f"paired feature shapes must match, got {frame0.shape} and {frame1.shape}"
            )
        mean = (frame0 + frame1) * 0.5
        motion = frame1 - frame0
        residual = self.residual(torch.cat((frame0, frame1, motion), dim=1))
        return mean + residual


def load_temporal_pair_fusion(checkpoint_path, device="cpu"):
    """Load a fusion-only checkpoint produced by train_temporal_pair.py."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    fusion = TemporalPairFusion(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint.get("hidden_channels", 256),
    ).to(device)
    fusion.load_state_dict(checkpoint["temporal_fusion_state_dict"])
    return fusion, checkpoint
