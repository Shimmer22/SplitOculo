"""Lightweight motion-aided feature memories inspired by MMNet and LSFA.

The original codec accelerator only propagated a feature map with motion
vectors.  MMNet's important addition is an appearance update: motion aligns
the old state and the residual provides new information for the current
frame.  This module implements that update at the CNN feature resolution.

The module is deliberately small and starts as an identity correction when
newly initialized.  A separately trained checkpoint is therefore required to
enable the learned correction during inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


class MMNetFeatureMemory(nn.Module):
    """Gated residual correction after motion-compensated feature warping.

    Args:
        feature_channels: Number of channels in the edge CNN feature map.
        residual_channels: Channels in the decoded residual proxy, normally 3.
        motion_channels: Motion-vector channels, normally 2.
        hidden_channels: Width of the lightweight residual encoder.

    Inputs are all BCHW tensors at the edge CNN feature resolution:

    * ``warped_feature``: previous feature after MV alignment;
    * ``residual``: current decoded-frame residual proxy;
    * ``motion``: motion vectors expressed in feature-cell units;
    * ``valid_mask``: 1 where a past-reference MV covers the cell.
    """

    def __init__(
        self,
        feature_channels: int,
        residual_channels: int = 3,
        motion_channels: int = 2,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        feature_channels = int(feature_channels)
        residual_channels = int(residual_channels)
        motion_channels = int(motion_channels)
        hidden_channels = int(hidden_channels or max(32, min(feature_channels, 96)))

        self.feature_channels = feature_channels
        self.residual_channels = residual_channels
        self.motion_channels = motion_channels
        self.hidden_channels = hidden_channels

        update_input_channels = residual_channels + motion_channels + 1
        self.update_encoder = nn.Sequential(
            nn.Conv2d(update_input_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(self._group_count(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, feature_channels, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels + update_input_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self._init_identity()

    @staticmethod
    def _group_count(channels: int) -> int:
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def _init_identity(self):
        """Make an untrained memory an exact no-op over the warped feature."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # The last update layer is zero, so a fresh checkpoint cannot alter
        # the old warp path before it has been trained on temporal data.
        nn.init.zeros_(self.update_encoder[-1].weight)
        nn.init.zeros_(self.update_encoder[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(
        self,
        warped_feature: torch.Tensor,
        residual: torch.Tensor,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if warped_feature.ndim != 4:
            raise ValueError(f"warped_feature must be BCHW, got {warped_feature.shape}")
        if residual.shape[-2:] != warped_feature.shape[-2:]:
            raise ValueError("residual and warped_feature must have the same spatial size")
        if motion.shape[-2:] != warped_feature.shape[-2:]:
            raise ValueError("motion and warped_feature must have the same spatial size")

        device = warped_feature.device
        dtype = warped_feature.dtype
        residual = residual.to(device=device, dtype=dtype)
        motion = motion.to(device=device, dtype=dtype)
        valid_mask = valid_mask.to(device=device, dtype=dtype)

        # Codec MVs can be large in feature-cell units.  Bounded scaling keeps
        # the correction branch numerically stable while retaining direction.
        motion = torch.tanh(motion / 4.0)
        update_input = torch.cat((residual, motion, valid_mask), dim=1)
        delta = self.update_encoder(update_input)
        gate_input = torch.cat((warped_feature, update_input), dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        return warped_feature + gate * delta


class LSFAFeatureMemory(nn.Module):
    """LSFA-like fusion of warped, residual, and current-image features.

    LSFA [Wang et al., 2021] uses three complementary inputs for a non-key
    frame: a motion-vector-warped feature, a feature projected from the codec
    residual, and a small CNN applied to the current image.  The real codec
    residual is not portable through the current PyAV path, so the caller may
    still provide the decoded RGB residual proxy.  The current-image branch
    is intentionally shallow and operates at the edge feature resolution.

    The branch gates are spatial (two gates per feature cell) so the module
    can suppress residual/tiny-image cues in unreliable regions while keeping
    the old warped feature as the identity path.
    """

    def __init__(
        self,
        feature_channels: int,
        residual_channels: int = 3,
        motion_channels: int = 2,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        feature_channels = int(feature_channels)
        residual_channels = int(residual_channels)
        motion_channels = int(motion_channels)
        hidden_channels = int(hidden_channels or max(32, min(feature_channels, 96)))

        self.feature_channels = feature_channels
        self.residual_channels = residual_channels
        self.motion_channels = motion_channels
        self.hidden_channels = hidden_channels

        # The original LSFA residual path is a 1x1 convolution.
        self.residual_projection = nn.Conv2d(
            residual_channels, feature_channels, kernel_size=1
        )
        # A tiny current-frame branch.  Its input is resized to the edge
        # feature grid by forward(), keeping the runtime cost small.
        self.tiny_encoder = nn.Sequential(
            nn.Conv2d(residual_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self._group_count(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=3, padding=1),
        )

        # Use motion and coverage as confidence cues for the two correction
        # branches.  The warped feature remains an explicit identity path.
        gate_input_channels = feature_channels + residual_channels + motion_channels + 1
        self.branch_gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self._init_identity()

    @staticmethod
    def _group_count(channels: int) -> int:
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def _init_identity(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Start as the existing MV warp path.  This makes an untrained
        # checkpoint safe and lets the learned branches be introduced by the
        # temporal training script.
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        nn.init.zeros_(self.tiny_encoder[-1].weight)
        nn.init.zeros_(self.tiny_encoder[-1].bias)
        nn.init.zeros_(self.branch_gate[-1].weight)
        nn.init.constant_(self.branch_gate[-1].bias, -1.0)

    def forward(
        self,
        warped_feature: torch.Tensor,
        residual: torch.Tensor,
        current_rgb: torch.Tensor,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if warped_feature.ndim != 4:
            raise ValueError(f"warped_feature must be BCHW, got {warped_feature.shape}")
        if current_rgb.ndim != 4:
            raise ValueError(f"current_rgb must be BCHW, got {current_rgb.shape}")
        feature_size = warped_feature.shape[-2:]
        if residual.shape[-2:] != feature_size or motion.shape[-2:] != feature_size:
            raise ValueError("residual and motion must match warped_feature spatial size")

        device = warped_feature.device
        dtype = warped_feature.dtype
        residual = residual.to(device=device, dtype=dtype)
        current_rgb = current_rgb.to(device=device, dtype=dtype)
        motion = motion.to(device=device, dtype=dtype)
        valid_mask = valid_mask.to(device=device, dtype=dtype)
        if current_rgb.shape[-2:] != feature_size:
            current_rgb = F.interpolate(
                current_rgb, size=feature_size, mode="bilinear", align_corners=False
            )

        normalized_motion = torch.tanh(motion / 4.0)
        gate_input = torch.cat(
            (warped_feature, residual, normalized_motion, valid_mask), dim=1
        )
        gates = torch.sigmoid(self.branch_gate(gate_input))
        residual_feature = self.residual_projection(residual)
        tiny_feature = self.tiny_encoder(current_rgb)
        return (
            warped_feature
            + gates[:, 0:1] * residual_feature
            + gates[:, 1:2] * tiny_feature
        )
