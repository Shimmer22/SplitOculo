"""Pooling token projector.

Projects CNN feature maps into a fixed number of transmission tokens using
adaptive pooling and lightweight convolutions.
"""

import math

import torch.nn as nn


class PoolingTokenProjector(nn.Module):
    """Project CNN features into transmission tokens with adaptive pooling."""

    def __init__(
        self,
        in_channels,
        hidden_size=1280,
        hidden_channels=512,
        transmission_tokens=49,
    ):
        super().__init__()

        self.transmission_size = int(math.sqrt(transmission_tokens))
        assert self.transmission_size ** 2 == transmission_tokens

        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((self.transmission_size, self.transmission_size))
        self.dw_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_channels, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        return x.flatten(2).transpose(1, 2)
