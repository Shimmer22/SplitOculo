"""
Strided token projector.

Uses strided depthwise convolution to reduce spatial resolution while
preserving more local structure than pure pooling.
"""

import math

import torch
import torch.nn as nn


class StridedTokenProjector(nn.Module):
    """Project CNN features into transmission tokens with strided convolutions."""

    def __init__(
        self,
        in_channels,
        hidden_size=1280,
        hidden_channels=512,
        transmission_tokens=49,
        input_resolution=14,
    ):
        super().__init__()

        self.transmission_size = int(math.sqrt(transmission_tokens))
        assert self.transmission_size ** 2 == transmission_tokens

        stride = input_resolution // self.transmission_size
        if input_resolution % self.transmission_size != 0:
            print(
                f"Warning: Input {input_resolution} not divisible by target "
                f"{self.transmission_size}. Using stride {stride} and adaptive pooling fallback."
            )

        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()

        self.dw_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()

        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
                nn.AvgPool2d(kernel_size=stride, stride=stride),
            )
        else:
            self.downsample = nn.Identity()

        self.pw_conv2 = nn.Conv2d(hidden_channels, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        self.fallback_pool = nn.AdaptiveAvgPool2d((self.transmission_size, self.transmission_size))

    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))

        residual = self.downsample(x)
        x = self.act2(self.bn2(self.dw_conv(x)))

        if x.shape == residual.shape:
            x = x + residual

        if x.shape[2] != self.transmission_size:
            x = self.fallback_pool(x)

        x = self.bn3(self.pw_conv2(x))

        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2)
