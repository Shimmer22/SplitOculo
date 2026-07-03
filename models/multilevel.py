"""Utilities for multi-level SplitOculo payloads.

The training checkpoint keeps the largest payload shape, for example 196x128.
Lower payload levels are represented as prefixes:

* fewer tokens are produced by spatial pooling;
* fewer channels are produced by taking the first bottleneck dimensions;
* cloud-side reconstruction pads the missing dimensions with zeros and adapts
  token count back to the upsampler input size.
"""

import math
import random

import torch
import torch.nn.functional as F


def parse_payload_levels(spec):
    """Parse a level spec like ``49x64,49x128,196x64``."""
    levels = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid payload level: {item}")
        tokens, dim = item.split("x", 1)
        levels.append((int(tokens), int(dim)))
    if not levels:
        raise ValueError("At least one payload level is required")
    return levels


def format_payload_level(level):
    tokens, dim = level
    return f"{tokens}x{dim}"


def choose_payload_level(levels, mode="random"):
    """Choose a payload level for one forward pass."""
    if mode == "max":
        return max(levels, key=lambda x: (x[0] * x[1], x[0], x[1]))
    if mode == "min":
        return min(levels, key=lambda x: (x[0] * x[1], x[0], x[1]))
    return random.choice(levels)


def _square_side(num_tokens):
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Token count must be square, got {num_tokens}")
    return side


def resize_tokens(tokens, target_tokens, mode="avg"):
    """Resize a square token grid to another square token count."""
    if tokens.shape[1] == target_tokens:
        return tokens

    b, n, c = tokens.shape
    source_side = _square_side(n)
    target_side = _square_side(target_tokens)
    grid = tokens.view(b, source_side, source_side, c).permute(0, 3, 1, 2)

    if target_tokens < n or mode == "avg":
        grid = F.adaptive_avg_pool2d(grid, (target_side, target_side))
    else:
        grid = F.interpolate(grid, size=(target_side, target_side), mode="bilinear", align_corners=False)

    return grid.permute(0, 2, 3, 1).reshape(b, target_tokens, c)


def truncate_dim(tokens, target_dim):
    if target_dim > tokens.shape[-1]:
        raise ValueError(f"Cannot truncate {tokens.shape[-1]} dims to larger target {target_dim}")
    return tokens[..., :target_dim]


def pad_dim(tokens, target_dim):
    if tokens.shape[-1] == target_dim:
        return tokens
    if tokens.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad {tokens.shape[-1]} dims down to {target_dim}")
    pad = target_dim - tokens.shape[-1]
    return F.pad(tokens, (0, pad))


def payload_bytes(level, dtype_bytes=1):
    tokens, dim = level
    return tokens * dim * dtype_bytes
