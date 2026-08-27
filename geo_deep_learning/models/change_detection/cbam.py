"""CBAM — Convolutional Block Attention Module for SAR Change Detection.

Applies sequential *channel attention* then *spatial attention* on the input
feature maps.  When placed after FiLM conditioning and before the Transformer
encoder, it allows the model to emphasize the most informative bands and
spatial regions for a given acquisition geometry.

Lightweight: adds ~0.01–0.1% extra parameters depending on channel count.

Reference:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

Usage in ChangeDetectionModel:
    self.cbam = CBAM(in_channels=in_channels) if use_cbam else None
    # In forward:
    if self.cbam is not None:
        x1 = self.cbam(x1)
        x2 = self.cbam(x2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ChannelAttention(nn.Module):
    """Channel attention sub-module (squeeze → excitation per channel)."""

    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C, H, W] → channel weights [B, C, 1, 1]."""
        # Global average + max pooling
        avg_pool = x.mean(dim=(2, 3))  # [B, C]
        max_pool = x.amax(dim=(2, 3))  # [B, C]

        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module (where to focus)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C, H, W] → spatially weighted features."""
        avg_out = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        max_out = x.amax(dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_desc = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attn = torch.sigmoid(self.conv(spatial_desc))  # [B, 1, H, W]
        return x * attn


class CBAM(nn.Module):
    """CBAM: Channel attention → Spatial attention (residual).

    Args:
        in_channels: Number of input feature channels.
        reduction: Channel attention reduction ratio.
        spatial_kernel: Kernel size for spatial attention conv.
        residual: If True, output = input + CBAM(input). Ensures the module
            starts as near-identity and doesn't break pre-trained weights.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        spatial_kernel: int = 7,
        *,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)
        self.residual = residual

        if residual:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply CBAM to input features [B, C, H, W]."""
        out = self.channel_attn(x)
        out = self.spatial_attn(out)
        if self.residual:
            return x + self.gamma * out

        return out
