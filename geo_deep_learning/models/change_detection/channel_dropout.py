"""Stochastic Channel Dropout — SAR-specific input regularization.

During training, randomly drops entire input channels (bands) with a given
probability.  This forces the model to not over-rely on any single Stokes
parameter and improves robustness when bands are noisy or partially missing.

For SAR data this is especially useful because:
  - Individual bands can be dominated by speckle
  - Some acquisitions may have lower-quality channels
  - It acts as an implicit ensemble over band subsets

This is applied AFTER FiLM/CBAM but BEFORE the encoder.

Usage in ChangeDetectionModel:
    self.channel_dropout = ChannelDropout(drop_prob=0.1) if use_channel_dropout else None
    # In forward:
    if self.training and self.channel_dropout is not None:
        x1 = self.channel_dropout(x1)
        x2 = self.channel_dropout(x2)  # same mask for temporal consistency
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ChannelDropout(nn.Module):
    """Drop entire channels (bands) randomly during training.

    Unlike standard Dropout2d which drops spatial feature maps in hidden layers,
    this is designed for INPUT channels and guarantees at least `min_channels`
    remain active (never drops below half the input bands).

    Args:
        drop_prob: Probability of dropping each channel independently.
        min_channels: Minimum number of channels that must remain active.
            Prevents degenerate cases where too few channels survive.
            If None, defaults to ceil(num_channels / 2) at runtime
            (computed in _generate_mask).
        consistent_temporal: If True, uses the SAME dropout mask for both
            pre and post images (call with same random state). This is handled
            externally by calling with a shared mask.
    """

    def __init__(
        self,
        drop_prob: float = 0.1,
        min_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self._min_channels_override = min_channels

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply channel dropout.

        Args:
            x: [B, C, H, W] input features.
            mask: Optional pre-computed mask [1, C, 1, 1] (for temporal consistency
                between pre/post images). If None, a new mask is generated.

        Returns:
            Tensor with some channels zeroed out (scaled by 1/(1-effective_drop_rate)).
        """
        if not self.training or self.drop_prob <= 0:
            return x

        C = x.shape[1]

        if mask is None:
            mask = self._generate_mask(C, x.device)

        # Scale remaining channels to maintain expected value
        active_ratio = mask.sum() / C
        if active_ratio > 0:
            return x * mask / active_ratio
        return x

    def _generate_mask(self, num_channels: int, device: torch.device) -> Tensor:
        """Generate a channel mask guaranteeing min_channels remain.

        If min_channels was not set explicitly, uses ceil(num_channels / 2)
        with a hard floor of 3. This ensures at least half the bands survive.

        Returns:
            mask: [1, C, 1, 1] binary mask (1=keep, 0=drop).
        """
        import math
        if self._min_channels_override is not None:
            min_ch = self._min_channels_override
        else:
            # Default: at least half the channels, minimum 3
            min_ch = max(3, math.ceil(num_channels / 2))

        mask = torch.bernoulli(
            torch.full((num_channels,), 1.0 - self.drop_prob, device=device)
        )

        # Ensure minimum channels are active
        if mask.sum() < min_ch:
            # Randomly activate channels until minimum is met
            inactive = (mask == 0).nonzero(as_tuple=True)[0]
            n_to_activate = min_ch - int(mask.sum().item())
            perm = torch.randperm(len(inactive), device=device)[:n_to_activate]
            mask[inactive[perm]] = 1.0

        return mask.view(1, -1, 1, 1)

    def generate_shared_mask(self, num_channels: int, device: torch.device) -> Tensor:
        """Generate a mask to be shared between pre and post images.

        Call this once per batch, then pass the result to forward() for both images.
        """
        return self._generate_mask(num_channels, device)
