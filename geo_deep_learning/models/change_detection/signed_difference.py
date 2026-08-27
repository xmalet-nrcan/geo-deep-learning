"""Signed Difference Channel — inject the *direction* of change as extra input bands.

Most Siamese change-detection decoders combine the two dates either by
``torch.abs(f1 - f2)`` (which discards the sign) or by ``torch.cat((f1, f2))``
(which lets the network learn the direction only implicitly).  For SAR wildfire
mapping the *sign* carries strong physical meaning: a **drop** of backscatter on
certain bands is a good indicator of a burned area, whereas a rise usually is
not.

This module makes that cue explicit by computing the signed temporal difference
``x1 - x2`` and appending it as additional channels to *both* images before the
Siamese encoder.  Because the extra channels are identical for the pre and post
image, the (nonlinear) encoder still produces distinct features, but every stage
now has direct access to the signed change signal.

Two modes:
  * **raw** (``project_channels=None``): append the full ``in_channels`` signed
    difference.  Encoder input becomes ``2 * in_channels``.
  * **projected** (``project_channels=k``): a learnable 1x1 conv compresses the
    signed difference to ``k`` channels (e.g. to emphasise the physically
    relevant bands).  Encoder input becomes ``in_channels + k``.

This is applied AFTER FiLM/CBAM/ChannelDropout but BEFORE the encoder, so those
modules keep operating on the original ``in_channels``.

Usage in ChangeDetectionModel::

    self.signed_difference = SignedDifferenceChannel(in_channels)
    encoder_in = in_channels + self.signed_difference.extra_channels
    # In forward, right before the encoder:
    x1, x2 = self.signed_difference(x1, x2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SignedDifferenceChannel(nn.Module):
    """Append the signed temporal difference (x1 - x2) as extra input channels.

    Args:
        in_channels: Number of data-only input channels per image.
        project_channels: If given, a learnable 1x1 conv maps the signed
            difference from ``in_channels`` to ``project_channels`` channels.
            If ``None``, the raw signed difference is appended unchanged.
        normalize: If True, apply a ``tanh`` to keep the signed-difference
            channels bounded to ``[-1, 1]`` (helps when band scales differ).
    """

    def __init__(
        self,
        in_channels: int,
        project_channels: int | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.project_channels = project_channels
        self.normalize = normalize

        if project_channels is not None:
            self.proj = nn.Conv2d(
                in_channels,
                project_channels,
                kernel_size=1,
                bias=True,
            )
            self.extra_channels = project_channels
        else:
            self.proj = nn.Identity()
            self.extra_channels = in_channels

    def compute_difference(
        self,
        x1: Tensor,
        x2: Tensor,
    ) -> Tensor:
        """Compute signed temporal difference from pre/post images."""

        # x1 = pre
        # x2 = post
        #
        # diff > 0 : signal decreased after the event
        # diff < 0 : signal increased after the event
        diff = self.proj(x1 - x2)

        if self.normalize:
            diff = torch.tanh(diff)

        return diff

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
    ) -> tuple[Tensor, Tensor]:

        diff = self.compute_difference(x1, x2)

        x1_aug = torch.cat((x1, diff), dim=1)
        x2_aug = torch.cat((x2, diff), dim=1)

        return x1_aug, x2_aug
