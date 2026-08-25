"""
ChangeStar2: Universal Remote Sensing Change Detection Architecture.

Paper:
    "Single-Temporal Supervised Learning for Universal Remote Sensing
    Change Detection"
    International Journal of Computer Vision (IJCV), 2024.
    https://link.springer.com/article/10.1007/s11263-024-02141-4

Original repo:
    https://github.com/Z-Zheng/pytorch-change-models
    File: torchange/models/changestar2_5.py

Core idea — ChangeMixin2.5:
    Three complementary pathways for temporal feature comparison:
    1. Concat path:     linear_cat(cat(t1, t2))
    2. Symmetric path:  linear_cat(cat(t2, t1))  (optional, temporal_symmetric)
    3. Difference path: linear_diff(|t1 - t2|)
    Final change features = sum of all three paths.

    Uses ConvNeXt blocks for feature refinement within each pathway.

Adapted for geo-deep-learning pipeline:
    - Removed `ever`, `torchange`, `einops` dependencies
    - Standalone dense encoder (ConvNeXt-based + FPN) replaces SwinFarSeg
    - Interface: forward(x1, x2) → list[Tensor]  (5 outputs for deep supervision)
    - Accepts arbitrary input channels (e.g. 10 for RCM SAR)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (replaces ever.module.LayerNorm2d)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    """Stochastic Depth (replaces timm.layers.DropPath)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) < keep
        return x / keep * mask


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block — from the ChangeStar2.5 paper.

    DwConv → LayerNorm → Linear → GELU → Linear → scale → DropPath + residual.
    """

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(self.dwconv(x))
        x = x.permute(0, 2, 3, 1)          # (B, C, H, W) → (B, H, W, C)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # (B, H, W, C) → (B, C, H, W)
        return shortcut + self.drop_path(x)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHANGEMIXIN 2.5 — paper's core contribution
# ═══════════════════════════════════════════════════════════════════════════

class ChangeMixin2_5(nn.Module):
    """ChangeMixin2.5: temporal feature comparison module.

    Faithfully reproduced from the paper / original code.

    Three complementary pathways:
        1. linear_cat(cat(t1, t2))          — concatenation
        2. linear_cat(cat(t2, t1))          — temporal symmetry (optional)
        3. linear_diff(cat(t1 - t2, |t1 - t2|))  — signed diff + magnitude

    The sum of all paths produces the final change features.

    Args:
        dim: Feature channel count.
        change_classes: Number of change classes (2 for binary with softmax).
        temporal_symmetric: Process both (t1,t2) and (t2,t1) orders.
        n_blocks: Number of ConvNeXt refinement blocks per pathway.
    """

    def __init__(
        self,
        dim: int,
        change_classes: int = 2,
        temporal_symmetric: bool = True,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.temporal_symmetric = temporal_symmetric

        refine_cat = (
            nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])
            if n_blocks > 0 else nn.Identity()
        )
        refine_diff = (
            nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])
            if n_blocks > 0 else nn.Identity()
        )

        self.linear_cat = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
            refine_cat,
        )
        # Difference path input = [signed diff | magnitude] → 2 * dim channels.
        # Keeping the signed difference preserves the *direction* of change
        # (e.g. a drop vs. rise in SAR backscatter), while |diff| keeps the
        # magnitude / temporal-order invariance.
        self.linear_diff = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
            refine_diff,
        )

        self.change_conv = nn.Conv2d(dim, change_classes, 1)

    def forward(self, t1_feat: torch.Tensor, t2_feat: torch.Tensor) -> torch.Tensor:
        """Temporal comparison.

        Args:
            t1_feat: T1 features [B, D, H, W]
            t2_feat: T2 features [B, D, H, W]

        Returns:
            Change logits [B, change_classes, H, W]
        """
        # Path 1: concatenation
        bi = self.linear_cat(torch.cat([t1_feat, t2_feat], dim=1))
        # Path 2: temporal symmetry
        if self.temporal_symmetric:
            bi = bi + self.linear_cat(torch.cat([t2_feat, t1_feat], dim=1))
        # Path 3: signed difference + magnitude
        # Concatenate the signed difference (preserves change direction) with
        # its absolute value (magnitude / order-invariant) so the model keeps
        # both cues instead of discarding the sign via .abs().
        diff = t1_feat - t2_feat
        bi = bi + self.linear_diff(torch.cat([diff, diff.abs()], dim=1))

        return self.change_conv(bi)


# ═══════════════════════════════════════════════════════════════════════════
# 3. DENSE ENCODER (ConvNeXt-based + FPN)
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtStage(nn.Module):
    """One ConvNeXt stage: optional downsample + N blocks."""

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2,
                 downsample: bool = True, drop_path: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if downsample:
            layers.append(nn.Sequential(
                LayerNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 2, 2),
            ))
        elif in_ch != out_ch:
            layers.append(nn.Conv2d(in_ch, out_ch, 1))
        for _ in range(num_blocks):
            layers.append(ConvNeXtBlock(out_ch, drop_path=drop_path))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseEncoder(nn.Module):
    """Dense feature encoder: ConvNeXt backbone + FPN → single feature map at H/4.

    Replaces the FarSeg / SwinFarSeg encoder from the original ChangeStar2.5.
    Produces a single dense feature map by fusing multi-scale features via FPN.

    Args:
        in_channels: Number of input channels.
        out_channels: Output feature dimension.
        stage_channels: Channel count per stage.
        stage_blocks: Number of ConvNeXt blocks per stage.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        stage_channels: tuple[int, ...] = (64, 128, 256, 512),
        stage_blocks: tuple[int, ...] = (2, 2, 4, 2),
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        # Stem: H → H/4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], 4, 4, bias=False),
            LayerNorm2d(stage_channels[0]),
        )

        # 4 stages (stage 1 at H/4, then each downsamples 2×)
        self.stage1 = ConvNeXtStage(
            stage_channels[0], stage_channels[0], stage_blocks[0],
            downsample=False, drop_path=drop_path,
        )
        self.stage2 = ConvNeXtStage(
            stage_channels[0], stage_channels[1], stage_blocks[1],
            downsample=True, drop_path=drop_path,
        )
        self.stage3 = ConvNeXtStage(
            stage_channels[1], stage_channels[2], stage_blocks[2],
            downsample=True, drop_path=drop_path,
        )
        self.stage4 = ConvNeXtStage(
            stage_channels[2], stage_channels[3], stage_blocks[3],
            downsample=True, drop_path=drop_path,
        )

        # FPN laterals
        self.lat1 = nn.Conv2d(stage_channels[0], out_channels, 1, bias=False)
        self.lat2 = nn.Conv2d(stage_channels[1], out_channels, 1, bias=False)
        self.lat3 = nn.Conv2d(stage_channels[2], out_channels, 1, bias=False)
        self.lat4 = nn.Conv2d(stage_channels[3], out_channels, 1, bias=False)

        # FPN smoothing
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Extract dense features.

        Returns:
            dense: FPN-fused features [B, out_channels, H/4, W/4]
            intermediates: [c1, c2, c3, c4] for deep supervision
        """
        x = self.stem(x)
        c1 = self.stage1(x)     # H/4
        c2 = self.stage2(c1)    # H/8
        c3 = self.stage3(c2)    # H/16
        c4 = self.stage4(c3)    # H/32

        # FPN top-down fusion
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        dense = self.smooth(p1)  # [B, out_channels, H/4, W/4]
        return dense, [c1, c2, c3, c4]


# ═══════════════════════════════════════════════════════════════════════════
# 4. CHANGESTAR2 — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════

class ChangeStar2(nn.Module):
    """ChangeStar2.5 adapted for the geo-deep-learning pipeline.

    Paper:
        "Single-Temporal Supervised Learning for Universal Remote Sensing
        Change Detection" — IJCV 2024.

    Pipeline:
        1. Siamese dense encoder (shared weights, ConvNeXt + FPN)
        2. ChangeMixin2.5 (concat + symmetric + diff pathways)
        3. Upsample to full resolution
        4. Deep supervision via auxiliary heads on encoder intermediates

    Returns 5 predictions for deep supervision compatibility.

    Args:
        input_nc: Number of input channels (e.g. 10 for RCM SAR).
        output_nc: Number of output classes (e.g. 2 for binary).
        embed_dim: Dense feature dimension (encoder output / mixin input).
        stage_channels: Per-stage channel counts for the encoder.
        stage_blocks: Number of ConvNeXt blocks per encoder stage.
        n_mixin_blocks: Number of ConvNeXt blocks in ChangeMixin pathways.
        temporal_symmetric: Enable temporal symmetry in mixin.
        drop_path: Stochastic depth rate.
        decoder_softmax: If True, apply softmax to outputs.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        embed_dim: int = 256,
        stage_channels: tuple[int, ...] = (64, 128, 256, 512),
        stage_blocks: tuple[int, ...] = (2, 2, 4, 2),
        n_mixin_blocks: int = 2,
        temporal_symmetric: bool = True,
        drop_path: float = 0.1,
        decoder_softmax: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.output_nc = output_nc

        # 1. Shared dense encoder (Siamese)
        self.encoder = DenseEncoder(
            in_channels=input_nc,
            out_channels=embed_dim,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            drop_path=drop_path,
        )

        # 2. ChangeMixin2.5 (paper's core module)
        self.mixin = ChangeMixin2_5(
            dim=embed_dim,
            change_classes=output_nc,
            temporal_symmetric=temporal_symmetric,
            n_blocks=n_mixin_blocks,
        )

        # 3. Auxiliary heads for deep supervision (on encoder intermediates)
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(ch, output_nc, 1) for ch in stage_channels
        ])

        self.apply_softmax = decoder_softmax
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, LayerNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x1: Pre-event image  [B, C, H, W]
            x2: Post-event image [B, C, H, W]

        Returns:
            List of 5 predictions [B, output_nc, H, W]:
                [aux_c4, aux_c3, aux_c2, aux_c1, pred_final]
        """
        target_size = (x1.shape[2], x1.shape[3])

        # Efficient Siamese encoding: batch both images together
        B = x1.shape[0]
        both = torch.cat([x1, x2], dim=0)                      # [2B, C, H, W]
        dense_both, intermediates_both = self.encoder(both)     # [2B, D, H/4, W/4]
        t1_embed, t2_embed = dense_both[:B], dense_both[B:]     # [B, D, H/4, W/4] each

        # Split encoder intermediates for auxiliary heads
        intermediates_t1 = [feat[:B] for feat in intermediates_both]
        intermediates_t2 = [feat[B:] for feat in intermediates_both]

        # ChangeMixin2.5 — core temporal comparison
        c_logit = self.mixin(t1_embed, t2_embed)                # [B, nc, H/4, W/4]

        # Upsample to full resolution
        final = F.interpolate(c_logit, size=target_size, mode='bilinear', align_corners=False)

        # Auxiliary predictions (deep supervision)
        outputs: list[torch.Tensor] = []
        for i in range(len(self.aux_heads) - 1, -1, -1):  # coarsest → finest
            diff = intermediates_t1[i] - intermediates_t2[i]
            pred = self.aux_heads[i](diff)
            pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            outputs.append(pred)
        outputs.append(final)

        if self.apply_softmax:
            outputs = [torch.softmax(o, dim=1) for o in outputs]

        return outputs


# ═══════════════════════════════════════════════════════════════════════════
# 5. VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

class ChangeStar2Small(ChangeStar2):
    """ChangeStar2-Small: lightweight variant."""

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 128,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim,
            stage_channels=(48, 96, 192, 384),
            stage_blocks=(1, 1, 2, 1),
            n_mixin_blocks=1,
            decoder_softmax=decoder_softmax, **kwargs,
        )


class ChangeStar2Base(ChangeStar2):
    """ChangeStar2-Base: balanced performance and efficiency."""

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 256,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim,
            stage_channels=(64, 128, 256, 512),
            stage_blocks=(2, 2, 4, 2),
            n_mixin_blocks=2,
            decoder_softmax=decoder_softmax, **kwargs,
        )


class ChangeStar2Large(ChangeStar2):
    """ChangeStar2-Large: higher capacity."""

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 384,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim,
            stage_channels=(96, 192, 384, 768),
            stage_blocks=(2, 3, 6, 2),
            n_mixin_blocks=3,
            decoder_softmax=decoder_softmax, **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    for name, cls in [("Small", ChangeStar2Small), ("Base", ChangeStar2Base),
                      ("Large", ChangeStar2Large)]:
        m = cls(input_nc=10, output_nc=2)
        x1 = torch.randn(2, 10, 256, 256)
        x2 = torch.randn(2, 10, 256, 256)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        o = m(x1, x2)
        print(  # noqa: T201
            f"ChangeStar2-{name}: {n / 1e6:.1f}M params, "
            f"{len(o)} outputs, final={o[-1].shape}"
        )

    # Gradient check
    print("\n--- Gradient check ---")  # noqa: T201
    m = ChangeStar2Base(input_nc=10, output_nc=2)
    m.train()
    o = m(torch.randn(2, 10, 256, 256), torch.randn(2, 10, 256, 256))
    sum(out.sum() for out in o).backward()
    nt = sum(1 for p in m.parameters() if p.requires_grad)
    ng = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None)
    assert ng == nt, f"{ng}/{nt}"
    print(f"✓ All {nt} params have gradients!")  # noqa: T201
