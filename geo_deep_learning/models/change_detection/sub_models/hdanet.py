"""
HDANet: High-resolution feature Difference Attention Network for Change Detection.

Paper:
    "A high-resolution feature difference attention network for the application
    of building change detection"
    International Journal of Applied Earth Observation and Geoinformation, 2022.
    https://www.sciencedirect.com/science/article/pii/S1569843222001479

Architecture (from the paper):
    1. Siamese HRNet backbone (shared weights) — 4 parallel resolution branches
       that maintain high-resolution representations throughout.
    2. ASPP (Atrous Spatial Pyramid Pooling) — multi-scale feature extraction
       with dilation rates 1, 6, 12 and a 1×1 convolution.
    3. DAM (Difference Attention Module) — computes a Change Intensity Map (CIM)
       via pixel-wise Euclidean distance, applies spatial attention (conv3×3 + σ)
       and channel attention (SE-style) on the difference features.
    4. Classification head — pixel-wise change/no-change prediction.

Interface: Compatible with ChangeFormerV6/V7
    forward(x1, x2) → list[Tensor]
    Returns 5 predictions (4 auxiliary + 1 final) for deep supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. RESIDUAL BLOCKS (HRNet building blocks)
# ═══════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (used in HRNet stages 2–4)."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """Standard ResNet Bottleneck with expansion=4 (used in HRNet stage 1)."""

    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ═══════════════════════════════════════════════════════════════════════════
# 2. HIGH-RESOLUTION MODULE (multi-branch + cross-resolution fusion)
# ═══════════════════════════════════════════════════════════════════════════

class HighResolutionModule(nn.Module):
    """One HRNet module: parallel multi-resolution branches + cross-resolution fusion.

    Each branch processes features at its own resolution using BasicBlocks.
    After processing, all branches are fused: each output branch receives
    contributions from all input branches (upsampled or downsampled as needed).

    Fusion rules:
        - Same resolution: identity
        - High → Low: cascade of stride-2 3×3 convolutions
        - Low → High: 1×1 conv (channel projection) + bilinear upsample
    """

    def __init__(self, num_branches: int, num_blocks: int,
                 channels: list[int]) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.channels = channels

        # Parallel branches (each: num_blocks BasicBlocks at same resolution)
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            layers = [BasicBlock(channels[i], channels[i]) for _ in range(num_blocks)]
            self.branches.append(nn.Sequential(*layers))

        # Cross-resolution fusion layers (N×N matrix)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_fuse_layers(self) -> nn.ModuleList:
        """Build N×N fusion matrix between branches."""
        fuse_layers = nn.ModuleList()
        for j in range(self.num_branches):          # output branch
            fuse_j = nn.ModuleList()
            for i in range(self.num_branches):      # input branch
                if i == j:
                    fuse_j.append(nn.Identity())
                elif i < j:
                    # High-res → Low-res: cascade of stride-2 3×3 convolutions
                    convs: list[nn.Module] = []
                    for k in range(j - i):
                        in_c = self.channels[i] if k == 0 else self.channels[j]
                        convs.append(nn.Conv2d(in_c, self.channels[j], 3, 2, 1, bias=False))
                        convs.append(nn.BatchNorm2d(self.channels[j]))
                        if k < j - i - 1:          # ReLU between steps, not on last
                            convs.append(nn.ReLU(inplace=True))
                    fuse_j.append(nn.Sequential(*convs))
                else:
                    # Low-res → High-res: 1×1 channel projection (upsample in forward)
                    fuse_j.append(nn.Sequential(
                        nn.Conv2d(self.channels[i], self.channels[j], 1, bias=False),
                        nn.BatchNorm2d(self.channels[j]),
                    ))
            fuse_layers.append(fuse_j)
        return fuse_layers

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        # Process each branch independently
        branch_out = [self.branches[i](x_list[i]) for i in range(self.num_branches)]

        # Cross-resolution fusion
        fused = []
        for j in range(self.num_branches):
            y = torch.zeros_like(branch_out[j])
            for i in range(self.num_branches):
                contrib = self.fuse_layers[j][i](branch_out[i])
                if i > j:
                    # Low→High: spatial upsample after channel projection
                    contrib = F.interpolate(
                        contrib, size=branch_out[j].shape[2:],
                        mode='bilinear', align_corners=False,
                    )
                y = y + contrib
            fused.append(self.relu(y))

        return fused


# ══════════════════��════════════════════════════════════════════════════════
# 3. HRNET BACKBONE
# ═══════════════════════════════════════════════════════════════════════════

class HRNetBackbone(nn.Module):
    """HRNet backbone with 4 parallel resolution branches.

    From Sun et al. (2019), "Deep High-Resolution Representation Learning for
    Visual Recognition," TPAMI.

    Maintains high-resolution representations throughout the network.
    At the end, all branches are upsampled to the highest resolution and
    concatenated, producing features at H/4 × W/4.

    Args:
        in_channels: Number of input channels.
        width: Base channel width W. Branches use [W, 2W, 4W, 8W].
        stage1_blocks: Number of Bottleneck blocks in Stage 1.
        stage_modules: (stage2, stage3, stage4) number of HRModules per stage.
        stage_blocks: Number of BasicBlocks per HRModule.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 18,
        stage1_blocks: int = 4,
        stage_modules: tuple[int, int, int] = (1, 4, 3),
        stage_blocks: int = 4,
    ) -> None:
        super().__init__()

        self.width = width
        channels = [width, width * 2, width * 4, width * 8]
        self.channels = channels
        self.output_channels = sum(channels)  # after final concat

        # ── Stem: 2× stride-2 convolutions (H,W → H/4,W/4, 64 ch) ──
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Stage 1: single branch, Bottleneck blocks (64 → 256 ch) ──
        s1_layers: list[nn.Module] = []
        in_ch = 64
        for _ in range(stage1_blocks):
            s1_layers.append(Bottleneck(in_ch, 64))
            in_ch = 64 * Bottleneck.expansion   # 256
        self.stage1 = nn.Sequential(*s1_layers)

        # ── Transition 1: 1 branch (256) → 2 branches ([W, 2W]) ──
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, channels[0], 1, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_ch, channels[1], 3, 2, 1, bias=False),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
            ),
        ])

        # ── Stage 2: 2 branches ──
        self.stage2 = nn.ModuleList([
            HighResolutionModule(2, stage_blocks, channels[:2])
            for _ in range(stage_modules[0])
        ])

        # ── Transition 2: add 3rd branch from last existing branch ──
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(channels[1], channels[2], 3, 2, 1, bias=False),
                nn.BatchNorm2d(channels[2]),
                nn.ReLU(inplace=True),
            ),
        ])

        # ── Stage 3: 3 branches ──
        self.stage3 = nn.ModuleList([
            HighResolutionModule(3, stage_blocks, channels[:3])
            for _ in range(stage_modules[1])
        ])

        # ── Transition 3: add 4th branch ──
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(channels[2], channels[3], 3, 2, 1, bias=False),
                nn.BatchNorm2d(channels[3]),
                nn.ReLU(inplace=True),
            ),
        ])

        # ── Stage 4: 4 branches ──
        self.stage4 = nn.ModuleList([
            HighResolutionModule(4, stage_blocks, channels)
            for _ in range(stage_modules[2])
        ])

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-resolution features.

        Returns:
            final: Concatenated features [B, sum(channels), H/4, W/4]
            stage2_hr: High-res branch after Stage 2 [B, W, H/4, W/4]
            stage3_hr: High-res branch after Stage 3 [B, W, H/4, W/4]
        """
        x = self.stem(x)
        x = self.stage1(x)

        # Stage 2
        x_list = [self.transition1[0](x), self.transition1[1](x)]
        for module in self.stage2:
            x_list = module(x_list)
        stage2_hr = x_list[0]

        # Stage 3
        x_list = [x_list[0], x_list[1], self.transition2[2](x_list[-1])]
        for module in self.stage3:
            x_list = module(x_list)
        stage3_hr = x_list[0]

        # Stage 4
        x_list = [
            x_list[0], x_list[1], x_list[2],
            self.transition3[3](x_list[-1]),
        ]
        for module in self.stage4:
            x_list = module(x_list)

        # Upsample all branches to highest resolution and concatenate
        target_size = x_list[0].shape[2:]
        final = torch.cat([
            x_list[0],
            F.interpolate(x_list[1], size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(x_list[2], size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(x_list[3], size=target_size, mode='bilinear', align_corners=False),
        ], dim=1)

        return final, stage2_hr, stage3_hr


# ══════════════════════════════════════��════════════════════════════════════
# 4. ASPP (Atrous Spatial Pyramid Pooling)
# ═══════════════════════════════════════════════════════════════════════════

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLab v2/v3).

    As described in the paper: 4 parallel branches with different receptive fields,
    followed by concatenation and 1×1 projection.

    Branches:
        1. Conv 1×1 (global context)
        2. Conv 3×3, dilation=1 (local context)
        3. Conv 3×3, dilation=6 (medium context)
        4. Conv 3×3, dilation=12 (large context)
    """

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_d1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_d6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_d12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([
            self.branch_1x1(x),
            self.branch_d1(x),
            self.branch_d6(x),
            self.branch_d12(x),
        ], dim=1))


# ═══════════════════════════════════════════════════════════════════════════
# 5. DIFFERENCE ATTENTION MODULE (DAM) — paper's key contribution
# ══════════════════════════════════════════════════��════════════════════════

class DifferenceAttentionModule(nn.Module):
    """Difference Attention Module (DAM) from the paper.

    Combines spatial attention based on change intensity with channel attention:

    1. **Change Intensity Map (CIM)**: pixel-wise Euclidean distance between
       bi-temporal features.  Highlights WHERE changes are strongest.
       ``CIM_i = sqrt(Σ_c (f1_{c,i} − f2_{c,i})²)``    (Eq. 11)

    2. **Difference Attention Weight**: conv3×3 + sigmoid on CIM.
       ``W_DA = σ(conv3×3(CIM))``                        (Eq. 12)

    3. **Channel Attention** (SE-style) on difference features DI = f1 − f2.
       Highlights WHICH channels are most informative for change.
       ``CA = σ(FC(ReLU(FC(GAP(DI)))))``

    4. **Combined output**: per-channel difference weighted by both attentions.
       ``output = DI × W_DA × CA``                       (Eq. 13–14)

    The DAM replaces the CBAM-style spatial attention (avg/max pooling) with the
    CIM-based attention, which better captures change intensity patterns.

    Args:
        channels: Number of feature channels (C).
        reduction: Channel attention reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()

        # Spatial difference attention: CIM → conv3×3 → sigmoid
        self.cim_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # Channel attention (SE-style) on difference features
        mid = max(channels // reduction, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Compute change-aware difference features.

        Args:
            f1: Features from time 1 (pre-event)  [B, C, H, W]
            f2: Features from time 2 (post-event) [B, C, H, W]

        Returns:
            Enhanced difference features [B, C, H, W]
        """
        # Per-channel difference (Eq. 14)
        di = f1 - f2                                                        # [B, C, H, W]

        # Change Intensity Map: pixel-wise Euclidean distance (Eq. 11)
        cim = torch.sqrt(torch.sum(di ** 2, dim=1, keepdim=True) + 1e-8)   # [B, 1, H, W]

        # Difference attention weights (Eq. 12)
        w_da = self.cim_conv(cim)                                           # [B, 1, H, W]

        # Channel attention on DI (Eq. 13)
        ca = self.channel_attn(di).unsqueeze(-1).unsqueeze(-1)              # [B, C, 1, 1]

        # Combined attention (Eq. 13): spatial × channel
        return di * w_da * ca                                               # [B, C, H, W]


# ═══════════════════════════════════════════════════════════════════════════
# 6. HDANET — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════

class HDANet(nn.Module):
    """High-resolution feature Difference Attention Network (HDANet).

    Paper:
        "A high-resolution feature difference attention network for the
        application of building change detection"
        International Journal of Applied Earth Observation and Geoinformation, 2022.
        https://www.sciencedirect.com/science/article/pii/S1569843222001479

    Pipeline (from the paper, Section 3.3):
        1. Siamese HRNet backbone (shared weights) → multi-resolution features
        2. ASPP (shared weights) → multi-scale features
        3. DAM → change-intensity-weighted difference features
        4. Classification head → per-pixel change prediction

    Returns 5 predictions (4 auxiliary + 1 final) for deep supervision
    compatibility with the ChangeFormer training pipeline.  The auxiliary
    heads are NOT part of the original paper — they are added for pipeline
    compatibility and can be ignored by setting ``deep_supervision=False``
    in the LightningModule config.

    Args:
        input_nc: Number of input channels (e.g. 10 for RCM SAR).
        output_nc: Number of output classes (e.g. 2 for binary change).
        embed_dim: ASPP output dimension (analogous to ChangeFormer's embed_dim).
        width: HRNet base width W. Branches use [W, 2W, 4W, 8W].
        stage1_blocks: Bottleneck blocks in HRNet Stage 1.
        stage_modules: (stage2, stage3, stage4) HRModule count per stage.
        stage_blocks: BasicBlocks per HRModule.
        decoder_softmax: If True, apply softmax to outputs.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        embed_dim: int = 256,
        width: int = 18,
        stage1_blocks: int = 4,
        stage_modules: tuple[int, int, int] = (1, 4, 3),
        stage_blocks: int = 4,
        decoder_softmax: bool = False,
        **kwargs,  # noqa: ARG002 — accept extra kwargs for compatibility
    ) -> None:
        super().__init__()
        self.output_nc = output_nc

        # 1. Siamese HRNet backbone (shared weights)
        self.backbone = HRNetBackbone(
            in_channels=input_nc,
            width=width,
            stage1_blocks=stage1_blocks,
            stage_modules=stage_modules,
            stage_blocks=stage_blocks,
        )
        hrnet_out_ch = self.backbone.output_channels  # W + 2W + 4W + 8W = 15W

        # 2. ASPP for multi-scale feature learning (shared weights)
        self.aspp = ASPP(in_channels=hrnet_out_ch, out_channels=embed_dim)

        # 3. Difference Attention Module
        self.dam = DifferenceAttentionModule(channels=embed_dim)

        # 4. Classification head
        head_mid = max(embed_dim // 4, 16)
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, head_mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(head_mid, output_nc, 1),
        )

        # ── Auxiliary heads for deep supervision (not in original paper) ──
        self.aux_head_s2 = nn.Conv2d(width, output_nc, 1)          # Stage 2 high-res
        self.aux_head_s3 = nn.Conv2d(width, output_nc, 1)          # Stage 3 high-res
        self.aux_head_full = nn.Conv2d(hrnet_out_ch, output_nc, 1) # Full HRNet concat
        self.aux_head_aspp = nn.Conv2d(embed_dim, output_nc, 1)    # After ASPP

        self.apply_softmax = decoder_softmax

        # Weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x1: Pre-event image  [B, C, H, W]
            x2: Post-event image [B, C, H, W]

        Returns:
            List of 5 prediction tensors [B, output_nc, H, W]
            (all at original input resolution):
                [pred_stage2, pred_stage3, pred_full, pred_aspp, pred_final]
        """
        target_size = (x1.shape[2], x1.shape[3])

        # 1. Siamese HRNet feature extraction (shared weights)
        f1, s2_f1, s3_f1 = self.backbone(x1)
        f2, s2_f2, s3_f2 = self.backbone(x2)

        # 2. ASPP multi-scale feature learning (shared weights)
        ms1 = self.aspp(f1)     # [B, embed_dim, H/4, W/4]
        ms2 = self.aspp(f2)     # [B, embed_dim, H/4, W/4]

        # 3. Difference Attention Module (paper Eq. 11–14)
        enhanced_diff = self.dam(ms1, ms2)    # [B, embed_dim, H/4, W/4]

        # 4. Classification head
        final_logits = self.head(enhanced_diff)
        final_logits = F.interpolate(
            final_logits, size=target_size, mode='bilinear', align_corners=False,
        )

        # ── Auxiliary predictions for deep supervision ──
        def _aux_pred(head: nn.Module, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
            diff = feat1 - feat2
            pred = head(diff)
            return F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)

        pred_s2 = _aux_pred(self.aux_head_s2, s2_f1, s2_f2)
        pred_s3 = _aux_pred(self.aux_head_s3, s3_f1, s3_f2)
        pred_full = _aux_pred(self.aux_head_full, f1, f2)
        pred_aspp = _aux_pred(self.aux_head_aspp, ms1, ms2)

        outputs = [pred_s2, pred_s3, pred_full, pred_aspp, final_logits]

        if self.apply_softmax:
            outputs = [torch.softmax(o, dim=1) for o in outputs]

        return outputs


# ═══════════════════════════════════════════════════════════════════════════
# 7. HDANET VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

class HDANetSmall(HDANet):
    """HDANet-Small: lighter variant for faster training/inference.

    Uses narrower HRNet (W=12) and fewer modules/blocks.
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 128,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            embed_dim=embed_dim,
            width=12,
            stage1_blocks=2,
            stage_modules=(1, 1, 1),
            stage_blocks=2,
            decoder_softmax=decoder_softmax,
            **kwargs,
        )


class HDANetBase(HDANet):
    """HDANet-Base: default variant with HRNet-W18.

    Balanced performance and efficiency.  Recommended starting point.
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 256,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            embed_dim=embed_dim,
            width=18,
            stage1_blocks=4,
            stage_modules=(1, 4, 3),
            stage_blocks=4,
            decoder_softmax=decoder_softmax,
            **kwargs,
        )


class HDANetLarge(HDANet):
    """HDANet-Large: higher capacity with HRNet-W32.

    For maximum accuracy when compute is not a constraint.
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 384,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            embed_dim=embed_dim,
            width=32,
            stage1_blocks=4,
            stage_modules=(1, 4, 3),
            stage_blocks=4,
            decoder_softmax=decoder_softmax,
            **kwargs,
        )


# ═══════════════════���═══════════════════════════════════════════════════════
# 8. QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    for name, cls in [("Small", HDANetSmall), ("Base", HDANetBase), ("Large", HDANetLarge)]:
        model = cls(input_nc=10, output_nc=2)
        x1 = torch.randn(2, 10, 256, 256)
        x2 = torch.randn(2, 10, 256, 256)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        outputs = model(x1, x2)
        print(  # noqa: T201
            f"HDANet-{name}: {n_params / 1e6:.1f}M params, "
            f"{len(outputs)} outputs, "
            f"final shape: {outputs[-1].shape}"
        )

    # Interface compatibility check
    print("\n--- Interface check ---")  # noqa: T201
    model = HDANetBase(input_nc=10, output_nc=2)
    x1 = torch.randn(2, 10, 256, 256)
    x2 = torch.randn(2, 10, 256, 256)
    outputs = model(x1, x2)
    assert isinstance(outputs, list), "Output must be a list"
    assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"
    for i, o in enumerate(outputs):
        assert o.shape == (2, 2, 256, 256), f"Output {i} shape mismatch: {o.shape}"

    # Gradient check (deep supervision: all heads)
    model.train()
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    assert n_grad == n_total, f"Only {n_grad}/{n_total} params have grad"
    print("✓ All checks passed!")  # noqa: T201
