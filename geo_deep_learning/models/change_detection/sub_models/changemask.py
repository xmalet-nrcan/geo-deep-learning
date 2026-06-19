"""
ChangeMask: Deep Multi-task Encoder-Transformer-Decoder for Change Detection.

Paper:
    "ChangeMask: Deep Multi-task Encoder-Transformer-Decoder Architecture
    for Semantic Change Detection"
    ISPRS Journal of Photogrammetry and Remote Sensing, 2022.

Original repo (Z-Zheng): https://github.com/Z-Zheng/pytorch-change-models

Architecture (from the paper):
    1. Siamese ResNet encoder (shared weights) — multi-scale features
    2. Bitemporal Feature Interaction — difference + concat at each scale
    3. Transformer-enhanced decoder — self-attention on change features
    4. Change detection head — pixel-wise binary change map
    5. (Optional) Semantic segmentation heads — per-image semantic labels

Adapted for SAR change detection:
    - Configurable input channels (not limited to RGB)
    - Deep supervision (5 outputs) for pipeline compatibility
    - Same interface as ChangeFormer/HDANet: forward(x1, x2) → list[Tensor]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. RESNET ENCODER (standalone — no torchvision dependency)
# ═══════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """ResNet BasicBlock (expansion=1)."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_ch != out_ch * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BottleneckBlock(nn.Module):
    """ResNet Bottleneck (expansion=4)."""

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


# ResNet configurations: {name: (block_class, layers)}
RESNET_CONFIGS = {
    'resnet18': (BasicBlock, [2, 2, 2, 2]),
    'resnet34': (BasicBlock, [3, 4, 6, 3]),
    'resnet50': (BottleneckBlock, [3, 4, 6, 3]),
}


class ResNetEncoder(nn.Module):
    """ResNet feature extractor (4 stages).

    Produces multi-scale features at [H/4, H/8, H/16, H/32].

    Args:
        in_channels: Number of input channels.
        variant: 'resnet18', 'resnet34', or 'resnet50'.
    """

    def __init__(self, in_channels: int = 3, variant: str = 'resnet18') -> None:
        super().__init__()
        block_cls, layers = RESNET_CONFIGS[variant]
        self.expansion = block_cls.expansion

        # Channel counts at each stage
        base_channels = [64, 128, 256, 512]
        self.stage_channels = [c * self.expansion for c in base_channels]

        # Stem: conv7×7 stride-2 + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        # Build 4 stages
        self.layer1 = self._make_layer(block_cls, 64, base_channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(
            block_cls, base_channels[0] * self.expansion, base_channels[1], layers[1], stride=2,
        )
        self.layer3 = self._make_layer(
            block_cls, base_channels[1] * self.expansion, base_channels[2], layers[2], stride=2,
        )
        self.layer4 = self._make_layer(
            block_cls, base_channels[2] * self.expansion, base_channels[3], layers[3], stride=2,
        )

    @staticmethod
    def _make_layer(block_cls, in_ch: int, mid_ch: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        if block_cls.expansion == 1:
            layers.append(block_cls(in_ch, mid_ch, stride))
            for _ in range(1, num_blocks):
                layers.append(block_cls(mid_ch, mid_ch))
        else:
            layers.append(block_cls(in_ch, mid_ch, stride))
            for _ in range(1, num_blocks):
                layers.append(block_cls(mid_ch * block_cls.expansion, mid_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            [c1, c2, c3, c4] at [H/4, H/8, H/16, H/32] respectively.
        """
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


# ═══════════════════════════════════════════════════════════════════════════
# 2. BITEMPORAL FEATURE INTERACTION
# ═══════════════════════════════════════════════════════════════════════════

class BitemporalInteraction(nn.Module):
    """Compute change-aware features from bi-temporal feature pairs.

    At each scale, computes:
        - Absolute difference: |f1 - f2|
        - Concatenation: [f1, f2]
    Then fuses them through a conv block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Input: concat(|f1-f2|, f1, f2) → 3 × in_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        return self.fuse(torch.cat([diff, f1, f2], dim=1))


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMER CHANGE MODULE
# ═══════════════════════════════════════════════════════════════════════════

class SpatialPositionEncoding(nn.Module):
    """Learnable 2D positional encoding for spatial features."""

    def __init__(self, channels: int, max_h: int = 64, max_w: int = 64) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(max_h, channels // 2)
        self.col_embed = nn.Embedding(max_w, channels // 2)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        row_pos = self.row_embed(torch.arange(h, device=x.device))  # [H, C/2]
        col_pos = self.col_embed(torch.arange(w, device=x.device))  # [W, C/2]
        # Broadcast to [H, W, C]
        pos = torch.cat([
            row_pos.unsqueeze(1).expand(-1, w, -1),
            col_pos.unsqueeze(0).expand(h, -1, -1),
        ], dim=-1)
        # [H, W, C] → [1, C, H, W]
        return pos.permute(2, 0, 1).unsqueeze(0)


class TransformerChangeModule(nn.Module):
    """Transformer module for change feature enhancement.

    Applies multi-head self-attention on spatially-flattened features
    with learnable positional encoding.

    Applied at deeper scales (H/16, H/32) where sequence length is manageable.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.pos_encoding = SpatialPositionEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process spatial features through transformer.

        Args:
            x: [B, C, H, W] change features

        Returns:
            Enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Add positional encoding
        pos = self.pos_encoding(x)
        x = x + pos

        # Flatten spatial dims → sequence: [B, H*W, C]
        seq = x.flatten(2).permute(0, 2, 1)

        # Transformer
        out = self.transformer(seq)
        out = self.norm(out)

        # Reshape back to spatial: [B, C, H, W]
        return out.permute(0, 2, 1).reshape(B, C, H, W)


# ═══════════════════════════════════════════════════════════════════════════
# 4. FPN CHANGE DECODER
# ═══════════════════════════════════════════════════════════════════════════

class FPNChangeDecoder(nn.Module):
    """Feature Pyramid Network decoder for change detection.

    Top-down pathway with lateral connections.
    Produces predictions at multiple scales for deep supervision.

    Args:
        in_channels: Channel counts at each scale [c1, c2, c3, c4].
        embed_dim: Internal decoder dimension.
        output_nc: Number of output classes.
    """

    def __init__(self, in_channels: list[int], embed_dim: int = 256,
                 output_nc: int = 2) -> None:
        super().__init__()
        self.num_scales = len(in_channels)

        # Lateral connections (project each scale to embed_dim)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels
        ])

        # Top-down smoothing convolutions
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_scales - 1)
        ])

        # Per-scale prediction heads (for deep supervision)
        self.pred_heads = nn.ModuleList([
            nn.Conv2d(embed_dim, output_nc, 1) for _ in range(self.num_scales)
        ])

        # Final fusion and prediction
        self.final_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_scales, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, output_nc, 1),
        )

    def forward(self, features: list[torch.Tensor],
                target_size: tuple[int, int]) -> list[torch.Tensor]:
        """Decode multi-scale change features.

        Args:
            features: [f1, f2, f3, f4] from coarsest to finest
                      (actually [finest, ..., coarsest] matching encoder output order)
            target_size: (H, W) for final upsampling.

        Returns:
            List of 5 predictions: [pred_c4, pred_c3, pred_c2, pred_c1, pred_final]
        """
        outputs = []

        # Project through lateral connections
        projected = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down pathway: start from coarsest
        td = [None] * self.num_scales
        td[-1] = projected[-1]  # coarsest scale

        for i in range(self.num_scales - 2, -1, -1):
            # Upsample coarser + add lateral
            up = F.interpolate(td[i + 1], size=projected[i].shape[2:],
                               mode='bilinear', align_corners=False)
            td[i] = self.smooth[i](projected[i] + up)

        # Per-scale predictions (coarsest first for deep supervision ordering)
        for i in range(self.num_scales - 1, -1, -1):
            pred = self.pred_heads[i](td[i])
            pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            outputs.append(pred)

        # Final: fuse all scales
        all_at_finest = []
        for i in range(self.num_scales):
            f = F.interpolate(td[i], size=td[0].shape[2:], mode='bilinear', align_corners=False)
            all_at_finest.append(f)
        fused = self.final_fuse(torch.cat(all_at_finest, dim=1))
        fused = F.interpolate(fused, size=target_size, mode='bilinear', align_corners=False)
        outputs.append(fused)

        return outputs  # [pred_c4, pred_c3, pred_c2, pred_c1, pred_final]


# ═══════════════════════════════════════════════════════════════════════════
# 5. CHANGEMASK — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════

class ChangeMask(nn.Module):
    """ChangeMask: Encoder-Transformer-Decoder for Change Detection.

    Paper:
        "ChangeMask: Deep Multi-task Encoder-Transformer-Decoder Architecture
        for Semantic Change Detection" — ISPRS P&RS 2022.

    Pipeline:
        1. Siamese ResNet encoder (shared weights) → multi-scale features
        2. Bitemporal interaction at each scale → change features
        3. Transformer enhancement at deep scales (H/16, H/32)
        4. FPN decoder → multi-scale change predictions
        5. (5 outputs for deep supervision compatibility)

    Args:
        input_nc: Number of input channels.
        output_nc: Number of output classes.
        embed_dim: Decoder embedding dimension.
        backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50').
        transformer_heads: Number of attention heads in transformer module.
        transformer_layers: Number of transformer encoder layers.
        decoder_softmax: If True, apply softmax to outputs.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        embed_dim: int = 256,
        backbone: str = 'resnet18',
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        decoder_softmax: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.output_nc = output_nc

        # 1. Siamese ResNet encoder (shared weights)
        self.encoder = ResNetEncoder(in_channels=input_nc, variant=backbone)
        enc_channels = self.encoder.stage_channels  # e.g. [64, 128, 256, 512] for resnet18

        # 2. Bitemporal interaction at each scale
        self.interactions = nn.ModuleList([
            BitemporalInteraction(ch, ch) for ch in enc_channels
        ])

        # 3. Transformer enhancement at deep scales only
        # (shallow scales have too many tokens for efficient self-attention)
        self.transformer_c3 = TransformerChangeModule(
            dim=enc_channels[2], num_heads=transformer_heads,
            num_layers=transformer_layers,
        )
        self.transformer_c4 = TransformerChangeModule(
            dim=enc_channels[3], num_heads=transformer_heads,
            num_layers=transformer_layers,
        )

        # 4. FPN Change Decoder
        self.decoder = FPNChangeDecoder(
            in_channels=enc_channels,
            embed_dim=embed_dim,
            output_nc=output_nc,
        )

        self.apply_softmax = decoder_softmax

        # Weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x1: Pre-event image  [B, C, H, W]
            x2: Post-event image [B, C, H, W]

        Returns:
            List of 5 predictions [B, output_nc, H, W]:
                [pred_c4, pred_c3, pred_c2, pred_c1, pred_final]
        """
        target_size = (x1.shape[2], x1.shape[3])

        # 1. Siamese encoding
        feats1 = self.encoder(x1)  # [c1, c2, c3, c4]
        feats2 = self.encoder(x2)

        # 2. Bitemporal interaction at each scale
        change_feats = []
        for i, interaction in enumerate(self.interactions):
            cf = interaction(feats1[i], feats2[i])
            change_feats.append(cf)

        # 3. Transformer enhancement at deep scales
        change_feats[2] = self.transformer_c3(change_feats[2])  # H/16
        change_feats[3] = self.transformer_c4(change_feats[3])  # H/32

        # 4. FPN decode
        outputs = self.decoder(change_feats, target_size)

        if self.apply_softmax:
            outputs = [torch.softmax(o, dim=1) for o in outputs]

        return outputs


# ═══════════════════════════════════════════════════════════════════════════
# 6. VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

class ChangeMask18(ChangeMask):
    """ChangeMask with ResNet-18 backbone (~15M params). Recommended default."""

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 256,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim,
            backbone='resnet18', transformer_heads=8, transformer_layers=2,
            decoder_softmax=decoder_softmax, **kwargs,
        )


class ChangeMask34(ChangeMask):
    """ChangeMask with ResNet-34 backbone (~25M params)."""

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 256,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim,
            backbone='resnet34', transformer_heads=8, transformer_layers=2,
            decoder_softmax=decoder_softmax, **kwargs,
        )


class ChangeMask50(ChangeMask):
    """ChangeMask with ResNet-50 backbone (~40M params)."""

    def __init__(self, input_nc: int = 3, output_nc: int = 2,
                 decoder_softmax: bool = False, embed_dim: int = 256,
                 **kwargs) -> None:
        super().__init__(
            input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim,
            backbone='resnet50', transformer_heads=8, transformer_layers=3,
            decoder_softmax=decoder_softmax, **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 7. QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    for name, cls in [("R18", ChangeMask18), ("R34", ChangeMask34), ("R50", ChangeMask50)]:
        model = cls(input_nc=10, output_nc=2)
        x1 = torch.randn(2, 10, 256, 256)
        x2 = torch.randn(2, 10, 256, 256)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        outputs = model(x1, x2)
        print(  # noqa: T201
            f"ChangeMask-{name}: {n_params / 1e6:.1f}M params, "
            f"{len(outputs)} outputs, "
            f"final shape: {outputs[-1].shape}"
        )

    # Interface check
    print("\n--- Interface check ---")  # noqa: T201
    model = ChangeMask18(input_nc=10, output_nc=2)
    model.train()
    x1 = torch.randn(2, 10, 256, 256)
    x2 = torch.randn(2, 10, 256, 256)
    outputs = model(x1, x2)
    assert len(outputs) == 5
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    assert n_grad == n_total, f"{n_grad}/{n_total}"
    print("✓ All checks passed!")  # noqa: T201
