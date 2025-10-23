"""Mask2Former decoder."""
# Adapted from facebookresearch/dinov3

import torch
from torch import nn
from torch.nn import functional as fn

from .config import CONFIG
from .pixel_decoder import MSDeformAttnPixelDecoder
from .transformer_decoder import MultiScaleMaskedTransformerDecoder


class Mask2FormerHead(nn.Module):
    """Mask2Former head."""

    def __init__(
        self,
        input_shape: dict[str, tuple[int]],
        num_classes: int = 150,
    ) -> None:
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict

        """
        super().__init__()

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=CONFIG.transformer_dropout,
            transformer_nheads=CONFIG.transformer_nheads,
            transformer_dim_feedforward=CONFIG.transformer_dim_feedforward,
            transformer_enc_layers=CONFIG.transformer_enc_layers,
            conv_dim=CONFIG.conv_dim,
            mask_dim=CONFIG.mask_dim,
            norm=CONFIG.norm,
            transformer_in_features=CONFIG.transformer_in_features,
            common_stride=CONFIG.common_stride,
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=CONFIG.conv_dim,
            mask_classification=CONFIG.mask_classification,
            num_classes=num_classes,
            hidden_dim=CONFIG.hidden_dim,
            num_queries=CONFIG.num_queries,
            nheads=CONFIG.num_heads,
            dim_feedforward=CONFIG.dim_feedforward,
            dec_layers=CONFIG.decoder_layers - 1,
            pre_norm=CONFIG.pre_norm,
            mask_dim=CONFIG.mask_dim,
            enforce_input_project=CONFIG.enforce_input_project,
        )

    def forward_features(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward features."""
        return self.layers(features, mask)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass."""
        return self.forward_features(features, mask)

    def predict(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
        rescale_to: tuple[int, int] = (512, 512),
    ) -> dict[str, torch.Tensor]:
        """Predict."""
        output = self.forward_features(features, mask)
        output["pred_masks"] = fn.interpolate(
            output["pred_masks"],
            size=rescale_to,
            mode="bilinear",
            align_corners=False,
        )
        return output

    def layers(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Layers."""
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
            features,
        )
        return self.predictor(multi_scale_features, mask_features, mask)


# if __name__ == "__main__":
#     from geo_deep_learning.models.encoders.dino_v3 import DINOv3Adapter, vit_large

#     backbone = vit_large()
#     model = DINOv3Adapter(backbone=backbone)

#     image = torch.randn(1, 3, 224, 224)
#     features = model(image)

#     decoder = Mask2FormerHead(
#         input_shape={
#             "1": [1024, 64, 64, 4],
#             "2": [1024, 32, 32, 4],
#             "3": [1024, 16, 16, 4],
#             "4": [1024, 8, 8, 4],
#         },
#         hidden_dim=2048,
#         num_classes=150,
#         ignore_value=255,
#     )

#     output = decoder.predict(features, rescale_to=(224, 224))
#     print(output.keys())
#     print(output["pred_logits"].shape)
#     print(output["pred_masks"].shape)
