"""Mask2Former decoder."""
# Adapted from facebookresearch/dinov3

import torch
from torch import nn
from torch.nn import functional as fn

from .pixel_decoder import MSDeformAttnPixelDecoder
from .transformer_decoder import MultiScaleMaskedTransformerDecoder


class Mask2FormerHead(nn.Module):
    """Mask2Former head."""

    def __init__(  # noqa: PLR0913
        self,
        input_shape: dict[str, tuple[int]],
        hidden_dim: int = 2048,
        num_classes: int = 150,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_in_feature: str = "multi_scale_pixel_decoder",
    ) -> None:
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            hidden_dim: hidden dimension
            num_classes: number of classes to predict
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_in_feature: input feature name to the transformer_predictor.

        """
        super().__init__()
        orig_input_shape = input_shape
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][-1])
        self.in_features = [k for k, _ in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=orig_input_shape,
            transformer_dropout=0.0,
            transformer_nheads=16,
            transformer_dim_feedforward=4096,
            transformer_enc_layers=6,
            conv_dim=hidden_dim,
            mask_dim=hidden_dim,
            norm="GN",
            transformer_in_features=["1", "2", "3", "4"],
            common_stride=4,
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=hidden_dim,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=100,
            nheads=16,
            dim_feedforward=4096,
            dec_layers=9,
            pre_norm=False,
            mask_dim=hidden_dim,
            enforce_input_project=False,
        )

        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

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
