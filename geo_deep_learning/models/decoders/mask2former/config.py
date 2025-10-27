"""Config for Mask2Former decoder."""

from dataclasses import dataclass, field


@dataclass
class Mask2FormerConfig:
    """Config for Mask2Former decoder."""

    # mask2former
    mask_dim: int = 256
    conv_dim: int = 128

    # transformer decoder
    num_heads: int = 8
    hidden_dim: int = 256
    num_queries: int = 100
    decoder_layers: int = 6
    dim_feedforward: int = 2048
    pre_norm: bool = False
    mask_classification: bool = True
    enforce_input_project: bool = False

    # pixel decoder
    norm: str = "GN"
    common_stride: int = 4
    transformer_nheads: int = 8
    transformer_dropout: float = 0.1
    transformer_dim_feedforward: int = 1024
    transformer_enc_layers: int = 0
    transformer_in_features: list[str] = field(
        default_factory=lambda: ["1", "2", "3", "4"],
    )

    # loss
    deep_supervision: bool = True
    no_object_weight: float = 0.1

    # loss weights
    class_weight: float = 1.0
    dice_weight: float = 1.0
    mask_weight: float = 20.0

    # point sampling
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75

    # num points
    num_points: int = 12544


CONFIG = Mask2FormerConfig()
