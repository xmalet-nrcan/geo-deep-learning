"""Change Detection Model segmentation model."""
import torch
from torch import Tensor

from geo_deep_learning.models.change_detection.cbam import CBAM
from geo_deep_learning.models.change_detection.channel_dropout import ChannelDropout
from geo_deep_learning.models.change_detection.difference_feature_attention import DifferenceFeatureAttention
from geo_deep_learning.models.change_detection.metadata_film_conditioner import MetadataFiLMConditioner
from geo_deep_learning.models.change_detection.sub_models.changeformer.original_change_former import ChangeFormerV6, \
    ChangeFormerV5, ChangeFormerV7
from geo_deep_learning.models.change_detection.sub_models.hdanet import (
    HDANet, HDANetSmall, HDANetBase, HDANetLarge,
)
from geo_deep_learning.models.change_detection.sub_models.changemask import (
    ChangeMask, ChangeMask18, ChangeMask34, ChangeMask50,
)
from geo_deep_learning.models.change_detection.sub_models.changestar2 import (
    ChangeStar2, ChangeStar2Small, ChangeStar2Base, ChangeStar2Large,
)
from geo_deep_learning.models.segmentation.base import BaseSegmentationModel


class ChangeDetectionModel(BaseSegmentationModel):
    """Change Detection segmentation model.

    When ``use_metadata_film=True``, SAT_PASS and BEAM are no longer fed as
    constant spatial bands.  Instead a lightweight FiLM layer modulates the
    input channels *before* the Transformer encoder, letting the network learn
    acquisition-specific adjustments without wasting encoder capacity.
    """

    def __init__(self, change_detection_model: str = "changeformer",
                 in_channels: int = 3,
                 out_channels: int = 2,
                 use_metadata_film: bool = False,
                 film_embed_dim: int = 32,
                 film_metadata_fields: dict[str, int] | None = None,
                 use_cbam: bool = False,
                 cbam_reduction: int = 4,
                 use_channel_dropout: bool = False,
                 channel_dropout_prob: float = 0.1,
                 use_dfa: bool = False,
                 dfa_gate_hidden: int = 16,
                 **kwargs) -> None:
        """Initialize Change Detection segmentation model.

        Args:
            change_detection_model: Model variant key ('changeformer', 'changeformer_5', 'changeformer_6').
            in_channels: Number of *data-only* input channels (excluding metadata bands).
            out_channels: Number of output classes.
            use_metadata_film: If True, create a FiLM conditioner for metadata.
            film_embed_dim: Embedding dimension for the FiLM conditioner.
            film_metadata_fields: Dict mapping field name to num categories.
                Example: {"sat_pass": 2, "beam": 4, "season": 4}
                If None, defaults to {"sat_pass": 2, "beam": 4}.
            use_cbam: If True, apply CBAM (Channel & Spatial Attention) after FiLM.
                Helps the model focus on relevant bands and spatial regions.
            cbam_reduction: Channel attention reduction ratio for CBAM.
            use_channel_dropout: If True, randomly drop input channels during training.
                Improves robustness to noisy/missing SAR bands.
            channel_dropout_prob: Probability of dropping each channel.
            use_dfa: If True, apply Difference Feature Attention on decoder outputs.
                Learns to gate unreliable intermediate decoder predictions.
            dfa_gate_hidden: Hidden dim for DFA gating MLP.
        """
        super().__init__()

        model_selection = {'changeformer': ChangeFormerV6,
                           'changeformer_5': ChangeFormerV5,
                           'changeformer_6': ChangeFormerV6,
                           'changeformer_7': ChangeFormerV7,
                           'hdanet': HDANetBase,
                           'hdanet_small': HDANetSmall,
                           'hdanet_base': HDANetBase,
                           'hdanet_large': HDANetLarge,
                           'changemask': ChangeMask18,
                           'changemask_18': ChangeMask18,
                           'changemask_34': ChangeMask34,
                           'changemask_50': ChangeMask50,
                           'changestar2': ChangeStar2Base,
                           'changestar2_small': ChangeStar2Small,
                           'changestar2_base': ChangeStar2Base,
                           'changestar2_large': ChangeStar2Large,
                           }

        model_sub_name = {'changeformer': 'changeformer',
                          'changeformer_5': 'changeformer',
                          'changeformer_6': 'changeformer',
                          'changeformer_7': 'changeformer',
                          'hdanet': 'hdanet',
                          'hdanet_small': 'hdanet',
                          'hdanet_base': 'hdanet',
                          'hdanet_large': 'hdanet',
                          'changemask': 'changemask',
                          'changemask_18': 'changemask',
                          'changemask_34': 'changemask',
                          'changemask_50': 'changemask',
                          'changestar2': 'changestar2',
                          'changestar2_small': 'changestar2',
                          'changestar2_base': 'changestar2',
                          'changestar2_large': 'changestar2',
                          }

        model_parameters = {'changeformer': {'decoder_softmax': False, 'embed_dim': 256},
                            'hdanet': {'decoder_softmax': False, 'embed_dim': 256},
                            'changemask': {'decoder_softmax': False, 'embed_dim': 256},
                            'changestar2': {'decoder_softmax': False, 'embed_dim': 256}}
        model_kwargs = model_parameters.get(model_sub_name.get(change_detection_model))
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs.update(kwargs)
        model = model_selection[change_detection_model]

        self.change_detection_model = model(input_nc=in_channels,
                                            output_nc=out_channels,
                                            **model_kwargs)

        # FiLM conditioner for acquisition metadata
        self.use_metadata_film = use_metadata_film
        self.film_conditioner: MetadataFiLMConditioner | None = None
        if use_metadata_film:
            self.film_conditioner = MetadataFiLMConditioner(
                in_channels=in_channels,
                embed_dim=film_embed_dim,
                metadata_fields=film_metadata_fields,
            )

        # CBAM: Channel & Spatial Attention (after FiLM, before encoder)
        self.cbam: CBAM | None = None
        if use_cbam:
            self.cbam = CBAM(
                in_channels=in_channels,
                reduction=cbam_reduction,
                residual=True,
            )

        # Stochastic Channel Dropout (training-only regularization)
        self.channel_dropout: ChannelDropout | None = None
        if use_channel_dropout:
            self.channel_dropout = ChannelDropout(
                drop_prob=channel_dropout_prob,
            )

        # Difference Feature Attention (on decoder multi-scale outputs)
        self.dfa: DifferenceFeatureAttention | None = None
        if use_dfa:
            self.dfa = DifferenceFeatureAttention(
                num_classes=out_channels,
                num_scales=4,
                gate_hidden=dfa_gate_hidden,
            )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        sat_pass: Tensor | None = None,
        beam: Tensor | None = None,
        **metadata_kwargs: Tensor,
    ) -> Tensor:
        """Forward pass of the model.

        Pipeline order:
            1. FiLM conditioning (metadata → per-channel scale/shift)
            2. CBAM (channel + spatial attention)
            3. Channel Dropout (training-only regularization)
            4. ChangeFormer encoder + decoder
            5. DFA (gating on intermediate decoder outputs)

        Args:
            x1: Pre-image tensor [B, C, H, W].
            x2: Post-image tensor [B, C, H, W].
            sat_pass: [B] integer tensor for satellite pass (0=ASC, 1=DESC).
            beam: [B] integer tensor for beam (0=A, 1=B, 2=C, 3=D).
            **metadata_kwargs: Additional metadata fields (e.g. season=[B]).

        Returns:
            List of output tensors (one per decoder head + final).
        """
        # 1. FiLM conditioning
        if self.film_conditioner is not None and sat_pass is not None and beam is not None:
            x1 = self.film_conditioner(x1, sat_pass, beam, **metadata_kwargs)
            x2 = self.film_conditioner(x2, sat_pass, beam, **metadata_kwargs)

        # 2. CBAM attention
        if self.cbam is not None:
            x1 = self.cbam(x1)
            x2 = self.cbam(x2)

        # 3. Channel Dropout (same mask for pre/post to preserve change signal)
        if self.training and self.channel_dropout is not None:
            shared_mask = self.channel_dropout.generate_shared_mask(
                x1.shape[1], x1.device
            )
            x1 = self.channel_dropout(x1, mask=shared_mask)
            x2 = self.channel_dropout(x2, mask=shared_mask)

        # 4. ChangeFormer encoder + decoder
        outputs = self.change_detection_model(x1, x2)

        # 5. Difference Feature Attention (gate intermediate predictions)
        if self.dfa is not None and isinstance(outputs, list):
            outputs = self.dfa(outputs)

        return outputs



if __name__ == '__main__':
    # Test without any conditioning
    model = ChangeDetectionModel(change_detection_model='changeformer_6', in_channels=9, out_channels=2)
    x1 = torch.randn(5, 9, 512, 512)
    x2 = torch.randn(5, 9, 512, 512)
    outputs = model(x1, x2)[-1]
    print(f"Without extras    - outputs.shape: {outputs.shape}")  # noqa: T201

    # Test HDANet
    model_hda = ChangeDetectionModel(change_detection_model='hdanet', in_channels=9, out_channels=2)
    x1_sm = torch.randn(2, 9, 256, 256)
    x2_sm = torch.randn(2, 9, 256, 256)
    outputs_hda = model_hda(x1_sm, x2_sm)[-1]
    print(f"HDANet            - outputs.shape: {outputs_hda.shape}")  # noqa: T201

    # Test with FiLM
    model_film = ChangeDetectionModel(
        change_detection_model='changeformer_6', in_channels=9, out_channels=2,
        use_metadata_film=True,
    )
    sat_pass = torch.tensor([0, 1, 0, 1, 0])
    beam = torch.tensor([0, 1, 2, 3, 0])
    outputs_film = model_film(x1, x2, sat_pass=sat_pass, beam=beam)[-1]
    print(f"With FiLM         - outputs.shape: {outputs_film.shape}")  # noqa: T201

    # Test with CBAM
    model_cbam = ChangeDetectionModel(
        change_detection_model='changeformer_6', in_channels=9, out_channels=2,
        use_cbam=True,
    )
    outputs_cbam = model_cbam(x1, x2)[-1]
    print(f"With CBAM         - outputs.shape: {outputs_cbam.shape}")  # noqa: T201

    # Test with Channel Dropout (training mode)
    model_cd = ChangeDetectionModel(
        change_detection_model='changeformer_6', in_channels=9, out_channels=2,
        use_channel_dropout=True, channel_dropout_prob=0.2,
    )
    model_cd.train()
    outputs_cd = model_cd(x1, x2)[-1]
    print(f"With ChannelDrop  - outputs.shape: {outputs_cd.shape}")  # noqa: T201

    # Test with DFA
    model_dfa = ChangeDetectionModel(
        change_detection_model='changeformer_6', in_channels=9, out_channels=2,
        use_dfa=True,
    )
    outputs_dfa = model_dfa(x1, x2)[-1]
    print(f"With DFA          - outputs.shape: {outputs_dfa.shape}")  # noqa: T201

    # Test ALL modules combined
    model_all = ChangeDetectionModel(
        change_detection_model='changeformer_6', in_channels=9, out_channels=2,
        use_metadata_film=True,
        use_cbam=True,
        use_channel_dropout=True, channel_dropout_prob=0.15,
        use_dfa=True,
    )
    model_all.train()
    outputs_all = model_all(x1, x2, sat_pass=sat_pass, beam=beam)[-1]
    print(f"With ALL modules  - outputs.shape: {outputs_all.shape}")  # noqa: T201

