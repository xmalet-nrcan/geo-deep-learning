"""Change Detection Model segmentation model."""
import torch
from torch import Tensor

from geo_deep_learning.models.change_detection.metadata_film_conditioner import MetadataFiLMConditioner
from geo_deep_learning.models.change_detection.sub_models.changeformer.original_change_former import ChangeFormerV6, \
    ChangeFormerV5
from geo_deep_learning.models.segmentation.base import BaseSegmentationModel


class ChangeDetectionModel(BaseSegmentationModel):
    """Change Detection segmentation model.

    When ``use_metadata_film=True``, SAT_PASS and BEAM are no longer fed as
    constant spatial bands.  Instead a lightweight FiLM layer modulates the
    input channels *before* the Transformer encoder, letting the network learn
    acquisition-specific adjustments without wasting encoder capacity.
    """
    # TODO : For now, only use ChangeFormer. Add more models later.
    #  ChangeFormer: https://github.com/wgcban/ChangeFormer.git
    def __init__(self, change_detection_model: str = "changeformer",
                 in_channels: int = 3,
                 out_channels: int = 2,
                 use_metadata_film: bool = False,
                 film_embed_dim: int = 32,
                 film_metadata_fields: dict[str, int] | None = None,
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
        """
        super().__init__()

        model_selection = {'changeformer': ChangeFormerV6,
                           'changeformer_5': ChangeFormerV5,
                           'changeformer_6': ChangeFormerV6,
                           }

        model_sub_name = {'changeformer': 'changeformer',
                          'changeformer_5': 'changeformer',
                          'changeformer_6': 'changeformer',
                          }

        model_parameters = {'changeformer': {'decoder_softmax': False, 'embed_dim': 256}}
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

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        sat_pass: Tensor | None = None,
        beam: Tensor | None = None,
        **metadata_kwargs: Tensor,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            x1: Pre-image tensor [B, C, H, W].
            x2: Post-image tensor [B, C, H, W].
            sat_pass: [B] integer tensor for satellite pass (0=ASC, 1=DESC).
            beam: [B] integer tensor for beam (0=A, 1=B, 2=C, 3=D).
            **metadata_kwargs: Additional metadata fields (e.g. season=[B]).

        Returns:
            List of output tensors (one per decoder head + final).
        """
        if self.film_conditioner is not None and sat_pass is not None and beam is not None:
            x1 = self.film_conditioner(x1, sat_pass, beam, **metadata_kwargs)
            x2 = self.film_conditioner(x2, sat_pass, beam, **metadata_kwargs)

        return self.change_detection_model(x1, x2)



if __name__ == '__main__':
    # Test without FiLM
    model = ChangeDetectionModel(change_detection_model='changeformer_6', in_channels=9, out_channels=2)
    x1 = torch.randn(5, 9, 512, 512)
    x2 = torch.randn(5, 9, 512, 512)
    outputs = model(x1, x2)[-1]
    print(f"Without FiLM - outputs.shape: {outputs.shape}")  # noqa: T201

    # Test with FiLM
    model_film = ChangeDetectionModel(
        change_detection_model='changeformer_6', in_channels=9, out_channels=2,
        use_metadata_film=True,
    )
    sat_pass = torch.tensor([0, 1, 0, 1, 0])
    beam = torch.tensor([0, 1, 2, 3, 0])
    outputs_film = model_film(x1, x2, sat_pass=sat_pass, beam=beam)[-1]
    print(f"With FiLM    - outputs.shape: {outputs_film.shape}")  # noqa: T201

