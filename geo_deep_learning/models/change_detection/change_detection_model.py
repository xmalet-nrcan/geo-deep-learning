"""Change Detection Model segmentation model."""
import torch
from torch import Tensor

from geo_deep_learning.models.change_detection.sub_models.changeformer.original_change_former import ChangeFormerV6, \
    ChangeFormerV5
from geo_deep_learning.models.segmentation.base import BaseSegmentationModel


class ChangeDetectionModel(BaseSegmentationModel):
    """Change Detection segmentation model."""
    # TODO : For now, only use ChangeFormer. Add more models later.
    #  ChangeFormer: https://github.com/wgcban/ChangeFormer.git
    def __init__(self, change_detection_model: str = "changeformer",
                 in_channels: int = 3,
                 out_channels: int = 2,
                 **kwargs) -> None:
        """Initialize Change Detection segmentation model."""
        super().__init__()

        model_selection = {'changeformer': ChangeFormerV6,
                           'changeformer_5': ChangeFormerV5,
                           'changeformer_6': ChangeFormerV6,
                           }

        model_sub_name = {'changeformer': 'changeformer',
                          'changeformer_5': 'changeformer',
                          'changeformer_6': 'changeformer',
                          }

        model_parameters = {'changeformer': {'decoder_softmax': True, 'embed_dim': 256}}
        model_kwargs = model_parameters.get(model_sub_name.get(change_detection_model))
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs.update(kwargs)
        model = model_selection[change_detection_model]

        self.change_detection_model = model(in_channels=in_channels,
                                            out_channels=out_channels,
                                            **model_kwargs)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass of the model.
        Because it's a change detection model, it takes two inputs as PRE / POST or X1 / X2.
        Args:
            x1 (Tensor): First input tensor.
            x2 (Tensor): Second input tensor.
        Returns:
            Tensor: Output tensor.
        """
        return self.change_detection_model(x1, x2)



if __name__ == '__main__':
    model = ChangeDetectionModel(change_detection_model='changeformer_6', in_channels=3, out_channels=2)
    x1 = torch.randn(5, 3, 512, 512)
    x2 = torch.randn(5, 3, 512, 512)
    outputs = model(x1, x2)
    print(f"outputs.shape: {outputs.shape}")

