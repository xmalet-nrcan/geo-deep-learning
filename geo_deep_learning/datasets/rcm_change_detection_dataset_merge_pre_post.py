import torch

from geo_deep_learning.datasets.rcm_change_detection_dataset import  RCMChangeDetectionDataset


class RCMChangeDetectionDatasetMergePrePost(RCMChangeDetectionDataset):
    """RCM Change Detection Dataset with one band."""

    @classmethod
    def num_input_channels(cls, *args, **kwargs) -> int:
        """Pre and post are stacked along the channel axis → twice the channels."""
        return 2 * super().num_input_channels(*args, **kwargs)


    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)

        img_pre = sample['image_pre']          # [COMMON_MASK, pre_core..., SAT_PASS, BEAM]
        img_post = sample['image_post']        # [COMMON_MASK, post_core..., SAT_PASS, BEAM]

        sample['image'] = torch.cat([img_pre, img_post], dim=0)


        return sample

