import torch

from geo_deep_learning.datasets.rcm_change_detection_dataset import  RCMChangeDetectionDataset


class RCMChangeDetectionDatasetMergePrePost(RCMChangeDetectionDataset):
    """RCM Change Detection Dataset with one band."""


    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)

        img_pre = sample['image_pre']          # [COMMON_MASK, pre_core..., SAT_PASS, BEAM]
        img_post = sample['image_post']        # [COMMON_MASK, post_core..., SAT_PASS, BEAM]

        sample['image'] = torch.cat([img_pre, img_post], dim=0)


        return sample

