import torch

from geo_deep_learning.datasets.rcm_change_detection_dataset import  RCMChangeDetectionDataset


class RCMChangeDetectionDatasetOneOutput(RCMChangeDetectionDataset):
    """RCM Change Detection Dataset with one band."""

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        # Keep only the first band
        img_pre = sample['image_pre']
        img_post = sample['image_post']
        img = torch.cat([img_pre, img_post], dim=0)
        sample['image'] = img
        return sample

