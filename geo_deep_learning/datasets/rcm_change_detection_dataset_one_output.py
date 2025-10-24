import torch

from geo_deep_learning.datasets.rcm_change_detection_dataset import  RCMChangeDetectionDataset


class RCMChangeDetectionDatasetOneOutput(RCMChangeDetectionDataset):
    """RCM Change Detection Dataset with one band."""

    # def __getitem__(self, index: int) -> dict:
    #     sample = super().__getitem__(index)
    #     # Keep only the first band
    #     img_pre = sample['image_pre']
    #     img_post = sample['image_post']
    #     img = torch.cat([img_pre, img_post], dim=0)
    #     sample['image'] = img
    #     return sample

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)

        img_pre = sample['image_pre']          # [COMMON_MASK, pre_core..., SAT_PASS, BEAM]
        img_post = sample['image_post']        # [COMMON_MASK, post_core..., SAT_PASS, BEAM]

        # Indices assuming channel order described above
        common_mask = img_pre[0:1]                    # keep dimension
        pre_core = img_pre[1:-2]
        post_core = img_post[1:-2]
        sat_pass = img_pre[-2:-1]              # take from pre (assumed identical)
        beam = img_pre[-1:]                    # take from pre (assumed identical)

        merged = torch.cat([common_mask, pre_core, post_core, sat_pass, beam], dim=0)
        sample['image'] = merged

        # Optional: rebuild band names if available
        orig_band_names = sample.get('bands')
        if orig_band_names and len(orig_band_names) >= 4:
            common = orig_band_names[0]
            sat_name = orig_band_names[-2]
            beam_name = orig_band_names[-1]
            core = orig_band_names[1:-2]
            sample['bands'] = [
                [common] +
                [f'pre_{b}' for b in core] +
                [f'post_{b}' for b in core] +
                [sat_name, beam_name]
            ]

        return sample

