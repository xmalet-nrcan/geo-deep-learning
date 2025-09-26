from pathlib import Path

from utils.logger import get_logger
from utils.utils import get_key_def

# Set the logging file
logging = get_logger(__name__)  # import logging
from dataset.pre_post_change_dataset import PrePostChangeDataset


def create_pre_post_dataloader(csv_dataset_path: Path,
                               data_directory: Path,
                               batch_size: int,
                               gpu_devices_dict: dict,
                               bands: list,
                               cfg: dict,
                               beam_filter: str | list,
                               satellite_pass_filter: str | list,
                               train_test_val_split: list[float],
                               split_seed: int = 42):
    """
    Function to create dataloader objects for training, validation and test datasets for pre-post change detection.
    @param csv_dataset_path: path to csv file containting list of pre-post change detection patches
    @param batch_size: (int) batch size
    @param gpu_devices_dict: (dict) dictionary where each key contains an available GPU with

    """
    logging.info('Creating dataloader for pre-post change detection')
    logging.info('CSV dataset path: {}'.format(csv_dataset_path))
    rcm_pre_post_dataset = PrePostChangeDataset.build_dataset_from_csv(csv_dataset_path,
                                                                       data_directory=str(data_directory),
                                                                       band_names=bands,
                                                                       beams_filter=beam_filter,
                                                                       sat_pass_filter=satellite_pass_filter,
                                                                       )
    # Number of workers
    if cfg.training.num_workers:
        num_workers = cfg.training.num_workers
    else:  # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
        num_workers = len(gpu_devices_dict.keys()) * 4 if len(gpu_devices_dict.keys()) > 1 else 4
    logging.info("Dataset contains {} patches".format(len(rcm_pre_post_dataset)))
    logging.info(f"Using {num_workers} workers for data loading")
    logging.info("Data path: {}".format(data_directory))
    # TODO : Implement Sampler weights for pre-post change detection ??
    # patches_weight = torch.from_numpy(patches_weight)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(patches_weight.type('torch.DoubleTensor'),
    #                                                        len(patches_weight))
    if train_test_val_split is None:
        train_test_val_split = get_key_def("train_test_val_split", cfg['dataset'], default=[0.7, 0.15, 0.15])

    trn_dataloader, tst_dataloader, val_dataloader = rcm_pre_post_dataset.get_train_val_test_dataloaders(
        batch_size=batch_size,
        ratios=train_test_val_split,
        seed=split_seed,
        num_workers=num_workers,
        drop_last=True,

    )
    if len(trn_dataloader) == 0 or len(val_dataloader) == 0:
        raise ValueError(f"\nTrain and validation dataloader should contain at least one data item."
                         f"\nTrain dataloader's length: {len(trn_dataloader)}"
                         f"\nVal dataloader's length: {len(val_dataloader)}")

    return trn_dataloader, val_dataloader, tst_dataloader
