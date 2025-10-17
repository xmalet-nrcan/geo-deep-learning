import pathlib
from typing import Iterable

import hydra
import numpy as np
import pandas
from omegaconf import DictConfig, OmegaConf

from geo_deep_learning.utils.rasters import compute_dataset_all_stats_from_list


@hydra.main(version_base="1.3", config_path="../../configs", config_name="rcm_change_detection")
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))


    csv_file_path = pathlib.Path(cfg['stats_calculations']['csv_file_path'])
    df = pandas.read_csv(csv_file_path)
    filters = cfg['stats_calculations']['filters']
    filtered_df = df.copy()

    for key, value in filters.items():
        print(f"VALUE IS A : {value}")
        print(f"VALUE IS A : {isinstance(value, Iterable)}")
        if value is None:
            continue  # on ignore cette clé
        # Si c’est une liste (plusieurs valeurs possibles)
        if isinstance(value, Iterable):
            filtered_df = filtered_df[filtered_df[key].isin(value)]
        else:
            filtered_df = filtered_df[filtered_df[key] == value]

    tiff_folder_path = pathlib.Path(cfg['data']['init_args']['patches_root_folder'])
    path_key_to_change = cfg['stats_calculations']['path_key_to_change']
    tiffs = []
    print(filtered_df.columns)
    print( cfg['stats_calculations']['tiff_paths_keys'])

    for tiff_paths in cfg['stats_calculations']['tiff_paths_keys']:
        print(filtered_df[tiff_paths])
        tiffs.extend((filtered_df[tiff_paths].tolist()))


    tiffs = list(set(tiffs))
    tiffs = [str(pathlib.Path(i.replace(path_key_to_change, str(tiff_folder_path)).strip())) for i in tiffs]
    print(f"Number of TIFF files to process: {len(tiffs)}")
    print(tiffs)

    print(f"Min and max values saved to {cfg.stats_calculations.csv_save_path}")

    mean, std, min_val, max_val = compute_dataset_all_stats_from_list(tiffs, np.int16)
    print(f"Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")

if __name__ == "__main__":
    main()
