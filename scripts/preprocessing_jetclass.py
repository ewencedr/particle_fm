"""script to preprocess the JetClass dataset based on the npz files created by
prepare_dataset.py. See also the repository of the paper: https://github.com/uhh-pd-ml/beyond_kinematics
"""

import glob
import logging
import os

import h5py
import hydra
from omegaconf import DictConfig, OmegaConf

from particle_fm.preprocessing.plotting import plot_h5file
from particle_fm.preprocessing.utils import calc_means_and_stds, merge_files, standardize_data

logger = logging.getLogger("preprocessing")


@hydra.main(version_base=None, config_path="../configs/preprocessing", config_name="data.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info("Preprocessing the files found in %s:", cfg.output_dir)

    # merge the different jet types files (for each train/val/test)
    for split_folder in ["train_100M", "val_5M", "test_20M"]:
        # for split_folder in ["train_100M"]:
        merge_files(
            list(glob.glob(f"{cfg.output_dir}/{split_folder}/*.npz")),
            f"{cfg.output_dir}/{split_folder}/merged.h5",
        )

    merged_files_dict = {
        "train": f"{cfg.output_dir}/train_100M/merged.h5",
        "val": f"{cfg.output_dir}/val_5M/merged.h5",
        "test": f"{cfg.output_dir}/test_20M/merged.h5",
    }
    merged_plots_dir = f"{cfg.output_dir}/plots_merged"
    os.makedirs(merged_plots_dir, exist_ok=True)
    plot_h5file(
        {key: h5py.File(value) for key, value in merged_files_dict.items()},
        merged_plots_dir,
        n_plot=200_000,
    )

    # standardize the data
    standardize_data(
        filename_dict=merged_files_dict,
        standardize_particle_features=True,
        standardize_jet_features=False,
    )

    logger.info(
        "Calc means and std after standardization: should be 0 and 1 for "
        "train and close to that for val/test"
    )
    # just as a cross-check, check means and std of the data afterwards
    calc_means_and_stds(filename=merged_files_dict["train"].replace(".h5", "_standardized.h5"))
    calc_means_and_stds(filename=merged_files_dict["val"].replace(".h5", "_standardized.h5"))
    calc_means_and_stds(filename=merged_files_dict["test"].replace(".h5", "_standardized.h5"))
    merged_std_plots_dir = f"{cfg.output_dir}/plots_merged_std"
    os.makedirs(merged_std_plots_dir, exist_ok=True)
    plot_h5file(
        {
            key: h5py.File(value.replace(".h5", "_standardized.h5"))
            for key, value in merged_files_dict.items()
        },
        merged_std_plots_dir,
        n_plot=200_000,
    )


if __name__ == "__main__":
    main()
