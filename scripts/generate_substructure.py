import os
import sys

sys.path.append("../")

import argparse
from os.path import join

import energyflow as ef
import h5py
import hydra

# plots and metrics
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import KernelDensity

from src.data.components import (
    calculate_all_wasserstein_metrics,
    inverse_normalize_tensor,
    normalize_tensor,
)
from src.utils.data_generation import generate_data
from src.utils.plotting import apply_mpl_styles, plot_data, prepare_data_for_plotting

apply_mpl_styles()

# set env variable DATA_DIR again because of hydra
from dotenv import load_dotenv
from jetnet.evaluation import w1efp, w1m, w1p

from src.data.components.metrics import wasserstein_distance_batched
from src.utils.jet_substructure import dump_hlvs
from src.utils.plotting import plot_full_substructure, plot_substructure

load_dotenv()
os.environ["DATA_DIR"] = os.environ.get("DATA_DIR")

data_folder = os.environ.get("DATA_DIR")


def main(params):
    # Load vinicius data
    path_v = f"{data_folder}/lhco/generated/FPCD_LHCO_SR.h5"
    with h5py.File(path_v, "r") as f:
        print(f.keys())
        jet_features_v = f["jet_features"][:]
        particle_data_v = f["particle_features"][:]
        mjj_v = f["mjj"][:]
    print(jet_features_v.shape)
    print(particle_data_v.shape)
    print(mjj_v.shape)

    # Load idealized data
    path_id = f"{data_folder}/lhco/generated/idealized_LHCO.h5"
    with h5py.File(path_id, "r") as f:
        print(f.keys())
        jet_features_id = f["jet_features"][:]
        particle_data_id = f["particle_features"][:]
        mjj_id = f["mjj"][:]
    print(jet_features_id.shape)
    print(particle_data_id.shape)
    print(mjj_id.shape)

    # Load ced data
    # path_ced = f"{data_folder}/lhco/generated/FM_LHCO_SR.h5"
    # path_ced = f"{data_folder}/lhco/generated/lhco_both_jets-midpoint-250.h5"
    # path_ced = f"{data_folder}/lhco/generated/latent64-midpoint-200.h5"
    # path_ced = f"{data_folder}/lhco/generated/lhco-xy-midpoint-300.h5"
    path_ced = f"{data_folder}/lhco/generated/lhco-xy-256-logpt_sr-midpoint-500.h5"
    # path_ced = f"{data_folder}/lhco/generated/FPCD_LHCO_SR_2.h5"

    with h5py.File(path_ced, "r") as f:
        print(f.keys())
        jet_features_ced = f["jet_features"][:]
        particle_data_ced = f["particle_features"][:]
        mjj_ced = f["mjj"][:]
        raw_ced = f["data_raw"][:]
    print(jet_features_ced.shape)
    print(particle_data_ced.shape)
    # print(mjj_ced.shape)

    particle_data_v = particle_data_v[: len(particle_data_id)]
    jet_features_v = jet_features_v[: len(jet_features_id)]
    mjj_v = mjj_v[: len(mjj_id)]

    particle_data_ced = particle_data_ced[..., [1, 2, 0]]
    particle_data_id = particle_data_id[..., [1, 2, 0]]
    particle_data_v = particle_data_v[..., [1, 2, 0]]

    print(particle_data_ced.shape)
    print(particle_data_id.shape)
    print(particle_data_v.shape)

    print(particle_data_v.reshape(-1, particle_data_v.shape[-2], particle_data_v.shape[-1]).shape)

    if params.data == "vinicius":
        particle_data = particle_data_v
        jet_features = jet_features_v
        mjj = mjj_v
    elif params.data == "idealized":
        particle_data = particle_data_id
        jet_features = jet_features_id
        mjj = mjj_id
    elif params.data == "ced":
        particle_data = particle_data_ced
        jet_features = jet_features_ced
        mjj = mjj_ced
    else:
        raise ValueError("Invalid data argument")

    print(f"Using {params.data} data for substructure calculation")
    print(f"Saving to /beegfs/desy/user/ewencedr/data/lhco/substructure/full_{params.data}")

    dump_hlvs(
        particle_data.reshape(-1, particle_data.shape[-2], particle_data.shape[-1]),
        f"/beegfs/desy/user/ewencedr/data/lhco/substructure/full_{params.data}",
        plot=False,
    )
    print(f"Saved to /beegfs/desy/user/ewencedr/data/lhco/substructure/full_{params.data}")


if __name__ == "__main__":
    # ARGUMENTS
    # define parser
    parser = argparse.ArgumentParser(description="Generating Samples from a trained model.")

    parser.add_argument(
        "--data",
        "-d",
        default="vinicius",
        help="which data to use for calculation of substructure",
        type=str,
    )

    params = parser.parse_args()
    main(params)
