import os
import sys

sys.path.append("../")

from os.path import join

import argparse

import energyflow as ef
import h5py
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import KernelDensity

# plots and metrics
import matplotlib.pyplot as plt

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

load_dotenv()
os.environ["DATA_DIR"] = os.environ.get("DATA_DIR")

data_folder = os.environ.get("DATA_DIR")


def main(params):
    folder = params.folder
    save_file = params.save_file

    cfg_backup_file = join(folder, "config.yaml")

    # load everything from experiment config
    with hydra.initialize(version_base=None, config_path="../configs/"):
        if os.path.exists(cfg_backup_file):
            print("config file already exists --> loading from run directory")
        else:
            raise FileNotFoundError("config file not found")

    cfg = OmegaConf.load(cfg_backup_file)
    print(type(cfg))
    print(OmegaConf.to_yaml(cfg))

    print("Instantiating model and data module")
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    datamodule.setup()

    ckpt = join(folder, "checkpoints", "last-EMA.ckpt")

    model = model.load_from_checkpoint(ckpt)
    print(f"Model loaded from {ckpt}")

    cond_x = datamodule.jet_data_sr_raw[:, 0]
    mask_x = datamodule.mask_sr_raw[:, 0]
    cond_y = datamodule.jet_data_sr_raw[:, 1]
    mask_y = datamodule.mask_sr_raw[:, 1]

    normalized_cond_x = normalize_tensor(
        torch.Tensor(cond_x).clone(),
        datamodule.cond_means,
        datamodule.cond_stds,
        datamodule.hparams.normalize_sigma,
    )

    normalized_cond_y = normalize_tensor(
        torch.Tensor(cond_y).clone(),
        datamodule.cond_means,
        datamodule.cond_stds,
        datamodule.hparams.normalize_sigma,
    )

    print("Generating data first jet")
    torch.manual_seed(9999)
    data_x, generation_time_x = generate_data(
        model,
        num_jet_samples=len(mask_x),
        batch_size=2048,
        cond=torch.Tensor(normalized_cond_x),
        variable_set_sizes=datamodule.hparams.variable_jet_sizes,
        mask=torch.Tensor(mask_x),
        normalized_data=datamodule.hparams.normalize,
        means=datamodule.means,
        stds=datamodule.stds,
        ode_solver=params.ode_solver,
        ode_steps=params.ode_steps,
    )

    print("Generating data second jet")
    torch.manual_seed(9999)
    data_y, generation_time_y = generate_data(
        model,
        num_jet_samples=len(mask_y),
        batch_size=2048,
        cond=torch.Tensor(normalized_cond_y),
        variable_set_sizes=datamodule.hparams.variable_jet_sizes,
        mask=torch.Tensor(mask_y),
        normalized_data=datamodule.hparams.normalize,
        means=datamodule.means,
        stds=datamodule.stds,
        ode_solver=params.ode_solver,
        ode_steps=params.ode_steps,
    )

    data_x_raw = np.copy(data_x)
    data_y_raw = np.copy(data_y)
    data_raw = np.stack([data_x_raw, data_y_raw], axis=1)

    print("Preparing data for saving")
    # remove unphysical values
    data_x[data_x[:, :, 2] < 0] = np.min(
        datamodule.tensor_train.numpy()[:, :, 2][datamodule.tensor_train.numpy()[:, :, 2] > 0.0]
    )
    data_x[data_x[:, :, 2] > 1] = np.max(
        datamodule.tensor_train.numpy()[:, :, 2][datamodule.tensor_train.numpy()[:, :, 2] < 1.0]
    )

    data_y[data_y[:, :, 2] < 0] = np.min(
        datamodule.tensor_train.numpy()[:, :, 2][datamodule.tensor_train.numpy()[:, :, 2] > 0.0]
    )
    data_y[data_y[:, :, 2] > 1] = np.max(
        datamodule.tensor_train.numpy()[:, :, 2][datamodule.tensor_train.numpy()[:, :, 2] < 1.0]
    )

    # back to non-rel coordinates

    pt_x = cond_x[:, 0].reshape(-1, 1)
    eta_x = cond_x[:, 1].reshape(-1, 1)
    phi_x = cond_x[:, 2].reshape(-1, 1)
    m_x = cond_x[:, 3].reshape(-1, 1)

    pt_y = cond_y[:, 0].reshape(-1, 1)
    eta_y = cond_y[:, 1].reshape(-1, 1)
    phi_y = cond_y[:, 2].reshape(-1, 1)
    m_y = cond_y[:, 3].reshape(-1, 1)

    mask_x_nonrel = np.expand_dims((data_x[..., 2] > 0).astype(int), axis=-1)
    non_rel_eta_x = np.expand_dims(data_x.copy()[:, :, 0] + eta_x, axis=-1)
    non_rel_phi_x = np.expand_dims(data_x.copy()[:, :, 1] + phi_x, axis=-1)
    # wrap phi between -pi and pi
    non_rel_phi_x = np.where(
        non_rel_phi_x > np.pi,
        non_rel_phi_x - 2 * np.pi,
        non_rel_phi_x,
    )
    non_rel_phi_x = np.where(
        non_rel_phi_x < -np.pi,
        non_rel_phi_x + 2 * np.pi,
        non_rel_phi_x,
    )
    non_rel_pt_x = np.expand_dims(data_x.copy()[:, :, 2] * pt_x, axis=-1)
    # fix the masking
    non_rel_eta_x = non_rel_eta_x * mask_x_nonrel
    non_rel_phi_x = non_rel_phi_x * mask_x_nonrel
    data_x_nonrel = np.concatenate([non_rel_eta_x, non_rel_phi_x, non_rel_pt_x], axis=-1)

    mask_y_nonrel = np.expand_dims((data_y[..., 2] > 0).astype(int), axis=-1)
    non_rel_eta_y = np.expand_dims(data_y.copy()[:, :, 0] + eta_y, axis=-1)
    non_rel_phi_y = np.expand_dims(data_y.copy()[:, :, 1] + phi_y, axis=-1)
    # wrap phi between -pi and pi
    non_rel_phi_y = np.where(
        non_rel_phi_y > np.pi,
        non_rel_phi_y - 2 * np.pi,
        non_rel_phi_y,
    )
    non_rel_phi_y = np.where(
        non_rel_phi_y < -np.pi,
        non_rel_phi_y + 2 * np.pi,
        non_rel_phi_y,
    )
    non_rel_pt_y = np.expand_dims(data_y.copy()[:, :, 2] * pt_y, axis=-1)
    # fix the masking
    non_rel_eta_y = non_rel_eta_y * mask_y_nonrel
    non_rel_phi_y = non_rel_phi_y * mask_y_nonrel
    data_y_nonrel = np.concatenate([non_rel_eta_y, non_rel_phi_y, non_rel_pt_y], axis=-1)

    # stack both jets
    particle_data = np.stack([data_x, data_y], axis=1)
    particle_data_nonrel = np.stack([data_x_nonrel, data_y_nonrel], axis=1)
    jet_features = np.stack([cond_x, cond_y], axis=1)

    # add particle multiplicity as feature
    mask = particle_data[..., 0] != 0
    particle_multiplicity = np.sum(mask, axis=-1)
    jet_features = np.concatenate([jet_features, particle_multiplicity[..., None]], axis=-1)

    # shuffle data
    pt = jet_features[:, :, 0]
    args = np.argsort(pt, axis=1)[:, ::-1]
    perm = np.random.permutation(args.shape[0])
    args = args[perm]
    # pt_sorted = np.take_along_axis(pt, args, axis=1)

    sorted_jets = np.take_along_axis(jet_features, args[..., None], axis=1)
    sorted_consts = np.take_along_axis(particle_data, args[..., None, None], axis=1)
    sorted_consts_nonrel = np.take_along_axis(particle_data_nonrel, args[..., None, None], axis=1)

    save_file = join(
        data_folder, "lhco", "generated", f"{save_file}-{params.ode_solver}-{params.ode_steps}.h5"
    )
    with h5py.File(save_file, "w") as f:
        f.create_dataset("particle_features", data=sorted_consts[..., [2, 0, 1]])
        f.create_dataset("jet_features", data=sorted_jets)
        f.create_dataset("particle_features_nonrel", data=sorted_consts_nonrel[..., [2, 0, 1]])
        f.create_dataset("data_raw", data=data_raw[..., [2, 0, 1]])

    print(f"Saved data to {save_file}")

    print(f"ODE solver: {params.ode_solver}")
    print(f"ODE steps: {params.ode_steps}")
    print(f"Generation time: {generation_time_x + generation_time_y} s")
    print(f"Number of generated samples: {len(particle_data)}")

    # TODO Wasserstein metrics


if __name__ == "__main__":
    # ARGUMENTS
    # define parser
    parser = argparse.ArgumentParser(description="Generating Samples from a trained model.")

    parser.add_argument(
        "--folder",
        "-f",
        default="/beegfs/desy/user/ewencedr/deep-learning/logs/lhco all jets/runs/2023-08-25_14-42-49",
        help="folder of the model to generate from",
        type=str,
    )

    parser.add_argument(
        "--save_file",
        "-s",
        default="lhco_both_jets",
        help="name of the file to save the data to",
        type=str,
    )

    parser.add_argument(
        "--ode_solver",
        "-ode",
        default="midpoint",
        help="ode_solver for sampling",
        type=str,
    )
    parser.add_argument(
        "--ode_steps",
        "-steps",
        default=250,
        help="steps for ode_solver",
        type=int,
    )
    params = parser.parse_args()
    main(params)