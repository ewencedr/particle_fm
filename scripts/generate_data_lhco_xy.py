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

from src.data.components.metrics import wasserstein_distance_batched
from src.utils.jet_substructure import dump_hlvs

from src.data.components import (
    calculate_all_wasserstein_metrics,
    inverse_normalize_tensor,
    normalize_tensor,
)
from src.utils.data_generation import generate_data
from src.utils.plotting import (
    apply_mpl_styles,
    plot_data,
    prepare_data_for_plotting,
    plot_substructure,
    plot_full_substructure,
)

apply_mpl_styles()

# set env variable DATA_DIR again because of hydra
from dotenv import load_dotenv

load_dotenv()
os.environ["DATA_DIR"] = os.environ.get("DATA_DIR")

data_folder = os.environ.get("DATA_DIR")


def main(params):
    folder_x = params.folder_x
    folder_y = params.folder_y
    save_file = params.save_file

    cfg_backup_file_x = join(folder_x, "config.yaml")
    cfg_backup_file_y = join(folder_y, "config.yaml")

    # load everything from experiment config
    with hydra.initialize(version_base=None, config_path="../configs/"):
        if os.path.exists(cfg_backup_file_x):
            print("config file already exists --> loading from run directory")
        else:
            raise FileNotFoundError("config file not found")

    # load everything from experiment config
    with hydra.initialize(version_base=None, config_path="../configs/"):
        if os.path.exists(cfg_backup_file_y):
            print("config file already exists --> loading from run directory")
        else:
            raise FileNotFoundError("config file not found")

    cfg_x = OmegaConf.load(cfg_backup_file_x)
    print(type(cfg_x))
    print(OmegaConf.to_yaml(cfg_x))
    cfg_y = OmegaConf.load(cfg_backup_file_y)
    print(type(cfg_y))
    print(OmegaConf.to_yaml(cfg_y))

    print("Instantiating model and data module")
    datamodule_x = hydra.utils.instantiate(cfg_x.data)
    model_x = hydra.utils.instantiate(cfg_x.model)
    datamodule_y = hydra.utils.instantiate(cfg_y.data)
    model_y = hydra.utils.instantiate(cfg_y.model)

    datamodule_x.setup()
    datamodule_y.setup()

    ckpt_x = join(folder_x, "checkpoints", "last-EMA.ckpt")
    ckpt_y = join(folder_y, "checkpoints", "last-EMA.ckpt")

    model_x = model_x.load_from_checkpoint(ckpt_x)
    print(f"Model loaded from {ckpt_x}")
    model_y = model_y.load_from_checkpoint(ckpt_y)
    print(f"Model loaded from {ckpt_y}")

    cond_x = datamodule_x.jet_data_sr_raw
    mask_x = datamodule_x.mask_sr_raw
    cond_y = datamodule_y.jet_data_sr_raw
    mask_y = datamodule_y.mask_sr_raw
    mjj = datamodule_x.mjj_sr

    normalized_cond_x = normalize_tensor(
        torch.Tensor(cond_x).clone(),
        datamodule_x.cond_means,
        datamodule_x.cond_stds,
        datamodule_x.hparams.normalize_sigma,
    )

    normalized_cond_y = normalize_tensor(
        torch.Tensor(cond_y).clone(),
        datamodule_y.cond_means,
        datamodule_y.cond_stds,
        datamodule_y.hparams.normalize_sigma,
    )

    print("Generating data first jet")
    torch.manual_seed(9999)
    data_x, generation_time_x = generate_data(
        model_x,
        num_jet_samples=len(mask_x),
        batch_size=1024,
        cond=torch.Tensor(normalized_cond_x),
        variable_set_sizes=datamodule_x.hparams.variable_jet_sizes,
        mask=torch.Tensor(mask_x),
        normalized_data=datamodule_x.hparams.normalize,
        means=datamodule_x.means,
        stds=datamodule_x.stds,
        log_pt=datamodule_x.hparams.log_pt,
        pt_standardization=datamodule_x.hparams.pt_standardization,
        ode_solver=params.ode_solver,
        ode_steps=params.ode_steps,
    )

    print("Generating data second jet")
    torch.manual_seed(9999)
    data_y, generation_time_y = generate_data(
        model_y,
        num_jet_samples=len(mask_y),
        batch_size=1024,
        cond=torch.Tensor(normalized_cond_y),
        variable_set_sizes=datamodule_y.hparams.variable_jet_sizes,
        mask=torch.Tensor(mask_y),
        normalized_data=datamodule_y.hparams.normalize,
        means=datamodule_y.means,
        stds=datamodule_y.stds,
        log_pt=datamodule_y.hparams.log_pt,
        pt_standardization=datamodule_y.hparams.pt_standardization,
        ode_solver=params.ode_solver,
        ode_steps=params.ode_steps,
    )

    data_x_raw = np.copy(data_x)
    data_y_raw = np.copy(data_y)
    data_raw = np.stack([data_x_raw, data_y_raw], axis=1)

    print("Preparing data for saving")
    # remove unphysical values
    data_x[data_x[:, :, 2] < 0] = np.min(
        datamodule_x.tensor_train.numpy()[:, :, 2][
            datamodule_x.tensor_train.numpy()[:, :, 2] > 0.0
        ]
    )
    data_x[data_x[:, :, 2] > 1] = np.max(
        datamodule_x.tensor_train.numpy()[:, :, 2][
            datamodule_x.tensor_train.numpy()[:, :, 2] < 1.0
        ]
    )

    data_y[data_y[:, :, 2] < 0] = np.min(
        datamodule_y.tensor_train.numpy()[:, :, 2][
            datamodule_y.tensor_train.numpy()[:, :, 2] > 0.0
        ]
    )
    data_y[data_y[:, :, 2] > 1] = np.max(
        datamodule_y.tensor_train.numpy()[:, :, 2][
            datamodule_y.tensor_train.numpy()[:, :, 2] < 1.0
        ]
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

    save_path = join(
        data_folder, "lhco", "generated", f"{save_file}-{params.ode_solver}-{params.ode_steps}.h5"
    )
    with h5py.File(save_path, "w") as f:
        f.create_dataset("particle_features", data=sorted_consts[..., [2, 0, 1]])
        f.create_dataset("jet_features", data=sorted_jets)
        f.create_dataset("particle_features_nonrel", data=sorted_consts_nonrel[..., [2, 0, 1]])
        f.create_dataset("data_raw", data=data_raw[..., [2, 0, 1]])
        f.create_dataset("mjj", data=mjj)

    print(f"Saved data to {save_path}")

    print(f"ODE solver: {params.ode_solver}")
    print(f"ODE steps: {params.ode_steps}")
    print(f"Generation time: {generation_time_x + generation_time_y} s")
    print(f"Number of generated samples: {len(particle_data)}")

    # TODO Wasserstein metrics
    print("Calculating Wasserstein metrics")
    # Load idealized data
    path_id = f"{data_folder}/lhco/generated/idealized_LHCO.h5"
    with h5py.File(path_id, "r") as f:
        print(f.keys())
        jet_features_id = f["jet_features"][:]
        particle_data_id = f["particle_features"][:]
        mjj_id = f["mjj"][:]
    # print(jet_features_id.shape)
    # print(particle_data_id.shape)
    # print(mjj_id.shape)
    id_etaphipt = particle_data_id[..., [1, 2, 0]]
    w_dists = calculate_all_wasserstein_metrics(
        sorted_consts.reshape(-1, sorted_consts.shape[-2], sorted_consts.shape[-1]),
        id_etaphipt.reshape(-1, id_etaphipt.shape[-2], id_etaphipt.shape[-1]),
        num_eval_samples=50_000,
        num_batches=40,
        calculate_efps=True,
    )
    print(w_dists)

    # substructure
    # take only a subset of the data because calculating the substructure takes a long time
    n_substructure_events = 20000
    save_file_substr = join(
        data_folder,
        "lhco",
        "substructure",
        f"{save_file}-{params.ode_solver}-{params.ode_steps}_substr",
    )
    save_file_substr_id = join(
        data_folder,
        "lhco",
        "substructure",
        f"idealized_substr",
    )

    dump_hlvs(
        sorted_consts.reshape(-1, sorted_consts.shape[-2], sorted_consts.shape[-1])[
            :n_substructure_events
        ],
        save_file_substr,
        plot=False,
    )

    # calculate substructure for reference data
    dump_hlvs(
        id_etaphipt.reshape(-1, id_etaphipt.shape[-2], id_etaphipt.shape[-1])[
            :n_substructure_events
        ],
        save_file_substr_id,
        plot=False,
    )

    # load substructure for model generated data
    keys = []
    data_substructure = []
    with h5py.File(save_file_substr + ".h5", "r") as f:
        tau21 = np.array(f["tau21"])
        tau32 = np.array(f["tau32"])
        d2 = np.array(f["d2"])
        for key in f.keys():
            keys.append(key)
            data_substructure.append(np.array(f[key]))
    keys = np.array(keys)
    data_substructure = np.array(data_substructure)

    # load substructure for JetNet data
    data_substructure_jetnet = []
    with h5py.File(save_file_substr_id + ".h5", "r") as f:
        tau21_jetnet = np.array(f["tau21"])
        tau32_jetnet = np.array(f["tau32"])
        d2_jetnet = np.array(f["d2"])
        for key in f.keys():
            data_substructure_jetnet.append(np.array(f[key]))
    data_substructure_jetnet = np.array(data_substructure_jetnet)

    # calculate wasserstein distances
    w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
        tau21_jetnet, tau21, num_eval_samples=50_000, num_batches=40
    )
    w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
        tau32_jetnet, tau32, num_eval_samples=50_000, num_batches=40
    )
    w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(
        d2_jetnet, d2, num_eval_samples=50_000, num_batches=40
    )

    w_dist_tau21_mean_true, w_dist_tau21_std_true = wasserstein_distance_batched(
        tau21_jetnet, tau21_jetnet, num_eval_samples=50_000, num_batches=40
    )
    w_dist_tau32_mean_true, w_dist_tau32_std_true = wasserstein_distance_batched(
        tau32_jetnet, tau32_jetnet, num_eval_samples=50_000, num_batches=40
    )
    w_dist_d2_mean_true, w_dist_d2_std_true = wasserstein_distance_batched(
        d2_jetnet, d2_jetnet, num_eval_samples=50_000, num_batches=40
    )

    print(f"Wasserstein distance tau21: {w_dist_tau21_mean} +- {w_dist_tau21_std}")
    print(f"Wasserstein distance tau32: {w_dist_tau32_mean} +- {w_dist_tau32_std}")
    print(f"Wasserstein distance d2: {w_dist_d2_mean} +- {w_dist_d2_std}")
    print()
    print(f"Wasserstein distance tau21: {w_dist_tau21_mean_true} +- {w_dist_tau21_std_true}")
    print(f"Wasserstein distance tau32: {w_dist_tau32_mean_true} +- {w_dist_tau32_std_true}")
    print(f"Wasserstein distance d2: {w_dist_d2_mean_true} +- {w_dist_d2_std_true}")

    save_file_substr_img = join(
        data_folder,
        "lhco",
        "substructure",
    )
    file_name_1 = f"{save_file}-{params.ode_solver}-{params.ode_steps}_img"
    save_file_substr_img_full = join(
        data_folder,
        "lhco",
        "substructure",
    )
    file_name_2 = f"{save_file}-{params.ode_solver}-{params.ode_steps}_img_full"

    plot_substructure(
        tau21=tau21,
        tau32=tau32,
        d2=d2,
        tau21_jetnet=tau21_jetnet,
        tau32_jetnet=tau32_jetnet,
        d2_jetnet=d2_jetnet,
        save_fig=True,
        save_folder=save_file_substr_img,
        save_name=file_name_1,
        close_fig=True,
    )
    plot_full_substructure(
        data_substructure=data_substructure,
        data_substructure_jetnet=data_substructure_jetnet,
        keys=keys,
        save_fig=True,
        save_folder=save_file_substr_img_full,
        save_name=file_name_2,
        close_fig=True,
    )


if __name__ == "__main__":
    # ARGUMENTS
    # define parser
    parser = argparse.ArgumentParser(description="Generating Samples from a trained model.")

    parser.add_argument(
        "--folder_x",
        "-fx",
        default="/beegfs/desy/user/ewencedr/deep-learning/logs/lhco x jet hyperhigh ptsorted/runs/2023-09-03_22-49-22",
        help="folder of the x model to generate from",
        type=str,
    )

    parser.add_argument(
        "--folder_y",
        "-fy",
        default="/beegfs/desy/user/ewencedr/deep-learning/logs/lhco y jet hyperhigh ptsorted/runs/2023-09-03_22-49-22",
        help="folder of the y model to generate from",
        type=str,
    )

    parser.add_argument(
        "--save_file",
        "-s",
        default="lhco_both_jets_xy",
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
