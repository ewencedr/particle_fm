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
from src.data.components.metrics import wasserstein_distance_batched
from src.utils.data_generation import generate_data
from src.utils.jet_substructure import dump_hlvs
from src.utils.plotting import (
    apply_mpl_styles,
    plot_data,
    plot_full_substructure,
    plot_substructure,
    prepare_data_for_plotting,
)

apply_mpl_styles()

# set env variable DATA_DIR again because of hydra
from dotenv import load_dotenv

load_dotenv()
os.environ["DATA_DIR"] = os.environ.get("DATA_DIR")

data_folder = os.environ.get("DATA_DIR")


def string_to_bool(str: str) -> bool:
    if str == "True":
        return True
    else:
        return False


def main(params):
    folder = params.folder
    save_file = params.save_file

    use_signal_region = string_to_bool(params.signal_region)

    if use_signal_region:
        save_file += "_sr"
    else:
        save_file += "_sb"

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

    if params.conditioning_file == "data":
        print("Use data as conditioning data")
        if use_signal_region:
            cond_x = datamodule.jet_data_sr_raw[:, 0]
            mask_x = datamodule.mask_sr_raw[:, 0]
            cond_y = datamodule.jet_data_sr_raw[:, 1]
            mask_y = datamodule.mask_sr_raw[:, 1]
            mjj = datamodule.mjj_sr
        else:
            cond_x = datamodule.jet_data_raw[:, 0]
            mask_x = datamodule.mask_raw[:, 0]
            cond_y = datamodule.jet_data_raw[:, 1]
            mask_y = datamodule.mask_raw[:, 1]
            mjj = datamodule.mjj
    else:
        print(f"Use {params.conditioning_file} as conditioning data")
        with h5py.File(params.conditioning_file, "r") as f:
            cond_x = f["jet_features_x"][:]
            mask_x = f["mask_x"][:]
            cond_y = f["jet_features_y"][:]
            mask_y = f["mask_y"][:]
            mjj = f["mjj"][:]

            cond_x = cond_x[:, : datamodule.jet_data_raw[:, 0].shape[-1]]
            cond_y = cond_y[:, : datamodule.jet_data_raw[:, 1].shape[-1]]
            print(f"Using {datamodule.jet_data_raw[:, 1].shape[-1]} variables for conditioning")
            print(f"cond x shape: {cond_x.shape}")
            print(f"cond y shape: {cond_y.shape}")
            print(f"mask x shape: {mask_x.shape}")
            print(f"mask y shape: {mask_y.shape}")
            print(f"mjj shape: {mjj.shape}")

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
    torch.manual_seed(1111)
    data_x, generation_time_x = generate_data(
        model,
        num_jet_samples=len(mask_x),
        batch_size=1024,
        cond=torch.Tensor(normalized_cond_x),
        variable_set_sizes=datamodule.hparams.variable_jet_sizes,
        mask=torch.Tensor(mask_x),
        normalized_data=datamodule.hparams.normalize,
        normalize_sigma=datamodule.hparams.normalize_sigma,
        means=datamodule.means,
        stds=datamodule.stds,
        log_pt=datamodule.hparams.log_pt,
        pt_standardization=datamodule.hparams.pt_standardization,
        ode_solver=params.ode_solver,
        ode_steps=params.ode_steps,
    )

    print("Generating data second jet")
    data_y, generation_time_y = generate_data(
        model,
        num_jet_samples=len(mask_y),
        batch_size=1024,
        cond=torch.Tensor(normalized_cond_y),
        variable_set_sizes=datamodule.hparams.variable_jet_sizes,
        mask=torch.Tensor(mask_y),
        normalized_data=datamodule.hparams.normalize,
        normalize_sigma=datamodule.hparams.normalize_sigma,
        means=datamodule.means,
        stds=datamodule.stds,
        log_pt=datamodule.hparams.log_pt,
        pt_standardization=datamodule.hparams.pt_standardization,
        ode_solver=params.ode_solver,
        ode_steps=params.ode_steps,
    )

    data_x_raw = np.copy(data_x)
    data_y_raw = np.copy(data_y)
    data_raw = np.stack([data_x_raw, data_y_raw], axis=1)

    print("Preparing data for saving")
    # remove unphysical values
    # data_x[..., 0][data_x[..., 0] > np.max(datamodule.tensor_train.numpy()[..., 0])] = np.max(
    #    datamodule.tensor_train.numpy()[..., 0]
    # )
    # data_x[..., 1][data_x[..., 1] > np.max(datamodule.tensor_train.numpy()[..., 1])] = np.max(
    #    datamodule.tensor_train.numpy()[..., 1]
    # )
    data_x[..., 2][data_x[..., 2] > np.max(datamodule.tensor_train.numpy()[..., 2])] = np.max(
        datamodule.tensor_train.numpy()[..., 2][datamodule.tensor_train.numpy()[..., 2] != 1]
    )
    # data_x[..., 0][data_x[..., 0] < np.min(datamodule.tensor_train.numpy()[..., 0])] = np.min(
    #    datamodule.tensor_train.numpy()[..., 0]
    # )
    # data_x[..., 1][data_x[..., 1] < np.min(datamodule.tensor_train.numpy()[..., 1])] = np.min(
    #    datamodule.tensor_train.numpy()[..., 1]
    # )
    data_x[..., 2][data_x[..., 2] < np.min(datamodule.tensor_train.numpy()[..., 2])] = np.min(
        datamodule.tensor_train.numpy()[..., 2][datamodule.tensor_train.numpy()[..., 2] != 0]
    )

    # data_y[..., 0][data_y[..., 0] > np.max(datamodule.tensor_train.numpy()[..., 0])] = np.max(
    #    datamodule.tensor_train.numpy()[..., 0]
    # )
    # data_y[..., 1][data_y[..., 1] > np.max(datamodule.tensor_train.numpy()[..., 1])] = np.max(
    #    datamodule.tensor_train.numpy()[..., 1]
    # )
    data_y[..., 2][data_y[..., 2] > np.max(datamodule.tensor_train.numpy()[..., 2])] = np.max(
        datamodule.tensor_train.numpy()[..., 2][datamodule.tensor_train.numpy()[..., 2] != 1]
    )
    # data_y[..., 0][data_y[..., 0] < np.min(datamodule.tensor_train.numpy()[..., 0])] = np.min(
    #    datamodule.tensor_train.numpy()[..., 0]
    # )
    # data_y[..., 1][data_y[..., 1] < np.min(datamodule.tensor_train.numpy()[..., 1])] = np.min(
    #    datamodule.tensor_train.numpy()[..., 1]
    # )
    data_y[..., 2][data_y[..., 2] < np.min(datamodule.tensor_train.numpy()[..., 2])] = np.min(
        datamodule.tensor_train.numpy()[..., 2][datamodule.tensor_train.numpy()[..., 2] != 0]
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
    w_dists_np = calculate_all_wasserstein_metrics(
        data_raw.reshape(-1, data_raw.shape[-2], data_raw.shape[-1]),
        id_etaphipt.reshape(-1, id_etaphipt.shape[-2], id_etaphipt.shape[-1]),
        num_eval_samples=50_000,
        num_batches=40,
        calculate_efps=True,
    )
    w_dists_id = calculate_all_wasserstein_metrics(
        id_etaphipt.reshape(-1, id_etaphipt.shape[-2], id_etaphipt.shape[-1]),
        id_etaphipt.reshape(-1, id_etaphipt.shape[-2], id_etaphipt.shape[-1]),
        num_eval_samples=50_000,
        num_batches=40,
        calculate_efps=True,
    )
    print("Wasserstein distances with post-processing:")
    print(w_dists)
    print("Wasserstein distances without post-processing:")
    print(w_dists_np)
    print("Wasserstein distances for idealized data:")
    print(w_dists_id)

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
        "idealized_substr",
    )

    dump_hlvs(
        np.reshape(
            sorted_consts, (-1, sorted_consts.shape[-2], sorted_consts.shape[-1]), order="F"
        )[:n_substructure_events],
        save_file_substr,
        plot=False,
    )

    # calculate substructure for reference data
    dump_hlvs(
        np.reshape(id_etaphipt, (-1, id_etaphipt.shape[-2], id_etaphipt.shape[-1]), order="F")[
            :n_substructure_events
        ],
        save_file_substr_id,
        plot=False,
    )

    # load substructure for model generated data
    keys = []
    data_substructure = []
    with h5py.File(save_file_substr + ".h5", "r") as f:
        tau21 = np.nan_to_num(np.array(f["tau21"]))
        tau32 = np.nan_to_num(np.array(f["tau32"]))
        d2 = np.nan_to_num(np.array(f["d2"]))
        for key in f.keys():
            keys.append(key)
            data_substructure.append(np.array(f[key]))
    keys = np.array(keys)
    data_substructure = np.array(data_substructure)

    # load substructure for JetNet data
    data_substructure_jetnet = []
    with h5py.File(save_file_substr_id + ".h5", "r") as f:
        tau21_jetnet = np.nan_to_num(np.array(f["tau21"]))
        tau32_jetnet = np.nan_to_num(np.array(f["tau32"]))
        d2_jetnet = np.nan_to_num(np.array(f["d2"]))
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
        "--folder",
        "-f",
        default=(
            "/beegfs/desy/user/ewencedr/deep-learning/logs/lhco all jets/runs/2023-08-25_14-42-49"
        ),
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
    parser.add_argument(
        "--signal_region",
        "-sr",
        default="True",
        help="sample in signal region",
    )

    parser.add_argument(
        "--conditioning_file",
        "-cond",
        default="data",
        help="file containing the conditioning data",
    )

    params = parser.parse_args()
    main(params)
