"""Script to evaluate a checkpoint and generate plots and metrics.

Usage: python particle_fm/eval_ckpt.py --ckpt <path_to_ckpt> --n_samples <int>
"""

import argparse
import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path

import awkward as ak
import h5py
import hydra
import numpy as np

# plots and metrics
import pandas as pd
import torch
import vector
import yaml

# set env variable DATA_DIR again because of hydra
from dotenv import load_dotenv
from jetnet.evaluation import w1p

# set env variable DATA_DIR again because of hydra
from omegaconf import OmegaConf

from particle_fm.data.components import calculate_all_wasserstein_metrics
from particle_fm.data.components.metrics import (
    reversed_kl_divergence_batched,
    wasserstein_distance_batched,
)

# from particle_fm.data.components.utils import calculate_jet_features
from particle_fm.utils.data_generation import generate_data
from particle_fm.utils.jet_substructure import calc_substructure  # , dump_hlvs
from particle_fm.utils.plotting import (  # create_and_plot_data,; plot_single_jets,; plot_data,
    apply_mpl_styles,
    plot_full_substructure,
    plot_jet_features,
    plot_particle_features,
    plot_substructure,
    prepare_data_for_plotting,
)

vector.register_awkward()


def reversed_kl_divergence_batched_different_variations(target, approx, **kwargs):
    """Calculate the reversed KL divergence between two datasets multiple times and return mean and
    std. Four different variations are calculated:

    - with clipping
    - without clipping
    - with scaling
    - without scaling

    Args:
        target (np.array): Target distribution
        approx (np.array): Approximation of the target distribution
        kwargs: Keyword arguments for the reversed_kl_divergence_batched function

    Returns:
        dict: Dictionary with mean and std for each variation (i.e. {"clipped":
        {"mean": ..., "std": ...}, ...})
    """
    # calculate 4 times: with/without clipping + with/without scaling
    # calculate reversed KL divergence
    suffixes = {
        "clipped": {"clip_approx": True, "rescale_pq": False},
        "notclipped_scaled": {"clip_approx": False, "rescale_pq": True},
        "clipped_scaled": {"clip_approx": True, "rescale_pq": True},
        "notclipped_notscaled": {"clip_approx": False, "rescale_pq": False},
    }

    results = {}
    for suffix, clipscale_kwargs in suffixes.items():
        kl_mean, kl_std = reversed_kl_divergence_batched(
            target=target,
            approx=approx,
            **kwargs,
            **clipscale_kwargs,
        )
        results[suffix] = {"mean": kl_mean, "std": kl_std}
    return results


# set up logging for jupyter notebook
pylogger = logging.getLogger("eval_ckpt")
logging.basicConfig(level=logging.INFO)

load_dotenv()
os.environ["DATA_DIR"] = os.environ.get("DATA_DIR")

apply_mpl_styles()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=100_000,
)
parser.add_argument(
    "--cond_gen_file",
    type=str,
    default=None,
)
parser.add_argument(
    "--suffix",
    type=str,
    default=None,
)
parser.add_argument(
    "--ode_steps",
    type=int,
    default=100,
)
parser.add_argument(
    "--used_jet_types",
    nargs="+",
    type=str,
    help=(
        "List of jet types to use for evaluation. If not specified, the jet types from the"
        " training are used."
    ),
    default=None,
)

VARIABLES_TO_CLIP = ["part_ptrel", "part_energyrel"]
W_DIST_CFG = {"num_eval_samples": 50_000, "num_batches": 10}


def main():
    args = parser.parse_args()
    ckpt = args.ckpt
    n_samples_gen = args.n_samples
    suffix = "" if args.suffix is None or args.suffix == "" else f"-{args.suffix}"

    pylogger.info(f"ckpt: {ckpt}")
    pylogger.info(f"n_samples: {n_samples_gen}")

    EVALUATE_SUBSTRUCTURE = True

    ckpt_path = Path(ckpt)
    run_dir = (
        ckpt_path.parent.parent.parent
        if "evaluated_ckpts" in str(ckpt_path)
        else ckpt_path.parent.parent
    )
    cfg_backup_file = f"{run_dir}/config.yaml"

    # load everything from run directory (safer in terms of reproducing results)
    cfg = OmegaConf.load(cfg_backup_file)
    print(type(cfg))
    print(OmegaConf.to_yaml(cfg))
    print(100 * "-")

    if args.cond_gen_file != "use_truth_cond":
        cfg.data.conditioning_gen_filename = args.cond_gen_file

    if args.used_jet_types is not None:
        cfg.data.used_jet_types = args.used_jet_types

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    if args.cond_gen_file == "use_truth_cond":
        pylogger.info("Using truth conditioning for generation.")
        pylogger.info(
            "--> setting mask_gen and tensor_conditioning_gen to mask_test and"
            " tensor_conditioning_test"
        )
        datamodule.mask_gen = datamodule.mask_test
        datamodule.tensor_conditioning_gen = datamodule.tensor_conditioning_test

    # load the model from the checkpoint
    model = hydra.utils.instantiate(cfg.model)
    model = model.load_from_checkpoint(ckpt)
    # model.to("cpu")

    # ------------------------------------------------
    data_sim = np.array(datamodule.tensor_test)
    mask_sim = np.array(datamodule.mask_test)
    cond_sim = np.array(datamodule.tensor_conditioning_test)

    n_samples_sim = n_samples_gen
    n_samples_gen = n_samples_gen

    if len(data_sim) < n_samples_sim:
        n_samples_sim = len(data_sim)
        pylogger.info(f"Only {len(data_sim)} samples available, using {n_samples_sim} samples.")
    else:
        data_sim = data_sim[:n_samples_sim]
        mask_sim = mask_sim[:n_samples_sim]
        cond_sim = cond_sim[:n_samples_sim]

    if args.cond_gen_file is not None and args.cond_gen_file != "use_truth_cond":
        pylogger.info(f"Using conditioning from file {args.cond_gen_file} for generation.")
        mask_gen = np.array(datamodule.mask_gen)
        cond_gen = np.array(datamodule.tensor_conditioning_gen)
        if len(cond_gen) < n_samples_gen:
            n_samples_gen = len(cond_gen)
            pylogger.info(
                f"Only {len(cond_gen)} generated masks available, using {n_samples_gen} samples."
            )
        mask_gen = mask_gen[:n_samples_gen]
        cond_gen = cond_gen[:n_samples_gen]

    else:
        pylogger.warning("Using conditioning from simulation for generation.")
        mask_gen = deepcopy(mask_sim)
        cond_gen = deepcopy(cond_sim)

    # check if the output already exists
    # --> only generate new data if it does not exist yet
    checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
    ckpt_epoch = checkpoint["epoch"]
    pylogger.info(f"Loaded checkpoint from epoch {ckpt_epoch}")

    ckpt_path = Path(ckpt)
    output_dir = (
        ckpt_path.parent
        if f"evaluated_ckpts/epoch_{ckpt_epoch}" in str(ckpt_path)
        else ckpt_path.parent.parent / "evaluated_ckpts" / f"epoch_{ckpt_epoch}"
    )
    plots_dir = output_dir / f"plots_{n_samples_gen}{suffix}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pylogger.info(f"Output directory: {output_dir}")
    pylogger.info(f"Plots directory: {plots_dir}")

    os.makedirs(output_dir, exist_ok=True)
    if not (output_dir / f"epoch_{ckpt_epoch}.ckpt").exists():
        pylogger.info(f"Copy checkpoint file to {output_dir}")
        shutil.copyfile(ckpt, output_dir / f"epoch_{ckpt_epoch}.ckpt")

    h5data_output_path = (
        output_dir / f"eval_output_epoch_{ckpt_epoch}_nsamples_{n_samples_gen}{suffix}.h5"
    )

    if h5data_output_path.exists():
        pylogger.info(f"h5 output file {h5data_output_path} already exists.")
        pylogger.info("--> Using existing file.")
    else:
        pylogger.info(f"Output file {h5data_output_path} doesn't exist.")
        pylogger.info("--> Generating data.")
        # set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        data_gen, generation_time = generate_data(
            model=model,
            num_jet_samples=len(mask_gen),
            cond=torch.tensor(cond_gen),
            variable_set_sizes=datamodule.hparams.variable_jet_sizes,
            mask=torch.tensor(mask_gen),
            normalized_data=datamodule.hparams.normalize,
            means=datamodule.means,
            stds=datamodule.stds,
            # device="cpu",
            ode_steps=args.ode_steps,
        )
        pylogger.info(f"Generated {len(data_gen)} samples in {generation_time:.0f} seconds.")

        # ------------------------------------------------
        # POSTPROCESSING
        # ------------------------------------------------
        # Correcting the generated data
        # Clipping
        for i, var_name in enumerate(datamodule.names_particle_features):
            if var_name not in VARIABLES_TO_CLIP:
                continue
            clip_min = datamodule.min_max_train_dict[var_name]["min"]
            clip_max = datamodule.min_max_train_dict[var_name]["max"]
            pylogger.info(
                f"Clipping outliers of generated {var_name} to original range: "
                f"[{clip_min}, {clip_max}]"
            )
            data_gen[mask_gen[..., 0] != 0, i] = np.clip(
                data_gen[mask_gen[..., 0] != 0, i],
                a_min=clip_min,
                a_max=clip_max,
            )
        # Argmax for particle-id features:
        # for the particle-id features, set the maximum value to 1 and the
        # others to 0 (a particle can only be one type, i.e. if isElectron=0,
        # isMuon=0, ...)

        names_part_features = list(datamodule.names_particle_features)
        indices_is_features = [
            names_part_features.index(name)
            for name in names_part_features
            if name.startswith("part_is")
        ]

        if len(indices_is_features) > 0:
            pylogger.info("Setting maximum value of particle-id features to 1, others to 0.")
            part_id_features = data_gen[:, :, indices_is_features]
            # find the index of the maximum value in the last axis
            max_indices = np.argmax(part_id_features, axis=-1)

            result = np.zeros_like(part_id_features)
            result[
                np.arange(part_id_features.shape[0])[:, None],
                np.arange(part_id_features.shape[1]),
                max_indices,
            ] = 1

            data_gen[:, :, indices_is_features] = result
            # correct masked values
            data_gen[mask_gen[..., 0] == 0, :] = 0

        # Round part_charge to nearest integer
        if "part_charge" in names_part_features:
            pylogger.info("Rounding part_charge to nearest integer.")
            idx_charge = names_part_features.index("part_charge")
            data_gen[:, :, idx_charge] = np.round(data_gen[:, :, idx_charge])

        # Remove jets with less than 3 particles (since we want to recluster
        # them later)
        more_than_3_particles_gen = np.sum(mask_gen[:, :, 0], axis=1) >= 3
        more_than_3_particles_sim = np.sum(mask_sim[:, :, 0], axis=1) >= 3

        data_gen = data_gen[more_than_3_particles_gen]
        data_sim = data_sim[more_than_3_particles_sim]
        mask_gen = mask_gen[more_than_3_particles_gen]
        mask_sim = mask_sim[more_than_3_particles_sim]
        cond_gen = cond_gen[more_than_3_particles_gen]
        cond_sim = cond_sim[more_than_3_particles_sim]

        # # ------------------------------------------------

        print(f"Saving data to {h5data_output_path}")

        # for jetnet compatibility
        if not hasattr(datamodule, "names_particle_features"):
            datamodule.names_particle_features = ["part_etarel", "part_dphi", "part_ptrel"]

        # Save all the data to an HDF file
        with h5py.File(h5data_output_path, mode="w") as h5file:
            # particle data
            h5file.create_dataset("part_data_sim", data=data_sim)
            h5file.create_dataset("part_data_gen", data=data_gen)
            h5file.create_dataset("part_mask_sim", data=mask_sim)
            h5file.create_dataset("part_mask_gen", data=mask_gen)
            for ds_key in ["part_data_sim", "part_data_gen"]:
                h5file[ds_key].attrs.create(
                    "names",
                    data=datamodule.names_particle_features,
                    dtype=h5py.special_dtype(vlen=str),
                )
            # jet data
            h5file.create_dataset("cond_data_sim", data=cond_sim)
            h5file.create_dataset("cond_data_gen", data=cond_gen)
            for ds_key in ["cond_data_sim", "cond_data_gen"]:
                if hasattr(datamodule, "names_conditioning"):
                    if datamodule.names_conditioning is not None:
                        h5file[ds_key].attrs.create(
                            "names",
                            data=datamodule.names_conditioning,
                            dtype=h5py.special_dtype(vlen=str),
                        )

    # read the file
    with h5py.File(h5data_output_path) as h5file:
        data_gen = h5file["part_data_gen"][:]
        mask_gen = h5file["part_mask_gen"][:]
        cond_gen = h5file["cond_data_gen"][:]
        data_sim = h5file["part_data_sim"][:]
        mask_sim = h5file["part_mask_sim"][:]
        cond_sim = h5file["cond_data_sim"][:]

        # jet_data_gen = h5file["jet_data_gen"][:]
        # jet_data_sim = h5file["jet_data_sim"][:]
        # pt_selected_particles_gen = h5file["pt_selected_particles_gen"][:]
        # pt_selected_particles_sim = h5file["pt_selected_particles_sim"][:]

        part_names_sim = h5file["part_data_sim"].attrs["names"][:]
        names_cond_features = list(datamodule.names_conditioning)
        names_part_features = list(datamodule.names_particle_features)

    # create awkward arrays and calculate the jet substructure
    idx_jet_pt = names_cond_features.index("jet_pt")
    idx_part_ptrel = names_part_features.index("part_ptrel")
    pt_gen = data_gen[:, :, idx_part_ptrel] * cond_gen[:, idx_jet_pt][:, None]
    pt_sim = data_sim[:, :, idx_part_ptrel] * cond_sim[:, idx_jet_pt][:, None]

    idx_jet_eta = names_cond_features.index("jet_eta")
    idx_part_etarel = names_part_features.index("part_etarel")
    eta_gen = data_gen[:, :, idx_part_etarel] + cond_gen[:, idx_jet_eta][:, None]
    eta_sim = data_sim[:, :, idx_part_etarel] + cond_sim[:, idx_jet_eta][:, None]

    idx_part_dphi = names_part_features.index("part_dphi")
    dphi_gen = data_gen[:, :, idx_part_dphi]
    dphi_sim = data_sim[:, :, idx_part_dphi]

    # create awkward arrays
    particles_gen = ak.zip(
        {
            "pt": pt_gen,
            "eta": eta_gen,
            "phi": dphi_gen,
            "mass": np.zeros_like(pt_gen),
        },
        with_name="Momentum4D",
    )
    particles_sim = ak.zip(
        {
            "pt": pt_sim,
            "eta": eta_sim,
            "phi": dphi_sim,
            "mass": np.zeros_like(pt_sim),
        },
        with_name="Momentum4D",
    )
    # remove zero-padded entries
    particles_gen_mask = ak.mask(particles_gen, particles_gen.pt > 0)
    particles_sim_mask = ak.mask(particles_sim, particles_sim.pt > 0)
    particles_gen = ak.drop_none(particles_gen_mask)
    particles_sim = ak.drop_none(particles_sim_mask)

    h5data_output_path_subs = str(h5data_output_path).replace(".h5", "_substructure.h5")
    # calculate the substrucutre
    calc_substructure(
        particles_sim=particles_sim,
        particles_gen=particles_gen,
        R=0.8,
        filename=h5data_output_path_subs,
    )

    # calculate substructure (old code)
    # substructure_path = output_dir
    # substr_filename_gen = (
    #     f"substructure_generated_epoch_{ckpt_epoch}_nsamples_{n_samples_gen}{suffix}"
    # )
    # print(f"Saving substructure to {substructure_path / substr_filename_gen}")
    # substructure_full_path = substructure_path / substr_filename_gen
    # substr_filename_jetclass = (
    #     f"substructure_simulated_epoch_{ckpt_epoch}_nsamples_{n_samples_gen}{suffix}"
    # )
    # substructure_full_path_jetclass = substructure_path / substr_filename_jetclass

    # # calculate substructure for generated data
    # if not os.path.isfile(str(substructure_full_path) + ".h5"):
    #     pylogger.info("Calculating substructure.")
    #     dump_hlvs(data_gen, str(substructure_full_path), plot=False)
    # # calculate substructure for reference data
    # if not os.path.isfile(str(substructure_full_path_jetclass) + ".h5"):
    #     pylogger.info("Calculating substructure.")
    #     dump_hlvs(data_sim, str(substructure_full_path_jetclass), plot=False)

    # load substructure for model generated data
    keys = ["tau1", "tau2", "tau3", "tau21", "tau32", "d2", "jet_mass", "jet_pt"]
    data_substructure = []
    with h5py.File(h5data_output_path_subs) as f:
        tau21 = np.array(f["tau21_gen"])
        tau32 = np.array(f["tau32_gen"])
        d2 = np.array(f["d2_gen"])
        jet_mass = np.array(f["jet_mass_gen"])
        jet_pt = np.array(f["jet_pt_gen"])
        tau21_isnan = np.isnan(tau21)
        tau32_isnan = np.isnan(tau32)
        d2_isnan = np.isnan(d2)
        if np.sum(tau21_isnan) > 0 or np.sum(tau32_isnan) > 0 or np.sum(d2_isnan) > 0:
            pylogger.warning(f"Found {np.sum(tau21_isnan)} nan values in tau21")
            pylogger.warning(f"Found {np.sum(tau32_isnan)} nan values in tau32")
            pylogger.warning(f"Found {np.sum(d2_isnan)} nan values in d2")
            pylogger.warning("Setting nan values to zero.")
        tau21[tau21_isnan] = 0
        tau32[tau32_isnan] = 0
        d2[d2_isnan] = 0
        for key in keys:
            data_substructure.append(np.array(f[key + "_gen"]))
    data_substructure = np.array(data_substructure)

    # load substructure for JetClass data
    data_substructure_jetclass = []
    with h5py.File(h5data_output_path_subs) as f:
        tau21_jetclass = np.array(f["tau21_sim"])
        tau32_jetclass = np.array(f["tau32_sim"])
        d2_jetclass = np.array(f["d2_sim"])
        jet_mass_jetclass = np.array(f["jet_mass_sim"])
        jet_pt_jetclass = np.array(f["jet_pt_sim"])
        tau21_jetclass_isnan = np.isnan(tau21_jetclass)
        tau32_jetclass_isnan = np.isnan(tau32_jetclass)
        d2_jetclass_isnan = np.isnan(d2_jetclass)
        if (
            np.sum(tau21_jetclass_isnan) > 0
            or np.sum(tau32_jetclass_isnan) > 0
            or np.sum(d2_jetclass_isnan) > 0
        ):
            pylogger.warning(f"Found {np.sum(tau21_jetclass_isnan)} nan values in tau21")
            pylogger.warning(f"Found {np.sum(tau32_jetclass_isnan)} nan values in tau32")
            pylogger.warning(f"Found {np.sum(d2_jetclass_isnan)} nan values in d2")
            pylogger.warning("Setting nan values to zero.")
        tau21_jetclass[tau21_jetclass_isnan] = 0
        tau32_jetclass[tau32_jetclass_isnan] = 0
        d2_jetclass[d2_jetclass_isnan] = 0
        for key in keys:
            data_substructure_jetclass.append(np.array(f[key + "_sim"]))
    data_substructure_jetclass = np.array(data_substructure_jetclass)

    # -----------------------------------------------
    # ----- calculate metrics and plot features -----
    metrics = {}
    # save the filename used for jet-conditioning for the generated jets
    metrics["cond_gen_file"] = cfg.data.conditioning_gen_filename

    pylogger.info("Calculating metrics")
    pylogger.info("Calculating Wasserstein distances of jet substructure")
    # calculate wasserstein distances
    w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
        tau21_jetclass, tau21, **W_DIST_CFG
    )
    w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
        tau32_jetclass, tau32, **W_DIST_CFG
    )
    w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(d2_jetclass, d2, **W_DIST_CFG)
    w_dist_jetmass_mean, w_dist_jetmass_std = wasserstein_distance_batched(
        jet_mass_jetclass, jet_mass, **W_DIST_CFG
    )
    w_dist_jetpt_mean, w_dist_jetpt_std = wasserstein_distance_batched(
        jet_pt_jetclass, jet_pt, **W_DIST_CFG
    )

    # add to metrics
    metrics["w_dist_tau21_mean"] = w_dist_tau21_mean
    metrics["w_dist_tau21_std"] = w_dist_tau21_std
    metrics["w_dist_tau32_mean"] = w_dist_tau32_mean
    metrics["w_dist_tau32_std"] = w_dist_tau32_std
    metrics["w_dist_d2_mean"] = w_dist_d2_mean
    metrics["w_dist_d2_std"] = w_dist_d2_std
    metrics["w_dist_jetmass_mean"] = w_dist_jetmass_mean
    metrics["w_dist_jetmass_std"] = w_dist_jetmass_std
    metrics["w_dist_jetpt_mean"] = w_dist_jetpt_mean
    metrics["w_dist_jetpt_std"] = w_dist_jetpt_std

    pylogger.info("Calculating KLD of jet substructure")
    # calculate reversed KL divergence
    kl_tau21_results = reversed_kl_divergence_batched_different_variations(
        approx=tau21,
        target=tau21_jetclass,
        num_batches=10,
    )
    kl_tau32_results = reversed_kl_divergence_batched_different_variations(
        approx=tau32,
        target=tau32_jetclass,
        num_batches=10,
    )
    kl_d2_results = reversed_kl_divergence_batched_different_variations(
        approx=d2,
        target=d2_jetclass,
        num_batches=10,
    )
    kl_jetmass_results = reversed_kl_divergence_batched_different_variations(
        approx=jet_mass,
        target=jet_mass_jetclass,
        num_batches=10,
    )
    kl_jetpt_results = reversed_kl_divergence_batched_different_variations(
        approx=jet_pt,
        target=jet_pt_jetclass,
        num_batches=10,
    )
    for kl_suffix, kl_variant in kl_tau21_results.items():
        metrics[f"kl_tau21_{kl_suffix}_mean"] = kl_variant["mean"]
        metrics[f"kl_tau21_{kl_suffix}_std"] = kl_variant["std"]
    for kl_suffix, kl_variant in kl_tau32_results.items():
        metrics[f"kl_tau32_{kl_suffix}_mean"] = kl_variant["mean"]
        metrics[f"kl_tau32_{kl_suffix}_std"] = kl_variant["std"]
    for kl_suffix, kl_variant in kl_d2_results.items():
        metrics[f"kl_d2_{kl_suffix}_mean"] = kl_variant["mean"]
        metrics[f"kl_d2_{kl_suffix}_std"] = kl_variant["std"]
    for kl_suffix, kl_variant in kl_jetmass_results.items():
        metrics[f"kl_jetmass_{kl_suffix}_mean"] = kl_variant["mean"]
        metrics[f"kl_jetmass_{kl_suffix}_std"] = kl_variant["std"]
    for kl_suffix, kl_variant in kl_jetpt_results.items():
        metrics[f"kl_jetpt_{kl_suffix}_mean"] = kl_variant["mean"]
        metrics[f"kl_jetpt_{kl_suffix}_std"] = kl_variant["std"]

    # plot substructure
    file_name_substructure = "substructure_3plots"
    file_name_full_substructure = "substructure_full"
    img_path = str(plots_dir) + "/"
    plot_substructure(
        tau21=tau21,
        tau32=tau32,
        d2=d2,
        tau21_jetnet=tau21_jetclass,
        tau32_jetnet=tau32_jetclass,
        d2_jetnet=d2_jetclass,
        save_fig=True,
        save_folder=img_path,
        save_name=file_name_substructure,
        close_fig=True,
        simulation_name="JetClass",
        model_name="Generated",
    )
    plot_full_substructure(
        data_substructure=data_substructure,
        data_substructure_jetnet=data_substructure_jetclass,
        keys=keys,
        save_fig=True,
        save_folder=img_path,
        save_name=file_name_full_substructure,
        close_fig=True,
        simulation_name="JetClass",
        model_name="Generated",
    )

    # If there are multiple jet types, plot them separately
    jet_types_dict = {
        var_name.split("_")[-1]: i
        for i, var_name in enumerate(datamodule.names_conditioning)
        if "jet_type" in var_name
    }
    jet_types_dict["all_jet_types"] = None
    pylogger.info(f"List of jet types: {jet_types_dict.keys()}")

    for jet_type, jet_type_idx in jet_types_dict.items():
        pylogger.info(f"Plotting substructure for jet type {jet_type}")
        if jet_type == "all_jet_types":
            jet_type_mask_sim = np.ones(len(cond_sim), dtype=bool)
            jet_type_mask_gen = np.ones(len(cond_gen), dtype=bool)
        else:
            jet_type_mask_sim = cond_sim[:, jet_type_idx] == 1
            jet_type_mask_gen = cond_gen[:, jet_type_idx] == 1

        if np.sum(jet_type_mask_sim) == 0 or np.sum(jet_type_mask_gen) == 0:
            pylogger.warning(f"No samples for jet type {jet_type} found -> continue.")
            continue

        pylogger.info("Calculating Wasserstein distances of jet substructure")
        w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
            tau21_jetclass[jet_type_mask_sim], tau21[jet_type_mask_gen], **W_DIST_CFG
        )
        w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
            tau32_jetclass[jet_type_mask_sim], tau32[jet_type_mask_gen], **W_DIST_CFG
        )
        w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(
            d2_jetclass[jet_type_mask_sim], d2[jet_type_mask_gen], **W_DIST_CFG
        )
        w_dist_jetmass_mean, w_dist_jetmass_std = wasserstein_distance_batched(
            jet_mass_jetclass[jet_type_mask_sim], jet_mass[jet_type_mask_gen], **W_DIST_CFG
        )
        w_dist_jetpt_mean, w_dist_jetpt_std = wasserstein_distance_batched(
            jet_pt_jetclass[jet_type_mask_sim], jet_pt[jet_type_mask_gen], **W_DIST_CFG
        )
        # add to metrics
        metrics[f"w_dist_tau21_mean_{jet_type}"] = w_dist_tau21_mean
        metrics[f"w_dist_tau21_std_{jet_type}"] = w_dist_tau21_std
        metrics[f"w_dist_tau32_mean_{jet_type}"] = w_dist_tau32_mean
        metrics[f"w_dist_tau32_std_{jet_type}"] = w_dist_tau32_std
        metrics[f"w_dist_d2_mean_{jet_type}"] = w_dist_d2_mean
        metrics[f"w_dist_d2_std_{jet_type}"] = w_dist_d2_std
        metrics[f"w_dist_jetmass_mean_{jet_type}"] = w_dist_jetmass_mean
        metrics[f"w_dist_jetmass_std_{jet_type}"] = w_dist_jetmass_std
        metrics[f"w_dist_jetpt_mean_{jet_type}"] = w_dist_jetpt_mean
        metrics[f"w_dist_jetpt_std_{jet_type}"] = w_dist_jetpt_std

        pylogger.info("Calculating KLD of jet substructure")

        kl_tau21_results = reversed_kl_divergence_batched_different_variations(
            approx=tau21[jet_type_mask_gen],
            target=tau21_jetclass[jet_type_mask_sim],
            num_batches=10,
        )
        kl_tau32_results = reversed_kl_divergence_batched_different_variations(
            approx=tau32[jet_type_mask_gen],
            target=tau32_jetclass[jet_type_mask_sim],
            num_batches=10,
        )
        kl_d2_results = reversed_kl_divergence_batched_different_variations(
            approx=d2[jet_type_mask_gen],
            target=d2_jetclass[jet_type_mask_sim],
            num_batches=10,
        )
        kl_jetmass_results = reversed_kl_divergence_batched_different_variations(
            approx=jet_mass[jet_type_mask_gen],
            target=jet_mass_jetclass[jet_type_mask_sim],
            num_batches=10,
        )
        kl_jetpt_results = reversed_kl_divergence_batched_different_variations(
            approx=jet_pt[jet_type_mask_gen],
            target=jet_pt_jetclass[jet_type_mask_sim],
            num_batches=10,
        )
        for kl_suffix, kl_variant in kl_tau21_results.items():
            metrics[f"kl_tau21_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_tau21_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]
        for kl_suffix, kl_variant in kl_tau32_results.items():
            metrics[f"kl_tau32_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_tau32_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]
        for kl_suffix, kl_variant in kl_d2_results.items():
            metrics[f"kl_d2_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_d2_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]
        for kl_suffix, kl_variant in kl_jetmass_results.items():
            metrics[f"kl_jetmass_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_jetmass_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]
        for kl_suffix, kl_variant in kl_jetpt_results.items():
            metrics[f"kl_jetpt_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_jetpt_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]

        plot_substructure(
            tau21=tau21[jet_type_mask_gen],
            tau32=tau32[jet_type_mask_gen],
            d2=d2[jet_type_mask_gen],
            tau21_jetnet=tau21_jetclass[jet_type_mask_sim],
            tau32_jetnet=tau32_jetclass[jet_type_mask_sim],
            d2_jetnet=d2_jetclass[jet_type_mask_sim],
            save_fig=True,
            save_folder=img_path,
            save_name=file_name_substructure + "_" + jet_type,
            close_fig=True,
            simulation_name="JetClass",
            model_name="Generated",
        )
        plot_full_substructure(
            data_substructure=[
                data_substructure[i][jet_type_mask_gen] for i in range(len(data_substructure))
            ],
            data_substructure_jetnet=[
                data_substructure_jetclass[i][jet_type_mask_sim]
                for i in range(len(data_substructure_jetclass))
            ],
            keys=keys,
            save_fig=True,
            save_folder=img_path,
            save_name=file_name_full_substructure + "_" + jet_type,
            close_fig=True,
            simulation_name="JetClass",
            model_name="Generated",
        )

    pylogger.info("Plotting particle features")
    plot_particle_features(
        data_gen=data_gen,
        data_sim=data_sim,
        mask_gen=mask_gen,
        mask_sim=mask_sim,
        feature_names=datamodule.names_particle_features,
        legend_label_sim="JetClass",
        legend_label_gen="Generated",
        plot_path=plots_dir / f"epoch_{ckpt_epoch}_particle_features.pdf",
    )

    # for jetnet compatibility
    if not hasattr(datamodule, "names_conditioning"):
        if len(cond_sim.shape) == 1:
            datamodule.names_conditioning = []
        else:
            datamodule.names_conditioning = [f"cond_var_{i}" for i in range(cond_sim.shape[1])]
    else:
        if datamodule.names_conditioning is None:
            datamodule.names_conditioning = []

    # remove this for now, since calculation of EFPs takes ages...
    # pylogger.info("Calculating Wasserstein distances for all jet types")
    # metrics.update(calculate_all_wasserstein_metrics(data_sim, data_gen, **W_DIST_CFG))

    for jet_type, jet_type_idx in jet_types_dict.items():
        pylogger.info(f"Plotting jet type {jet_type}")
        if jet_type == "all_jet_types":
            jet_type_mask_sim = np.ones(len(cond_sim), dtype=bool)
            jet_type_mask_gen = np.ones(len(cond_gen), dtype=bool)
        else:
            jet_type_mask_sim = cond_sim[:, jet_type_idx] == 1
            jet_type_mask_gen = cond_gen[:, jet_type_idx] == 1

        print(f"Sum of jet_type_mask_sim: {np.sum(jet_type_mask_sim)}")
        print(f"Sum of jet_type_mask_gen: {np.sum(jet_type_mask_gen)}")

        metrics[f"n_samples_sim_{jet_type}"] = np.sum(jet_type_mask_sim)
        metrics[f"n_samples_gen_{jet_type}"] = np.sum(jet_type_mask_gen)

        if np.sum(jet_type_mask_sim) == 0 or np.sum(jet_type_mask_gen) == 0:
            pylogger.warning(f"No samples for jet type {jet_type} found -> continue.")
            continue

        # calculate metrics and add to dict
        # metrics_this_type = calculate_all_wasserstein_metrics(
        #     data_sim[jet_type_mask_sim], data_gen[jet_type_mask_gen], **W_DIST_CFG
        # )
        # for key, value in metrics_this_type.items():
        #     metrics[f"{key}_{jet_type}"] = value

        # select the particle features for this jet type
        data_sim_this_type = data_sim[jet_type_mask_sim]
        data_gen_this_type = data_gen[jet_type_mask_gen]
        mask_sim_this_type = mask_sim[jet_type_mask_sim]
        mask_gen_this_type = mask_gen[jet_type_mask_gen]

        plot_particle_features(
            data_gen=data_gen[jet_type_mask_gen],
            data_sim=data_sim[jet_type_mask_sim],
            mask_gen=mask_gen[jet_type_mask_gen],
            mask_sim=mask_sim[jet_type_mask_sim],
            feature_names=datamodule.names_particle_features,
            legend_label_sim="JetClass",
            legend_label_gen="Generated",
            plot_path=plots_dir / f"epoch_{ckpt_epoch}_particle_features_{jet_type}.pdf",
        )

        pylogger.info("Calculating KL divergence for each particle feature")

        for i, part_feature_name in enumerate(part_names_sim):
            particle_feat_kld = reversed_kl_divergence_batched_different_variations(
                target=data_sim_this_type[:, :, i],
                approx=data_gen_this_type[:, :, i],
                mask_target=mask_sim_this_type[..., 0] == 1,
                mask_approx=mask_gen_this_type[..., 0] == 1,
                num_batches=10,
            )
            for kl_suffix, kl_variant in particle_feat_kld.items():
                metrics[f"kl_{part_feature_name}_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
                metrics[f"kl_{part_feature_name}_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]

        # calculate IP significance of charged particles
        def idx_part(name):
            return list(part_names_sim).index(name)

        if (
            "part_d0val" not in part_names_sim
            or "part_d0err" not in part_names_sim
            or "part_dzval" not in part_names_sim
            or "part_dzerr" not in part_names_sim
            or "part_isPhoton" not in part_names_sim
            or "part_isNeutralHadron" not in part_names_sim
        ):
            pylogger.warning(
                "Not calculating IP significance, since necessary features are missing."
            )
            pylogger.warning("--> skipping")
            continue

        # fmt: off
        is_charged_sim = np.logical_and(
            (data_sim_this_type[:, :, idx_part("part_isPhoton")]) == 0,
            (data_sim_this_type[:, :, idx_part("part_isNeutralHadron")]) == 0,
        )
        is_charged_gen = np.logical_and(
            (data_gen_this_type[:, :, idx_part("part_isPhoton")]) == 0,
            (data_gen_this_type[:, :, idx_part("part_isNeutralHadron")]) == 0,
        )
        is_charged_and_nonmasked_sim = np.logical_and(is_charged_sim, mask_sim_this_type[..., 0])
        is_charged_and_nonmasked_gen = np.logical_and(is_charged_gen, mask_gen_this_type[..., 0])

        sd0_sim = data_sim_this_type[:, :, idx_part("part_d0val")] / data_sim_this_type[:, :, idx_part("part_d0err")]
        sd0_gen = data_gen_this_type[:, :, idx_part("part_d0val")] / data_gen_this_type[:, :, idx_part("part_d0err")]
        sdz_sim = data_sim_this_type[:, :, idx_part("part_dzval")] / data_sim_this_type[:, :, idx_part("part_dzerr")]
        sdz_gen = data_gen_this_type[:, :, idx_part("part_dzval")] / data_gen_this_type[:, :, idx_part("part_dzerr")]
        # fmt: on

        sd0_charged_kld = reversed_kl_divergence_batched_different_variations(
            target=sd0_sim,
            approx=sd0_gen,
            mask_target=is_charged_and_nonmasked_sim,
            mask_approx=is_charged_and_nonmasked_gen,
            num_batches=W_DIST_CFG["num_batches"],
        )
        sd0_kld = reversed_kl_divergence_batched_different_variations(
            target=sd0_sim,
            approx=sd0_gen,
            mask_target=mask_sim_this_type[..., 0] == 1,
            mask_approx=mask_gen_this_type[..., 0] == 1,
            num_batches=W_DIST_CFG["num_batches"],
        )
        sdz_charged_kld = reversed_kl_divergence_batched_different_variations(
            target=sdz_sim,
            approx=sdz_gen,
            mask_target=is_charged_and_nonmasked_sim,
            mask_approx=is_charged_and_nonmasked_gen,
            num_batches=W_DIST_CFG["num_batches"],
        )
        sdz_kld = reversed_kl_divergence_batched_different_variations(
            target=sdz_sim,
            approx=sdz_gen,
            mask_target=mask_sim_this_type[..., 0] == 1,
            mask_approx=mask_gen_this_type[..., 0] == 1,
            num_batches=W_DIST_CFG["num_batches"],
        )
        for kl_suffix, kl_variant in sd0_charged_kld.items():
            metrics[f"kl_part_sd0_charged_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_part_sd0_charged_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]
        for kl_suffix, kl_variant in sd0_kld.items():
            metrics[f"kl_part_sd0_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_part_sd0_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]
        for kl_suffix, kl_variant in sdz_charged_kld.items():
            metrics[f"kl_part_sdz_charged_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_part_sdz_charged_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]
        for kl_suffix, kl_variant in sdz_kld.items():
            metrics[f"kl_part_sdz_{kl_suffix}_mean_{jet_type}"] = kl_variant["mean"]
            metrics[f"kl_part_sdz_{kl_suffix}_std_{jet_type}"] = kl_variant["std"]

        # calculate the w1 distance for each particle feature
        # pylogger.info("Calculating w1 distance for each particle feature.")
        # w1p_means_this_type, w1p_stds_this_type = w1p(
        #     jets1=data_sim[jet_type_mask_sim],
        # jets2=data_gen[jet_type_mask_gen],
        # mask1=mask_sim[jet_type_mask_sim],
        # mask2=mask_gen[jet_type_mask_gen],
        # exclude_zeros=True,
        # **W_DIST_CFG,
        # )
        # add to dict
        # for i, part_feature_name in enumerate(part_names_sim):
        #     w1_mean, w1_std = w1p_means_this_type[i], w1p_stds_this_type[i]
        #     metrics[f"w_dist_{part_feature_name}_mean_{jet_type}"] = w1_mean
        #     metrics[f"w_dist_{part_feature_name}_std_{jet_type}"] = w1_std

    yaml_path = output_dir / f"eval_metrics_{n_samples_gen}{suffix}.yml"
    pylogger.info(f"Writing final evaluation metrics to {yaml_path}")

    # transform numpy.float64 for better readability in yaml file
    metrics_final = {}
    for key, value in metrics.items():
        if not isinstance(value, str):
            metrics_final[key] = float(value)
        else:
            metrics_final[key] = value
    metrics_final["w1_kld_calc_num_eval_samples"] = W_DIST_CFG["num_eval_samples"]
    metrics_final["w1_kld_calc_num_batches"] = W_DIST_CFG["num_batches"]
    # write to yaml file
    with open(yaml_path, "w") as outfile:
        yaml.dump(metrics_final, outfile, default_flow_style=False)
    # also print to terminal
    print(pd.DataFrame.from_dict(metrics_final, orient="index"))


if __name__ == "__main__":
    main()
