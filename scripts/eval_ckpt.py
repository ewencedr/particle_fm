"""Script to evaluate a checkpoint and generate plots and metrics.

Usage: python src/eval_ckpt.py --ckpt <path_to_ckpt> --n_samples <int>
"""
import argparse
import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path

import h5py
import hydra
import numpy as np

# plots and metrics
import pandas as pd
import torch
import yaml

# set env variable DATA_DIR again because of hydra
from dotenv import load_dotenv

# set env variable DATA_DIR again because of hydra
from omegaconf import OmegaConf

from src.data.components import calculate_all_wasserstein_metrics
from src.data.components.metrics import wasserstein_distance_batched

# from src.data.components.utils import calculate_jet_features
from src.utils.data_generation import generate_data
from src.utils.jet_substructure import dump_hlvs
from src.utils.plotting import (  # create_and_plot_data,; plot_single_jets,; plot_data,
    apply_mpl_styles,
    plot_full_substructure,
    plot_jet_features,
    plot_particle_features,
    plot_substructure,
    prepare_data_for_plotting,
)

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

VARIABLES_TO_CLIP = ["part_ptrel"]


def main():
    args = parser.parse_args()
    ckpt = args.ckpt
    n_samples_gen = args.n_samples
    suffix = f"-{args.suffix}" if args.suffix is not None else ""

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

    cfg.data.conditioning_gen_filename = args.cond_gen_file

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

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

    if args.cond_gen_file is not None:
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
    pylogger.info(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    if not (output_dir / f"epoch_{ckpt_epoch}.ckpt").exists():
        pylogger.info(f"Copy checkpoint file to {output_dir}")
        shutil.copyfile(ckpt, output_dir / f"epoch_{ckpt_epoch}.ckpt")

    h5data_output_path = (
        output_dir / f"generated_data_epoch_{ckpt_epoch}_nsamples_{n_samples_gen}{suffix}.h5"
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
        # Correcting the generated data
        # Clipping
        for i, var_name in enumerate(datamodule.names_particle_features):
            if var_name not in VARIABLES_TO_CLIP:
                continue
            pylogger.info(f"Clipping outliers of generated {var_name} to original range.")
            data_gen[mask_gen[..., 0] != 0, i] = np.clip(
                data_gen[mask_gen[..., 0] != 0, i],
                a_min=datamodule.min_max_train_dict[var_name]["min"],
                a_max=datamodule.min_max_train_dict[var_name]["max"],
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

        # # ------------------------------------------------

        pylogger.info("Calculating jet features")
        plot_prep_config = {"calculate_efps": True}
        (
            jet_data_gen,
            efps_values_gen,
            pt_selected_particles_gen,
            _,
        ) = prepare_data_for_plotting([data_gen[:, :, :3]], **plot_prep_config)
        (
            jet_data_sim,
            efps_values_sim,
            pt_selected_particles_sim,
            _,
        ) = prepare_data_for_plotting([data_sim[:, :, :3]], **plot_prep_config)
        # prepare_data_for_plotting returns lists of arrays of the following shape:
        # jet_data: (n_datasets, n_jets, n_jet_features)
        # efps_values: (n_datasets, n_jets, n_efp_features)
        # pt_selected_particles: (n_datasets, n_selected_particles, n_jets) --> swap axes here!
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
            # calculated jet data
            h5file.create_dataset("jet_data_gen", data=jet_data_gen[0])
            h5file.create_dataset("jet_data_sim", data=jet_data_sim[0])
            for ds_key in ["jet_data_sim", "jet_data_gen"]:
                h5file[ds_key].attrs.create(
                    "names",
                    data=["jet_pt", "jet_y", "jet_phi", "jet_mass"],
                    dtype=h5py.special_dtype(vlen=str),
                )
            h5file.create_dataset("efp_values_gen", data=efps_values_gen[0])
            h5file.create_dataset("efp_values_sim", data=efps_values_sim[0])
            h5file.create_dataset(
                "pt_selected_particles_gen", data=np.swapaxes(pt_selected_particles_gen, 1, 2)[0]
            )
            h5file.create_dataset(
                "pt_selected_particles_sim", data=np.swapaxes(pt_selected_particles_sim, 1, 2)[0]
            )

    # read the file
    with h5py.File(h5data_output_path) as h5file:
        data_gen = h5file["part_data_gen"][:]
        mask_gen = h5file["part_mask_gen"][:]
        cond_gen = h5file["cond_data_gen"][:]
        data_sim = h5file["part_data_sim"][:]
        mask_sim = h5file["part_mask_sim"][:]
        cond_sim = h5file["cond_data_sim"][:]

        jet_data_gen = h5file["jet_data_gen"][:]
        jet_data_sim = h5file["jet_data_sim"][:]
        pt_selected_particles_gen = h5file["pt_selected_particles_gen"][:]
        pt_selected_particles_sim = h5file["pt_selected_particles_sim"][:]

        part_names_sim = h5file["part_data_sim"].attrs["names"][:]

    pylogger.info("Plotting particle features")
    plot_particle_features(
        data_gen=data_gen,
        data_sim=data_sim,
        mask_gen=mask_gen,
        mask_sim=mask_sim,
        feature_names=datamodule.names_particle_features,
        legend_label_sim="JetClass",
        legend_label_gen="Generated",
        plot_path=output_dir / f"epoch_{ckpt_epoch}_particle_features.pdf",
    )
    pylogger.info("Plotting jet features")
    plot_jet_features(
        jet_data_gen=jet_data_gen,
        jet_data_sim=jet_data_sim,
        jet_feature_names=["jet_pt", "jet_y", "jet_phi", "jet_mrel"],
        legend_label_sim="JetClass",
        legend_label_gen="Generated",
        plot_path=output_dir / f"epoch_{ckpt_epoch}_jet_features.pdf",
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

    pylogger.info("Calculating Wasserstein distances.")
    metrics = calculate_all_wasserstein_metrics(data_sim, data_gen)
    # metrics = {}

    # If there are multiple jet types, plot them separately
    jet_types_dict = {
        var_name.split("_")[-1]: i
        for i, var_name in enumerate(datamodule.names_conditioning)
        if "jet_type" in var_name
    }
    jet_types_dict["all_jet_types"] = None
    pylogger.info(f"Used jet types: {jet_types_dict.keys()}")

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
        # calculate metrics and add to dict
        metrics_this_type = calculate_all_wasserstein_metrics(
            data_sim[jet_type_mask_sim], data_gen[jet_type_mask_gen]
        )
        for key, value in metrics_this_type.items():
            metrics[f"{key}_{jet_type}"] = value

        for i, part_feature_name in enumerate(part_names_sim):
            w_dist_config = {"num_eval_samples": 50_000, "num_batches": 10}
            w1_mean, w1_std = wasserstein_distance_batched(
                data_sim[jet_type_mask_sim, :, i][mask_sim[jet_type_mask_sim, :, 0] == 1],
                data_gen[jet_type_mask_gen, :, i][mask_gen[jet_type_mask_gen, :, 0] == 1],
                **w_dist_config,
            )
            metrics[f"w_dist_{part_feature_name}_mean_{jet_type}"] = w1_mean
            metrics[f"w_dist_{part_feature_name}_std_{jet_type}"] = w1_std

        plot_particle_features(
            data_gen=data_gen[jet_type_mask_gen],
            data_sim=data_sim[jet_type_mask_sim],
            mask_gen=mask_gen[jet_type_mask_gen],
            mask_sim=mask_sim[jet_type_mask_sim],
            feature_names=datamodule.names_particle_features,
            legend_label_sim="JetClass",
            legend_label_gen="Generated",
            plot_path=output_dir / f"epoch_{ckpt_epoch}_particle_features_{jet_type}.pdf",
        )
        plot_jet_features(
            jet_data_gen=jet_data_gen[jet_type_mask_gen],
            jet_data_sim=jet_data_sim[jet_type_mask_sim],
            jet_feature_names=["jet_pt", "jet_y", "jet_phi", "jet_mrel"],
            legend_label_sim="JetClass",
            legend_label_gen="Generated",
            plot_path=output_dir / f"epoch_{ckpt_epoch}_jet_features_{jet_type}.pdf",
        )

    if EVALUATE_SUBSTRUCTURE:
        substructure_path = output_dir
        substr_filename_gen = (
            f"substructure_generated_epoch_{ckpt_epoch}_nsamples_{n_samples_gen}{suffix}"
        )
        substructure_full_path = substructure_path / substr_filename_gen
        substr_filename_jetclass = (
            f"substructure_simulated_epoch_{ckpt_epoch}_nsamples_{n_samples_gen}{suffix}"
        )
        substructure_full_path_jetclass = substructure_path / substr_filename_jetclass

        # calculate substructure for generated data
        if not os.path.isfile(str(substructure_full_path) + ".h5"):
            pylogger.info("Calculating substructure.")
            dump_hlvs(data_gen, str(substructure_full_path), plot=False)
        # calculate substructure for reference data
        if not os.path.isfile(str(substructure_full_path_jetclass) + ".h5"):
            pylogger.info("Calculating substructure.")
            dump_hlvs(data_sim, str(substructure_full_path_jetclass), plot=False)

        # load substructure for model generated data
        keys = []
        data_substructure = []
        with h5py.File(str(substructure_full_path) + ".h5", "r") as f:
            tau21 = np.array(f["tau21"])
            tau32 = np.array(f["tau32"])
            d2 = np.array(f["d2"])
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
            for key in f.keys():
                keys.append(key)
                data_substructure.append(np.array(f[key]))
        keys = np.array(keys)
        data_substructure = np.array(data_substructure)

        # load substructure for JetClass data
        data_substructure_jetclass = []
        with h5py.File(str(substructure_full_path_jetclass) + ".h5", "r") as f:
            tau21_jetclass = np.array(f["tau21"])
            tau32_jetclass = np.array(f["tau32"])
            d2_jetclass = np.array(f["d2"])
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
            for key in f.keys():
                data_substructure_jetclass.append(np.array(f[key]))
        data_substructure_jetclass = np.array(data_substructure_jetclass)

        w_dist_config = {"num_eval_samples": 50_000, "num_batches": 10}
        # calculate wasserstein distances
        w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
            tau21_jetclass, tau21, **w_dist_config
        )
        w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
            tau32_jetclass, tau32, **w_dist_config
        )
        w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(
            d2_jetclass, d2, **w_dist_config
        )

        # add to metrics
        metrics["w_dist_tau21_mean"] = w_dist_tau21_mean
        metrics["w_dist_tau21_std"] = w_dist_tau21_std
        metrics["w_dist_tau32_mean"] = w_dist_tau32_mean
        metrics["w_dist_tau32_std"] = w_dist_tau32_std
        metrics["w_dist_d2_mean"] = w_dist_d2_mean
        metrics["w_dist_d2_std"] = w_dist_d2_std

        # plot substructure
        file_name_substructure = "substructure_3plots"
        file_name_full_substructure = "substructure_full"
        img_path = str(output_dir) + "/"
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
        for jet_type, jet_type_idx in jet_types_dict.items():
            pylogger.info(f"Plotting substructure for jet type {jet_type}")
            if jet_type == "all_jet_types":
                jet_type_mask_sim = np.ones(len(cond_sim), dtype=bool)
                jet_type_mask_gen = np.ones(len(cond_gen), dtype=bool)
            else:
                jet_type_mask_sim = cond_sim[:, jet_type_idx] == 1
                jet_type_mask_gen = cond_gen[:, jet_type_idx] == 1
            w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
                tau21_jetclass[jet_type_mask_sim], tau21[jet_type_mask_gen], **w_dist_config
            )
            w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
                tau32_jetclass[jet_type_mask_sim], tau32[jet_type_mask_gen], **w_dist_config
            )
            w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(
                d2_jetclass[jet_type_mask_sim], d2[jet_type_mask_gen], **w_dist_config
            )
            # add to metrics
            metrics[f"w_dist_tau21_mean_{jet_type}"] = w_dist_tau21_mean
            metrics[f"w_dist_tau21_std_{jet_type}"] = w_dist_tau21_std
            metrics[f"w_dist_tau32_mean_{jet_type}"] = w_dist_tau32_mean
            metrics[f"w_dist_tau32_std_{jet_type}"] = w_dist_tau32_std
            metrics[f"w_dist_d2_mean_{jet_type}"] = w_dist_d2_mean
            metrics[f"w_dist_d2_std_{jet_type}"] = w_dist_d2_std

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

    yaml_path = output_dir / f"eval_metrics_{n_samples_gen}{suffix}.yml"
    pylogger.info(f"Writing final evaluation metrics to {yaml_path}")

    # transform numpy.float64 for better readability in yaml file
    metrics = {k: float(v) for k, v in metrics.items()}
    # write to yaml file
    with open(yaml_path, "w") as outfile:
        yaml.dump(metrics, outfile, default_flow_style=False)
    # also print to terminal
    print(pd.DataFrame.from_dict(metrics, orient="index"))


if __name__ == "__main__":
    main()
