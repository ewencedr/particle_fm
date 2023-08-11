import logging
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import cplt
import hydra

# plots and metrics
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

# set env variable DATA_DIR again because of hydra
from omegaconf import OmegaConf

from src.data.components import (
    calculate_all_wasserstein_metrics,
    inverse_normalize_tensor,
    normalize_tensor,
)
from src.data.components.utils import calculate_jet_features
from src.utils.data_generation import generate_data
from src.utils.plotting import (
    apply_mpl_styles,
    create_and_plot_data,
    plot_data,
    plot_full_substructure,
    plot_particle_features,
    plot_single_jets,
    plot_substructure,
    prepare_data_for_plotting,
)

# set up logging for jupyter notebook
pylogger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logging.info("test")

apply_mpl_styles()

# specify here the path to the run directory of the model you want to evaluate
run_dir = "/beegfs/desy/user/birkjosc/epic-fm/logs/jetclass_cond_jettype/runs/2023-08-10_16-26-03"
cfg_backup_file = f"{run_dir}/config.yaml"

# load everything from run directory (safer in terms of reproducing results)
cfg = OmegaConf.load(cfg_backup_file)
print(type(cfg))
print(OmegaConf.to_yaml(cfg))

datamodule = hydra.utils.instantiate(cfg.data)
datamodule.setup()

# load the model from the checkpoint
model = hydra.utils.instantiate(cfg.model)
ckpt = f"{run_dir}/checkpoints/last-EMA.ckpt"
ckpt = f"{run_dir}/evaluated_ckpts/epoch_273/epoch_273.ckpt"
model = model.load_from_checkpoint(ckpt)

# ------------------------------------------------
data_sim = np.array(datamodule.tensor_test)
mask_sim = np.array(datamodule.mask_test)
cond_sim = np.array(datamodule.tensor_conditioning_test)

n_generated_samples = 100_000
data_sim = data_sim[:n_generated_samples]
mask_sim = mask_sim[:n_generated_samples]
cond_sim = cond_sim[:n_generated_samples]

means = np.array(datamodule.means)
stds = np.array(datamodule.stds)

# for now use the same mask/cond for the generated data
mask_gen = deepcopy(mask_sim)
cond_gen = deepcopy(cond_sim)

# check if the output already exists
# --> only generate new data if it does not exist yet
checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
ckpt_epoch = checkpoint["epoch"]
pylogger.info(f"Loaded checkpoint from epoch {ckpt_epoch}")

ckpt_path = Path(ckpt)
output_dir = (
    ckpt_path.parent  # this should then be the "evaluated_ckpts" folder
    if f"evaluated_ckpts/epoch_{ckpt_epoch}" in str(ckpt_path)
    else ckpt_path.parent.parent / "evaluated_ckpts" / f"epoch_{ckpt_epoch}"
)
os.makedirs(output_dir, exist_ok=True)
if not (output_dir / f"epoch_{ckpt_epoch}.ckpt").exists():
    pylogger.info(f"Copy checkpoint file to {output_dir}")
    shutil.copyfile(ckpt, output_dir / f"epoch_{ckpt_epoch}.ckpt")

data_output_path = output_dir / f"generated_data_epoch_{ckpt_epoch}_nsamples_{len(data_sim)}.npz"
if data_output_path.exists():
    pylogger.info(
        f"Output file {data_output_path} already exists. "
        "Will use existing file instead of generating again."
    )
    npfile = np.load(data_output_path, allow_pickle=True)
    data_gen = npfile["data_gen"]
    mask_gen = npfile["mask_gen"]
    cond_gen = npfile["cond_gen"]
    data_sim = npfile["data_sim"]
    mask_sim = npfile["mask_sim"]
    cond_sim = npfile["cond_sim"]
else:
    # Generate data
    data_gen, generation_time = generate_data(
        model=model,
        num_jet_samples=len(mask_gen),
        cond=torch.tensor(cond_gen),
        variable_set_sizes=datamodule.hparams.variable_jet_sizes,
        mask=torch.tensor(mask_gen),
        normalized_data=datamodule.hparams.normalize,
        means=datamodule.means,
        stds=datamodule.stds,
    )
    pylogger.info(f"Generated {len(data_gen)} samples in {generation_time:.0f} seconds.")

    pylogger.info(f"Saving generated data to {data_output_path}")
    np.savez_compressed(
        data_output_path.resolve(),
        data_gen=data_gen,
        mask_gen=mask_gen,
        cond_gen=cond_gen,
        data_sim=data_sim,
        mask_sim=mask_sim,
        cond_sim=cond_sim,
        names_particles_features=datamodule.names_particle_features,
        names_conditioning=datamodule.names_conditioning,
    )

# If there are multiple jet types, plot them separately
jet_types_dict = {
    var_name.split("_")[-1]: i
    for i, var_name in enumerate(datamodule.names_conditioning)
    if "jet_type" in var_name
}
pylogger.info(f"Used jet types: {jet_types_dict.keys()}")

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

for jet_type, jet_type_idx in jet_types_dict.items():
    jet_type_mask_sim = cond_sim[:, jet_type_idx] == 1
    jet_type_mask_gen = cond_gen[:, jet_type_idx] == 1
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

# remove the additional particle features for compatibility with the rest of the code
data_sim = data_sim[:, :, :3]
data_gen = data_gen[:, :, :3]

# Wasserstein distances

pylogger.info("Calculating Wasserstein distances.")
metrics = calculate_all_wasserstein_metrics(
    data_sim, data_gen
)  # TODO: should we add a config?, **self.w_dist_config)

# Prepare Data for Plotting
pylogger.info("Preparing data for plotting.")
data_gen_plotting = data_gen
# plot_prep_config = {
#     "calculate_efps" if key == "plot_efps" else key: value
#     for key, value in self.plot_config.items()
#     if key in ["plot_efps", "selected_particles", "selected_multiplicities"]
# }
plot_prep_config = {"calculate_efps": True}

(
    jet_data_gen,
    efps_values_gen,
    pt_selected_particles_gen,
    pt_selected_multiplicities_gen,
) = prepare_data_for_plotting(np.array([data_gen_plotting]), **plot_prep_config)

(
    jet_data_sim,
    efps_sim,
    pt_selected_particles_sim,
    pt_selected_multiplicities_sim,
) = prepare_data_for_plotting(
    [data_sim],
    **plot_prep_config,
)
jet_data_sim, efps_sim, pt_selected_particles_sim = (
    jet_data_sim[0],
    efps_sim[0],
    pt_selected_particles_sim[0],
)

# Plotting
pylogger.info("Plotting distributions.")
plot_name = f"epoch_{ckpt_epoch}_overview_all_jet_types"
img_path = output_dir
plot_data(
    particle_data=np.array([data_gen_plotting]),
    sim_data=data_sim,
    jet_data_sim=jet_data_sim,
    jet_data=jet_data_gen,
    efps_sim=efps_sim,
    efps_values=efps_values_gen,
    # num_samples=num_plot_samples,
    pt_selected_particles=pt_selected_particles_gen,
    pt_selected_multiplicities=pt_selected_multiplicities_gen,
    pt_selected_particles_sim=pt_selected_particles_sim,
    pt_selected_multiplicities_sim=pt_selected_multiplicities_sim,
    save_fig=True,
    save_folder=img_path,
    save_name=plot_name,
    close_fig=True,
    # **self.plot_config,
)

if len(jet_types_dict) > 0:
    pylogger.info("Plotting jet types separately")
    for jet_type, index in jet_types_dict.items():
        pylogger.info(f"Plotting jet type {jet_type}")
        mask_this_jet_type_sim = cond_sim[:, index] == 1
        # TODO: check if this still works once we use generated conditioning
        mask_this_jet_type_gen = cond_gen[:, index] == 1
        (
            jet_data_this_jet_type,
            efps_values_this_jet_type,
            pt_selected_particles_this_jet_type,
            pt_selected_multiplicities_this_jet_type,
        ) = prepare_data_for_plotting(
            np.array([data_gen_plotting[mask_this_jet_type_gen]]), **plot_prep_config
        )

        (
            jet_data_this_jet_type_sim,
            efps_this_jet_type_sim,
            pt_selected_particles_this_jet_type_sim,
            pt_selected_multiplicities_this_jet_type_sim,
        ) = prepare_data_for_plotting(
            [data_sim[mask_this_jet_type_sim]],
            **plot_prep_config,
        )
        (
            jet_data_this_jet_type_sim,
            efps_this_jet_type_sim,
            pt_selected_particles_this_jet_type_sim,
        ) = (
            jet_data_this_jet_type_sim[0],
            efps_this_jet_type_sim[0],
            pt_selected_particles_this_jet_type_sim[0],
        )

        plot_name = f"epoch_{ckpt_epoch}_overview_{jet_type}"
        plot_data(
            particle_data=np.array([data_gen_plotting[mask_this_jet_type_gen]]),
            sim_data=data_sim[mask_this_jet_type_sim],
            jet_data_sim=jet_data_this_jet_type_sim,
            jet_data=jet_data_this_jet_type,
            efps_sim=efps_this_jet_type_sim,
            efps_values=efps_values_this_jet_type,
            num_samples=-1,
            pt_selected_particles=pt_selected_particles_this_jet_type,
            pt_selected_multiplicities=pt_selected_multiplicities_this_jet_type,
            pt_selected_particles_sim=pt_selected_particles_this_jet_type_sim,
            pt_selected_multiplicities_sim=pt_selected_multiplicities_this_jet_type_sim,
            save_fig=True,
            save_folder=img_path,
            save_name=plot_name,
            close_fig=True,
            # **self.plot_config,
        )
