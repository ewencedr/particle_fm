import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path

import h5py
import hydra

# plots and metrics
import numpy as np
import torch
import yaml

# set env variable DATA_DIR again because of hydra
from omegaconf import OmegaConf

from src.data.components import calculate_all_wasserstein_metrics
from src.data.components.metrics import wasserstein_distance_batched

# from src.data.components.utils import calculate_jet_features
from src.utils.data_generation import generate_data
from src.utils.jet_substructure import dump_hlvs
from src.utils.plotting import (  # create_and_plot_data,; plot_single_jets,
    apply_mpl_styles,
    plot_data,
    plot_full_substructure,
    plot_particle_features,
    plot_substructure,
    prepare_data_for_plotting,
)

# set up logging for jupyter notebook
pylogger = logging.getLogger("eval_ckpt")
logging.basicConfig(level=logging.INFO)
logging.info("test")

apply_mpl_styles()

# TODO:
# - make ckpt_path a command_line argument
# improve the looping over the jet types (takes way too long at the moment)

# specify here the path to the run directory of the model you want to evaluate
ckpt = "/beegfs/desy/user/birkjosc/epic-fm/logs/jetclass_cond_jettype/runs/2023-08-10_16-26-03/evaluated_ckpts/epoch_273/epoch_273.ckpt"
EVALUATE_SUBSTRUCTURE = True
N_GENERATED_SAMPLES = 100_000

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

datamodule = hydra.utils.instantiate(cfg.data)
datamodule.setup()

# load the model from the checkpoint
model = hydra.utils.instantiate(cfg.model)
model = model.load_from_checkpoint(ckpt)

# ------------------------------------------------
data_sim = np.array(datamodule.tensor_test)
mask_sim = np.array(datamodule.mask_test)
cond_sim = np.array(datamodule.tensor_conditioning_test)

data_sim = data_sim[:N_GENERATED_SAMPLES]
mask_sim = mask_sim[:N_GENERATED_SAMPLES]
cond_sim = cond_sim[:N_GENERATED_SAMPLES]

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
    ckpt_path.parent
    if f"evaluated_ckpts/epoch_{ckpt_epoch}" in str(ckpt_path)
    else ckpt_path.parent.parent / "evaluated_ckpts" / f"epoch_{ckpt_epoch}"
)
pylogger.info(f"Output directory: {output_dir}")

os.makedirs(output_dir, exist_ok=True)
if not (output_dir / f"epoch_{ckpt_epoch}.ckpt").exists():
    pylogger.info(f"Copy checkpoint file to {output_dir}")
    shutil.copyfile(ckpt, output_dir / f"epoch_{ckpt_epoch}.ckpt")

data_output_path = output_dir / f"generated_data_epoch_{ckpt_epoch}_nsamples_{len(data_sim)}.npz"
# data_output_path = output_dir / f"generated_data_epoch_{ckpt_epoch}_nsamples_{1000}.npz"
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
# metrics = {}

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
img_path = str(output_dir) + "/"
pylogger.warning(f"Saving plots to {img_path}")
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

        # TODO: the calculation below slows things down a lot
        # --> can this be done using the calculated values from above and
        # just apply the mask to the calculated values?
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

if EVALUATE_SUBSTRUCTURE:
    pylogger.info("Calculating substructure.")
    substructure_path = output_dir
    substr_filename_gen = "substructure_generated"
    substructure_full_path = substructure_path / substr_filename_gen

    # calculate substructure for generated data
    dump_hlvs(data_gen, str(substructure_full_path), plot=False)

    substr_filename_jetclass = "substructure_jetclass"
    substructure_full_path_jetclass = substructure_path / substr_filename_jetclass

    # calculate substructure for reference data
    dump_hlvs(data_sim, str(substructure_full_path_jetclass), plot=False)

    # load substructure for model generated data
    keys = []
    data_substructure = []
    with h5py.File(str(substructure_full_path) + ".h5", "r") as f:
        tau21 = np.array(f["tau21"])
        tau32 = np.array(f["tau32"])
        d2 = np.array(f["d2"])
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
        for key in f.keys():
            data_substructure_jetclass.append(np.array(f[key]))
    data_substructure_jetclass = np.array(data_substructure_jetclass)

    w_dist_config = {"num_eval_samples": 50_000, "num_batches": 40}
    # calculate wasserstein distances
    w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
        tau21_jetclass, tau21, **w_dist_config
    )
    w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
        tau32_jetclass, tau32, **w_dist_config
    )
    w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(d2_jetclass, d2, **w_dist_config)

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

    # log substructure images
    img_path_substructure = f"{img_path}{file_name_substructure}.png"
    img_path_substructure_full = f"{img_path}{file_name_full_substructure}.png"

yaml_path = output_dir / "final_eval_metrics.yml"
pylogger.info(f"Writing final evaluation metrics to {yaml_path}")

# transform numpy.float64 for better readability in yaml file
metrics = {k: float(v) for k, v in metrics.items()}
# write to yaml file
with open(yaml_path, "w") as outfile:
    yaml.dump(metrics, outfile, default_flow_style=False)

# rename wasserstein distances for better distinction
metrics_final = {}
for key, value in metrics.items():
    metrics_final[key + "_final"] = value
