"""Do final evaluation of the model after training.

Specific to JetClass dataset.
"""

import os
import shutil
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import yaml

from particle_fm.data.components import calculate_all_wasserstein_metrics, normalize_tensor
from particle_fm.data.components.metrics import wasserstein_distance_batched
from particle_fm.utils.data_generation import generate_data
from particle_fm.utils.jet_substructure import dump_hlvs
from particle_fm.utils.plotting import (
    apply_mpl_styles,
    plot_data,
    plot_full_substructure,
    plot_particle_features,
    plot_substructure,
    prepare_data_for_plotting,
)
from particle_fm.utils.pylogger import get_pylogger

from .ema import EMA, EMAModelCheckpoint

pylogger = get_pylogger("JetClassTestEvaluationCallback")


# TODO cond_path is currently only working for mass and pt
class JetClassTestEvaluationCallback(pl.Callback):
    """Callback to do final evaluation of the model after training. Specific to JetClass dataset.

    Args:
        use_ema (bool, optional): Use exponential moving average weights for logging.
            Defaults to False.
        dataset (str, optional): Dataset to evaluate on. Defaults to "test".
        nr_checkpoint_callbacks (int, optional): Number of checkpoint callback that is used to
            select best epoch. Will only be used when ckpt_path is None. Defaults to 0.
        use_last_checkpoint (bool, optional): Use last checkpoint instead of best checkpoint.
            Defaults to True.
        ckpt_path (Optional[str], optional): Path to checkpoint. If given, this ckpt will be
            used for evaluation. Defaults to None.
        num_jet_samples (int, optional): How many jet samples to generate. Negative values define
            the amount of times the whole dataset is taken, e.g. -2 would use 2*len(dataset)
            samples. Defaults to -1.
        fix_seed (bool, optional): Fix seed for data generation to have better reproducibility
            and comparability between epochs. Defaults to True.
        evaluate_substructure (bool, optional): Evaluate substructure metrics. Takes very long.
            Defaults to True.
        suffix (str, optional): Suffix for logging. Defaults to "".
        cond_path (Optional[str], optional): Path for conditioning that is used during generation.
            If not provided, the selected dataset will be used for conditioning. Defaults to None.
        w_dist_config (Mapping, optional): Configuration for Wasserstein distance calculation.
            Defaults to {'num_jet_samples': 10_000, 'num_batches': 40}.
        generation_config (Mapping, optional): Configuration for data generation.
            Defaults to {"batch_size": 256, "ode_solver": "midpoint", "ode_steps": 100}.
        plot_config (Mapping, optional): Configuration for plotting. Defaults to {}.
    """

    def __init__(
        self,
        use_ema: bool = True,
        dataset: str = "test",
        nr_checkpoint_callbacks: int = 0,
        use_last_checkpoint: bool = True,
        ckpt_path: Optional[str] = None,
        num_jet_samples: int = -1,
        fix_seed: bool = True,
        evaluate_substructure: bool = True,
        suffix: str = "",
        cond_path: Optional[str] = None,  # TODO: figure out when to use this
        w_dist_config: Mapping = {
            "num_eval_samples": 50_000,
            "num_batches": 40,
        },
        generation_config: Mapping = {
            "batch_size": 1024,
            "ode_solver": "midpoint",
            "ode_steps": 100,
        },
        plot_config: Mapping = {"plot_efps": False},
    ):
        super().__init__()

        apply_mpl_styles()

        self.use_ema = use_ema
        self.dataset = dataset
        self.ckpt_path = ckpt_path
        self.nr_checkpoint_callbacks = nr_checkpoint_callbacks
        self.use_last_checkpoint = use_last_checkpoint
        self.num_jet_samples = num_jet_samples
        self.fix_seed = fix_seed
        self.evaluate_substructure = evaluate_substructure
        self.suffix = suffix
        self.cond_path = cond_path
        # loggers
        self.comet_logger = None
        self.wandb_logger = None

        # configs
        self.w_dist_config = w_dist_config
        self.generation_config = generation_config
        self.plot_config = plot_config

    def on_test_start(self, trainer, pl_module) -> None:
        pylogger.info(
            "JetClassFinalEvaluationCallback will be used for evaluating the model after training."
        )

        # set number of jet samples if negative
        if self.num_jet_samples < 0:
            self.datasets_multiplier = abs(self.num_jet_samples)
            if self.dataset == "test":
                self.num_jet_samples = len(trainer.datamodule.tensor_test) * abs(
                    self.num_jet_samples
                )
            if self.dataset == "val":
                self.num_jet_samples = len(trainer.datamodule.tensor_val) * abs(
                    self.num_jet_samples
                )
        else:
            self.datasets_multiplier = -1
        if self.cond_path is not None:
            self.datasets_multiplier = -1

        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pylogger.info(f"Evaluating model on {self.dataset} dataset.")

        ckpt = self._get_checkpoint(trainer, use_last_checkpoint=self.use_last_checkpoint)

        pylogger.info(f"Loading checkpoint from {ckpt}")
        model = pl_module.load_from_checkpoint(ckpt)

        if self.fix_seed:
            # fix seed for better reproducibility and comparable results
            torch.manual_seed(9999)

        # load conditioning data if provided
        # TODO: when to use this?
        if self.cond_path is not None:
            with h5py.File(self.cond_path) as f:
                pt_c = f["pt"][:]
                mass_c = f["mass"][:]
                num_particles_c = f["num_particles"][:].squeeze()

            # masking for jet size
            jet_size = trainer.datamodule.hparams.num_particles
            num_particles_ctemp = np.array(
                [n if n <= jet_size else jet_size for n in num_particles_c]
            )

            mask_c = np.expand_dims(
                np.tri(jet_size)[num_particles_ctemp.astype(int) - 1], axis=-1
            ).astype(np.float32)

            # get conditioning data
            # TODO implement other conditioning options
            if trainer.datamodule.num_cond_features != 0:
                cond_means = np.array(trainer.datamodule.cond_means)
                cond_stds = np.array(trainer.datamodule.cond_stds)
                pt_norm = normalize_tensor(pt_c.copy(), [cond_means[0]], [cond_stds[0]])
                mass_norm = normalize_tensor(mass_c.copy(), [cond_means[1]], [cond_stds[1]])
                cond_c = np.concatenate((pt_norm, mass_norm), axis=-1)
            else:
                cond_c = np.concatenate((pt_c, mass_c), axis=-1)

        # Get background data for plotting and calculating Wasserstein distances
        if self.dataset == "test":
            data_sim = np.array(trainer.datamodule.tensor_test)[: self.num_jet_samples]
            mask_sim = np.array(trainer.datamodule.mask_test)[: self.num_jet_samples]
            cond_sim = np.array(trainer.datamodule.tensor_conditioning_test)[
                : self.num_jet_samples
            ]
        elif self.dataset == "val":
            data_sim = np.array(trainer.datamodule.tensor_val)[: self.num_jet_samples]
            mask_sim = np.array(trainer.datamodule.mask_val)[: self.num_jet_samples]
            cond_sim = np.array(trainer.datamodule.tensor_conditioning_val)[: self.num_jet_samples]
        if self.cond_path is not None:
            mask_gen = mask_c[: self.num_jet_samples]
            cond_gen = cond_c[: self.num_jet_samples]
        else:
            mask_gen = mask_sim
            cond_gen = cond_sim

        # maximum number of samples to plot is the number of samples in the dataset
        num_plot_samples = len(data_sim)

        if self.datasets_multiplier > 1:
            mask_gen = np.repeat(mask_gen, self.datasets_multiplier, axis=0)
            cond_gen = np.repeat(cond_gen, self.datasets_multiplier, axis=0)

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

        data_output_path = (
            output_dir / f"generated_data_epoch_{ckpt_epoch}_nsamples_{len(mask_gen)}.npz"
        )
        if data_output_path.exists():
            pylogger.info(
                f"Output file {data_output_path} already exists. "
                "Will use existing file instead of generating again."
            )
            npfile = np.load(data_output_path, allow_pickle=True)
            data_gen = npfile["data_gen"]
        else:
            # Generate data
            data_gen, generation_time = generate_data(
                model=model,
                num_jet_samples=len(mask_gen),
                cond=torch.tensor(cond_gen),
                variable_set_sizes=trainer.datamodule.hparams.variable_jet_sizes,
                mask=torch.tensor(mask_gen),
                normalized_data=trainer.datamodule.hparams.normalize,
                means=trainer.datamodule.means,
                stds=trainer.datamodule.stds,
                **self.generation_config,
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
                names_particles_features=trainer.datamodule.names_particle_features,
                names_conditioning=trainer.datamodule.names_conditioning,
            )

        # If there are multiple jet types, plot them separately
        jet_types_dict = {
            var_name.split("_")[-1]: i
            for i, var_name in enumerate(trainer.datamodule.names_conditioning)
            if "jet_type" in var_name
        }
        pylogger.info(f"Used jet types: {jet_types_dict.keys()}")

        plot_particle_features(
            data_gen=data_gen,
            data_sim=data_sim,
            mask_gen=mask_gen,
            mask_sim=mask_sim,
            feature_names=trainer.datamodule.names_particle_features,
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
                feature_names=trainer.datamodule.names_particle_features,
                legend_label_sim="JetClass",
                legend_label_gen="Generated",
                plot_path=output_dir / f"epoch_{ckpt_epoch}_particle_features_{jet_type}.pdf",
            )

        # remove the additional particle features for compatibility with the rest of the code
        data_sim = data_sim[:, :, :3]
        data_gen = data_gen[:, :, :3]

        # save generated data
        # path = "/".join(ckpt.split("/")[:-2]) + "/"
        # file_name = f"final_generated_data{self.suffix}.npy"
        # full_path = path + file_name
        # np.save(full_path, data_gen)

        # Wasserstein distances
        pylogger.info("Calculating Wasserstein distances.")
        metrics = calculate_all_wasserstein_metrics(data_sim, data_gen, **self.w_dist_config)

        # Prepare Data for Plotting
        pylogger.info("Preparing data for plotting.")
        data_gen_plotting = data_gen[:num_plot_samples]
        plot_prep_config = {
            "calculate_efps" if key == "plot_efps" else key: value
            for key, value in self.plot_config.items()
            if key in ["plot_efps", "selected_particles", "selected_multiplicities"]
        }

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
        plot_name = f"final_plot_all_jet_types{self.suffix}"
        img_path = "/".join(ckpt.split("/")[:-2]) + "/"
        plot_data(
            particle_data=np.array([data_gen_plotting]),
            sim_data=data_sim,
            jet_data_sim=jet_data_sim,
            jet_data=jet_data_gen,
            efps_sim=efps_sim,
            efps_values=efps_values_gen,
            num_samples=num_plot_samples,
            pt_selected_particles=pt_selected_particles_gen,
            pt_selected_multiplicities=pt_selected_multiplicities_gen,
            pt_selected_particles_sim=pt_selected_particles_sim,
            pt_selected_multiplicities_sim=pt_selected_multiplicities_sim,
            save_fig=True,
            save_folder=img_path,
            save_name=plot_name,
            close_fig=True,
            **self.plot_config,
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

                plot_name = f"final_plot_{jet_type}{self.suffix}"
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
                    **self.plot_config,
                )

        if self.evaluate_substructure:
            substructure_path = "/".join(ckpt.split("/")[:-2]) + "/"
            substructure_file_name = f"substructure{self.suffix}"
            substructure_full_path = substructure_path + substructure_file_name

            # calculate substructure for generated data
            dump_hlvs(data_gen, substructure_full_path, plot=False)

            substructure_path_jetclass = "/".join(ckpt.split("/")[:-2]) + "/"
            substructure_file_name_jetclass = f"substructure_jetclass{self.suffix}"
            substructure_full_path_jetclass = (
                substructure_path_jetclass + substructure_file_name_jetclass
            )

            # calculate substructure for reference data
            dump_hlvs(data_sim, substructure_full_path_jetclass, plot=False)

            # load substructure for model generated data
            keys = []
            data_substructure = []
            with h5py.File(substructure_full_path + ".h5", "r") as f:
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
            with h5py.File(substructure_full_path_jetclass + ".h5", "r") as f:
                tau21_jetclass = np.array(f["tau21"])
                tau32_jetclass = np.array(f["tau32"])
                d2_jetclass = np.array(f["d2"])
                for key in f.keys():
                    data_substructure_jetclass.append(np.array(f[key]))
            data_substructure_jetclass = np.array(data_substructure_jetclass)

            # calculate wasserstein distances
            w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
                tau21_jetclass, tau21, **self.w_dist_config
            )
            w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
                tau32_jetclass, tau32, **self.w_dist_config
            )
            w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(
                d2_jetclass, d2, **self.w_dist_config
            )

            # add to metrics
            metrics["w_dist_tau21_mean"] = w_dist_tau21_mean
            metrics["w_dist_tau21_std"] = w_dist_tau21_std
            metrics["w_dist_tau32_mean"] = w_dist_tau32_mean
            metrics["w_dist_tau32_std"] = w_dist_tau32_std
            metrics["w_dist_d2_mean"] = w_dist_d2_mean
            metrics["w_dist_d2_std"] = w_dist_d2_std

            # plot substructure
            file_name_substructure = f"substructure_3plots{self.suffix}"
            file_name_full_substructure = f"substructure_full{self.suffix}"
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
            if self.comet_logger is not None:
                self.comet_logger.log_image(
                    img_path_substructure, name=f"A_final_substructure{self.suffix}"
                )
                self.comet_logger.log_image(
                    img_path_substructure_full, name=f"A_final_substructure_full{self.suffix}"
                )
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {f"A_final_substructure{self.suffix}": wandb.Image(img_path_substructure)}
                )
                self.wandb_logger.log(
                    {
                        f"A_final_substructure_full{self.suffix}": wandb.Image(
                            img_path_substructure_full
                        )
                    }
                )

        yaml_path = "/".join(ckpt.split("/")[:-2]) + f"/final_eval_metrics{self.suffix}.yml"
        pylogger.info(f"Writing final evaluation metrics to {yaml_path}")

        # transform numpy.float64 for better readability in yaml file
        metrics = {k: float(v) for k, v in metrics.items()}
        # write to yaml file
        with open(yaml_path, "w") as outfile:
            yaml.dump(metrics, outfile, default_flow_style=False)

        # rename wasserstein distances for better distinction
        metrics_final = {}
        for key, value in metrics.items():
            metrics_final[key + f"_final{self.suffix}"] = value

        # log metrics and image to loggers
        img_path_data = f"{img_path}{plot_name}.png"
        if self.comet_logger is not None:
            self.comet_logger.log_image(img_path_data, name=f"A_final_plot{self.suffix}")
            self.comet_logger.log_metrics(metrics_final)
        if self.wandb_logger is not None:
            self.wandb_logger.log({f"A_final_plot{self.suffix}": wandb.Image(img_path_data)})
            self.wandb_logger.log(metrics_final)

    def _get_checkpoint(self, trainer: pl.Trainer, use_last_checkpoint: bool = True) -> None:
        """Get checkpoint path based on the selected checkpoint callback."""
        if self.ckpt_path is None:
            if self.use_ema:
                if isinstance(
                    trainer.checkpoint_callbacks[self.nr_checkpoint_callbacks], EMAModelCheckpoint
                ):
                    if use_last_checkpoint:
                        return trainer.checkpoint_callbacks[
                            self.nr_checkpoint_callbacks
                        ].last_model_path_ema
                    else:
                        return trainer.checkpoint_callbacks[
                            self.nr_checkpoint_callbacks
                        ].best_model_path_ema

                else:
                    raise ValueError(
                        "JetClassFinalEvaluationCallback was told to use EMA weights for"
                        " evaluation but the provided checkpoint callback is not of type"
                        " EMAModelCheckpoint"
                    )
            else:
                if use_last_checkpoint:
                    return trainer.checkpoint_callbacks[
                        self.nr_checkpoint_callbacks
                    ].last_model_path
                else:
                    return trainer.checkpoint_callbacks[
                        self.nr_checkpoint_callbacks
                    ].best_model_path
        else:
            return self.ckpt_path
