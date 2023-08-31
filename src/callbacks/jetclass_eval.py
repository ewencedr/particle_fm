"""Callback for evaluating the model on the JetClass dataset."""
import os
import time
import warnings
from typing import Callable, Mapping, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from src.data.components import calculate_all_wasserstein_metrics
from src.data.components.metrics import wasserstein_distance_batched
from src.schedulers.logging_scheduler import (
    custom1,
    custom5000epochs,
    custom10000epochs,
    epochs10000,
    nolog10000,
)
from src.utils.data_generation import generate_data
from src.utils.jet_substructure import dump_hlvs
from src.utils.plotting import (
    apply_mpl_styles,
    plot_data,
    plot_full_substructure,
    plot_particle_features,
    plot_substructure,
    prepare_data_for_plotting,
)
from src.utils.pylogger import get_pylogger

from .ema import EMA

pylogger = get_pylogger("JetClassEvaluationCallback")


def load_substructure(filepath):
    """Load substructure data from a h5 file.

    Args:
        filepath (str): Path to the h5 file.

    Returns:
        data_substructure (np.ndarray): Substructure data.
        keys (np.ndarray): Keys of the substructure data.
        tau21 (np.ndarray): Tau21 values (nan set to 0)
        tau32 (np.ndarray): Tau32 values (nan set to 0).
        d2 (np.ndarray): D2 values (nan set to 0).
    """
    keys = []
    data_substructure = []
    with h5py.File(filepath) as f:
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

    return data_substructure, keys, tau21, tau32, d2


class JetClassEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the JetClass dataset and log
    the results to loggers. Currently supported are CometLogger and WandbLogger.

    Args:
        every_n_epochs (int, optional): Log every n epochs. Defaults to 10.
        additional_eval_epochs (list, optional): Log additional epochs. Defaults to [].
        num_jet_samples (int, optional): How many jet samples to generate.
            Negative values define the amount of times the whole dataset is taken,
            e.g. -2 would use 2*len(dataset) samples. Defaults to -1.
        image_path (str, optional): Folder where the images are saved. Defaults
            to "./logs/callback_images/".
        model_name (str, optional): Name for saving the model. Defaults to "model-test".
        log_times (bool, optional): Log generation times of data. Defaults to True.
        log_epoch_zero (bool, optional): Log in first epoch. Default to False.
        data_type (str, optional): Type of data to plot. Options are 'test' and 'val'.
            Defaults to "test".
        use_ema (bool, optional): Use exponential moving average weights for logging.
            Defaults to False.
        fix_seed (bool, optional): Fix seed for data generation to have better
            reproducibility and comparability between epochs. Defaults to True.
        w_dist_config (Mapping, optional): Configuration for Wasserstein distance
            calculation. Defaults to {'num_jet_samples': 10_000, 'num_batches': 40}.
        generation_config (Mapping, optional): Configuration for data generation.
            Defaults to {"batch_size": 256, "ode_solver": "midpoint", "ode_steps": 100}.
        plot_config (Mapping, optional): Configuration for plotting. Defaults to {}.
    """

    def __init__(
        self,
        every_n_epochs: int | Callable = 10,
        additional_eval_epochs: list[int] = None,
        num_jet_samples: int = -1,
        image_path: str = None,
        model_name: str = "model",
        log_times: bool = True,
        log_epoch_zero: bool = False,
        data_type: str = "val",
        use_ema: bool = False,
        fix_seed: bool = True,
        w_dist_config: Mapping = {
            "num_jet_samples": 10_000,
            "num_batches": 40,
        },
        generation_config: Mapping = {
            "batch_size": 256,
            "ode_solver": "midpoint",
            "ode_steps": 100,
        },
        plot_config: Mapping = {"plot_efps": False},
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.additional_eval_epochs = additional_eval_epochs
        self.num_jet_samples = num_jet_samples
        self.log_times = log_times
        self.log_epoch_zero = log_epoch_zero
        self.use_ema = use_ema
        self.fix_seed = fix_seed

        self.model_name = model_name
        self.data_type = data_type

        self.image_path = image_path
        apply_mpl_styles()

        self.w_dist_config = w_dist_config
        self.generation_config = generation_config
        self.plot_config = plot_config

        # loggers
        self.comet_logger = None
        self.wandb_logger = None

        # available custom logging schedulers
        self.available_custom_logging_scheduler = {
            "custom1": custom1,
            "custom5000epochs": custom5000epochs,
            "custom10000epochs": custom10000epochs,
            "nolog10000": nolog10000,
            "epochs10000": epochs10000,
        }

    def on_train_start(self, trainer, pl_module) -> None:
        # log something, so that metrics exists and the checkpoint callback doesn't crash
        self.log("w1m_mean", 0.005)
        self.log("w1p_mean", 0.005)

        if self.image_path is None:
            self.image_path = f"{trainer.default_root_dir}/plots/"
            os.makedirs(self.image_path, exist_ok=True)

        pylogger.info("Logging plots during training to %s", self.image_path)

        # set number of jet samples if negative
        if self.num_jet_samples < 0:
            self.datasets_multiplier = abs(self.num_jet_samples)
            if self.data_type == "test":
                self.num_jet_samples = len(trainer.datamodule.tensor_test) * abs(
                    self.num_jet_samples
                )
            if self.data_type == "val":
                self.num_jet_samples = len(trainer.datamodule.tensor_val) * abs(
                    self.num_jet_samples
                )
        else:
            self.datasets_multiplier = -1

        hparams_to_log = {
            "training_dataset_size": float(len(trainer.datamodule.tensor_train)),
            "validation_dataset_size": float(len(trainer.datamodule.tensor_val)),
            "test_dataset_size": float(len(trainer.datamodule.tensor_test)),
            "number_of_generated_val_jets": float(self.num_jet_samples),
        }
        # get loggers
        for logger in trainer.loggers:
            logger.log_hyperparams(hparams_to_log)
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        # get ema callback
        self.ema_callback = self._get_ema_callback(trainer)
        if self.ema_callback is None and self.use_ema:
            warnings.warn(
                "JetClass Evaluation Callbacks was told to use EMA weights, but EMA callback was"
                " not found. Using normal weights."
            )
        elif self.ema_callback is not None and self.use_ema:
            pylogger.info("Using EMA weights for evaluation.")

        # TODO: maybe add here crosscheck plots (e.g. the jet mass of different
        # jet types to ensure the labels are not messed up etc (+ other variables))

    def on_train_epoch_end(self, trainer, pl_module):
        if self.fix_seed:
            # fix seed for better reproducibility and comparable results
            torch.manual_seed(9999)

        # Skip for all other epochs
        log_epoch = True
        if not self.log_epoch_zero and trainer.current_epoch == 0:
            log_epoch = False

        # determine if logging should happen
        log = False
        if type(self.every_n_epochs) is int:
            if trainer.current_epoch % self.every_n_epochs == 0 and log_epoch:
                log = True
        else:
            try:
                custom_logging_schedule = self.available_custom_logging_scheduler[
                    self.every_n_epochs
                ]
                log = custom_logging_schedule(trainer.current_epoch)
            except KeyError:
                raise KeyError("Custom logging schedule not available.")
        # log at additional epochs
        if self.additional_eval_epochs is not None:
            if trainer.current_epoch in self.additional_eval_epochs and log_epoch:
                log = True

        if log:
            time_eval_start = time.time()
            pylogger.info(f"Evaluating model after epoch {trainer.current_epoch}.")
            # Get background data for plotting and calculating Wasserstein distances
            # fmt: off
            if self.data_type == "test":
                pylogger.info("Using test data for evaluation.")
                background_data = np.array(trainer.datamodule.tensor_test)[: self.num_jet_samples]  # noqa: E501
                background_mask = np.array(trainer.datamodule.mask_test)[: self.num_jet_samples]
                background_cond = np.array(trainer.datamodule.tensor_conditioning_test)[
                    : self.num_jet_samples
                ]
            elif self.data_type == "val":
                pylogger.info("Using validation data for evaluation.")
                background_data = np.array(trainer.datamodule.tensor_val)[: self.num_jet_samples]  # noqa: E501
                background_mask = np.array(trainer.datamodule.mask_val)[: self.num_jet_samples]
                background_cond = np.array(trainer.datamodule.tensor_conditioning_val)[
                    : self.num_jet_samples
                ]
            # fmt: on

            if trainer.datamodule.mask_gen is None:
                pylogger.info(
                    "No mask for generated data found. Using the same mask as for simulated data."
                )
                mask = background_mask
                cond = background_cond
            else:
                mask = trainer.datamodule.mask_gen
                cond = trainer.datamodule.tensor_conditioning_gen

            # maximum number of samples to plot is the number of samples in the dataset
            num_plot_samples = len(background_data)

            # Get EMA weights if available
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.replace_model_weights(pl_module)
            elif self.ema_callback and self.use_ema:
                warnings.warn("EMA Callback is not initialized. Using normal weights.")

            # Generate data
            data, generation_time = generate_data(
                model=pl_module,
                num_jet_samples=len(mask),
                cond=torch.tensor(cond),
                variable_set_sizes=trainer.datamodule.hparams.variable_jet_sizes,
                mask=torch.tensor(mask),
                normalized_data=trainer.datamodule.hparams.normalize,
                means=trainer.datamodule.means,
                stds=trainer.datamodule.stds,
                **self.generation_config,
            )
            pylogger.info(f"Generated {len(data)} samples in {generation_time:.0f} seconds.")

            # If there are multiple jet types, plot them separately
            if trainer.datamodule.names_conditioning is not None:
                jet_types_dict = {
                    var_name.split("_")[-1]: i
                    for i, var_name in enumerate(trainer.datamodule.names_conditioning)
                    if "jet_type" in var_name and np.sum(background_cond[:, i] == 1) > 0
                }
            else:
                jet_types_dict = {}
            pylogger.info(f"Used jet types: {jet_types_dict.keys()}")

            plot_path_part_features = (
                f"{self.image_path}/particle_features_epoch_{trainer.current_epoch}.pdf"
            )
            plot_path_part_features_png = plot_path_part_features.replace(".pdf", ".png")
            plot_particle_features(
                data_sim=background_data,
                data_gen=data,
                mask_sim=background_mask,
                mask_gen=mask,
                feature_names=trainer.datamodule.names_particle_features,
                legend_label_sim="JetClass",
                legend_label_gen="Generated",
                plot_path=plot_path_part_features,
                also_png=True,
            )

            # todo: combine this later on with the image logging further down
            # fmt: off
            if self.comet_logger is not None:
                self.comet_logger.log_image(plot_path_part_features_png, name=f"epoch{trainer.current_epoch}_particle_features")  # noqa: E501
            if self.wandb_logger is not None:
                self.wandb_logger.log({f"epoch{trainer.current_epoch}_particle_features": wandb.Image(plot_path_part_features_png)})  # noqa: E501

            substructure_full_path = f"{self.image_path}/substructure_epoch_{trainer.current_epoch}_gen"  # noqa: E501
            substructure_full_path_jetclass = substructure_full_path.replace("_gen", "_sim")  # noqa: E501
            # calculate substructure for generated data
            pylogger.info("Calculating substructure for generated data.")
            dump_hlvs(data, substructure_full_path, plot=False)
            # calculate substructure for reference data
            pylogger.info("Calculating substructure for JetClass data.")
            dump_hlvs(background_data, substructure_full_path_jetclass, plot=False)
            # fmt: on

            # load substructure data
            pylogger.info("Loading substructure data.")
            data_substructure, keys, tau21, tau32, d2 = load_substructure(
                substructure_full_path + ".h5"
            )
            (
                data_substructure_jetclass,
                keys_jetclass,
                tau21_jetclass,
                tau32_jetclass,
                d2_jetclass,
            ) = load_substructure(substructure_full_path_jetclass + ".h5")

            # ---------------------------------------------------------------
            pylogger.info("Calculating Wasserstein distances for substructure.")

            # Wasserstein distances
            # mass and particle features averaged
            w_dists = calculate_all_wasserstein_metrics(
                background_data, data, **self.w_dist_config
            )
            # substructure
            w_dist_config = {
                "num_eval_samples": self.w_dist_config["num_eval_samples"],
                "num_batches": self.w_dist_config["num_batches"],
            }
            w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
                tau21_jetclass, tau21, **w_dist_config
            )
            w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
                tau32_jetclass, tau32, **w_dist_config
            )
            w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(
                d2_jetclass, d2, **w_dist_config
            )
            self.log("w_dist_tau21_mean", w_dist_tau21_mean)
            self.log("w_dist_tau21_std", w_dist_tau21_std)
            self.log("w_dist_tau32_mean", w_dist_tau32_mean)
            self.log("w_dist_tau32_std", w_dist_tau32_std)
            self.log("w1m_mean", w_dists["w1m_mean"])
            self.log("w1p_mean", w_dists["w1p_mean"])
            self.log("w1m_std", w_dists["w1m_std"])
            self.log("w1p_std", w_dists["w1p_std"])

            if self.comet_logger is not None:
                text = (
                    f"W-Dist epoch:{trainer.current_epoch} "
                    f"W1m: {w_dists['w1m_mean']}+-{w_dists['w1m_std']}, "
                    f"W1p: {w_dists['w1p_mean']}+-{w_dists['w1p_std']}, "
                    f"W1efp: {w_dists['w1efp_mean']}+-{w_dists['w1efp_std']}"
                )
                self.comet_logger.log_text(text)

            for jet_type, jet_type_idx in jet_types_dict.items():
                jet_type_mask_sim = background_cond[:, jet_type_idx] == 1
                jet_type_mask_gen = cond[:, jet_type_idx] == 1
                path_part_feats_this_type = plot_path_part_features.replace(
                    ".pdf", f"_{jet_type}.pdf"
                )
                path_part_feats_this_type_png = path_part_feats_this_type.replace(".pdf", ".png")
                plot_particle_features(
                    data_sim=background_data[jet_type_mask_sim],
                    data_gen=data[jet_type_mask_gen],
                    mask_sim=background_mask[jet_type_mask_sim],
                    mask_gen=mask[jet_type_mask_gen],
                    feature_names=trainer.datamodule.names_particle_features,
                    legend_label_sim="JetClass",
                    legend_label_gen="Generated",
                    plot_path=path_part_feats_this_type,
                    also_png=True,
                )
                if self.comet_logger is not None:
                    self.comet_logger.log_image(
                        path_part_feats_this_type_png,
                        name=f"epoch{trainer.current_epoch}_particle_features_{jet_type}",
                    )  # noqa: E501
                if self.wandb_logger is not None:
                    self.wandb_logger.log(
                        {
                            f"epoch{trainer.current_epoch}_particle_features_{jet_type}": (
                                wandb.Image(path_part_feats_this_type_png)
                            )
                        }
                    )
                # calculate the wasserstein distances for this jet type
                pylogger.info(f"Calculating Wasserstein distances for {jet_type} jets.")
                w_dists_tt = calculate_all_wasserstein_metrics(
                    background_data[jet_type_mask_sim],
                    data[jet_type_mask_gen],
                    **self.w_dist_config,
                )
                w_dist_tau21_mean_tt, w_dist_tau21_std_tt = wasserstein_distance_batched(
                    tau21_jetclass[jet_type_mask_sim],
                    tau21[jet_type_mask_gen],
                    **w_dist_config,
                )
                w_dist_tau32_mean_tt, w_dist_tau32_std_tt = wasserstein_distance_batched(
                    tau32_jetclass[jet_type_mask_sim],
                    tau32[jet_type_mask_gen],
                    **w_dist_config,
                )
                w_dist_d2_mean_tt, w_dist_d2_std_tt = wasserstein_distance_batched(
                    d2_jetclass[jet_type_mask_sim],
                    d2[jet_type_mask_gen],
                    **w_dist_config,
                )
                self.log(f"w_dist_tau21_mean_{jet_type}", w_dist_tau21_mean_tt)
                self.log(f"w_dist_tau21_std_{jet_type}", w_dist_tau21_std_tt)
                self.log(f"w_dist_tau32_mean_{jet_type}", w_dist_tau32_mean_tt)
                self.log(f"w_dist_tau32_std_{jet_type}", w_dist_tau32_std_tt)
                self.log(f"w1m_mean_{jet_type}", w_dists_tt["w1m_mean"])
                self.log(f"w1p_mean_{jet_type}", w_dists_tt["w1p_mean"])
                self.log(f"w1m_std_{jet_type}", w_dists_tt["w1m_std"])
                self.log(f"w1p_std_{jet_type}", w_dists_tt["w1p_std"])

                # todo: plot substructure for different jet types and log them
                # plot substructure
                file_name_substructure = f"epoch{trainer.current_epoch}_subs_3plots_{jet_type}"
                file_name_full_substructure = f"epoch{trainer.current_epoch}_subs_full_{jet_type}"
                plot_substructure(
                    tau21=tau21[jet_type_mask_gen],
                    tau32=tau32[jet_type_mask_gen],
                    d2=d2[jet_type_mask_gen],
                    tau21_jetnet=tau21_jetclass[jet_type_mask_sim],
                    tau32_jetnet=tau32_jetclass[jet_type_mask_sim],
                    d2_jetnet=d2_jetclass[jet_type_mask_sim],
                    save_fig=True,
                    save_folder=self.image_path,
                    save_name=file_name_substructure,
                    close_fig=True,
                    simulation_name="JetClass",
                    model_name="Generated",
                )
                plot_full_substructure(
                    data_substructure=[
                        data_substructure[i][jet_type_mask_gen]
                        for i in range(len(data_substructure))
                    ],
                    data_substructure_jetnet=[
                        data_substructure_jetclass[i][jet_type_mask_sim]
                        for i in range(len(data_substructure_jetclass))
                    ],
                    keys=keys,
                    save_fig=True,
                    save_folder=self.image_path,
                    save_name=file_name_full_substructure,
                    close_fig=True,
                    simulation_name="JetClass",
                    model_name="Generated",
                )
                # upload image to comet
                img_path_3plots = f"{self.image_path}/{file_name_substructure}.png"
                img_path_full = f"{self.image_path}/{file_name_full_substructure}.png"
                if self.comet_logger is not None:
                    self.comet_logger.log_image(
                        img_path_3plots,
                        name=f"epoch{trainer.current_epoch}_substructure_3plots_{jet_type}",
                    )
                    self.comet_logger.log_image(
                        img_path_full,
                        name=f"epoch{trainer.current_epoch}_substructure_full_{jet_type}",
                    )

            # remove the additional particle features for compatibility with the rest of the code
            background_data = background_data[:, :, :3]
            data = data[:, :, :3]

            # Get normal weights back after sampling
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.restore_original_weights(pl_module)

            # Prepare Data for Plotting
            plot_prep_config = {
                "calculate_efps" if key == "plot_efps" else key: value
                for key, value in self.plot_config.items()
                if key in ["plot_efps", "selected_particles", "selected_multiplicities"]
            }

            (
                jet_data,
                efps_values,
                pt_selected_particles,
                pt_selected_multiplicities,
            ) = prepare_data_for_plotting(np.array([data]), **plot_prep_config)

            (
                jet_data_sim,
                efps_sim,
                pt_selected_particles_sim,
                pt_selected_multiplicities_sim,
            ) = prepare_data_for_plotting(
                [background_data],
                **plot_prep_config,
            )
            jet_data_sim, efps_sim, pt_selected_particles_sim = (
                jet_data_sim[0],
                efps_sim[0],
                pt_selected_particles_sim[0],
            )

            # Plotting
            plot_name = f"{self.model_name}_epoch{trainer.current_epoch}"
            _ = plot_data(
                particle_data=np.array([data]),
                sim_data=background_data,
                jet_data_sim=jet_data_sim,
                jet_data=jet_data,
                efps_sim=efps_sim,
                efps_values=efps_values,
                num_samples=num_plot_samples,
                pt_selected_particles=pt_selected_particles,
                pt_selected_multiplicities=pt_selected_multiplicities,
                pt_selected_particles_sim=pt_selected_particles_sim,
                pt_selected_multiplicities_sim=pt_selected_multiplicities_sim,
                save_fig=True,
                save_folder=self.image_path,
                save_name=plot_name,
                close_fig=True,
                **self.plot_config,
            )

            # Log plots
            img_path = f"{self.image_path}{plot_name}.png"
            if self.comet_logger is not None:
                self.comet_logger.log_image(img_path, name=f"epoch{trainer.current_epoch}")
            if self.wandb_logger is not None:
                self.wandb_logger.log({f"epoch{trainer.current_epoch}": wandb.Image(img_path)})

            time_eval_end = time.time()
            eval_time = time_eval_end - time_eval_start
            # Log jet generation time
            if self.log_times:
                if self.comet_logger is not None:
                    self.comet_logger.log_metrics({"Jet generation time": generation_time})
                    self.comet_logger.log_metrics({"Evaluation time": eval_time})
                if self.wandb_logger is not None:
                    self.wandb_logger.log({"Jet generation time": generation_time})
                    self.wandb_logger.log({"Evaluation time": eval_time})

        if self.fix_seed:
            torch.manual_seed(torch.seed())

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback
