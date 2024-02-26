import warnings
from typing import Callable, Mapping, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from src.data.components.metrics import (
    calculate_all_wasserstein_metrics,
    calculate_wasserstein_metrics_jets,
)
from src.schedulers.logging_scheduler import (
    custom1,
    custom5000epochs,
    custom10000epochs,
    epochs10000,
    nolog10000,
)
from src.utils.data_generation import generate_data
from src.utils.lhco_utils import cluster_data, plot_unprocessed_data_lhco
from src.utils.plotting import apply_mpl_styles, plot_data, prepare_data_for_plotting
from src.utils.pylogger import get_pylogger

from .ema import EMA

log = get_pylogger("LHCOEvaluationCallback")


class LHCOEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the LHCO dataset when using the more complex datastructures where clustering is required and log
    the results to loggers. Currently supported are CometLogger and WandbLogger.

    Args:
        every_n_epochs (int, optional): Log every n epochs. Defaults to 10.
        num_jet_samples (int, optional): How many jet samples to generate. Negative values define the amount of times the whole dataset is taken, e.g. -2 would use 2*len(dataset) samples. Defaults to -1.
        image_path (str, optional): Folder where the images are saved. Defaults to "/beegfs/desy/user/ewencedr/comet_logs".
        model_name (str, optional): Name for saving the model. Defaults to "model-test".
        log_times (bool, optional): Log generation times of data. Defaults to True.
        log_epoch_zero (bool, optional): Log in first epoch. Default to False.
        data_type (str, optional): Type of data to plot. Options are 'test' and 'val'. Defaults to "test".
        use_ema (bool, optional): Use exponential moving average weights for logging. Defaults to False.
        fix_seed (bool, optional): Fix seed for data generation to have better reproducibility and comparability between epochs. Defaults to True.
        w_dist_config (Mapping, optional): Configuration for Wasserstein distance calculation. Defaults to {'num_jet_samples': 10_000, 'num_batches': 40}.
        generation_config (Mapping, optional): Configuration for data generation. Defaults to {"batch_size": 256, "ode_solver": "midpoint", "ode_steps": 100}.
        plot_config (Mapping, optional): Configuration for plotting. Defaults to {}.
    """

    def __init__(
        self,
        every_n_epochs: int | Callable = 10,
        num_jet_samples: int = -1,
        image_path: str = "./logs/callback_images/",
        model_name: str = "model",
        log_times: bool = True,
        log_epoch_zero: bool = False,
        data_type: str = "val",
        use_ema: bool = False,
        fix_seed: bool = True,
        w_dist_config: Mapping = {
            "num_eval_samples": 50_000,
            "num_batches": 40,
            "calculate_efps": False,
        },
        generation_config: Mapping = {
            "batch_size": 2048,
            "ode_solver": "midpoint",
            "ode_steps": 100,
        },
        plot_config: Mapping = {"plot_efps": False, "plottype": "", "plot_jet_features": True},
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
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
        self.log("w1m_mean", 0.005, sync_dist=True)
        self.log("w1p_mean", 0.005, sync_dist=True)

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

        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        # get ema callback
        self.ema_callback = self._get_ema_callback(trainer)
        if self.ema_callback is None and self.use_ema:
            warnings.warn(
                "JetNet Evaluation Callbacks was told to use EMA weights, but EMA callback was not"
                " found. Using normal weights."
            )
        elif self.ema_callback is not None and self.use_ema:
            log.info("Using EMA weights for logging.")

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

        if log:
            # Get background data for plotting and calculating Wasserstein distances
            if self.data_type == "test":
                background_data = np.array(trainer.datamodule.tensor_test)[: self.num_jet_samples]
                background_mask = np.array(trainer.datamodule.mask_test)[: self.num_jet_samples]
                background_cond = np.array(trainer.datamodule.tensor_conditioning_test)[
                    : self.num_jet_samples
                ]
            elif self.data_type == "val":
                background_data = np.array(trainer.datamodule.tensor_val)[: self.num_jet_samples]
                background_mask = np.array(trainer.datamodule.mask_val)[: self.num_jet_samples]
                background_cond = np.array(trainer.datamodule.tensor_conditioning_val)[
                    : self.num_jet_samples
                ]

            mask = background_mask
            cond = background_cond

            # maximum number of samples to plot is the number of samples in the dataset
            num_plot_samples = len(background_data)

            if self.datasets_multiplier > 1:
                mask = np.repeat(mask, self.datasets_multiplier, axis=0)
                cond = np.repeat(cond, self.datasets_multiplier, axis=0)

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
                normalize_sigma=trainer.datamodule.hparams.normalize_sigma,
                means=trainer.datamodule.means,
                stds=trainer.datamodule.stds,
                log_pt=trainer.datamodule.hparams.log_pt,
                pt_standardization=trainer.datamodule.hparams.pt_standardization,
                **self.generation_config,
            )

            # Get normal weights back after sampling
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.restore_original_weights(pl_module)

            # go to pt, eta, phi format for clustering
            data_full = data[..., [2, 0, 1]]
            background_data = background_data[..., [2, 0, 1]]

            plot_name1 = f"{self.model_name}--epoch{trainer.current_epoch}_unprocessed_data_lhco"
            plot_unprocessed_data_lhco(
                particle_data=np.array([data_full]),
                sim_data=background_data,
                num_samples=num_plot_samples,
                plottype="",
                save_fig=True,
                save_folder=self.image_path,
                save_name=plot_name1,
                close_fig=True,
            )

            # Clustering

            consts, max_consts_gen = cluster_data(
                data_full,
                max_jets=2,
                max_consts=trainer.datamodule.hparams.num_particles,
                return_max_consts_gen=True,
            )

            # calculate mask for jet constituents
            mask = np.expand_dims((consts[..., 0] > 0).astype(int), axis=-1)

            x_consts = consts[:, 0]
            y_consts = consts[:, 1]

            # compare to originally clustered data
            path = "/beegfs/desy/user/ewencedr/data/lhco/final_data/processed_data_background.h5"
            with h5py.File(path, "r") as f:
                jet_data = f["jet_data"][:]
                particle_data = f["constituents"][:]
                mask_ref = f["mask"][:]

            x_consts_ref = particle_data[: self.num_jet_samples, 0]
            y_consts_ref = particle_data[: self.num_jet_samples, 1]

            # X-Jets

            data = x_consts[..., [1, 2, 0]]
            background_data = x_consts_ref[..., [1, 2, 0]]

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
            ) = prepare_data_for_plotting(
                np.array([data]),
                **plot_prep_config,
            )

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

            w_dists_x = calculate_all_wasserstein_metrics(
                background_data, data, **self.w_dist_config
            )
            w_dists_jet_x = calculate_wasserstein_metrics_jets(
                jet_data_sim, jet_data[0], **self.w_dist_config
            )

            plot_name_x = f"{self.model_name}--epoch{trainer.current_epoch}_clustered_x_jets"
            plot_data(
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
                save_name=plot_name_x,
                close_fig=True,
                **self.plot_config,
            )

            # Y-Jets

            data_y = y_consts[..., [1, 2, 0]]
            background_data_y = y_consts_ref[..., [1, 2, 0]]

            (
                jet_data_y,
                efps_values_y,
                pt_selected_particles_y,
                pt_selected_multiplicities_y,
            ) = prepare_data_for_plotting(
                np.array([data_y]),
                **plot_prep_config,
            )

            (
                jet_data_sim_y,
                efps_sim_y,
                pt_selected_particles_sim_y,
                pt_selected_multiplicities_sim_y,
            ) = prepare_data_for_plotting(
                [background_data_y],
                **plot_prep_config,
            )
            jet_data_sim_y, efps_sim_y, pt_selected_particles_sim_y = (
                jet_data_sim_y[0],
                efps_sim_y[0],
                pt_selected_particles_sim_y[0],
            )

            w_dists_y = calculate_all_wasserstein_metrics(
                background_data_y, data_y, **self.w_dist_config
            )
            w_dists_jet_y = calculate_wasserstein_metrics_jets(
                jet_data_sim_y, jet_data_y[0], **self.w_dist_config
            )

            plot_name_y = f"{self.model_name}--epoch{trainer.current_epoch}_clustered_y_jets"
            plot_data(
                particle_data=np.array([data_y]),
                sim_data=background_data_y,
                jet_data_sim=jet_data_sim_y,
                jet_data=jet_data_y,
                efps_sim=efps_sim_y,
                efps_values=efps_values_y,
                num_samples=num_plot_samples,
                pt_selected_particles=pt_selected_particles_y,
                pt_selected_multiplicities=pt_selected_multiplicities_y,
                pt_selected_particles_sim=pt_selected_particles_sim_y,
                pt_selected_multiplicities_sim=pt_selected_multiplicities_sim_y,
                save_fig=True,
                save_folder=self.image_path,
                save_name=plot_name_y,
                close_fig=True,
                **self.plot_config,
            )

            self.log("w1m_mean", w_dists_x["w1m_mean"], sync_dist=True)
            self.log("w1p_mean", w_dists_x["w1p_mean"], sync_dist=True)
            self.log("w1m_std", w_dists_x["w1m_std"], sync_dist=True)
            self.log("w1p_std", w_dists_x["w1p_std"], sync_dist=True)

            self.log("w1m_mean_y", w_dists_y["w1m_mean"], sync_dist=True)
            self.log("w1p_mean_y", w_dists_y["w1p_mean"], sync_dist=True)
            self.log("w1m_std_y", w_dists_y["w1m_std"], sync_dist=True)
            self.log("w1p_std_y", w_dists_y["w1p_std"], sync_dist=True)

            self.log("w1pt_jet_mean", w_dists_jet_x["w1pt_jet_mean"], sync_dist=True)
            self.log("w1pt_jet_std", w_dists_jet_x["w1pt_jet_std"], sync_dist=True)
            self.log("w1eta_jet_mean", w_dists_jet_x["w1eta_jet_mean"], sync_dist=True)
            self.log("w1eta_jet_std", w_dists_jet_x["w1eta_jet_std"], sync_dist=True)
            self.log("w1phi_jet_mean", w_dists_jet_x["w1phi_jet_mean"], sync_dist=True)
            self.log("w1phi_jet_std", w_dists_jet_x["w1phi_jet_std"], sync_dist=True)
            self.log("w1mass_jet_mean", w_dists_jet_x["w1mass_jet_mean"], sync_dist=True)
            self.log("w1mass_jet_std", w_dists_jet_x["w1mass_jet_std"], sync_dist=True)

            self.log("w1pt_jet_mean_y", w_dists_jet_y["w1pt_jet_mean"], sync_dist=True)
            self.log("w1pt_jet_std_y", w_dists_jet_y["w1pt_jet_std"], sync_dist=True)
            self.log("w1eta_jet_mean_y", w_dists_jet_y["w1eta_jet_mean"], sync_dist=True)
            self.log("w1eta_jet_std_y", w_dists_jet_y["w1eta_jet_std"], sync_dist=True)
            self.log("w1phi_jet_mean_y", w_dists_jet_y["w1phi_jet_mean"], sync_dist=True)
            self.log("w1phi_jet_std_y", w_dists_jet_y["w1phi_jet_std"], sync_dist=True)
            self.log("w1mass_jet_mean_y", w_dists_jet_y["w1mass_jet_mean"], sync_dist=True)
            self.log("w1mass_jet_std_y", w_dists_jet_y["w1mass_jet_std"], sync_dist=True)

            self.log("max_consts_gen", float(max_consts_gen), sync_dist=True)

            # Log plots
            img_path1 = f"{self.image_path}{plot_name1}.png"
            img_path_x = f"{self.image_path}{plot_name_x}.png"
            img_path_y = f"{self.image_path}{plot_name_y}.png"
            if self.comet_logger is not None:
                self.comet_logger.log_image(
                    img_path1, name=f"epoch{trainer.current_epoch}_unprocessed"
                )
                self.comet_logger.log_image(img_path_x, name=f"epoch{trainer.current_epoch}_x")
                self.comet_logger.log_image(img_path_y, name=f"epoch{trainer.current_epoch}_y")
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {f"epoch{trainer.current_epoch}_unprocessed": wandb.Image(img_path1)}
                )
                self.wandb_logger.log({f"epoch{trainer.current_epoch}_x": wandb.Image(img_path_x)})
                self.wandb_logger.log({f"epoch{trainer.current_epoch}_y": wandb.Image(img_path_y)})

            # Log jet generation time
            if self.log_times:
                if self.comet_logger is not None:
                    self.comet_logger.log_metrics({"Jet generation time": generation_time})
                if self.wandb_logger is not None:
                    self.wandb_logger.log({"Jet generation time": generation_time})

        if self.fix_seed:
            torch.manual_seed(torch.seed())

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback
