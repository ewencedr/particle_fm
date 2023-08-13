import warnings
from typing import Callable, Mapping, Optional

import awkward as ak
import energyflow as ef
import fastjet as fj
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import vector
import wandb

from src.callbacks.ema import EMA
from src.data.components import inverse_normalize_tensor
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
from src.utils.lhco_utils import plot_unprocessed_data_lhco, sort_by_pt
from src.utils.plotting import apply_mpl_styles, plot_data, prepare_data_for_plotting
from src.utils.pylogger import get_pylogger

log = get_pylogger("LHCOEvaluationCallback")

# TODO wandb logging min and max values
# TODO wandb logging video of jets, histograms, and point clouds
# TODO fix efp logging
# TODO use ema can be taken from ema callback and should be removed here


class LHCOJetFeaturesEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the LHCO dataset and log the
    results to loggers. Currently supported are CometLogger and WandbLogger.

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
            "verbose": False,
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
        self.log("w1m_mean", 0.005)
        self.log("w1p_mean", 0.005)

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
                background_cond = np.array(trainer.datamodule.tensor_conditioning_test)[
                    : self.num_jet_samples
                ]
            elif self.data_type == "val":
                background_data = np.array(trainer.datamodule.tensor_val)[: self.num_jet_samples]
                background_cond = np.array(trainer.datamodule.tensor_conditioning_val)[
                    : self.num_jet_samples
                ]

            cond = background_cond
            # maximum number of samples to plot is the number of samples in the dataset
            num_plot_samples = len(background_data)

            if self.datasets_multiplier > 1:
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
                num_jet_samples=len(cond),
                cond=torch.tensor(cond),
                normalized_data=trainer.datamodule.hparams.normalize,
                means=trainer.datamodule.means,
                stds=trainer.datamodule.stds,
                **self.generation_config,
            )

            # Get normal weights back after sampling
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.restore_original_weights(pl_module)

            # de-standardize conditioning
            cond_true = inverse_normalize_tensor(
                torch.tensor(cond),
                trainer.datamodule.cond_means,
                trainer.datamodule.cond_stds,
                trainer.datamodule.hparams.normalize_sigma,
            )
            cond_true = cond_true.numpy()

            # Calculate mjj
            p4_x_jet = ef.p4s_from_ptyphims(data[:, :4])
            p4_y_jet = ef.p4s_from_ptyphims(data[:, 4:])
            # get mjj from p4_jets
            pj_x = np.sqrt(np.sum(p4_x_jet**2, axis=1))
            pj_y = np.sqrt(np.sum(p4_y_jet**2, axis=1))
            mjj = (pj_x + pj_y) ** 2

            fig, axs = plt.subplots()
            hist = axs.hist(
                cond_true,
                bins=np.arange(0.005e8, 1.8e8, 0.005e8),
                histtype="stepfilled",
                label="train data",
                alpha=0.5,
            )
            axs.hist(mjj, bins=hist[1], histtype="step", label="generated")
            axs.set_xlabel(r"$m_{jj}$")
            axs.legend()
            plt.tight_layout()
            plot_name_mjj = "_lhco_jet_features_mjj"
            plt.savefig(f"{self.image_path}{plot_name_mjj}.png")
            plt.close()

            # Compare generated data to background data

            label_map = {
                "0": r"${p_T}_1$",
                "1": r"$\eta_1$",
                "2": r"$\phi_1$",
                "3": r"$m_1$",
                "4": r"${p_T}_2$",
                "5": r"$\eta_2$",
                "6": r"$\phi_2$",
                "7": r"$m_2$",
            }
            fig, axs = plt.subplots(2, 4, figsize=(15, 10))
            for index, ax in enumerate(axs.reshape(-1)):
                x_min, x_max = min(np.min(background_data[:, index]), np.min(data[:, index])), max(
                    np.max(background_data[:, index]), np.max(data[:, index])
                )
                hist1 = ax.hist(
                    background_data[:, index],
                    bins=100,
                    label="train data",
                    range=[x_min, x_max],
                    alpha=0.5,
                )
                ax.hist(data[:, index], bins=hist1[1], label="generated", histtype="step")
                ax.set_xlabel(f"{label_map[str(index)]}")
                ax.set_yscale("log")
                if index == 2 or index == 6:
                    ax.legend(frameon=False)
                    ax.set_ylim(1e-1, 1e6)
            plt.tight_layout()
            plot_name = "_lhco_jet_features"
            plt.savefig(f"{self.image_path}{plot_name}.png")
            plt.close()

            # Log plots
            img_path = f"{self.image_path}{plot_name}.png"
            img_path_mjj = f"{self.image_path}{plot_name_mjj}.png"
            if self.comet_logger is not None:
                self.comet_logger.log_image(img_path, name=f"epoch{trainer.current_epoch}")
                self.comet_logger.log_image(img_path_mjj, name=f"epoch{trainer.current_epoch}_mjj")
            if self.wandb_logger is not None:
                self.wandb_logger.log({f"epoch{trainer.current_epoch}": wandb.Image(img_path)})
                self.wandb_logger.log(
                    {f"epoch{trainer.current_epoch}_mjj": wandb.Image(img_path_mjj)}
                )

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
