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

from particle_fm.callbacks.ema import EMA
from particle_fm.data.components import inverse_normalize_tensor, normalize_tensor
from particle_fm.data.components.metrics import (
    calculate_all_wasserstein_metrics,
    calculate_wasserstein_metrics_jets,
)
from particle_fm.schedulers.logging_scheduler import (
    custom1,
    custom5000epochs,
    custom10000epochs,
    epochs10000,
    nolog10000,
)
from particle_fm.utils.data_generation import generate_data, generate_data_v2
from particle_fm.utils.lhco_utils import plot_unprocessed_data_lhco, sort_by_pt
from particle_fm.utils.plotting import (
    apply_mpl_styles,
    plot_data,
    prepare_data_for_plotting,
)
from particle_fm.utils.pylogger import get_pylogger

log = get_pylogger("LHCOEvaluationCallback")

# TODO wandb logging min and max values
# TODO wandb logging video of jets, histograms, and point clouds
# TODO fix efp logging
# TODO use ema can be taken from ema callback and should be removed here


class GenChallengeEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the GenChallenge and log the
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
            data, generation_time = generate_data_v2(
                model=pl_module,
                num_jet_samples=len(cond),
                cond=torch.tensor(cond),
                preprocessing_pipeline=trainer.datamodule.preprocessing_pipeline,
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

            # reshape in case of set data
            data = np.reshape(data, (data.shape[0], -1))
            background_data = np.reshape(background_data, (background_data.shape[0], -1))

            # Calculate mjj
            # p4_x_jet = ef.p4s_from_ptyphims(data[:, 0:4])
            # p4_y_jet = ef.p4s_from_ptyphims(data[:, 5:9])
            # get mjj from p4_jets
            # sum_p4 = p4_x_jet + p4_y_jet
            # mjj = ef.ms_from_p4s(sum_p4)

            # fig, axs = plt.subplots()
            # hist = axs.hist(
            #    cond_true,
            #    bins=np.arange(1e3, 9.5e3, 0.1e3),
            #    histtype="stepfilled",
            #    label="train data",
            #    alpha=0.5,
            # )
            # axs.hist(mjj, bins=hist[1], histtype="step", label="generated")
            # axs.set_xlabel(r"$m_{jj}$ [GeV]")
            # axs.set_yscale("log")
            # axs.legend(frameon=False)
            # plt.tight_layout()
            # plot_name_mjj = "_lhco_jet_features_mjj"
            # plt.savefig(f"{self.image_path}{plot_name_mjj}.png")
            # plt.close()

            # Compare generated data to background data

            samples = np.concatenate([cond, data], axis=1)
            bckg_data = np.concatenate([background_cond, background_data], axis=1)

            label_map = {
                "0": r"$m_{jj}$",
                "1": r"$m_{J_1}$",
                "2": r"$\Delta m_J$",
                "3": r"$\tau_{41}^{J_1}$",
                "4": r"$\tau_{41}^{J_2}$",
            }
            fig, ax = plt.subplots(2, 5, figsize=(35, 8), gridspec_kw={"height_ratios": [3, 1]})
            for i in range(5):
                hist_data = ax[0, i].hist(
                    bckg_data[:, i],
                    bins=100,
                    label="data background",
                    density=True,
                    histtype="stepfilled",
                )
                binning = hist_data[1]
                hist_samples = ax[0, i].hist(
                    samples[:, i],
                    bins=binning,
                    label="sampled background",
                    density=True,
                    histtype="step",
                )
                data_hist = np.histogram(bckg_data[:, i], bins=binning, density=False)[0]
                sample_hist = np.histogram(samples[:, i], bins=binning, density=False)[0]

                data_scale_factor = np.sum(data_hist) * np.diff(binning)
                sample_scale_factor = np.sum(sample_hist) * np.diff(binning)
                if i == 2:
                    ax[0, i].legend(loc="best", frameon=False)
                ax[0, i].set_xlabel(f"{label_map[str(i)]}")
                ax[0, i].set_yscale("log")
                with np.errstate(divide="ignore", invalid="ignore"):
                    ax[1, i].axhline(1.0, color="black", linestyle="-", alpha=0.8)
                    next(ax[1, i]._get_lines.prop_cycler)
                    ax[1, i].errorbar(
                        0.5 * (binning[:-1] + binning[1:]),
                        data_hist / sample_hist * sample_scale_factor / data_scale_factor,
                        linestyle="none",
                        marker=".",
                        yerr=np.sqrt(data_hist)
                        / sample_hist
                        * sample_scale_factor
                        / data_scale_factor,
                    )
                    ax[1, i].set_ylim(0.85, 1.15)
                    # ax[1,i].set_ylim(0.3, 1.7)

                if i == 0:
                    ax[0, i].set_ylabel("Events (norm.)")
                    ax[1, i].set_ylabel("Data/Sample (norm.)")
            plt.tight_layout()
            plot_name = "_lhco_jet_features"
            plt.savefig(f"{self.image_path}{plot_name}.png")
            plt.close()

            # Log plots
            img_path = f"{self.image_path}{plot_name}.png"
            # img_path_mjj_cond_sr = f"{self.image_path}{plot_name_mjj_cond_sr}.png"
            if self.comet_logger is not None:
                self.comet_logger.log_image(img_path, name=f"epoch{trainer.current_epoch}")
            if self.wandb_logger is not None:
                self.wandb_logger.log({f"epoch{trainer.current_epoch}": wandb.Image(img_path)})

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
