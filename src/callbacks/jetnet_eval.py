import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from src.data.components import calculate_all_wasserstein_metrics
from src.data.components.utils import jet_masses
from src.schedulers.logging_scheduler import (
    custom1,
    custom5000epochs,
    custom10000epochs,
    epochs10000,
    nolog10000,
)
from src.utils import apply_mpl_styles, create_and_plot_data
from src.utils.data_generation import generate_data
from src.utils.pylogger import get_pylogger

from .ema import EMA

log = get_pylogger("JetNetEvaluationCallback")

# TODO wandb logging min and max values
# TODO wandb logging video of jets, histograms, and point clouds
# TODO fix efp logging
# ! High statistics are hardcoded, generation should be done in a separate function


class JetNetEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the JetNet dataset and log
    the results to loggers. Currently supported are CometLogger and WandbLogger.

    Args:
        every_n_epochs (int, optional): Log every n epochs. Defaults to 10.
        num_jet_samples (int, optional): How many jet samples to generate. Negative values define the amount of times the whole dataset is taken, e.g. -2 would use 2*len(dataset) samples. Defaults to -1.
        w_dists_batches (int, optional): How many batches to calculate Wasserstein distances. Jet samples for each batch are num_jet_samples // w_dists_batches. Defaults to 5.
        image_path (str, optional): Folder where the images are saved. Defaults to "/beegfs/desy/user/ewencedr/comet_logs".
        model_name (str, optional): Name for saving the model. Defaults to "model-test".
        calculate_efps (bool, optional): Calculate EFPs for the jets. Defaults to False.
        log_w_dists (bool, optional): Calculate and log wasserstein distances Defaults to False.
        log_times (bool, optional): Log generation times of data. Defaults to True.
        log_epoch_zero (bool, optional): Log in first epoch. Default to False.
        data_type (str, optional): Type of data to plot. Options are 'test' and 'val'. Defaults to "test".
        use_ema (bool, optional): Use exponential moving average weights for logging. Defaults to False.
        **kwargs: Arguments for create_and_plot_data
    """

    def __init__(
        self,
        every_n_epochs: int | Callable = 10,
        num_jet_samples: int = -1,
        w_dists_batches: int = 5,
        image_path: str = "./logs/callback_images/",
        model_name: str = "model",
        calculate_efps: bool = False,
        log_w_dists: bool = False,
        log_times: bool = True,
        log_epoch_zero: bool = False,
        data_type: str = "val",
        use_ema: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_jet_samples = -5  # ! FIX
        self.w_dists_batches = w_dists_batches
        self.log_w_dists = log_w_dists
        self.log_times = log_times
        self.log_epoch_zero = log_epoch_zero
        self.use_ema = use_ema

        # Parameters for plotting
        self.model_name = model_name
        self.calculate_efps = calculate_efps
        self.data_type = data_type
        self.kwargs = kwargs

        self.image_path = image_path
        apply_mpl_styles()

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
        self.log("w1m_mean_1b", 0.005)
        self.log("w1p_mean_1b", 0.005)

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
                "JetNet Evaluation Callbacks was told to use EMA weights, but EMA callback was not found. Using normal weights."
            )
        elif self.ema_callback is not None and self.use_ema:
            log.info("Using EMA weights for logging.")

    def on_train_epoch_end(self, trainer, pl_module):
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
                background_data = np.array(trainer.datamodule.tensor_test)
                background_mask = np.array(trainer.datamodule.mask_test)
                background_cond = np.array(trainer.datamodule.tensor_conditioning_test)
            elif self.data_type == "val":
                background_data = np.array(trainer.datamodule.tensor_val)
                background_mask = np.array(trainer.datamodule.mask_val)
                background_cond = np.array(trainer.datamodule.tensor_conditioning_val)
            # if self.datasets_multiplier > 1:
            #    background_data = np.repeat(background_data, self.datasets_multiplier, axis=0)
            #    background_mask = np.repeat(background_mask, self.datasets_multiplier, axis=0)
            big_mask = np.repeat(background_mask, 5, axis=0)
            big_data = np.repeat(background_data, 5, axis=0)
            big_cond = np.repeat(background_cond, 5, axis=0)

            plot_name = f"{self.model_name}--epoch{trainer.current_epoch}"

            # Get EMA weights if available
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.replace_model_weights(pl_module)
            elif self.ema_callback and self.use_ema:
                warnings.warn("EMA Callback is not initialized. Using normal weights.")

            # Maximum number of samples to plot is the number of samples in the dataset
            # if self.datasets_multiplier == -1:
            #    if self.data_type == "test":
            #        background_data_plot = background_data[: len(trainer.datamodule.tensor_test)]
            #        background_mask_plot = background_mask[: len(trainer.datamodule.mask_test)]
            #        jet_samples_plot = len(trainer.datamodule.tensor_test)
            #    if self.data_type == "val":
            #        background_data_plot = background_data[: len(trainer.datamodule.tensor_val)]
            #        background_mask_plot = background_mask[: len(trainer.datamodule.mask_val)]
            #        jet_samples_plot = len(trainer.datamodule.tensor_val)
            # else:
            # background_data_plot = background_data
            # background_mask_plot = background_mask
            # jet_samples_plot = self.num_jet_samples

            # fig, particle_data, times = create_and_plot_data(
            #    background_data_plot,
            #    [pl_module],
            #    cond=cond,
            #    save_name=plot_name,
            #    labels=["Model"],
            #    normalized_data=[trainer.datamodule.hparams.normalize],
            #    normalize_sigma=trainer.datamodule.hparams.normalize_sigma,
            #    variable_set_sizes=trainer.datamodule.hparams.variable_jet_sizes,
            #    mask=background_mask_plot,
            #    num_jet_samples=jet_samples_plot,
            #    means=trainer.datamodule.means,
            #    stds=trainer.datamodule.stds,
            #    save_folder=self.image_path,
            #    print_parameters=False,
            #    plot_efps=self.calculate_efps,
            #    close_fig=True,
            #    **self.kwargs,
            # )
            print(f"big-cond: {big_cond.shape}")
            data, generation_time = generate_data(
                model=pl_module,
                num_jet_samples=5 * len(background_mask),
                batch_size=256,
                cond=torch.tensor(big_cond),
                variable_set_sizes=True,
                mask=torch.tensor(big_mask),
                normalized_data=trainer.datamodule.hparams.normalize,
                means=trainer.datamodule.means,
                stds=trainer.datamodule.stds,
                ode_solver="midpoint",
                ode_steps=100,
            )

            # Get normal weights back after sampling
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.restore_original_weights(pl_module)

            w_dists_big = calculate_all_wasserstein_metrics(
                background_data[..., :3],
                data,
                None,
                None,
                num_eval_samples=len(background_data),
                num_batches=5,
                calculate_efps=False,
                use_masks=False,
            )
            self.log("w1m_mean_1b", w_dists_big["w1m_mean"])
            self.log("w1p_mean_1b", w_dists_big["w1p_mean"])

            # particle_data = particle_data[0]
            # mask_data = (particle_data[..., 0] == 0).astype(int)
            # mask_data = np.expand_dims(mask_data, axis=-1)
            # mask_data = 1 - mask_data

            # if self.log_w_dists:
            #    # 1 batch
            #    # ! Do this properly
            #    w_dists_1b_temp = calculate_all_wasserstein_metrics(
            #        background_data[: len(trainer.datamodule.tensor_val), :, :3],
            #        particle_data,
            #        None,
            #        None,
            #        num_eval_samples=len(trainer.datamodule.tensor_val),
            #        num_batches=self.datasets_multiplier,
            #        calculate_efps=True,
            #        use_masks=False,
            #    )
            #    # create new dict with _1b suffix to not log the same values twice
            #    w_dists_1b = {}
            #    for key, value in w_dists_1b_temp.items():
            #        w_dists_1b[key + "_1b"] = value
            #
            #    # divide into batches
            #    w_dists = calculate_all_wasserstein_metrics(
            #        background_data[: len(particle_data), :, :3],
            #        particle_data,
            #        background_mask[: len(particle_data)],
            #        mask_data,
            #        num_eval_samples=self.num_jet_samples // self.w_dists_batches,
            #        num_batches=self.w_dists_batches,
            #        calculate_efps=self.calculate_efps,
            #        use_masks=False,
            #    )
            #
            #    # Wasserstein Metrics
            #    text = f"W-Dist epoch:{trainer.current_epoch} W1m: {w_dists['w1m_mean']}+-{w_dists['w1m_std']}, W1p: {w_dists['w1p_mean']}+-{w_dists['w1p_std']}, W1efp: {w_dists['w1efp_mean']}+-{w_dists['w1efp_std']}"
            #    text_1b = f"1 BATCH W-Dist epoch:{trainer.current_epoch} W1m: {w_dists_1b['w1m_mean_1b']}+-{w_dists_1b['w1m_std_1b']}, W1p: {w_dists_1b['w1p_mean_1b']}+-{w_dists_1b['w1p_std_1b']}, W1efp: {w_dists_1b['w1efp_mean_1b']}+-{w_dists_1b['w1efp_std_1b']}"
            #    if self.comet_logger is not None:
            #        self.comet_logger.log_text(text)
            #        self.comet_logger.log_metrics(w_dists)
            #        self.comet_logger.log_text(text_1b)
            #        self.comet_logger.log_metrics(w_dists_1b)
            #    if self.wandb_logger is not None:
            #        self.wandb_logger.log({"Wasserstein Metrics": w_dists})
            #        self.wandb_logger.log({"Wasserstein Metrics 1b": w_dists_1b})
            #    self.log("w1m_mean_1b", w_dists_1b["w1m_mean_1b"])
            #    self.log("w1p_mean_1b", w_dists_1b["w1p_mean_1b"])

            # Jet genereation time
            # if self.log_times:
            #    if self.comet_logger is not None:
            #        self.comet_logger.log_metrics({"Jet generation time": times})
            #    if self.wandb_logger is not None:
            #        self.wandb_logger.log({"Jet generation time": times})

            # Histogram Plots
            # img_path = f"{self.image_path}{plot_name}.png"
            # if self.comet_logger is not None:
            #    self.comet_logger.log_image(img_path, name=f"epoch{trainer.current_epoch}")
            # if self.wandb_logger is not None:
            #    self.wandb_logger.log({f"epoch{trainer.current_epoch}": wandb.Image(img_path)})

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback
