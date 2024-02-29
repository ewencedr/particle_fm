import traceback
import warnings
from typing import Any, Callable, Dict, Mapping, Optional

import hist
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from particle_fm.data.components import calculate_all_wasserstein_metrics
from particle_fm.schedulers.logging_scheduler import (
    custom1,
    custom5000epochs,
    custom10000epochs,
    epochs10000,
    nolog10000,
)
from particle_fm.utils.calo_challenge_utils import (
    generate_data_calochallenge,
    plotting_point_cloud,
)
from particle_fm.utils.data_generation import generate_data
from particle_fm.utils.plotting import (
    apply_mpl_styles,
    plot_data,
    prepare_data_for_plotting,
)
from particle_fm.utils.pylogger import get_pylogger

from ..ema import EMA

log = get_pylogger("CaloChallengeEvaluationCallback")


class CaloChallengeEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the CaloChallenge dataset and
    log the results to loggers. Currently supported are CometLogger and WandbLogger.

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

        self.log("w1p_mean", 0.005)

        # set number of jet samples if negative

        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.experiment.log_code(".")

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

        if log:
            bins = [100, 45, 16, 9]
            # Generate data
            hists = {}
            hists["hists_real"] = []
            hists["hists_fake"] = []
            hists["hists_real_unscaled"] = []
            hists["hists_fake_unscaled"] = []

            hists["weighted_hists_real"] = []
            hists["weighted_hists_fake"] = []
            hists["response_real"] = hist.Hist(hist.axis.Regular(100, 0.6, 1.1))
            hists["response_fake"] = hist.Hist(hist.axis.Regular(100, 0.6, 1.1))

            hists["hists_real_unscaled"].append(hist.Hist(hist.axis.Regular(100, 0, 6500)))
            hists["hists_fake_unscaled"].append(hist.Hist(hist.axis.Regular(100, 0, 6500)))

            hists["hists_real"].append(hist.Hist(hist.axis.Regular(100, 0, 6500)))
            hists["hists_fake"].append(hist.Hist(hist.axis.Regular(100, 0, 6500)))

            for n in bins[1:]:
                hists["hists_real"].append(hist.Hist(hist.axis.Integer(0, n)))
                hists["hists_fake"].append(hist.Hist(hist.axis.Integer(0, n)))
                hists["hists_real_unscaled"].append(hist.Hist(hist.axis.Regular(100, -10, 10)))
                hists["hists_fake_unscaled"].append(hist.Hist(hist.axis.Regular(100, -10, 10)))
            for n in bins[1:]:
                hists["weighted_hists_real"].append(hist.Hist(hist.axis.Integer(0, n)))
                hists["weighted_hists_fake"].append(hist.Hist(hist.axis.Integer(0, n)))
            hists, generation_time = generate_data_calochallenge(
                model=pl_module,
                dl=trainer.datamodule.val_dataloader(),
                scaler=trainer.datamodule.scaler,
                hists=hists,
                **self.generation_config,
            )

            w1ps = []
            weighted_w1ps = []
            plot = False
            if not hasattr(self, "min_w1p") or not hasattr(self, "min_z"):
                self.min_w1p = 10
                self.min_z = 0.01
            bins = [100, 45, 16, 9]
            names = ["E", "z", "alpha", "R"]
            (
                hists_fake,
                hists_real,
                weighted_hists_fake,
                weighted_hists_real,
                response_fake,
                response_real,
                hists_fake_unscaled,
                hists_real_unscaled,
            ) = (
                hists["hists_fake"],
                hists["hists_real"],
                hists["weighted_hists_fake"],
                hists["weighted_hists_real"],
                hists["response_fake"],
                hists["response_real"],
                hists["hists_fake_unscaled"],
                hists["hists_real_unscaled"],
            )
            for i in range(4):
                cdf_fake = hists_fake[i].values().cumsum().astype(float)
                cdf_real = hists_real[i].values().cumsum().astype(float)
                cdf_fake /= float(cdf_fake[-1])
                cdf_real /= float(cdf_real[-1])
                w1p = np.mean(np.abs(cdf_fake - cdf_real))
                w1ps.append(w1p)
                if i != 0:
                    self.log("features/" + names[i], w1p, on_step=False, on_epoch=True)
                    weighted_cdf_fake = weighted_hists_fake[i - 1].values().cumsum()
                    weighted_cdf_real = weighted_hists_real[i - 1].values().cumsum()
                    weighted_cdf_fake /= weighted_cdf_fake[-1]
                    weighted_cdf_real /= weighted_cdf_real[-1]
                    weighted_w1p = np.mean(np.abs(weighted_cdf_fake - weighted_cdf_real))
                    weighted_w1ps.append(weighted_w1p)
                    self.log(
                        "features/" + names[i] + "_weighted",
                        weighted_w1p,
                        on_step=False,
                        on_epoch=True,
                    )
                if i == 1:
                    self.log("weighted_z", weighted_w1p, on_step=False, on_epoch=True)
                    if weighted_w1p < self.min_z:
                        self.min_z = weighted_w1p
                        plot = True
                if i == 0:
                    self.log("features_E", w1p, on_step=False, on_epoch=True)
            self.log("w1p_mean", np.mean(w1ps), on_step=False, on_epoch=True)
            # self.log("weighted_w1p_mean", np.mean(weighted_w1ps), on_step=False, on_epoch=True)
            self.plot = plotting_point_cloud(step=trainer.global_step, logger=self.wandb_logger)
            try:
                self.plot.plot_response(response_real, response_fake)
                self.plot.plot_calo(hists_fake, hists_real, weighted=False)
                self.plot.plot_calo(
                    hists_fake_unscaled, hists_real_unscaled, weighted=False, unscaled=True
                )
                self.plot.plot_calo(weighted_hists_fake, weighted_hists_real, weighted=True)
                # self.plot.plot_scores(torch.cat(self.scores_real).numpy().reshape(-1), torch.cat(self.scores_fake.reshape(-1)).numpy(), False, self.global_step)

            except Exception as e:
                plt.close()
                traceback.print_exc()
            if self.log_times:
                if self.comet_logger is not None:
                    self.comet_logger.log_metrics({"Jet generation time": generation_time})
                if self.wandb_logger is not None:
                    self.log("Jet generation time", generation_time)

            if self.fix_seed:
                torch.manual_seed(torch.seed())

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback
