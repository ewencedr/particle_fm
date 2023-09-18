"""Callback for evaluating the classifier on the JetClass dataset."""
import os
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from src.utils.pylogger import get_pylogger

from .ema import EMA

pylogger = get_pylogger("JetClassClassifierEvaluationCallback")


class JetClassClassifierEvaluationCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int | Callable = 10,
        additional_eval_epochs: list[int] = None,
        image_path: str = None,
        log_times: bool = True,
        log_epoch_zero: bool = False,
    ):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        plot_dir = trainer.default_root_dir + "/plots/"
        os.makedirs(plot_dir, exist_ok=True)
        # Save the plots
        fig, ax = plt.subplots(figsize=(5, 3))
        val_cnt = pl_module.validation_cnt
        val_dict = pl_module.validation_output[str(val_cnt)]
        is_fake = val_dict["labels"] == 1
        p_fake = val_dict["model_predictions"][:, 1]

        hist_kwargs = dict(bins=np.linspace(0, 1, 50), histtype="step", density=True)

        ax.hist(p_fake[is_fake], label="fake", **hist_kwargs)
        ax.hist(p_fake[~is_fake], label="real", **hist_kwargs)
        ax.set_xlabel("$p_\\mathrm{fake}$")
        ax.legend(frameon=False)
        fig.tight_layout()
        plot_filename = f"{plot_dir}/p_fake_{trainer.current_epoch}_{val_cnt}.png"
        fig.savefig(plot_filename, dpi=300)
        if self.comet_logger is not None:
            self.comet_logger.log_image(
                plot_filename, name=plot_filename.split("/")[-1]
            )  # noqa: E501
