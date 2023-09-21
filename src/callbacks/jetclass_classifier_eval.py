"""Callback for evaluating the classifier on the JetClass dataset."""
import os
from typing import Callable

import cplt
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score, roc_curve

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
        if not hasattr(pl_module, "val_preds") or not hasattr(pl_module, "val_labels"):
            pylogger.info("No validation predictions found. Skipping plotting.")
            return

        pylogger.info(
            f"Running JetClassClassifierEvaluationCallback epoch: {trainer.current_epoch} step:"
            f" {trainer.global_step}"
        )
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
        if pl_module.val_labels.shape[1] == 2:
            labels = pl_module.val_labels[:, 1]
            p_fake = pl_module.val_preds[:, 1]
        else:
            labels = pl_module.val_labels
            p_fake = pl_module.val_preds
        is_fake = labels == 1

        roc_auc = roc_auc_score(labels, p_fake)

        hist_kwargs = dict(bins=np.linspace(0, 1, 50), histtype="step", density=True)

        ax.hist(p_fake[is_fake], label="Fake jets", **hist_kwargs, color="darkred")
        ax.hist(p_fake[~is_fake], label="Real jets", **hist_kwargs, color="darkblue")
        ax.set_xlabel("$p_\\mathrm{fake}$")
        ax.set_ylabel("Normalized")
        ax.legend(frameon=False, loc="upper right")
        ax.set_yscale("log")
        cplt.utils.decorate_ax(ax, text=f"AUC: {roc_auc:.3f}")
        fig.tight_layout()
        plot_filename = f"{plot_dir}/p_fake_{trainer.current_epoch}_{trainer.global_step}.png"
        fig.savefig(plot_filename, dpi=300)
        if self.comet_logger is not None:
            self.comet_logger.log_image(
                plot_filename, name=plot_filename.split("/")[-1]
            )  # noqa: E501
