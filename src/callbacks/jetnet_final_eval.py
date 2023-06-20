import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import yaml

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

from .ema import EMA, EMAModelCheckpoint

log = get_pylogger("JetNetFinalEvaluationCallback")


class JetNetFinalEvaluationCallback(pl.Callback):
    """Callback to do final evaluation of the model after training. Specific to JetNet dataset.

    Args:
        use_ema (bool, optional): Use exponential moving average weights for logging. Defaults to False.
        dataset (str, optional): Dataset to evaluate on. Defaults to "test".
        nr_checkpoint_callbacks (int, optional): Number of checkpoint callback that is used to select best epoch. Will only be used when ckpt_path is None. Defaults to 1.
        ckpt_path (Optional[str], optional): Path to checkpoint. If given, this ckpt will be used for evaluation. Defaults to None.
        **kwargs: Arguments for create_and_plot_data
    """

    def __init__(
        self,
        use_ema: bool = True,
        dataset: str = "test",
        nr_checkpoint_callbacks: int = 1,
        ckpt_path: Optional[str] = None,
        num_samples: int = -5,  # TODO
        **kwargs,
    ):
        super().__init__()

        # self.image_path = image_path
        apply_mpl_styles()

        self.use_ema = use_ema
        self.dataset = dataset
        self.ckpt_path = ckpt_path
        self.nr_checkpoint_callbacks = nr_checkpoint_callbacks
        self.num_samples = num_samples
        # loggers
        self.comet_logger = None
        self.wandb_logger = None

    def on_train_start(self, trainer, pl_module) -> None:
        log.info(
            "JetNetFinalEvaluationCallback will be used for evaluating the model after training."
        )

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    # def on_train_end(self, trainer, pl_module) -> None:
    # log.info("Evaluating model on test dataset.")
    # ema_callback = self._get_ema_callback(trainer)
    # if ema_callback is not None:
    #    ema_callback.apply_ema_weights(pl_module)
    # self._evaluate_model(trainer, pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info(f"Evaluating model on {self.dataset} dataset.")

        ckpt = self._get_checkpoint(trainer)

        log.info(f"Loading checkpoint from {ckpt}")
        model = pl_module.load_from_checkpoint(ckpt)

        factor = 2

        # Get background data for plotting and calculating Wasserstein distances
        if self.dataset == "test":
            background_data = np.array(trainer.datamodule.tensor_test)
            background_mask = np.array(trainer.datamodule.mask_test)
            background_cond = np.array(trainer.datamodule.tensor_conditioning_test)
        elif self.dataset == "val":
            background_data = np.array(trainer.datamodule.tensor_val)
            background_mask = np.array(trainer.datamodule.mask_val)
            background_cond = np.array(trainer.datamodule.tensor_conditioning_val)

        big_mask = np.repeat(background_mask, factor, axis=0)
        big_data = np.repeat(background_data, factor, axis=0)
        big_cond = np.repeat(background_cond, factor, axis=0)

        # plot_name = f"{self.model_name}--epoch{trainer.current_epoch}"

        data, generation_time = generate_data(
            model=model,
            num_jet_samples=factor * len(background_mask),
            batch_size=256,
            cond=torch.tensor(big_cond),
            variable_set_sizes=True,
            mask=torch.tensor(big_mask),
            normalized_data=trainer.datamodule.hparams.normalize,
            means=trainer.datamodule.means,
            stds=trainer.datamodule.stds,
            ode_solver="midpoint",
            ode_steps=200,
        )

        w_dists_big = calculate_all_wasserstein_metrics(
            background_data[..., :3],
            data,
            None,
            None,
            num_eval_samples=len(background_data),
            num_batches=factor,
            calculate_efps=True,
            use_masks=False,
        )

        yaml_path = "/".join(ckpt.split("/")[:-2]) + "/final_eval_metrics.yml"
        log.info(f"Writing final evaluation metrics to {yaml_path}")

        # transform numpy.float64 for better readability in yaml file
        w_dists_big = {k: float(v) for k, v in w_dists_big.items()}
        # write to yaml file
        with open(yaml_path, "w") as outfile:
            yaml.dump(w_dists_big, outfile, default_flow_style=False)

    def _get_checkpoint(self, trainer: pl.Trainer) -> None:
        if self.ckpt_path is None:
            if self.use_ema:
                if (
                    type(trainer.checkpoint_callbacks[self.nr_checkpoint_callbacks])
                    == EMAModelCheckpoint
                ):
                    return trainer.checkpoint_callbacks[
                        self.nr_checkpoint_callbacks
                    ].best_model_path_ema
                else:
                    raise ValueError(
                        "JetNetFinalEvaluationCallback was told to use EMA weights for evaluation but the provided checkpoint callback is not of type EMAModelCheckpoint"
                    )
            else:
                return trainer.checkpoint_callbacks[self.nr_checkpoint_callbacks].best_model_path
        else:
            return self.ckpt_path
