# Code in this file is adapted from:
# https://github.com/NVIDIA/NeMo/pull/5169
# https://github.com/BioinfoMachineLearning/bio-diffusion/blob/e4bad15139815e562a27fb94dab0c31907522bc5/src/utils/__init__.py#L71
# https://github.com/NVIDIA/NeMo/blob/be0804f61e82dd0f63da7f9fe8a4d8388e330b18/nemo/utils/exp_manager.py#L744
import os
import os.path
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from particle_fm.utils.pylogger import get_pylogger

log = get_pylogger("EMA")


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.
    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        apply_ema_every_n_steps: Apply EMA every n global steps.
        start_step: Start applying EMA from ``start_step`` global step onwards.
        save_ema_weights_in_callback_state: Enable saving EMA weights in callback state.
        evaluate_ema_weights_instead: Validate the EMA weights instead of the original weights.
            Note this means that when saving the model, the validation metrics are calculated with the EMA weights.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        log.info("Creating EMA weights copy.")
        if self._ema_model_weights is None:
            self._ema_model_weights = [p.detach().clone() for p in pl_module.state_dict().values()]
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        return self.apply_ema(pl_module)

    def apply_ema(self, pl_module: "pl.LightningModule") -> None:
        for orig_weight, ema_weight in zip(
            list(pl_module.state_dict().values()), self._ema_model_weights
        ):
            if ema_weight.data.dtype != torch.long and orig_weight.data.dtype != torch.long:
                # ensure that non-trainable parameters (e.g., feature distributions) are not included in EMA weight averaging
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - self.decay)
                ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        return (
            step != self._cur_step
            and step >= self.start_step
            and step % self.apply_ema_every_n_steps == 0
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        if self.save_ema_weights_in_callback_state:
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cur_step = state_dict["cur_step"]
        # when loading within apps such as NeMo, EMA weights will be loaded by the experiment manager separately
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        if trainer.ckpt_path and checkpoint_callback is not None:
            ext = checkpoint_callback.FILE_EXTENSION
            if trainer.ckpt_path.endswith(f"-EMA{ext}"):
                log.info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = trainer.ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                self._ema_model_weights = ema_state_dict["state_dict"].values()
                del ema_state_dict
                log.info(
                    "EMA weights have been loaded successfully. Continuing training with saved EMA"
                    " weights."
                )
            else:
                warnings.warn(
                    (
                        "we were unable to find the associated EMA weights when re-loading, "
                        "training will start with new EMA weights."
                    ),
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        self._weights_buffer = [
            p.detach().clone().to("cpu") for p in pl_module.state_dict().values()
        ]
        new_state_dict = {
            k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)
        }
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)


# TODO breaks when task_name contains metric_map keys
class EMAModelCheckpoint(ModelCheckpoint):
    """Light wrapper around Lightning's `ModelCheckpoint` to, upon request, save an EMA copy of the
    model as well. Should only be used with `EMACallback`. Should only work for trainings with a
    single GPU. For custom checkpoint names use the metric map.

    Args:
        metric_map (Dict[str, str]): A dictionary mapping the metric name that is logged to the name of the metric in the checkpoint file name.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/be0804f61e82dd0f63da7f9fe8a4d8388e330b18/nemo/utils/exp_manager.py#L744
    """

    def __init__(self, metric_map={"val/loss": "loss"}, **kwargs):
        # call the parent class constructor with the provided kwargs
        super().__init__(**kwargs)
        self.metric_map = metric_map
        self.best_k_models_ema = {}
        self.kth_best_model_path_ema = ""
        self.best_model_score_ema = None
        self.best_model_path_ema = ""
        self.model_parallel_size_ema = None
        self.last_model_path_ema = ""

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None:
            # save EMA copy of the model as well
            ema_callback.replace_model_weights(trainer.lightning_module)
            filepath = self._ema_format_filepath(filepath)
            if self.verbose:
                rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
            super()._save_checkpoint(trainer, filepath)
            if "last-EMA.ckpt" in filepath:
                self.last_model_path_ema = filepath
            ema_callback.restore_original_weights(trainer.lightning_module)
            if self.save_top_k != -1:
                self.topk_check_ema()

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    def topk_check_ema(self):
        checkpoints = list(Path(self.dirpath).rglob("*-EMA.ckpt"))
        log.debug(f"checkpoints: {checkpoints}")
        for checkpoint in checkpoints:
            checkpoint = str(checkpoint)
            if "last" in checkpoint:
                continue

            if self.monitor in self.metric_map:
                monitor = self.metric_map[self.monitor]
                if monitor not in checkpoint:
                    continue
            else:
                monitor = self.monitor

            index = checkpoint.find(monitor) + len(monitor) + 1  # Find monitor in str + 1 for '='
            log.debug(f"-----monitor: {monitor}")
            log.debug(f"index: {index}")
            log.debug(f"checkpoint[index:]: {checkpoint[index:]}")
            if index != -1:
                match = re.search("[A-z]", checkpoint[index:])
                log.debug(f"match: {match}")
                if match:
                    value = checkpoint[
                        index : index + match.start() - 1
                    ]  # -1 due to separator hyphen
                    log.debug(f"value: {value}")
                    self.best_k_models_ema[checkpoint] = float(value)
        if len(self.best_k_models_ema) < 1:
            return  # No saved checkpoints yet

        _reverse = False if self.mode == "min" else True

        best_k_models_ema = sorted(
            self.best_k_models_ema, key=self.best_k_models_ema.get, reverse=_reverse
        )

        log.debug(f"best_k_models_ema: {best_k_models_ema}")

        # This section should be ok as rank zero will delete all excess checkpoints, since all other ranks are
        # instantiated after rank zero. models_to_delete should be 0 for all other ranks.
        if self.model_parallel_size_ema is not None:
            models_to_delete = (
                len(best_k_models_ema) - self.model_parallel_size_ema * self.save_top_k
            )
        else:
            models_to_delete = len(best_k_models_ema) - self.save_top_k
        log.debug(f"Number of models to delete: {models_to_delete}")
        for _ in range(models_to_delete):
            model = best_k_models_ema.pop(-1)
            self.best_k_models_ema.pop(model)
            self._del_model_without_trainer(model)
            log.debug(f"Removed checkpoint: {model}")

        self.kth_best_model_path_ema = best_k_models_ema[-1]
        self.best_model_path_ema = best_k_models_ema[0]
        self.best_model_score_ema = self.best_k_models_ema[self.best_model_path_ema]

    def _del_model_without_trainer(self, filepath: str) -> None:
        try:
            self._fs.rm(filepath)
            if self.verbose:
                log.info(f"Removed checkpoint: {filepath}")
        except Exception as ex:
            log.info(f"Tried to remove checkpoint: {filepath} but failed with exception: {ex}")
