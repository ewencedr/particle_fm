import copy
import sys

import lightning as L
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, "/home/birkjosc/repositories/weaver-core")
from typing import Any, Dict, Tuple

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import AUROC, Accuracy
from weaver.nn.model.ParticleNet import ParticleNet
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.train import optim
from weaver.utils.nn.tools import _flatten_label, _flatten_preds

from .components.epic import EPiC_discriminator

# standard model configuration from
# https://github.com/jet-universe/particle_transformer/blob/main/networks/example_ParticleTransformer.py#L26-L44  # noqa: E501
part_default_kwargs = dict(
    input_dim=7,
    num_classes=10,
    # network configurations
    pair_input_dim=4,
    use_pre_activation_pair=False,
    embed_dims=[128, 512, 128],
    pair_embed_dims=[64, 64, 64],
    num_heads=8,
    num_layers=8,
    num_cls_layers=2,
    block_params=None,
    cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
    fc_params=[],
    activation="gelu",
    # misc
    trim=True,
    for_inference=False,
)


class ParticleTransformerPL(pl.LightningModule):
    """Pytorch-lightning wrapper for ParticleTransformer."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        kwargs.pop("input_dims", None)  # it's `input_dim` in ParT

        self.mod = ParticleTransformer(**kwargs)
        self.loss_func = torch.nn.CrossEntropyLoss()
        # self.data_config = data_config
        self.fc_params = kwargs["fc_params"]
        self.num_classes = kwargs["num_classes"]
        self.last_embed_dim = kwargs["embed_dims"][-1]
        # self.set_learning_rates()
        self.test_output_list = []
        self.test_labels_list = []
        self.test_output = None
        self.test_labels = None

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.metrics_dict = {}

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auc_best = MaxMetric()

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        labels = jet_labels.squeeze()
        logits = self.forward(
            points=None,
            features=pf_features.to("cuda"),
            lorentz_vectors=pf_vectors.to("cuda"),
            mask=pf_mask.to("cuda"),
        )
        loss = self.criterion(logits.to("cuda"), labels.float().to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(logits, targets)
        self.train_auc(logits, targets)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        print(f"Epoch {self.trainer.current_epoch} finished.", end="\r")

    def on_train_end(self):
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, targets = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(logits, targets)
        self.val_auc(logits, targets)
        self.log("val_loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        auc = self.val_auc.compute()  # get current val auc
        self.val_auc_best(auc)  # update best so far val auc
        self.log("val_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val_auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_start(self):
        self.test_loop_preds_list = []
        self.test_loop_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.test_loop_preds_list.append(preds.detach().cpu().numpy())
        self.test_loop_labels_list.append(targets.detach().cpu().numpy())
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_auc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_end(self):
        self.test_loop_preds = np.concatenate(self.test_loop_preds_list)
        self.test_loop_labels = np.concatenate(self.test_loop_labels_list)

    # def set_learning_rates(self, lr_fc=0.001, lr=0.0001):
    #     """Set the learning rates for the fc layer and the rest of the model."""
    #     self.lr_fc = lr_fc
    #     self.lr = lr
    #     print(f"Setting learning rates to lr_fc={self.lr_fc} and lr={self.lr}")

    def reinitialise_fc(self):
        """Reinitialise the final fully connected network of the model."""
        if self.fc_params is not None:
            fcs = []
            in_dim = self.last_embed_dim
            for out_dim, drop_rate in self.fc_params:
                fcs.append(
                    nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate))
                )
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, self.num_classes))
            self.mod.fc = nn.Sequential(*fcs)
        else:
            self.mod.fc = None

    def set_args_for_optimizer(self, args, dev):
        self.opt_args = args
        self.dev = dev

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


# https://github.com/jet-universe/particle_transformer/blob/main/networks/example_ParticleNet.py
conv_params = [
    (16, (64, 64, 64)),
    (16, (128, 128, 128)),
    (16, (256, 256, 256)),
]
fc_params = [(256, 0.1)]
kwargs = {}
particlenet_default_kwargs = dict(
    input_dims=None,
    num_classes=10,
    conv_params=kwargs.get("conv_params", conv_params),
    fc_params=kwargs.get("fc_params", fc_params),
    use_fusion=kwargs.get("use_fusion", False),
    use_fts_bn=kwargs.get("use_fts_bn", True),
    use_counts=kwargs.get("use_counts", True),
    for_inference=kwargs.get("for_inference", False),
)
PARTICLENET_KIN_MODEL_PATH = (
    "/home/birkjosc/repositories/particle_transformer/models/ParticleNet_kin.pt"
)


class ParticleNetPL(pl.LightningModule):
    """Pytorch-lightning wrapper for ParticleNet."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.lr = kwargs.get("lr", 0.001)
        kwargs.pop("lr", None)
        kwargs.pop("input_dim", None)  # it's `input_dim` in ParticleNet

        cfg = copy.deepcopy(particlenet_default_kwargs)
        cfg["input_dims"] = kwargs["input_dims"]
        cfg["fc_params"] = kwargs["fc_params"]
        cfg["conv_params"] = kwargs["conv_params"]

        self.mod = ParticleNet(**cfg)
        if kwargs.get("load_pretrained", False):
            if cfg["input_dims"] == 7:
                ckpt = torch.load(PARTICLENET_KIN_MODEL_PATH, map_location="cuda")
                self.load_state_dict(ckpt)

        # TODO: can we go back to this instead of BCE? (i.e. is it equivalent to just
        # using one of the two output nodes and put it into BCE)
        # self.loss_func = torch.nn.CrossEntropyLoss()
        # self.data_config = data_config
        self.fc_params = kwargs["fc_params"]
        self.num_classes = kwargs.get("num_classes", 2)
        self.last_embed_dim = kwargs["conv_params"][-1][-1][0]

        self.reinitialise_fc()
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auc_best = MaxMetric()

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(points=points, features=features, mask=mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_auc.reset()
        self.val_auc_best.reset()

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        labels = jet_labels.squeeze()
        logits = self.forward(
            points=pf_points.to("cuda"),
            features=pf_features.to("cuda"),
            lorentz_vectors=None,
            mask=pf_mask.to("cuda"),
        )
        loss = self.criterion(logits.to("cuda"), labels.float().to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(logits, targets)
        self.train_auc(logits, targets)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        print(f"Epoch {self.trainer.current_epoch} finished.", end="\r")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, targets = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(logits, targets)
        self.val_auc(logits, targets)
        self.log("val_loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        auc = self.val_auc.compute()  # get current val auc
        self.val_auc_best(auc)  # update best so far val auc
        self.log("val_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val_auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_start(self):
        self.test_loop_preds_list = []
        self.test_loop_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.test_loop_preds_list.append(preds.detach().cpu().numpy())
        self.test_loop_labels_list.append(targets.detach().cpu().numpy())
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_auc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_end(self):
        self.test_loop_preds = np.concatenate(self.test_loop_preds_list)
        self.test_loop_labels = np.concatenate(self.test_loop_labels_list)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # def on_train_end(self):
    #     self.train_loss = np.array(self.train_loss_list)
    #     self.train_acc = np.array(self.train_acc_list)

    # def set_learning_rates(self, lr_fc=0.001, lr=0.0001):
    #     """Set the learning rates for the fc layer and the rest of the model."""
    #     self.lr_fc = lr_fc
    #     self.lr = lr
    #     print(f"Setting learning rates to lr_fc={self.lr_fc} and lr={self.lr}")

    def reinitialise_fc(self):
        """Reinitialise the final fully connected network of the model."""
        if self.fc_params is not None:
            fcs = []
            in_dim = self.last_embed_dim
            for out_dim, drop_rate in self.fc_params:
                fcs.append(
                    nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate))
                )
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, self.num_classes))
            self.mod.fc = nn.Sequential(*fcs)
        else:
            self.mod.fc = None


class EPiCClassifierLitModule(LightningModule):
    """Example of a `LightningModule` for PointCloud classification by using an EPiC
    architecture."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net_config: Dict[str, Any] = {},
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = EPiC_discriminator(**net_config)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auc_best = MaxMetric()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x_local=x, mask=mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_auc.reset()
        self.val_auc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        labels = jet_labels.squeeze()
        logits = self.forward(pf_features, mask=pf_mask).squeeze()
        loss = self.criterion(logits, labels)
        preds = logits
        return loss, preds, labels

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_auc(preds, targets)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_auc(preds, targets)
        self.log("val_loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        auc = self.val_auc.compute()  # get current val auc
        self.val_auc_best(auc)  # update best so far val auc
        self.log("val_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val_auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_auc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = EPiCClassifierLitModule(None, None, None)
