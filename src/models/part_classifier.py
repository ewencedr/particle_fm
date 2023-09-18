import copy
import sys

import lightning as L
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

sys.path.insert(0, "/home/birkjosc/repositories/weaver-core")
from weaver.nn.model.ParticleNet import ParticleNet
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.train import optim
from weaver.utils.nn.tools import _flatten_label, _flatten_preds

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

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.opt = kwargs.get("optimizer", None)
        kwargs.pop("optimizer", None)
        self.scheduler = kwargs.get("scheduler", None)
        kwargs.pop("scheduler", None)
        self.lr = kwargs.get("lr", 0.001)
        kwargs.pop("lr", None)

        self.mod = ParticleTransformer(**kwargs)
        self.loss_func = torch.nn.CrossEntropyLoss()
        # self.data_config = data_config
        self.fc_params = kwargs["fc_params"]
        self.num_classes = kwargs["num_classes"]
        self.last_embed_dim = kwargs["embed_dims"][-1]
        self.set_learning_rates()
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

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)

    def training_step(self, batch, batch_nb):
        self.mod.for_inference = False
        self.train()
        # X, y, _ = batch
        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        label = jet_labels.long().to("cuda")
        model_output = self(
            points=None,
            features=pf_features.to("cuda"),
            lorentz_vectors=pf_vectors.to("cuda"),
            mask=pf_mask.to("cuda"),
        )
        with torch.cuda.amp.autocast():
            logits = _flatten_preds(model_output)
            loss = self.loss_func(logits, label)
        _, preds = logits.max(1)
        correct = (preds == label).sum().item()
        accuracy = correct / label.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True)
        self.train_loss_list.append(loss.detach().cpu().numpy())
        self.train_acc_list.append(accuracy)
        return loss

    def on_train_epoch_end(self):
        print(f"Epoch {self.trainer.current_epoch} finished.", end="\r")

    def on_train_end(self):
        self.train_loss = np.array(self.train_loss_list)
        self.train_acc = np.array(self.train_acc_list)

    def validation_step(self, batch, batch_nb):
        self.mod.for_inference = True
        self.eval()

        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        label = jet_labels.long().to("cuda")
        model_output = self(
            points=None,
            features=pf_features.to("cuda"),
            lorentz_vectors=pf_vectors.to("cuda"),
            mask=pf_mask.to("cuda"),
        )

        key = str(self.validation_cnt)

        if batch_nb == 0:
            self.validation_cnt += 1
            key = str(self.validation_cnt)
            self.validation_output[key] = {
                "model_predictions": model_output.detach().cpu().numpy(),
                "labels": label.detach().cpu().numpy(),
                "global_step": self.trainer.global_step,
            }
        else:
            self.validation_output[key]["model_predictions"] = np.concatenate(
                [
                    self.validation_output[key]["model_predictions"],
                    model_output.detach().cpu().numpy(),
                ]
            )
            self.validation_output[key]["labels"] = np.concatenate(
                [self.validation_output[key]["labels"], label.detach().cpu().numpy()]
            )

        with torch.cuda.amp.autocast():
            logits = _flatten_preds(model_output)
            loss = self.loss_func(logits, label)
        _, preds = logits.max(1)
        correct = (preds == label).sum().item()
        accuracy = correct / label.size(0)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", accuracy, on_step=True, on_epoch=True)
        self.val_loss_list.append(loss.detach().cpu().numpy())
        self.val_acc_list.append(accuracy)

        return loss

    def on_validation_epoch_end(self):
        self.val_loss = np.array(self.val_loss_list)
        self.val_acc = np.array(self.val_acc_list)
        each_me = copy.deepcopy(self.trainer.callback_metrics)
        curr_step = str(self.trainer.global_step)
        if curr_step not in self.metrics_dict:
            self.metrics_dict[curr_step] = {}
        for k, v in each_me.items():
            self.metrics_dict[curr_step][k] = v.detach().cpu().numpy()

    def test_step(self, batch, batch_nb):
        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        label = jet_labels.long().to("cuda")
        self.mod.for_inference = True
        self.eval()
        model_output = self(
            points=None,
            features=pf_features.to("cuda"),
            lorentz_vectors=pf_vectors.to("cuda"),
            mask=pf_mask.to("cuda"),
        )
        # with torch.cuda.amp.autocast():
        #     logits = _flatten_preds(model_output)
        #     loss = self.loss_func(logits, label)
        # _, preds = logits.max(1)
        # correct = (preds == label).sum().item()
        # accuracy = correct / label.size(0)
        self.test_output_list.append(model_output.detach().cpu().numpy())
        self.test_labels_list.append(label.detach().cpu().numpy())

    def on_test_end(self):
        self.test_output = np.concatenate(self.test_output_list)
        self.test_labels = np.concatenate(self.test_labels_list)

    def set_learning_rates(self, lr_fc=0.001, lr=0.0001):
        """Set the learning rates for the fc layer and the rest of the model."""
        self.lr_fc = lr_fc
        self.lr = lr
        print(f"Setting learning rates to lr_fc={self.lr_fc} and lr={self.lr}")

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

    def configure_optimizers(self):
        # optimizer, scheduler = optim(self.opt_args, self, self.dev)
        # User adamw optimizer
        # TODO: use more sophisticated optimizer and scheduler?
        # compare to the standard ParT fine-tuning
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


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


class ParticleNetPL(pl.LightningModule):
    """Pytorch-lightning wrapper for ParticleNet."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        #
        self.opt = kwargs.get("optimizer", None)
        kwargs.pop("optimizer", None)
        self.scheduler = kwargs.get("scheduler", None)
        kwargs.pop("scheduler", None)
        self.lr = kwargs.get("lr", 0.001)
        kwargs.pop("lr", None)

        self.mod = ParticleNet(**kwargs)
        self.loss_func = torch.nn.CrossEntropyLoss()
        # self.data_config = data_config
        self.fc_params = kwargs["fc_params"]
        self.num_classes = kwargs["num_classes"]
        self.last_embed_dim = kwargs["conv_params"][-1][-1][0]
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

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(points=points, features=features, mask=mask)

    def training_step(self, batch, batch_nb):
        self.mod.for_inference = False
        self.train()
        # X, y, _ = batch
        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        label = jet_labels.long().to("cuda")
        model_output = self(
            points=pf_points,
            features=pf_features.to("cuda"),
            lorentz_vectors=None,
            mask=pf_mask.to("cuda"),
        )
        with torch.cuda.amp.autocast():
            logits = _flatten_preds(model_output)
            loss = self.loss_func(logits, label)
        _, preds = logits.max(1)
        correct = (preds == label).sum().item()
        accuracy = correct / label.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True)
        self.train_loss_list.append(loss.detach().cpu().numpy())
        self.train_acc_list.append(accuracy)
        return loss

    def on_train_epoch_end(self):
        print(f"Epoch {self.trainer.current_epoch} finished.", end="\r")

    def on_train_end(self):
        self.train_loss = np.array(self.train_loss_list)
        self.train_acc = np.array(self.train_acc_list)

    def validation_step(self, batch, batch_nb):
        self.mod.for_inference = True
        self.eval()

        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        label = jet_labels.long().to("cuda")
        model_output = self(
            points=pf_points,
            features=pf_features.to("cuda"),
            lorentz_vectors=None,
            mask=pf_mask.to("cuda"),
        )

        key = str(self.validation_cnt)

        if batch_nb == 0:
            self.validation_cnt += 1
            key = str(self.validation_cnt)
            self.validation_output[key] = {
                "model_predictions": model_output.detach().cpu().numpy(),
                "labels": label.detach().cpu().numpy(),
                "global_step": self.trainer.global_step,
            }
        else:
            self.validation_output[key]["model_predictions"] = np.concatenate(
                [
                    self.validation_output[key]["model_predictions"],
                    model_output.detach().cpu().numpy(),
                ]
            )
            self.validation_output[key]["labels"] = np.concatenate(
                [self.validation_output[key]["labels"], label.detach().cpu().numpy()]
            )

        with torch.cuda.amp.autocast():
            logits = _flatten_preds(model_output)
            loss = self.loss_func(logits, label)
        _, preds = logits.max(1)
        correct = (preds == label).sum().item()
        accuracy = correct / label.size(0)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", accuracy, on_step=True, on_epoch=True)
        self.val_loss_list.append(loss.detach().cpu().numpy())
        self.val_acc_list.append(accuracy)

        return loss

    def on_validation_epoch_end(self):
        self.val_loss = np.array(self.val_loss_list)
        self.val_acc = np.array(self.val_acc_list)
        each_me = copy.deepcopy(self.trainer.callback_metrics)
        curr_step = str(self.trainer.global_step)
        if curr_step not in self.metrics_dict:
            self.metrics_dict[curr_step] = {}
        for k, v in each_me.items():
            self.metrics_dict[curr_step][k] = v.detach().cpu().numpy()

    def test_step(self, batch, batch_nb):
        pf_points, pf_features, pf_vectors, pf_mask, cond, jet_labels = batch
        label = jet_labels.long().to("cuda")
        self.mod.for_inference = True
        self.eval()
        model_output = self(
            points=pf_points,
            features=pf_features.to("cuda"),
            lorentz_vectors=None,
            mask=pf_mask.to("cuda"),
        )
        # with torch.cuda.amp.autocast():
        #     logits = _flatten_preds(model_output)
        #     loss = self.loss_func(logits, label)
        # _, preds = logits.max(1)
        # correct = (preds == label).sum().item()
        # accuracy = correct / label.size(0)
        self.test_output_list.append(model_output.detach().cpu().numpy())
        self.test_labels_list.append(label.detach().cpu().numpy())

    def on_test_end(self):
        self.test_output = np.concatenate(self.test_output_list)
        self.test_labels = np.concatenate(self.test_labels_list)

    def set_learning_rates(self, lr_fc=0.001, lr=0.0001):
        """Set the learning rates for the fc layer and the rest of the model."""
        self.lr_fc = lr_fc
        self.lr = lr
        print(f"Setting learning rates to lr_fc={self.lr_fc} and lr={self.lr}")

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

    def configure_optimizers(self):
        # optimizer, scheduler = optim(self.opt_args, self, self.dev)
        # User adamw optimizer
        # TODO: use more sophisticated optimizer and scheduler?
        # compare to the standard ParT fine-tuning
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
