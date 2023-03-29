from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from zuko.utils import odeint

from .components import EPiC_generator, Transformer


class CNF(nn.Module):
    def __init__(
        self,
        features: int,
        model: str = "transformer",
        num_particles: int = 150,
        frequencies: int = 6,
        hidden_dim: int = 128,
        layers: int = 8,
        return_latent_space: bool = False,
        dropout: float = 0.0,
        heads: int = 4,
        mask=False,
        latent: int = 16,
    ):
        super().__init__()
        self.model = model
        self.latent = latent
        if self.model == "transformer":
            self.net = Transformer(
                input_dim=features + 2 * frequencies,
                output_dim=features,
                emb=hidden_dim,
                mask=mask,
                seq_length=num_particles,
                heads=heads,
                depth=layers,
                dropout=dropout,
            )
        elif self.model == "epic":

            self.net = EPiC_generator(
                latent_local=features + 2 * frequencies,
                feats=features,
                latent=latent,
                equiv_layers=layers,
                hid_d=hidden_dim,
                return_latent_space=return_latent_space,
            )

        self.register_buffer("frequencies", 2 ** torch.arange(frequencies) * torch.pi)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t = self.frequencies * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        x = torch.cat((t, x), dim=-1)
        if self.model == "epic":
            x_global = torch.randn_like(torch.ones(x.shape[0], self.latent, device=x.device))
            x_local = x
            x = self.net(x_global, x_local)
        else:
            x = self.net(x)

        return x

    def encode(self, x: Tensor) -> Tensor:
        return odeint(self, x, 0.0, 1.0, phi=self.parameters())

    def decode(self, z: Tensor) -> Tensor:
        return odeint(self, z, 1.0, 0.0, phi=self.parameters())

    def log_prob(self, x: Tensor) -> Tensor:
        i = torch.eye(x.shape[-1]).to(x)
        i = i.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, i, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum("i...i", jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj * 1e2


class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        super().__init__()

        self.v = v

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        return (self.v(t.squeeze(-1), y) - u).square().mean()


class FlowMatchingLoss2(nn.Module):
    def __init__(self, vs: nn.Module):
        super().__init__()

        self.vs = vs

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x
        for v in self.vs:
            y = v(t.squeeze(-1), y)
        return (y - u).square().mean()


class SetFlowMatchingLitModule(pl.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model: str = "epic",
        features: int = 3,
        hidden_dim: int = 128,
        num_particles: int = 150,
        frequencies: int = 6,
        layers: int = 8,
        n_transforms: int = 1,
        # epic
        latent: int = 16,
        return_latent_space: bool = False,
        # transformer
        dropout: float = 0.0,
        heads: int = 4,
        mask=False,
        **kwargs,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        flows = nn.ModuleList()
        # losses = nn.ModuleList()
        for _ in range(n_transforms):
            flows.append(
                CNF(
                    features=features,
                    model=model,
                    hidden_dim=hidden_dim,
                    num_particles=num_particles,
                    frequencies=frequencies,
                    layers=layers,
                    latent=latent,
                    return_latent_space=return_latent_space,
                    dropout=dropout,
                    heads=heads,
                    mask=mask,
                )
            )
            # losses.append(FlowMatchingLoss(flows[-1]))
        self.flows = flows
        # self.losses = losses
        self.loss = FlowMatchingLoss(self.flows[0])

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if reverse:
            for f in reversed(self.flows):
                x = f.decode(x)
        else:
            for f in self.flows:
                x = f.encode(x)
        return x

    # def loss(self, x: torch.Tensor):
    #    return sum(loss(x) for loss in self.losses) / len(self.losses)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        pass

    # def on_training_epoch_end(self, outputs: List[Any]):
    # `outputs` is a list of dicts returned from `training_step()`

    # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
    # this may not be an issue when training on mnist
    # but on larger datasets/models it's easy to run into out-of-memory errors

    # consider detaching tensors before returning them from `training_step()`
    # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

    #    pass

    def training_step(self, batch, batch_idx):
        x, mask = batch
        loss = self.loss(x)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        x, mask = batch
        loss = self.loss(x)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    # def on_validation_epoch_end(self, outputs: List[Any]):
    #    pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    # def on_test_epoch_end(self, outputs: List[Any]):
    #    pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def sample(self, n_samples: int):
        z = torch.randn(n_samples, self.hparams.num_particles, self.hparams.features).to(
            self.device
        )
        samples = self.forward(z, reverse=True)
        return samples
