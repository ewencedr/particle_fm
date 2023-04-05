from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from src.data.components.utils import jet_masses
from src.models.zuko.utils import odeint
from src.utils.pylogger import get_pylogger

from .components import EPiC_generator, Transformer

logger = get_pylogger("fm_module")

# TODO conditioned sampling


class CNF(nn.Module):
    """Continuous Normalizing Flow with EPiC Generator or Transformer.

    Args:
        features (int): Data features. Defaults to 3.
        model (str, optional): Use Transformer or EPiC Generator as architecture. Defaults to "transformer".
        num_particles (int, optional): Set cardinality. Defaults to 150.
        frequencies (int, optional): Frequency for time. Defaults to 6.
        hidden_dim (int, optional): Hidden dimensions. Defaults to 128.
        layers (int, optional): Number of Layers to use. Defaults to 8.
        mass_conditioning (bool, optional): Condition the model on the jet mass. Defaults to False.
        return_latent_space (bool, optional): Return latent space. Defaults to False.
        dropout (float, optional): Dropout value for dropout layers. Defaults to 0.0.
        heads (int, optional): Number of attention heads. Defaults to 4.
        mask (bool, optional): Use mask. Defaults to False.
        latent (int, optional): Latent dimension. Defaults to 16.
        activation (str, optional): Activation function. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper function. Defaults to "weight_norm".
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
    """

    def __init__(
        self,
        features: int = 3,
        model: str = "transformer",
        num_particles: int = 150,
        frequencies: int = 6,
        hidden_dim: int = 128,
        layers: int = 8,
        mass_conditioning: bool = False,
        return_latent_space: bool = False,
        dropout: float = 0.0,
        heads: int = 4,
        mask=False,
        latent: int = 16,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        t_local_cat: bool = False,
        t_global_cat: bool = False,
    ):
        super().__init__()
        self.model = model
        self.latent = latent
        self.mass_conditioning = mass_conditioning
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
            global_cond_dim = 1 if mass_conditioning else 0
            self.net = EPiC_generator(
                latent_local=features + 2 * frequencies,
                feats=features,
                latent=latent,
                equiv_layers=layers,
                hid_d=hidden_dim,
                return_latent_space=return_latent_space,
                activation=activation,
                wrapper_func=wrapper_func,
                frequencies=frequencies,
                t_local_cat=t_local_cat,
                t_global_cat=t_global_cat,
                global_cond_dim=global_cond_dim,
            )

        self.register_buffer("frequencies", 2 ** torch.arange(frequencies) * torch.pi)

    def forward(
        self,
        t: Tensor,
        x: Tensor,
    ) -> Tensor:
        logger.debug(f"self.mass_conditioning: {self.mass_conditioning}")
        # logger.debug(f"t.shape0: {t[:3]}")
        logger.debug(f"x.shape1: {x.shape}")
        # t: (batch_size,num_particles)
        t = self.frequencies * t[..., None]  # (batch_size,num_particles,frequencies)
        logger.debug(f"t.shape1: {t[:3]}")
        t = torch.cat((t.cos(), t.sin()), dim=-1)  # (batch_size,num_particles,2*frequencies)
        logger.debug(f"t.shape2: {t[:3]}")
        t = t.expand(*x.shape[:-1], -1)  # (batch_size,num_particles,2*frequencies)
        logger.debug(f"t.shape3: {t[:3]}")
        logger.debug(f"t.shape3: {t.shape}")
        x = torch.cat((t, x), dim=-1)  # (batch_size,num_particles,features+2*frequencies)
        logger.debug(f"x.shape2: {x[:3]}")

        if self.model == "epic":
            x_global = torch.randn_like(torch.ones(x.shape[0], self.latent, device=x.device))
            x_local = x
            if self.mass_conditioning:
                cond = jet_masses(x_local).unsqueeze(-1)
                logger.debug(f"mass.shape: {cond.shape}")
            else:
                cond = None
            x = self.net(t, x_global, x_local, cond)

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
    """Flow Matching loss objective for training CNFs.

    Args:
        v (nn.Module): Model
        use_mass_loss (bool, optional): Use mass in loss function. Defaults to False.
    """

    def __init__(self, v: nn.Module, use_mass_loss: bool = False):
        super().__init__()

        self.v = v
        self.use_mass_loss = use_mass_loss

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x
        v_res = self.v(t.squeeze(-1), y)
        out = (v_res - u).square().mean()
        if self.use_mass_loss:
            mass_scaling_factor = 0.001 * 1
            mass_mse = (jet_masses(v_res) - jet_masses(u)).square().mean()
            logger.debug(f"jet_mass_diff: {mass_mse*mass_scaling_factor}")
            logger.debug(f"out: {out}")
            return out + mass_mse * mass_scaling_factor, mass_mse * mass_scaling_factor
        else:
            return out


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
    """Pytorch Lightning module for training CNFs with Flow Matching loss.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler): Scheduler
        model (str, optional): Use Transformer or EPiC Generator as model. Defaults to "epic".
        features (int, optional): Features of data. Defaults to 3.
        hidden_dim (int, optional): Hidden dimensions. Defaults to 128.
        num_particles (int, optional): Set cardinality. Defaults to 150.
        frequencies (int, optional): Time frequencies. Defaults to 6.
        use_mass_loss (bool, optional): Add mass term to loss. Defaults to True.
        layers (int, optional): Number of layers. Defaults to 8.
        n_transforms (int, optional): Number of flow transforms. Defaults to 1.
        mass_conditioning (bool, optional): Condition the model on the jet mass. Defaults to False.
        activation (str, optional): Activation function. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper function. Defaults to "weight_norm".
        latent (int, optional): Latent dimension. Defaults to 16.
        return_latent_space (bool, optional): Return latent space. Defaults to False.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
        dropout (float, optional): Value for dropout layers. Defaults to 0.0.
        heads (int, optional): Number of attention heads. Defaults to 4.
        mask (bool, optional): Use Mask. Defaults to False.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model: str = "epic",
        features: int = 3,
        hidden_dim: int = 128,
        num_particles: int = 150,
        frequencies: int = 6,
        use_mass_loss: bool = True,
        layers: int = 8,
        n_transforms: int = 1,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        # epic
        latent: int = 16,
        return_latent_space: bool = False,
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        mass_conditioning: bool = False,
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
                    mass_conditioning=mass_conditioning,
                    latent=latent,
                    return_latent_space=return_latent_space,
                    dropout=dropout,
                    heads=heads,
                    mask=mask,
                    activation=activation,
                    wrapper_func=wrapper_func,
                    t_global_cat=t_global_cat,
                    t_local_cat=t_local_cat,
                )
            )
            # losses.append(FlowMatchingLoss(flows[-1]))
        self.flows = flows
        self.use_mass_loss = use_mass_loss
        # self.losses = losses
        self.loss = FlowMatchingLoss(self.flows[0], use_mass_loss=use_mass_loss)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, reverse: bool = False):
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
        if self.use_mass_loss:
            loss, mse_mass = self.loss(x)
            self.log("train/mse_mass", mse_mass, on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.loss(x)
        # loss = self.loss(x)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        x, mask = batch
        if self.use_mass_loss:
            loss, mse_mass = self.loss(x)
            self.log("val/mse_mass", mse_mass, on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.loss(x)
        # loss = self.loss(x)
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
        cond = torch.zeros(n_samples, self.hparams.num_particles, 1).to(self.device)
        samples = self.forward(z, cond=cond, reverse=True)
        return samples
