import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions import Normal
from torchdyn.core import NeuralODE
from zuko.utils import odeint

from .components.losses import FlowMatchingLoss
from .components.mlp import (
    MLP,
    small_cond_MLP_model,
    small_cond_ResNet_model,
    very_small_cond_MLP_model,
)


class ode_wrapper(torch.nn.Module):
    """Wraps model to ode solver compatible format.

    Args:
        model (torch.nn.Module): Model to wrap.
        mask (torch.Tensor, optional): Mask. Defaults to None.
        cond (torch.Tensor, optional): Condition. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        mask: torch.Tensor = None,
        cond: torch.Tensor = None,
    ):
        super().__init__()
        self.model = model
        self.mask = mask
        self.cond = cond

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x, mask=self.mask, cond=self.cond)


class CNF(nn.Module):
    def __init__(
        self,
        features: int,
        freqs: int = 3,
        activation: str = "Tanh",
    ):
        super().__init__()
        self.net = small_cond_MLP_model(
            features, features, dim_t=2 * freqs, dim_cond=1, activation=activation
        )

        self.register_buffer("freqs", torch.arange(1, freqs + 1) * torch.pi)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        cond: torch.Tensor = None,
    ) -> torch.Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(t, x, cond=cond)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        node = NeuralODE(x, solver="midpoint", sensitivity="adjoint")
        t_span = torch.linspace(0.0, 1.0, 50)
        traj = node.trajectory(x, t_span)
        return traj[-1]

    def decode(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor = None,
        ode_solver: str = "midpoint",
        ode_steps: int = 100,
    ) -> torch.Tensor:
        wrapped_cnf = ode_wrapper(
            model=self,
            mask=mask,
            cond=cond,
        )
        if ode_solver == "midpoint":
            node = NeuralODE(wrapped_cnf, solver="midpoint", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        else:
            raise NotImplementedError(f"Solver {ode_solver} not implemented")

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        i = torch.eye(x.shape[-1]).to(x)
        i = i.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t: torch.Tensor, x: torch.Tensor, ladj: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, i, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum("i...i", jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj * 1e2


class FLowMatchingNoSetsLitModule(pl.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        features: int = 10,
        n_transforms: int = 1,
        sigma: float = 1e-4,
        activation: str = "ELU",
        freqs: int = 3,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        flows = nn.ModuleList()
        for _ in range(n_transforms):
            flows.append(CNF(features, freqs=freqs, activation=activation))
        self.flows = flows

        self.loss = FlowMatchingLoss(flows=self.flows, sigma=sigma)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor = None,
        mask: torch.Tensor = None,
        reverse: bool = False,
        ode_solver: str = "midpoint",
        ode_steps: int = 100,
    ):
        if reverse:
            for f in reversed(self.flows):
                x = f.decode(x, cond, mask, ode_solver=ode_solver, ode_steps=ode_steps)
        else:
            for f in self.flows:
                x = f.encode(x, mask, ode_solver=ode_solver, ode_steps=ode_steps)
        return x

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        x, mask, cond = batch

        loss = self.loss(x, cond=cond)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        # set same seed for every validation epoch
        torch.manual_seed(9999)

    def on_validation_epoch_end(self) -> None:
        torch.manual_seed(torch.seed())

    def validation_step(self, batch, batch_idx: int):
        x, mask, cond = batch

        loss = self.loss(x, cond=cond)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx: int):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            opt = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            opt = {"optimizer": optimizer}
        return opt

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        mask: torch.Tensor = None,
        cond: torch.Tensor = None,
        ode_solver: str = "midpoint",
        ode_steps: int = 100,
    ):
        """Generate Samples.

        Args:
            n_samples (int): Number of samples to generate.
            cond (torch.Tensor, optional): Data on which the model is conditioned. Defaults to None.
            ode_solver (str, optonal): ODE solver to use. Defaults to "midpoint".

        Returns:
            torch.Tensor: Generated samples
        """
        z = torch.randn(n_samples, self.hparams.features).to(self.device)
        if cond is not None:
            cond = cond.to(self.device)

        samples = self.forward(
            z, cond=cond, mask=mask, reverse=True, ode_solver=ode_solver, ode_steps=ode_steps
        )

        return samples
