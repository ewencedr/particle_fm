from typing import Any, Mapping

import numpy as np
import ot as pot
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
from torchdyn.core import NeuralODE
from zuko.utils import odeint

from src.utils.pylogger import get_pylogger

from .components import EPiC_generator, IterativeNormLayer
from .components.time_emb import CosineEncoding, GaussianFourierProjection

logger = get_pylogger("fm_module")
logger_loss = get_pylogger("fm_module_loss")


class ode_wrapper(torch.nn.Module):
    """Wraps model to ode solver compatible format."""

    def __init__(self, model, cond, mask):
        super().__init__()
        self.model = model
        self.cond = cond
        self.mask = mask

    def forward(self, t, x):
        return self.model(t, x, cond=self.cond, mask=self.mask)


class CNF(nn.Module):
    """Continuous Normalizing Flow with EPiC Generator or Transformer.

    Args:
        features (int): Data features. Defaults to 3.
        num_particles (int, optional): Set cardinality. Defaults to 150.
        frequencies (int, optional): Frequency for time. Basically half the size of the time vector that is added to the model. Defaults to 6.
        hidden_dim (int, optional): Hidden dimensions. Defaults to 128.
        layers (int, optional): Number of Layers to use. Defaults to 8.
        global_cond_dim (int, optional): Dimension to concatenate to the global feature in EPiC Layer. Must be zero for no conditioning. Defaults to 0.
        local_cond_dim (int, optional): Dimension to concatenate to the Local MLPs in EPiC Model. Must be zero for no conditioning. Defaults to 0.
        return_latent_space (bool, optional): Return latent space. Defaults to False.
        dropout (float, optional): Dropout value for dropout layers. Defaults to 0.0.
        heads (int, optional): Number of attention heads. Defaults to 4.
        mask (bool, optional): Use mask. Defaults to False.
        latent (int, optional): Latent dimension. Defaults to 16.
        activation (str, optional): Activation function. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper function. Defaults to "weight_norm".
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
        add_time_to_input (bool, optional): Concat time to input. Defaults to True.
        t_emb (str, optional): Embedding for time. Defaults to "sincos".
    """

    def __init__(
        self,
        features: int = 3,
        num_particles: int = 150,
        frequencies: int = 6,
        hidden_dim: int = 128,
        layers: int = 8,
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        return_latent_space: bool = False,
        dropout: float = 0.0,
        heads: int = 4,
        mask=False,
        latent: int = 16,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        add_time_to_input: bool = True,
        t_emb: str = "sincos",
    ):
        super().__init__()
        self.latent = latent
        self.add_time_to_input = add_time_to_input
        input_dim = features + 2 * frequencies if self.add_time_to_input else features

        self.net = EPiC_generator(
            input_dim=input_dim,
            feats=features,
            latent=latent,
            equiv_layers=layers,
            hid_d=hidden_dim,
            return_latent_space=return_latent_space,
            activation=activation,
            wrapper_func=wrapper_func,
            frequencies=frequencies,
            num_points=num_particles,
            t_local_cat=t_local_cat,
            t_global_cat=t_global_cat,
            global_cond_dim=global_cond_dim,
            local_cond_dim=local_cond_dim,
            dropout=dropout,
        )

        self.register_buffer("frequencies", 2 ** torch.arange(frequencies) * torch.pi)
        self.activation = activation
        self.t_emb = t_emb
        # Gaussian random feature embedding layer for time
        if self.t_emb == "gaussian":
            self.embed = nn.Sequential(
                GaussianFourierProjection(embed_dim=hidden_dim), nn.Linear(hidden_dim, hidden_dim)
            )
            self.linear = nn.Linear(hidden_dim, 2 * frequencies)
        elif self.t_emb == "cosine":
            self.embed = CosineEncoding(
                outp_dim=2 * frequencies,
                min_value=0.0,
                max_value=1.0,
                frequency_scaling="exponential",
            )

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        cond: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        t = self.time_embedding(t, x, self.t_emb)
        if self.add_time_to_input:
            x = torch.cat((t, x), dim=-1)  # (batch_size,num_particles,features+2*frequencies)

        x_global = torch.randn_like(torch.ones(x.shape[0], self.latent, device=x.device))
        x_local = x

        x = self.net(t, x_global, x_local, cond, mask)

        return x

    def time_embedding(self, t: Tensor, x: Tensor, t_emb: str = "sincos") -> Tensor:
        """Time embedding."""
        if t_emb == "sincos":
            t = self.frequencies * t[..., None]  # (batch_size,num_particles,frequencies)
            t = torch.cat((t.cos(), t.sin()), dim=-1)  # (batch_size,num_particles,2*frequencies)
            t = t.expand(*x.shape[:-1], -1)  # (batch_size,num_particles,2*frequencies)

        elif t_emb == "gaussian":
            # Obtain the Gaussian random feature embedding for t

            # different shape for training
            if len(t.shape) == 2:
                t = t[:, 0]
            t = getattr(F, self.activation, lambda x: x)(self.embed(t))
            t = self.linear(t).unsqueeze(1)
            t = t.expand(*x.shape[:-1], -1)

        elif t_emb == "cosine":
            # different shape for sampling
            if t.dim() == 0:
                t = t.unsqueeze(0)
            t = self.embed(t)
            t = t.expand(*x.shape[:-1], -1)

        else:
            raise NotImplementedError(f"t_emb={t_emb} not implemented")

        return t

    def encode(
        self, x: Tensor, mask: Tensor = None, ode_solver: str = "dopri5_zuko", ode_steps: int = 100
    ) -> Tensor:
        wrapped_cnf = ode_wrapper(model=self, cond=None, mask=mask)
        node = NeuralODE(wrapped_cnf, solver="rk4", sensitivity="adjoint")
        t_span = torch.linspace(0.0, 1.0, 100)
        traj = node.trajectory(x, t_span)
        return traj[-1]
        # return odeint(wrapped_cnf, x, 0.0, 1.0, phi=self.parameters())

    # TODO make code cleaner by not repeating code, add code to encode and use config to configure ode_solver
    def decode(
        self,
        z: Tensor,
        cond: Tensor,
        mask: Tensor = None,
        ode_solver: str = "dopri5_zuko",
        ode_steps: int = 100,
    ) -> Tensor:
        wrapped_cnf = ode_wrapper(model=self, cond=cond, mask=mask)
        if ode_solver == "dopri5_zuko":
            return odeint(wrapped_cnf, z, 1.0, 0.0, phi=self.parameters())
        elif ode_solver == "rk4":
            node = NeuralODE(wrapped_cnf, solver="rk4", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        elif ode_solver == "dopri5":  # adaptive
            node = NeuralODE(
                wrapped_cnf,
                solver="dopri5",
                atol=1e-4,
                rtol=1e-4,
                seminorm=True,
            )
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        elif ode_solver == "euler":
            node = NeuralODE(wrapped_cnf, solver="euler", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        elif ode_solver == "midpoint":
            node = NeuralODE(wrapped_cnf, solver="midpoint", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        elif ode_solver == "tsit5":  # adaptive
            node = NeuralODE(wrapped_cnf, solver="tsit5", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        elif ode_solver == "ieuler":
            node = NeuralODE(wrapped_cnf, solver="ieuler", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        elif ode_solver == "alf":
            node = NeuralODE(wrapped_cnf, solver="alf", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        # elif solver == "scipy":
        # return solve_ivp(wrapped_cnf, [1.0, 0.0], z[:, 0, 0].cpu(), vectorized=True)
        else:
            raise NotImplementedError(f"Solver {ode_solver} not implemented")

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


class FlowMatchingLoss2(nn.Module):
    def __init__(self, vs: nn.Module):
        super().__init__()

        self.vs = vs

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        # y = (1 - t) * x + (1e-6 + (1 - 1e-6) * t) * z
        # u = (1 - 1e-6) * z - x
        y = (1 - (1 - 1e-4) * t) * z + t * x
        u = x - (1 - 1e-4) * z
        for v in self.vs:
            y = v(t.squeeze(-1), y)
        return (y - u).square().mean()


class SetFlowMatchingLitModule(pl.LightningModule):
    """Pytorch Lightning module for training CNFs with Flow Matching loss.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler): Scheduler
        features (int, optional): Features of data. Defaults to 3.
        hidden_dim (int, optional): Hidden dimensions. Defaults to 128.
        num_particles (int, optional): Set cardinality. Defaults to 150.
        frequencies (int, optional): Time frequencies. Basically half the size of the time vector that is added to the model. Defaults to 6.
        layers (int, optional): Number of layers. Defaults to 8.
        n_transforms (int, optional): Number of flow transforms. Defaults to 1.
        global_cond_dim (int, optional): Dimension to concatenate to the global feature in EPiC Layer. Must be zero for no conditioning. Defaults to 0.
        local_cond_dim (int, optional): Dimension to concatenate to the Local MLPs in EPiC Model. Must be zero for no conditioning. Defaults to 0.
        activation (str, optional): Activation function. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper function. Defaults to "weight_norm".
        use_normaliser (bool, optional): Use layers that learn to normalise the input and conditioning. Defaults to True.
        normaliser_config (Mapping, optional): Normaliser config. Defaults to None.
        latent (int, optional): Latent dimension. Defaults to 16.
        return_latent_space (bool, optional): Return latent space. Defaults to False.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
        add_time_to_input (bool, optional): Concat time to input. Defaults to False.
        dropout (float, optional): Value for dropout layers. Defaults to 0.0.
        heads (int, optional): Number of attention heads. Defaults to 4.
        mask (bool, optional): Use Mask. Defaults to False.
        loss_type (str, optional): Loss type. Defaults to "FM-OT".
        t_emb (str, optional): Embedding for time. Defaults to "sincos".
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        features: int = 3,
        hidden_dim: int = 128,
        num_particles: int = 150,
        frequencies: int = 6,
        layers: int = 8,
        n_transforms: int = 1,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        use_normaliser: bool = False,
        normaliser_config: Mapping = {},
        # epic
        latent: int = 16,
        return_latent_space: bool = False,
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        add_time_to_input: bool = True,
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        # transformer
        dropout: float = 0.0,
        heads: int = 4,
        mask=False,
        # loss
        loss_type: str = "FM-OT",
        sigma: float = 1e-4,
        t_emb: str = "sincos",
        **kwargs,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        flows = nn.ModuleList()

        for _ in range(n_transforms):
            flows.append(
                CNF(
                    features=features,
                    hidden_dim=hidden_dim,
                    num_particles=num_particles,
                    frequencies=frequencies,
                    layers=layers,
                    global_cond_dim=global_cond_dim,
                    local_cond_dim=local_cond_dim,
                    latent=latent,
                    return_latent_space=return_latent_space,
                    dropout=dropout,
                    heads=heads,
                    mask=mask,
                    activation=activation,
                    wrapper_func=wrapper_func,
                    t_global_cat=t_global_cat,
                    t_local_cat=t_local_cat,
                    add_time_to_input=add_time_to_input,
                    t_emb=t_emb,
                )
            )
        self.flows = flows

        self.conditioned = global_cond_dim > 0
        if use_normaliser:
            self.normaliser = IterativeNormLayer(
                (features,),
                **normaliser_config,
            )
            if self.conditioned:
                self.ctxt_normaliser = IterativeNormLayer((global_cond_dim,), **normaliser_config)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor = None,
        mask: torch.Tensor = None,
        reverse: bool = False,
        ode_solver: str = "dopri5_zuko",
        ode_steps: int = 100,
    ):
        if reverse:
            for f in reversed(self.flows):
                x = f.decode(x, cond, mask, ode_solver=ode_solver, ode_steps=ode_steps)
        else:
            for f in self.flows:
                x = f.encode(x, mask, ode_solver=ode_solver, ode_steps=ode_steps)
        return x

    def loss(self, x: torch.Tensor, mask: torch.Tensor = None, cond: torch.Tensor = None):
        """Loss function.

        Args:
            x (torch.Tensor): Values.

        Raises:
            NotImplementedError: If loss type is not supported.

        Returns:
            _type_: Loss.
        """
        if self.hparams.loss_type == "FM-OT":
            t = torch.rand_like(torch.ones(x.shape[0]))
            t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1).unsqueeze(-1)
            t = t.type_as(x)

            logger_loss.debug(f"t: {t.shape}")

            z = torch.randn_like(x)

            logger_loss.debug(f"z: {z.shape}")
            y = (1 - t) * x + (self.hparams.sigma + (1 - self.hparams.sigma) * t) * z

            logger_loss.debug(f"y: {y.shape}")
            logger_loss.debug(f"y grad: {y.requires_grad}")

            u_t = (1 - self.hparams.sigma) * z - x
            u_t = u_t * mask

            logger_loss.debug(f"u_t: {u_t.shape}")

            temp = y.clone()
            for v in self.flows:
                temp = v(t.squeeze(-1), temp, mask=mask, cond=cond)
            v_t = temp.clone()

            logger_loss.debug(f"v_t grad: {v_t.requires_grad}")
            logger_loss.debug(f"v_t: {v_t.shape}")

            out = (v_t - u_t).square().mean()

        elif self.hparams.loss_type == "CFM":
            # from https://arxiv.org/abs/2302.00482
            sigma_t = 0.1

            t = torch.rand_like(torch.ones(x.shape[0]))
            t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1).unsqueeze(-1)
            t = t.type_as(x)

            x_0 = torch.randn_like(x)  # sample from prior
            x_1 = x  # conditioning

            logger_loss.debug(f"t: {t.shape}")
            logger_loss.debug(f"x_0: {x_0.shape}")
            logger_loss.debug(f"x_1: {x_1.shape}")

            mu_t = (1 - t) * x_1 + t * x_0
            y = mu_t + sigma_t * torch.randn_like(mu_t)

            u_t = x_0 - x_1
            u_t = u_t * mask

            logger_loss.debug(f"mu_t: {mu_t.shape}")
            logger_loss.debug(f"y: {y.shape}")
            logger_loss.debug(f"u_t: {u_t.shape}")

            temp = y.clone()
            for v in self.flows:
                temp = v(t.squeeze(-1), temp, mask=mask, cond=cond)
            v_t = temp.clone()

            out = (v_t - u_t).square().mean()

            logger_loss.debug(f"t squeeze: {t.squeeze(-1).shape}")
            logger_loss.debug(f"v_t: {v_t.shape}")
            logger_loss.debug(f"out: {out.shape}")

        elif self.hparams.loss_type == "CFM-OT":
            # from https://arxiv.org/abs/2302.00482

            sigma = 0.1

            x0 = torch.randn_like(x)  # sample from prior
            x1 = x  # wanted distribution

            t = torch.rand_like(torch.ones(x0.shape[0]))
            t = t.unsqueeze(-1).repeat_interleave(x0.shape[1], dim=1).unsqueeze(-1)
            t = t.type_as(x0)

            a, b = pot.unif(x0.size()[1]), pot.unif(x1.size()[1])
            a = np.repeat(np.expand_dims(a, axis=0), x0.size()[0], axis=0)
            b = np.repeat(np.expand_dims(b, axis=0), x1.size()[0], axis=0)

            M = torch.cdist(x0, x1) ** 2

            # for each set
            for k in range(M.shape[0]):
                M[k] = M[k] / M[k].max()
                pi = pot.emd(a[k], b[k], M[k].detach().cpu().numpy())
                p = pi.flatten()
                p = p / p.sum()

                choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=pi.shape[0])
                i, j = np.divmod(choices, pi.shape[1])

                x0[k] = x0[k, i]
                x1[k] = x1[k, j]
                mask_ot = mask[k, j]

            mu_t = x0 * t + x1 * (1 - t)
            sigma_t = sigma
            y = mu_t + sigma_t * torch.randn_like(x0)
            ut = x0 - x1
            ut = ut * mask_ot

            temp = y.clone()
            for v in self.flows:
                temp = v(t.squeeze(-1), temp, mask=mask_ot, cond=cond)
            vt = temp.clone()

            out = torch.mean((vt - ut) ** 2)

        else:
            raise NotImplementedError(f"loss_type {self.hparams.loss_type} not implemented")

        return out

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
        x, mask, cond = batch
        if self.hparams.use_normaliser:
            bool_mask = (mask.clone().detach() == 1).squeeze()
            x = self.normaliser(x, bool_mask)
            if self.conditioned:
                cond = self.ctxt_normaliser(cond)
        if not self.hparams.mask:
            mask = None

        loss = self.loss(x, mask=mask, cond=cond)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        # set same seed for every validation epoch
        torch.manual_seed(9999)

    def on_validation_epoch_end(self) -> None:
        torch.manual_seed(torch.seed())

    def validation_step(self, batch: Any, batch_idx: int):
        x, mask, cond = batch
        if self.hparams.use_normaliser:
            bool_mask = (mask.clone().detach() == 1).squeeze()
            x = self.normaliser(x, bool_mask)
            if self.conditioned:
                cond = self.ctxt_normaliser(cond)
        if not self.hparams.mask:
            mask = None

        loss = self.loss(x, mask, cond=cond)

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
        cond: torch.Tensor = None,
        mask: torch.Tensor = None,
        ode_solver: str = "dopri5_zuko",
        ode_steps: int = 100,
    ):
        """Generate Samples.

        Args:
            n_samples (int): Number of samples to generate.
            cond (torch.Tensor, optional): Data on which the model is conditioned. Defaults to None.
            mask (torch.Tensor, optional): Mask for data generation. Defaults to None.
            ode_solver (str, optonal): ODE solver to use. Defaults to "dopri5_zuko".

        Returns:
            torch.Tensor: Generated samples
        """
        z = torch.randn(n_samples, self.hparams.num_particles, self.hparams.features).to(
            self.device
        )
        if cond is not None:
            cond = cond.to(self.device)
            if self.hparams.use_normaliser:
                cond = self.ctxt_normaliser(cond)
        if mask is not None:
            mask = mask[:n_samples]
            mask = mask.to(self.device)
            z = z * mask
        samples = self.forward(
            z, cond=cond, mask=mask, reverse=True, ode_solver=ode_solver, ode_steps=ode_steps
        )
        if self.hparams.use_normaliser:
            samples = self.normaliser.reverse(samples, mask)
        return samples
