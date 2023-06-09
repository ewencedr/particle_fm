import warnings
from typing import Any, List

import energyflow as ef
import numpy as np
import ot as pot
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from torch import Tensor
from torch.distributions import Normal
from torchdyn.core import NeuralODE
from zuko.utils import odeint

from src.data.components.utils import jet_masses
from src.utils.pylogger import get_pylogger

from .components import EPiC_discriminator, EPiC_generator, Transformer
from .components.utils import SWD, MMDLoss, calculate_gradient_penalty

logger = get_pylogger("fm_module")
logger_loss = get_pylogger("fm_module_loss")


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.

    Inspired by https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # logger.debug(f"GFP w: {self.W.shape}")
        # logger.debug(f"x[:,None]{x[:, None].shape}")
        # logger.debug(f"self.W[None,:]:{self.W[None,:].shape}")
        x_proj = x[..., None] * self.W[None, ...] * 2 * np.pi
        # logger.debug(f"x_proj: {x_proj.shape}")
        # logger.debug(f"torch cat: {torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).shape}")
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


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
        model (str, optional): Use Transformer or EPiC Generator as architecture. Defaults to "transformer".
        num_particles (int, optional): Set cardinality. Defaults to 150.
        frequencies (int, optional): Frequency for time. Basically half the size of the time vector that is added to the model. Defaults to 6.
        hidden_dim (int, optional): Hidden dimensions. Defaults to 128.
        layers (int, optional): Number of Layers to use. Defaults to 8.
        mass_conditioning (bool, optional): Condition the model on the jet mass. Defaults to False.
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
        t_emb (str, optional): Embedding for time. Defaults to "sincos".
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
        t_emb: str = "sincos",
    ):
        super().__init__()
        self.model = model
        self.latent = latent
        self.mass_conditioning = mass_conditioning
        if self.model == "transformer":
            # TODO doesn't work anymore
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
            if mass_conditioning and global_cond_dim == 0 and local_cond_dim == 0:
                raise ValueError(
                    "If mass_conditioning is True, global_cond_dim or local_cond_dim must be > 0"
                )
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

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        cond: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        # time embedding
        if self.t_emb == "sincos":
            logger.debug(f"t.shape: {t.shape}")
            t = self.frequencies * t[..., None]  # (batch_size,num_particles,frequencies)
            logger.debug(f"t.shape1: {t.shape}")

            t = torch.cat((t.cos(), t.sin()), dim=-1)  # (batch_size,num_particles,2*frequencies)
            # logger.debug(f"t.shape2: {t[:3]}")
            t = t.expand(*x.shape[:-1], -1)  # (batch_size,num_particles,2*frequencies)
            logger.debug(f"t.shape3: {t.shape}")
        elif self.t_emb == "gaussian":
            # Obtain the Gaussian random feature embedding for t
            # logger.debug(f"t.shape: {t.shape}")
            test = False
            # different shape for training and sampling
            if len(t.shape) == 2:
                t = t[:, 0]
                test = True
            # logger.debug(f"t:{t[:3]}")
            # logger.debug(f"t.shape: {t.shape}")
            t = getattr(F, self.activation, lambda x: x)(self.embed(t))
            # logger.debug(f"t2.shape: {t.shape}")
            t = self.linear(t).unsqueeze(1)
            logger.debug(f"t3.shape: {t.shape}")
            t = t.expand(*x.shape[:-1], -1)
            # if test:
            #    t = t.unsqueeze(1).repeat_interleave(x.shape[1], dim=1)
            logger.debug(f"t4.shape: {t.shape}")
        else:
            raise NotImplementedError(f"t_emb={self.t_emb} not implemented")

        x = torch.cat((t, x), dim=-1)  # (batch_size,num_particles,features+2*frequencies)
        # logger.debug(f"x.shape2: {x[:3]}")

        if self.model == "epic":
            x_global = torch.randn_like(torch.ones(x.shape[0], self.latent, device=x.device))
            x_local = x
            if self.mass_conditioning:
                if cond is None:
                    cond = jet_masses(x_local).unsqueeze(-1)
                logger.debug(f"mass.shape: {cond.shape}")
            else:
                cond = None
            x = self.net(t, x_global, x_local, cond, mask)

        else:
            x = self.net(x)

        return x

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
        optimizer_d (torch.optim.Optimizer): Optimizer for discriminator
        scheduler (torch.optim.lr_scheduler): Scheduler
        scheduler_d (torch.optim.lr_scheduler): Scheduler for discriminator
        model (str, optional): Use Transformer or EPiC Generator as model. Defaults to "epic".
        features (int, optional): Features of data. Defaults to 3.
        hidden_dim (int, optional): Hidden dimensions. Defaults to 128.
        num_particles (int, optional): Set cardinality. Defaults to 150.
        frequencies (int, optional): Time frequencies. Basically half the size of the time vector that is added to the model. Defaults to 6.
        use_mass_loss (bool, optional): Add mass term to loss. Defaults to True.
        layers (int, optional): Number of layers. Defaults to 8.
        n_transforms (int, optional): Number of flow transforms. Defaults to 1.
        mass_conditioning (bool, optional): Condition the model on the jet mass. Defaults to False.
        global_cond_dim (int, optional): Dimension to concatenate to the global feature in EPiC Layer. Must be zero for no conditioning. Defaults to 0.
        local_cond_dim (int, optional): Dimension to concatenate to the Local MLPs in EPiC Model. Must be zero for no conditioning. Defaults to 0.
        activation (str, optional): Activation function. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper function. Defaults to "weight_norm".
        latent (int, optional): Latent dimension. Defaults to 16.
        return_latent_space (bool, optional): Return latent space. Defaults to False.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
        dropout (float, optional): Value for dropout layers. Defaults to 0.0.
        heads (int, optional): Number of attention heads. Defaults to 4.
        mask (bool, optional): Use Mask. Defaults to False.
        loss_type (str, optional): Loss type. Defaults to "FM-OT".
        loss_comparison (str, optional): Which method to use for comparing the two VFs in FM loss. Defaults to "MSE".
        loss_type_d (str, optional): Loss type for discriminator. Defaults to "BCE".
        t_emb (str, optional): Embedding for time. Defaults to "sincos".
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_d: torch.optim.lr_scheduler = None,
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
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        # transformer
        dropout: float = 0.0,
        heads: int = 4,
        mask=False,
        # debug
        plot_loss_hist_debug: bool = False,
        # loss
        loss_type: str = "FM-OT",
        sigma: float = 1e-4,
        loss_comparison: str = "MSE",
        loss_type_d: str = "LSGAN",
        t_emb: str = "sincos",
        **kwargs,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        if loss_comparison == "adversarial":
            self.automatic_optimization = (
                False  # we will do our own optimization for Adversarial loss
            )
        flows = nn.ModuleList()
        # losses = nn.ModuleList()
        if optimizer_d is not None and self.hparams.loss_comparison != "adversarial":
            warnings.warn(
                "Optimizer for discriminator is not None but loss comparison is not adversarial!"
            )
        if scheduler_d is not None and self.hparams.loss_comparison != "adversarial":
            warnings.warn(
                "Scheduler for discriminator is not None but loss comparison is not adversarial!"
            )
        if loss_type_d is not None and self.hparams.loss_comparison != "adversarial":
            warnings.warn(
                "Loss type for discriminator is not None but loss comparison is not adversarial!"
            )
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
                    t_emb=t_emb,
                )
            )
            # losses.append(FlowMatchingLoss(flows[-1]))
        self.flows = flows
        self.u_mass = []
        self.v_mass = []
        self.data = []
        self.data_x = []
        self.data_ut = []
        self.data_vt = []
        self.v_mass_tensor = torch.empty(0, 30, 3)
        if loss_comparison == "SWD":
            self.swd = SWD()
        elif loss_comparison == "MMD":
            self.mmd = MMDLoss()
        elif loss_comparison == "adversarial":
            self.discriminator = EPiC_discriminator(
                feats=features,
                equiv_layers=layers,
                hid_d=hidden_dim,
                latent=latent,
                activation=activation,
                wrapper_func=wrapper_func,
                t_global_cat=t_global_cat,
                t_local_cat=t_local_cat,
            )

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

    def adversarial_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        loss_type_d: str = "BCE",
        generator: bool = True,
    ):
        """Discriminator loss. Thanks to Copilot for the code.

        Args:
            y_hat (torch.Tensor): Values.
            y (torch.Tensor): Values.
            loss_type_d (str, optional): Possible losses: BCE, LSGAN, WGAN, Hinge. Defaults to "BCE".
            generator (bool, optional): Whether to use generator or discriminator. Defaults to True.

        Raises:
            ValueError: If loss type is not supported.

        Returns:
            torch.Tensor [1]: Loss.
        """
        if loss_type_d == "LSGAN":
            # LSGAN (https://agustinus.kristia.de/techblog/2017/03/02/least-squares-gan/)
            # print(f"y_hat: {y_hat}")
            loss = torch.mean((y_hat - y) ** 2)
            if generator:
                loss = loss * 0.5
        elif loss_type_d == "WGAN":
            # WGAN (https://arxiv.org/abs/1701.07875)
            if generator:
                # print(f"y_hat: {y_hat}")
                loss = -torch.mean(y_hat)
                # print(f"loss: {loss}")
        elif loss_type_d == "BCE":
            # BCE (https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        elif loss_type_d == "Hinge":
            # Hinge (https://arxiv.org/abs/1705.07215)
            loss = torch.mean(F.relu(1 - y * y_hat))
        else:
            raise NotImplementedError("Loss type not supported!")
        return loss

    def loss(self, x: torch.Tensor, mask: torch.Tensor = None):
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
                temp = v(t.squeeze(-1), temp, mask=mask)
            v_t = temp.clone()

            logger_loss.debug(f"v_t grad: {v_t.requires_grad}")
            logger_loss.debug(f"v_t: {v_t.shape}")

            if self.hparams.loss_comparison == "MSE":
                out = (v_t - u_t).square().mean()
            elif self.hparams.loss_comparison == "SWD":
                out = self.swd(v_t, u_t)
            elif self.hparams.loss_comparison == "MMD":
                # TODO NOT WORKING YET
                out = self.mmd(v_t, u_t)
            elif self.hparams.loss_comparison == "adversarial":
                # TODO mask for adversarial loss
                clip_gradients = True
                epochs_pretrain_generator = 0

                optimizer, optimizer_d = self.optimizers()
                if self.trainer.current_epoch < epochs_pretrain_generator:
                    self.toggle_optimizer(optimizer)
                    out = (v_t - u_t).square().mean()
                    optimizer.zero_grad()
                    self.manual_backward(out)
                    if clip_gradients:
                        self.clip_gradients(
                            optimizer,
                            gradient_clip_val=0.5,
                            gradient_clip_algorithm="norm",
                        )
                    optimizer.step()

                    self.untoggle_optimizer(optimizer)
                    return out
                else:
                    if self.hparams.scheduler is not None and self.hparams.scheduler_d is not None:
                        scheduler = self.lr_schedulers()
                    elif (
                        self.hparams.scheduler is not None and self.hparams.scheduler_d is not None
                    ):
                        scheduler, scheduler_d = self.lr_schedulers()
                    elif self.hparams.scheduler is None and self.hparams.scheduler_d is not None:
                        scheduler_d = self.lr_schedulers()

                    # train generator
                    self.toggle_optimizer(optimizer)

                    # ground truth result (ie: all fake)
                    valid = torch.ones(v_t.size(0), 1)
                    # add noise to labels to stabilise training
                    noise = -torch.rand_like(valid) * 0.05
                    logger_loss.debug(f"valid: {valid.shape}")
                    valid = valid.type_as(v_t) + noise.type_as(v_t)

                    # Measure generator's ability to fool the discriminator
                    g_loss = self.adversarial_loss(
                        self.discriminator(t.squeeze(-1), self.flows[0](t.squeeze(-1), y)),
                        valid,
                        loss_type_d=self.hparams.loss_type_d,
                    )
                    self.log(
                        "train/g_loss",
                        g_loss,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                    )
                    logger_loss.debug(f"g_loss grad: {g_loss.requires_grad}")
                    self.manual_backward(g_loss)
                    if clip_gradients:
                        self.clip_gradients(
                            optimizer,
                            gradient_clip_val=0.5,
                            gradient_clip_algorithm="norm",
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    self.untoggle_optimizer(optimizer)

                    # train discriminator
                    # Measure discriminator's ability to classify between both vector fields
                    self.toggle_optimizer(optimizer_d)

                    if self.hparams.loss_type_d == "WGAN":
                        fake_data = self.flows[0](t.squeeze(-1).detach(), y.detach())
                        discr_out_real = self.discriminator(t.squeeze(-1), u_t)
                        discr_out_fake = self.discriminator(
                            t.squeeze(-1),
                            fake_data,
                        )

                        errD_real = torch.mean(discr_out_real)
                        errD_fake = torch.mean(discr_out_fake)
                        gradient_penalty = calculate_gradient_penalty(
                            self.discriminator,
                            t.squeeze(-1),
                            u_t,
                            fake_data,
                            device=self.device,
                        )
                        # print(f"errD_real: {errD_real}")
                        # print(f"errD_fake: {errD_fake}")
                        # print(f"gradient penalty: {gradient_penalty}")
                        d_loss = -errD_real + errD_fake + gradient_penalty * 10
                    else:
                        # how well can it label as real?
                        valid = torch.ones(v_t.size(0), 1)
                        noise = -torch.rand_like(valid) * 0.05
                        valid = valid.type_as(v_t) + noise.type_as(v_t)

                        real_loss = self.adversarial_loss(
                            self.discriminator(t.squeeze(-1), u_t),
                            valid,
                            loss_type_d=self.hparams.loss_type_d,
                            generator=False,
                        )
                        self.log(
                            "train/d_real_loss",
                            real_loss,
                            on_step=True,
                            on_epoch=True,
                            prog_bar=True,
                        )

                        # how well can it label as fake?
                        fake = torch.zeros(v_t.size(0), 1)
                        noise = torch.rand_like(fake) * 0.05
                        fake = fake.type_as(v_t) + noise.type_as(v_t)

                        fake_loss = self.adversarial_loss(
                            self.discriminator(
                                t.squeeze(-1),
                                self.flows[0](t.squeeze(-1).detach(), y.detach()),
                            ),
                            fake,
                            loss_type_d=self.hparams.loss_type_d,
                        )
                        self.log(
                            "train/d_fake_loss",
                            fake_loss,
                            on_step=True,
                            on_epoch=True,
                            prog_bar=True,
                        )
                        # discriminator loss is the average of these
                        d_loss = (real_loss + fake_loss) * 0.5
                    self.log(
                        "train/d_loss",
                        d_loss,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                    )
                    self.manual_backward(d_loss)
                    if clip_gradients:
                        self.clip_gradients(
                            optimizer_d,
                            gradient_clip_val=0.5,
                            gradient_clip_algorithm="norm",
                        )
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                    self.untoggle_optimizer(optimizer_d)

                    if self.trainer.is_last_batch:
                        if (
                            self.hparams.scheduler is not None
                            and self.hparams.scheduler_d is not None
                        ):
                            scheduler.step()
                        elif (
                            self.hparams.scheduler is not None
                            and self.hparams.scheduler_d is not None
                        ):
                            scheduler.step()
                            scheduler_d.step()
                        elif (
                            self.hparams.scheduler is None and self.hparams.scheduler_d is not None
                        ):
                            scheduler_d.step()

                    out = d_loss
            else:
                raise NotImplementedError
            logger_loss.debug(f"out: {out.shape}")

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
                temp = v(t.squeeze(-1), temp, mask=mask)
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

            mu_t = x0 * t + x1 * (1 - t)
            sigma_t = sigma
            y = mu_t + sigma_t * torch.randn_like(x0)
            ut = x0 - x1

            temp = y.clone()
            for v in self.flows:
                temp = v(t.squeeze(-1), temp, mask=mask)
            vt = temp.clone()

            out = torch.mean((vt - ut) ** 2)

        else:
            raise NotImplementedError(f"loss_type {self.hparams.loss_type} not implemented")

        if self.hparams.use_mass_loss:
            mass_scaling_factor = 0.0001 * 1
            jm_v = jet_masses(v_t)
            jm_u = jet_masses(u_t)
            mass_mse = (jm_v - jm_u).square().mean()
            logger.debug(f"jet_mass_diff: {mass_mse*mass_scaling_factor}")
            logger.debug(f"out: {out}")
            return (
                out,  # + mass_mse * mass_scaling_factor,
                mass_mse * mass_scaling_factor,
                u_t,
                v_t,
                x,
            )
        else:
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
        x, mask = batch
        if not self.hparams.mask:
            mask = None

        if self.hparams.use_mass_loss:
            loss, mse_mass, u_mass, v_mass, x = self.loss(x, mask)
            self.log("train/mse_mass", mse_mass, on_step=False, on_epoch=True, prog_bar=True)
            if self.hparams.plot_loss_hist_debug:
                if batch_idx == 0:
                    self.u_mass = []
                    self.v_mass = []
                    self.data = []
                    self.data_ut = torch.empty(0, 30, 3)
                    self.data_vt = torch.empty(0, 30, 3)
                    self.data_x = torch.empty(0, 30, 3)
                    self.v_mass_tensor = torch.empty(0, 30, 3)
                # if batch_idx = 400:
                #    pass
                # print(f"vmass: {v_mass.shape}")
                # self.v_mass.append(v_mass.cpu().detach())
                if batch_idx % 50 == 0:
                    self.v_mass_tensor = torch.cat((self.v_mass_tensor, u_mass.cpu().detach()), 0)
                    self.data_x = torch.cat((self.data_x, x.cpu().detach()), 0)
                    self.data_ut = torch.cat((self.data_ut, u_mass.cpu().detach()), 0)
                    self.data_vt = torch.cat((self.data_vt, v_mass.cpu().detach()), 0)

                # print(f"vmass tensor: {self.v_mass_tensor.shape}")
                # self.u_mass.append(u_mass.cpu().detach().numpy())
                # self.v_mass.append(v_mass.cpu().detach().numpy())
                # print(f"self.test_mass: {np.array(self.test_mass).shape}")

                # self.data.append(data)
                if (
                    batch_idx == 450
                    and self.trainer.current_epoch % 3 == 0
                    and self.trainer.current_epoch > 0
                ):
                    # TODO Still work in progress
                    # TODO Check if the mass loss is working
                    print(f"BATCH_IDX {batch_idx}")
                    # print(f"vmass self: {np.array(self.v_mass).shape}")
                    print(f"vmass tensor: {self.v_mass_tensor.shape}")
                    plot1 = False
                    if plot1:
                        data = (
                            odeint(
                                self.flows[0],
                                self.v_mass_tensor.cuda(),
                                None,
                                1.0,
                                0.0,
                                phi=self.parameters(),
                            )
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        # data = self.v_mass_tensor.cpu().detach().numpy()
                        print(f"data: {data.shape}")
                        self.data_x = self.data_x.numpy()
                        print(f"data_x: {self.data_x.shape}")
                        # self.v_mass.append(data)
                        # print(self.trainer.current_epoch)
                        # print(f"batch_idx: {batch_idx}")
                        # print(f"mse_mass: {mse_mass.shape}")
                        # plt.hist(
                        #    np.array(self.u_mass).flatten(),
                        #    bins=100,
                        #    label="u",
                        #    histtype="stepfilled",
                        # )
                        # data = np.array(self.data)
                        fig = plt.figure(figsize=(20, 4))
                        gs = GridSpec(1, 4)

                        #####

                        # eta
                        ax = fig.add_subplot(gs[0])
                        i_feat = 0
                        bins = np.linspace(-0.5, 0.5, 50)
                        # print(f"data1: {data.shape}")
                        # print(f"data2: {np.concatenate(data).shape}")
                        # print(f"data type: {type(data)}")
                        ax.hist(
                            (np.concatenate(data))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="Gen",
                        )
                        ax.hist(
                            (np.concatenate(self.data_x))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="x",
                        )
                        ax.set_xlabel(r"$\eta^\mathrm{rel}$")
                        ax.get_yaxis().set_ticklabels([])
                        ax.set_yscale("log")
                        ax.legend()

                        # phi
                        ax = fig.add_subplot(gs[1])

                        i_feat = 1

                        bins = np.linspace(-0.5, 0.5, 50)
                        ax.hist(
                            (np.concatenate(data))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="Gen",
                        )
                        ax.hist(
                            (np.concatenate(self.data_x))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="x",
                        )
                        ax.set_xlabel(r"$\phi^\mathrm{rel}$")
                        ax.get_yaxis().set_ticklabels([])
                        ax.set_yscale("log")
                        ax.legend()

                        # pt
                        ax = fig.add_subplot(gs[2])

                        i_feat = 2

                        bins = np.linspace(-0.1, 0.5, 100)
                        ax.hist(
                            (np.concatenate(data))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="Gen",
                        )
                        ax.hist(
                            (np.concatenate(self.data_x))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="x",
                        )

                        ax.set_xlabel(r"$p_\mathrm{T}^\mathrm{rel}$")
                        ax.get_yaxis().set_ticklabels([])
                        ax.set_yscale("log")
                        ax.legend()

                        # mass
                        def jet_masses_ef(jets_ary):
                            jets_p4s = ef.p4s_from_ptyphims(jets_ary)
                            masses = ef.ms_from_p4s(jets_p4s.sum(axis=1))
                            return masses

                        ax = fig.add_subplot(gs[3])

                        bins = np.linspace(0.0, 0.3, 100)

                        jet_mass = jet_masses_ef(
                            np.array([data[:, :, 2], data[:, :, 0], data[:, :, 1]]).transpose(
                                1, 2, 0
                            )
                        )
                        jet_mass_x = jet_masses_ef(
                            np.array(
                                [
                                    self.data_x[:, :, 2],
                                    self.data_x[:, :, 0],
                                    self.data_x[:, :, 1],
                                ]
                            ).transpose(1, 2, 0)
                        )
                        # print(f"data: {data[0]}")
                        # print(f"jet_mass: {jet_mass}")
                        ax.hist(
                            jet_mass,
                            histtype="step",
                            bins=100,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="Gen",
                        )
                        ax.hist(
                            jet_mass_x,
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="x",
                        )

                        ax.set_xlabel(r"Jet mass")
                        ax.set_yscale("log")
                        ax.legend()

                        plt.tight_layout()
                        plt.show()

                    plot_ut_vt = True
                    if plot_ut_vt:
                        data = self.data_ut.cpu().detach().numpy()
                        data1 = self.data_vt.cpu().detach().numpy()
                        fig = plt.figure(figsize=(20, 4))
                        gs = GridSpec(1, 4)

                        #####

                        # eta
                        ax = fig.add_subplot(gs[0])
                        i_feat = 0
                        bins = np.linspace(-0.5, 0.5, 50)
                        # print(f"data1: {data.shape}")
                        # print(f"data2: {np.concatenate(data).shape}")
                        # print(f"data type: {type(data)}")
                        ax.hist(
                            (np.concatenate(data))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="ut",
                        )
                        ax.hist(
                            (np.concatenate(data1))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="vt",
                        )
                        ax.set_xlabel(r"$\eta^\mathrm{rel}$")
                        ax.get_yaxis().set_ticklabels([])
                        ax.set_yscale("log")
                        ax.legend()

                        # phi
                        ax = fig.add_subplot(gs[1])

                        i_feat = 1

                        bins = np.linspace(-0.5, 0.5, 50)
                        ax.hist(
                            (np.concatenate(data))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="ut",
                        )
                        ax.hist(
                            (np.concatenate(data1))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="vt",
                        )
                        ax.set_xlabel(r"$\phi^\mathrm{rel}$")
                        ax.get_yaxis().set_ticklabels([])
                        ax.set_yscale("log")
                        ax.legend()

                        # pt
                        ax = fig.add_subplot(gs[2])

                        i_feat = 2

                        bins = np.linspace(-0.1, 0.5, 100)
                        ax.hist(
                            (np.concatenate(data))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="ut",
                        )
                        ax.hist(
                            (np.concatenate(data1))[:, i_feat],
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="vt",
                        )

                        ax.set_xlabel(r"$p_\mathrm{T}^\mathrm{rel}$")
                        ax.get_yaxis().set_ticklabels([])
                        ax.set_yscale("log")
                        ax.legend()

                        # mass
                        def jet_masses_ef(jets_ary):
                            jets_p4s = ef.p4s_from_ptyphims(jets_ary)
                            masses = ef.ms_from_p4s(jets_p4s.sum(axis=1))
                            return masses

                        ax = fig.add_subplot(gs[3])

                        bins = np.linspace(0.0, 0.3, 100)

                        jet_mass = jet_masses_ef(
                            np.array([data[:, :, 2], data[:, :, 0], data[:, :, 1]]).transpose(
                                1, 2, 0
                            )
                        )
                        jet_mass_x = jet_masses_ef(
                            np.array(
                                [
                                    data1[:, :, 2],
                                    data1[:, :, 0],
                                    data1[:, :, 1],
                                ]
                            ).transpose(1, 2, 0)
                        )
                        # print(f"data: {data[0]}")
                        # print(f"jet_mass: {jet_mass}")
                        ax.hist(
                            jet_mass,
                            histtype="step",
                            bins=100,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="ut",
                        )
                        ax.hist(
                            jet_mass_x,
                            histtype="step",
                            bins=bins,
                            density=True,
                            lw=2,
                            ls="--",
                            alpha=0.7,
                            label="vt",
                        )

                        ax.set_xlabel(r"Jet mass")
                        ax.set_yscale("log")
                        ax.legend()

                        plt.tight_layout()
                        plt.show()

        else:
            loss = self.loss(x, mask=mask)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        # set same seed for every validation epoch
        torch.manual_seed(9999)

    def on_validation_epoch_end(self) -> None:
        torch.manual_seed(torch.seed())

    def validation_step(self, batch: Any, batch_idx: int):
        x, mask = batch
        if self.trainer.current_epoch == 0:
            # Just to have something logged so that the checkpoint callback doesn't fail
            self.log("w1m_mean", 0.005)
            self.log("w1p_mean", 0.005)
        if not self.hparams.mask:
            mask = None

        if self.hparams.use_mass_loss:
            loss, mse_mass, u_mass, v_mass, x = self.loss(x, mask)
            self.log("val/mse_mass", mse_mass, on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.loss(x, mask)
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
        # TODO check if parameters are correctly passed to optimizer
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

        if self.hparams.loss_comparison == "adversarial":
            optimizer_d = self.hparams.optimizer(params=self.discriminator.parameters())
            if self.hparams.scheduler_d is not None:
                scheduler_d = self.hparams.scheduler(optimizer=optimizer)
                opt_d = {
                    "optimizer": optimizer_d,
                    "lr_scheduler": {
                        "scheduler": scheduler_d,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            else:
                opt_d = {"optimizer": optimizer_d}
            return [opt, opt_d]
        else:
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
        if mask is not None:
            mask = mask[:n_samples]
            mask = mask.to(self.device)
            z = z * mask
        samples = self.forward(
            z, cond=cond, mask=mask, reverse=True, ode_solver=ode_solver, ode_steps=ode_steps
        )
        return samples
