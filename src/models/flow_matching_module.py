from typing import Any, Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
from torchdyn.core import NeuralODE
from zuko.utils import odeint

from src.models.components.diffusion import VPDiffusionSchedule
from src.utils.pylogger import get_pylogger

from .components import EPiC_encoder, IterativeNormLayer, MDMA
from .components.droid_transformer import (
    FullCrossAttentionEncoder,
    FullTransformerEncoder,
)
from .components.losses import (
    ConditionalFlowMatchingLoss,
    ConditionalFlowMatchingOTLoss,
    DiffusionLoss,
    DroidLoss,
    FlowMatchingLoss,
)
from .components.solver import ddim_sampler, euler_maruyama_sampler
from .components.time_emb import CosineEncoding, GaussianFourierProjection
from .components.transformer import Transformer

logger = get_pylogger("fm_module")


# TODO put EPiC model config also in separate dictionary (net_config)
class ode_wrapper(torch.nn.Module):
    """Wraps model to ode solver compatible format. Also important for solving various types of
    ODEs.

    Args:
        model (torch.nn.Module): Model to wrap.
        mask (torch.Tensor, optional): Mask. Defaults to None.
        cond (torch.Tensor, optional): Condition. Defaults to None.
        loss_type (str, optional): Loss type. Defaults to "FM-OT".
        diff_config (Mapping, optional): Config for diffusion noise scheduling. Only necessary when using loss_type="diffusion". Defaults to {"max_sr": 0.999, "min_sr": 0.02}.
    """

    def __init__(
        self,
        model: nn.Module,
        mask: torch.Tensor = None,
        cond: torch.Tensor = None,
        loss_type: str = "FM-OT",
        diff_config: Mapping = {"max_sr": 0.999, "min_sr": 0.02},
    ):
        super().__init__()
        self.model = model
        self.mask = mask
        self.cond = cond
        self.loss_type = loss_type
        if self.loss_type == "diffusion":
            self.diff_sched = VPDiffusionSchedule(**diff_config)

    def forward(self, t, x, *args, **kwargs):
        if self.loss_type == "diffusion":
            expanded_shape = [-1] + [1] * (x.dim() - 1)
            _, noise_rates = self.diff_sched(t.view(expanded_shape))
            betas = self.diff_sched.get_betas(t.view(expanded_shape))
            return (
                -0.5 * betas * (x - self.model(t, x, mask=self.mask, cond=self.cond) / noise_rates)
            )
        else:
            return self.model(t, x, mask=self.mask, cond=self.cond)


class CNF(nn.Module):
    """Continuous Normalizing Flow with EPiC Generator or Transformer.

    Args:
        model (str, optional): Model to use. Defaults to "epic".
        features (int): Data features. Defaults to 3.
        num_particles (int, optional): Set cardinality. Defaults to 150.
        frequencies (int, optional): Frequency for time. Basically half the size of the time vector that is added to the model. Defaults to 6.
        hidden_dim (int, optional): Hidden dimensions. Defaults to 128.
        layers (int, optional): Number of Layers to use. Defaults to 8.
        global_cond_dim (int, optional): Dimension to concatenate to the global feature in EPiC Layer. Must be zero for no conditioning. Defaults to 0.
        local_cond_dim (int, optional): Dimension to concatenate to the Local MLPs in EPiC Model. Must be zero for no conditioning. Defaults to 0.
        dropout (float, optional): Dropout value for dropout layers. Defaults to 0.0.
        latent (int, optional): Latent dimension. Defaults to 16.
        activation (str, optional): Activation function. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper function. Defaults to "weight_norm".
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
        add_time_to_input (bool, optional): Concat time to input. Defaults to True.
        t_emb (str, optional): Embedding for time. Defaults to "sincos".
        loss_type (str, optional): Loss type. Defaults to "FM-OT".
        diff_config (Mapping, optional): Config for diffusion rate scheduling. Defaults to {"max_sr": 1, "min_sr": 1e-8}.
        sum_scale (float, optional): Factor that is multiplied with the sum pooling. Defaults to 1e-2.
        net_config (Mapping, optional): Config for Architecture. Defaults to {}.
    """

    def __init__(
        self,
        model: str = "epic",
        features: int = 3,
        num_particles: int = 150,
        frequencies: int = 6,
        hidden_dim: int = 128,
        layers: int = 8,
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        dropout: float = 0.0,
        latent: int = 16,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        add_time_to_input: bool = True,
        t_emb: str = "sincos",
        loss_type: str = "FM-OT",
        diff_config: Mapping[str, Any] = {"max_sr": 0.999, "min_sr": 0.02},
        sum_scale: float = 1e-2,
        net_config: Mapping[str, Any] = {},
    ):
        super().__init__()
        self.latent = latent
        self.add_time_to_input = add_time_to_input
        input_dim = features + 2 * frequencies if self.add_time_to_input else features

        if model == "epic":
            net_config = {
                "input_dim": input_dim,
                "feats": features,
                "latent": latent,
                "equiv_layers": layers,
                "hid_d": hidden_dim,
                "activation": activation,
                "wrapper_func": wrapper_func,
                "frequencies": frequencies,
                "num_points": num_particles,
                "t_local_cat": t_local_cat,
                "t_global_cat": t_global_cat,
                "global_cond_dim": global_cond_dim,
                "local_cond_dim": local_cond_dim,
                "dropout": dropout,
                "sum_scale": sum_scale,
            }
            self.net = EPiC_encoder(
                **net_config,
            )
        elif model == "transformer":
            self.net = Transformer(
                input_dim=input_dim,
                **net_config,
            )
        elif model == "droid_fulltransformer":
            self.net = FullTransformerEncoder(
                inpt_dim=input_dim,
                outp_dim=features,
                ctxt_dim=global_cond_dim + 2 * frequencies,
                **net_config,
            )
        elif model == "droid_fullcrossattention":
            self.net = FullCrossAttentionEncoder(
                inpt_dim=input_dim,
                outp_dim=features,
                ctxt_dim=global_cond_dim + 2 * frequencies,
                **net_config,
            )
        elif model == "mdma":
            self.net = MDMA(
                input_dim=input_dim,
                **net_config,
            )

        else:
            raise NotImplementedError(f"Model {model} not implemented.")

        self.register_buffer("frequencies", 2 ** torch.arange(frequencies) * torch.pi)
        self.activation = activation
        self.t_emb = t_emb
        self.loss_type = loss_type
        self.diff_config = diff_config
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

        x = self.net(t, x, cond, mask)

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

    # TODO make code cleaner by not repeating code, add code to encode and use config to configure ode_solver
    def decode(
        self,
        z: Tensor,
        cond: Tensor,
        mask: Tensor = None,
        ode_solver: str = "dopri5_zuko",
        ode_steps: int = 100,
    ) -> Tensor:
        wrapped_cnf = ode_wrapper(
            model=self,
            cond=cond,
            mask=mask,
            loss_type=self.loss_type,
            diff_config=self.diff_config,
        )
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
        elif ode_solver == "em" or ode_solver == "ddim":
            if self.loss_type == "diffusion":
                diff_sched = VPDiffusionSchedule(**self.diff_config)
                if ode_solver == "em":  # euler-maruyama
                    x = euler_maruyama_sampler(
                        self,
                        diff_sched=diff_sched,
                        initial_noise=z,
                        mask=mask,
                        cond=cond,
                        n_steps=ode_steps,
                    )
                elif ode_solver == "ddim":
                    x = ddim_sampler(
                        self,
                        diff_sched=diff_sched,
                        initial_noise=z,
                        mask=mask,
                        cond=cond,
                        n_steps=ode_steps,
                    )
                return x[0]
            else:
                raise SyntaxError(f"Solver {ode_solver} is only implemented for diffusion loss")
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
        net_config (Mapping, optional): Config for Architecture. Defaults to {}.
        latent (int, optional): Latent dimension. Defaults to 16.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
        add_time_to_input (bool, optional): Concat time to input. Defaults to False.
        dropout (float, optional): Value for dropout layers. Defaults to 0.0.
        sum_scale (float, optional): Factor that is multiplied with the sum pooling. Defaults to 1e-2.
        loss_type (str, optional): Loss type. Defaults to "FM-OT".
        t_emb (str, optional): Embedding for time. Defaults to "sincos".
        diff_config (Mapping, optional): Config for diffusion rate scheduling. Defaults to {"max_sr": 1, "min_sr": 1e-8}.
        criterion (str, optional): Criterion for loss. Defaults to "mse".
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model: str = "epic",
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
        net_config: Mapping = {},
        # epic
        latent: int = 16,
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        add_time_to_input: bool = True,
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        dropout: float = 0.0,
        sum_scale: float = 1e-2,
        # loss
        loss_type: str = "FM-OT",
        sigma: float = 1e-4,
        t_emb: str = "sincos",
        diff_config: Mapping = {"max_sr": 1, "min_sr": 1e-8},
        criterion: str = "mse",
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        flows = nn.ModuleList()

        for _ in range(n_transforms):
            flows.append(
                CNF(
                    model=model,
                    net_config=net_config,
                    features=features,
                    hidden_dim=hidden_dim,
                    num_particles=num_particles,
                    frequencies=frequencies,
                    layers=layers,
                    global_cond_dim=global_cond_dim,
                    local_cond_dim=local_cond_dim,
                    latent=latent,
                    dropout=dropout,
                    activation=activation,
                    wrapper_func=wrapper_func,
                    t_global_cat=t_global_cat,
                    t_local_cat=t_local_cat,
                    add_time_to_input=add_time_to_input,
                    t_emb=t_emb,
                    loss_type=loss_type,
                    diff_config=diff_config,
                    sum_scale=sum_scale,
                )
            )

        self.flows = flows
        self.conditioned = global_cond_dim > 0

        if loss_type == "FM-OT":
            self.loss = FlowMatchingLoss(flows=self.flows, sigma=sigma, criterion=criterion)
        elif loss_type == "CFM":
            self.loss = ConditionalFlowMatchingLoss(
                flows=self.flows, sigma=sigma, criterion=criterion
            )
        elif loss_type == "CFM-OT":
            self.loss = ConditionalFlowMatchingOTLoss(
                flows=self.flows, sigma=sigma, criterion=criterion
            )
        elif loss_type == "diffusion":
            self.loss = DiffusionLoss(
                flows=self.flows, sigma=sigma, diff_config=diff_config, criterion=criterion
            )
        elif loss_type == "droid":
            self.loss = DroidLoss(flows=self.flows, sigma=sigma, criterion=criterion)
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented.")

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
        if not self.trainer.datamodule.hparams.variable_jet_sizes:
            mask = None

        loss = self.loss(x, mask=mask, cond=cond)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.current_epoch % 20 == 0 and hasattr(
            self.trainer.datamodule.hparams, "loss_per_jettype"
        ):
            if self.trainer.datamodule.hparams.loss_per_jettype:
                jet_type_cond_mapping = {
                    jet_type: list(self.trainer.datamodule.names_conditioning).index(
                        f"jet_type_label_{jet_type}"
                    )
                    for jet_type in self.trainer.datamodule.hparams.used_jet_types
                }
                for jet_type, cond_idx in jet_type_cond_mapping.items():
                    mask_this_jet_type = cond[:, cond_idx] == 1
                    loss_this_jet_type = self.loss(
                        x[mask_this_jet_type][:10_000],
                        mask[mask_this_jet_type][:10_000],
                        cond=cond[mask_this_jet_type][:10_000],
                    )

                    self.log(
                        f"train/loss_{jet_type}",
                        loss_this_jet_type,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )

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
        if not self.trainer.datamodule.hparams.variable_jet_sizes:
            mask = None

        loss = self.loss(x, mask, cond=cond)

        # if specified, calculate loss for each jet type
        if self.current_epoch % 20 == 0 and hasattr(
            self.trainer.datamodule.hparams, "loss_per_jettype"
        ):
            if self.trainer.datamodule.hparams.loss_per_jettype:
                jet_type_cond_mapping = {
                    jet_type: list(self.trainer.datamodule.names_conditioning).index(
                        f"jet_type_label_{jet_type}"
                    )
                    for jet_type in self.trainer.datamodule.hparams.used_jet_types
                }
                for jet_type, cond_idx in jet_type_cond_mapping.items():
                    mask_this_jet_type = cond[:, cond_idx] == 1
                    loss_this_jet_type = self.loss(
                        x[mask_this_jet_type],
                        mask[mask_this_jet_type],
                        cond=cond[mask_this_jet_type],
                    )

                    self.log(
                        f"val/loss_{jet_type}",
                        loss_this_jet_type,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        ode_solver: str = "midpoint",
        ode_steps: int = 100,
    ):
        """Generate Samples.

        Args:
            n_samples (int): Number of samples to generate.
            cond (torch.Tensor, optional): Data on which the model is conditioned. Defaults to None.
            mask (torch.Tensor, optional): Mask for data generation. Defaults to None.
            ode_solver (str, optional): ODE solver to use. Defaults to "dopri5_zuko".

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
