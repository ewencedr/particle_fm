"""Losses for the models."""

from typing import Mapping

import numpy as np
import ot as pot
import torch
import torch.nn as nn

from src.models.components.diffusion import VPDiffusionSchedule
from src.utils.pylogger import get_pylogger

logger_loss = get_pylogger("loss")


class FlowMatchingLoss(nn.Module):
    """Flow matching loss.

    from: https://arxiv.org/abs/2210.02747

    Args:
        flows (nn.ModuleList): Module list of flows
        sigma (float, optional): Sigma. Defaults to 1e-4.
    """

    def __init__(self, flows: nn.ModuleList, sigma: float = 1e-4, criterion: str = "mse"):
        super().__init__()
        self.flows = flows
        self.sigma = sigma
        self.criterion: str
        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="sum")
        elif criterion == "huber":
            self.criterion = nn.HuberLoss(reduction="sum")
        else:
            raise NotImplementedError(f"criterion {criterion} not supported")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, cond: torch.Tensor = None
    ) -> torch.Tensor:
        t = torch.rand_like(torch.ones(x.shape[0]))
        t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1)
        if len(x.shape) == 3:
            # for set data
            t = t.unsqueeze(-1)
        t = t.type_as(x)
        logger_loss.debug(f"t: {t.shape}")

        z = torch.randn_like(x)

        logger_loss.debug(f"z: {z.shape}")
        y = (1 - t) * x + (self.sigma + (1 - self.sigma) * t) * z

        logger_loss.debug(f"y: {y.shape}")
        logger_loss.debug(f"y grad: {y.requires_grad}")

        u_t = (1 - self.sigma) * z - x
        u_t = u_t * mask

        logger_loss.debug(f"u_t: {u_t.shape}")

        temp = y.clone()
        for v in self.flows:
            temp = v(t.squeeze(-1), temp, mask=mask, cond=cond)
        v_t = temp.clone()

        logger_loss.debug(f"v_t grad: {v_t.requires_grad}")
        logger_loss.debug(f"v_t: {v_t.shape}")

        # out = self.criterion(v_t, u_t) / mask.sum()
        sqrd = (v_t - u_t).square()
        out = sqrd.sum() / mask.sum()  # mean with ignoring masked values
        return out


class ConditionalFlowMatchingLoss(nn.Module):
    """Conditional Flow matching loss.

    from: https://arxiv.org/abs/2302.00482

    Args:
        flows (nn.ModuleList): Module list of flows
        sigma (float, optional): Sigma. Defaults to 1e-4.
    """

    def __init__(self, flows: nn.ModuleList, sigma: float = 1e-4, criterion: str = "mse"):
        super().__init__()
        self.flows = flows
        self.sigma = sigma
        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="sum")
        elif criterion == "huber":
            self.criterion = nn.HuberLoss(reduction="sum")
        else:
            raise NotImplementedError(f"criterion {criterion} not supported")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, cond: torch.Tensor = None
    ) -> torch.Tensor:
        t = torch.rand_like(torch.ones(x.shape[0]))
        t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1).unsqueeze(-1)
        t = t.type_as(x)

        x_0 = torch.randn_like(x)  # sample from prior
        x_1 = x  # conditioning

        logger_loss.debug(f"t: {t.shape}")
        logger_loss.debug(f"x_0: {x_0.shape}")
        logger_loss.debug(f"x_1: {x_1.shape}")

        mu_t = (1 - t) * x_1 + t * x_0
        y = mu_t + self.sigma * torch.randn_like(mu_t)

        u_t = x_0 - x_1
        u_t = u_t * mask

        logger_loss.debug(f"mu_t: {mu_t.shape}")
        logger_loss.debug(f"y: {y.shape}")
        logger_loss.debug(f"u_t: {u_t.shape}")

        temp = y.clone()
        for v in self.flows:
            temp = v(t.squeeze(-1), temp, mask=mask, cond=cond)
        v_t = temp.clone()

        out = self.criterion(v_t, u_t) / mask.sum()

        logger_loss.debug(f"t squeeze: {t.squeeze(-1).shape}")
        logger_loss.debug(f"v_t: {v_t.shape}")
        logger_loss.debug(f"out: {out.shape}")

        return out


# TODO work in progress - why numpy functions?
class ConditionalFlowMatchingOTLoss(nn.Module):
    """Conditional Flow matching Optimal Transport loss.

    from: https://arxiv.org/abs/2302.00482

    Args:
        flows (nn.ModuleList): Module list of flows
        sigma (float, optional): Sigma. Defaults to 1e-4.
    """

    def __init__(self, flows: nn.ModuleList, sigma: float = 1e-4, criterion: str = "mse"):
        super().__init__()
        self.flows = flows
        self.sigma = sigma
        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="sum")
        elif criterion == "huber":
            self.criterion = nn.HuberLoss(reduction="sum")
        else:
            raise NotImplementedError(f"criterion {criterion} not supported")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, cond: torch.Tensor = None
    ) -> torch.Tensor:
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
        sigma_t = self.sigma
        y = mu_t + sigma_t * torch.randn_like(x0)
        ut = x0 - x1
        ut = ut * mask_ot

        temp = y.clone()
        for v in self.flows:
            temp = v(t.squeeze(-1), temp, mask=mask_ot, cond=cond)
        vt = temp.clone()

        out = self.criterion(vt, ut) / mask.sum()

        return out


class DiffusionLoss(nn.Module):
    """Diffusion loss.

    from https://github.com/rodem-hep/PC-JeDi/blob/main/src/models/pc_jedi.py
    Args:
        flows (nn.ModuleList): Module list of flows
        sigma (float, optional): Sigma. Defaults to 1e-4.
    """

    def __init__(
        self,
        flows: nn.ModuleList,
        sigma: float = 1e-4,
        criterion: str = "huber",
        diff_config: Mapping = {"max_sr": 1, "min_sr": 1e-8},
    ):
        super().__init__()
        self.flows = flows
        self.sigma = sigma
        self.mle_loss_weight = 0.001
        self.diff_sched = VPDiffusionSchedule(**diff_config)
        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        elif criterion == "huber":
            self.criterion = nn.HuberLoss(reduction="none")
        else:
            raise NotImplementedError(f"criterion {criterion} not supported")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, cond: torch.Tensor = None
    ) -> torch.Tensor:
        # sample random uniform times
        t = torch.rand_like(torch.ones(x.shape[0]))
        t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1).unsqueeze(-1)
        t = t.type_as(x)
        logger_loss.debug(f"t: {t.shape}")

        # Sample from the gaussian latent space to perturb the point clouds
        z = torch.randn_like(x) * mask
        logger_loss.debug(f"z: {z.shape}")

        # renaming for clarity
        nodes = x
        noises = z
        diffusion_times = t.clone()
        diffusion_times = diffusion_times[:, 0]

        logger_loss.debug(f"times2 {diffusion_times.shape}")
        logger_loss.debug(f"times3 {diffusion_times.view(-1, 1, 1).shape}")

        # Get the signal and noise rates from the diffusion schedule
        signal_rates, noise_rates = self.diff_sched(diffusion_times.view(-1, 1, 1))

        # Mix the signal and noise according to the diffusion equation
        noisy_nodes = signal_rates * nodes + noise_rates * noises
        logger_loss.debug(f"noisy_nodes: {noisy_nodes.shape}")

        # Predict the noise using the network
        temp = noisy_nodes.clone()
        for v in self.flows:
            temp = v(t.squeeze(-1), temp, mask=mask, cond=cond)
        pred_noises = temp.clone()
        logger_loss.debug(f"pred_noises: {pred_noises.shape}")

        # Simple noise loss is for "perceptual quality"
        simple_loss = self.criterion(noises, pred_noises) * mask

        # MLE loss is for maximum liklihood training
        if self.mle_loss_weight:
            betas = self.diff_sched.get_betas(diffusion_times.view(-1, 1, 1))
            mle_weights = betas / noise_rates
            mle_loss = mle_weights * simple_loss
            out = (
                simple_loss.sum() / mask.sum() + self.mle_loss_weight * mle_loss.sum() / mask.sum()
            )
        else:
            out = simple_loss.sum() / mask.sum()

        return out


class DroidLoss(nn.Module):
    """Droid loss.

    Args:
        flows (nn.ModuleList): Module list of flows
        sigma (float, optional): Sigma. Defaults to 1e-4.
    """

    def __init__(self, flows: nn.ModuleList, sigma: float = 1e-4, criterion: str = "mse"):
        super().__init__()
        self.flows = flows
        self.sigma = sigma
        self.criterion: str
        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="sum")
        elif criterion == "huber":
            self.criterion = nn.HuberLoss(reduction="sum")
        else:
            raise NotImplementedError(f"criterion {criterion} not supported")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, cond: torch.Tensor = None
    ) -> torch.Tensor:
        t = torch.rand_like(torch.ones(x.shape[0]))
        t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1).unsqueeze(-1)
        t = t.type_as(x)

        logger_loss.debug(f"t: {t.shape}")

        z = torch.randn_like(x)

        logger_loss.debug(f"z: {z.shape}")
        # y = (1 - t) * x + (self.sigma + (1 - self.sigma) * t) * z
        y = x + t * z

        logger_loss.debug(f"y: {y.shape}")
        logger_loss.debug(f"y grad: {y.requires_grad}")

        # u_t = (1 - self.sigma) * z - x
        u_t = z * mask

        logger_loss.debug(f"u_t: {u_t.shape}")

        temp = y.clone()
        for v in self.flows:
            temp = v(t.squeeze(-1), temp, mask=mask, cond=cond)
        v_t = temp.clone()

        logger_loss.debug(f"v_t grad: {v_t.requires_grad}")
        logger_loss.debug(f"v_t: {v_t.shape}")

        # out = self.criterion(v_t, u_t) / mask.sum()
        sqrd = (v_t - u_t).square()
        out = sqrd.sum() / mask.sum()  # mean with ignoring masked values
        return out
