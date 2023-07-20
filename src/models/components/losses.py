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
        t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1).unsqueeze(-1)
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
