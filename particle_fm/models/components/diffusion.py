# adapted from https://github.com/rodem-hep/PC-JeDi/blob/main/src/models/diffusion.py

import math
from typing import Tuple

import torch as T


class VPDiffusionSchedule:
    def __init__(self, max_sr: float = 1, min_sr: float = 1e-2) -> None:
        self.max_sr = max_sr
        self.min_sr = min_sr

    def __call__(self, time: T.Tensor) -> T.Tensor:
        return cosine_diffusion_shedule(time, self.max_sr, self.min_sr)

    def get_betas(self, time: T.Tensor) -> T.Tensor:
        return cosine_beta_shedule(time, self.max_sr, self.min_sr)


def cosine_diffusion_shedule(
    diff_time: T.Tensor, max_sr: float = 1, min_sr: float = 1e-2
) -> Tuple[T.Tensor, T.Tensor]:
    """Calculates the signal and noise rate for any point in the diffusion processes.

    Using continuous diffusion times between 0 and 1 which make switching between
    different numbers of diffusion steps between training and testing much easier.
    Returns only the values needed for the jump forward diffusion step and the reverse
    DDIM step.
    These are sqrt(alpha_bar) and sqrt(1-alphabar) which are called the signal_rate
    and noise_rate respectively.

    The jump forward diffusion process is simply a weighted sum of:
        input * signal_rate + eps * noise_rate

    Uses a cosine annealing schedule as proposed in
    Proposed in https://arxiv.org/abs/2102.09672

    Args:
        diff_time: The time used to sample the diffusion scheduler
            Output will match the shape
            Must be between 0 and 1
        max_sr: The initial rate at the first step
        min_sr: How much signal is preserved at end of diffusion
            (can't be zero due to log)
    """

    # Use cosine annealing, which requires switching from times -> angles
    start_angle = math.acos(max_sr)
    end_angle = math.acos(min_sr)
    diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
    signal_rates = T.cos(diffusion_angles)
    noise_rates = T.sin(diffusion_angles)
    return signal_rates, noise_rates


def cosine_beta_shedule(diff_time: T.Tensor, max_sr: float = 1, min_sr: float = 1e-2) -> T.Tensor:
    """Returns the beta values for the continuous flows using the above cosine scheduler."""
    start_angle = math.acos(max_sr)
    end_angle = math.acos(min_sr)
    diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
    return 2 * (end_angle - start_angle) * T.tan(diffusion_angles)
