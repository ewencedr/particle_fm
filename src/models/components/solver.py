"""Solver for the DDIM process adapted by https://github.com/rodem-hep/PC-
JeDi/blob/main/src/models/diffusion.py."""

from typing import Optional, Tuple

import torch
from particle_fm.models.components.diffusion import VPDiffusionSchedule
from tqdm import tqdm


def ddim_predict(
    noisy_data: torch.Tensor,
    pred_noises: torch.Tensor,
    signal_rates: torch.Tensor,
    noise_rates: torch.Tensor,
) -> torch.Tensor:
    """Use a single ddim step to predict the final image from anywhere in the diffusion process."""
    return (noisy_data - noise_rates * pred_noises) / signal_rates


@torch.no_grad()
def ddim_sampler(
    model,
    diff_sched: VPDiffusionSchedule,
    initial_noise: torch.Tensor,
    n_steps: int = 50,
    keep_all: bool = False,
    mask: Optional[torch.Tensor] = None,
    cond: Optional[torch.BoolTensor] = None,
    clip_predictions: Optional[tuple] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, list]:
    """Apply the DDIM sampling process to generate a batch of samples from noise.

    Args:
        model: A denoising diffusion model
            Requires: inpt_dim, device, forward() method that outputs pred noise
        diff_sched: A diffusion schedule object to calculate signal and noise rates
        initial_noise: The initial noise to pass through the process
            If none it will be generated here
        n_steps: The number of iterations to generate the samples
        keep_all: Return all stages of diffusion process
            Can be memory heavy for large batches
        num_samples: How many samples to generate
            Ignored if initial_noise is provided
        mask: The mask for the output point clouds
        cond: The context tensor for the output point clouds
        clip_predictions: Can stabalise generation by clipping the outputs
        verbose: Whether to show the progress bar
    """

    # Get the initial noise for generation and the number of sammples
    num_samples = initial_noise.shape[0]

    # The shape needed for expanding the time encodings
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)

    # Check the input argument for the n_steps, must be less than what was trained
    all_stages = []
    step_size = 1 / n_steps

    # The initial variables needed for the loop
    noisy_data = initial_noise
    diff_times = torch.ones(num_samples, device=noisy_data.device)
    next_signal_rates, next_noise_rates = diff_sched(diff_times.view(expanded_shape))
    for step in tqdm(range(n_steps), "DDIM-sampling", leave=False, disable=not verbose):
        # Update with the previous 'next' step
        signal_rates = next_signal_rates
        noise_rates = next_noise_rates

        # Keep track of the diffusion evolution
        if keep_all:
            all_stages.append(noisy_data)

        # Apply the denoise step to get X_0 and expected noise
        pred_noises = model(diff_times[0], noisy_data, mask=mask, cond=cond)
        pred_data = ddim_predict(noisy_data, pred_noises, signal_rates, noise_rates)

        # Get the next predicted components using the next signal and noise rates
        diff_times = diff_times - step_size
        next_signal_rates, next_noise_rates = diff_sched(diff_times.view(expanded_shape))

        # Clamp the predicted X_0 for stability
        if clip_predictions is not None:
            pred_data.clamp_(*clip_predictions)

        # Remix the predicted components to go from estimated X_0 -> X_{torch-1}
        noisy_data = next_signal_rates * pred_data + next_noise_rates * pred_noises

    return pred_data, all_stages


@torch.no_grad()
def euler_maruyama_sampler(
    model,
    diff_sched: VPDiffusionSchedule,
    initial_noise: torch.Tensor,
    n_steps: int = 50,
    keep_all: bool = False,
    mask: Optional[torch.Tensor] = None,
    cond: Optional[torch.BoolTensor] = None,
    clip_predictions: Optional[tuple] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, list]:
    """Apply the full reverse process to noise to generate a batch of samples."""

    # Get the initial noise for generation and the number of sammples
    num_samples = initial_noise.shape[0]

    # The shape needed for expanding the time encodings
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)

    # Check the input argument for the n_steps, must be less than what was trained
    all_stages = []
    delta_t = 1 / n_steps

    # The initial variables needed for the loop
    x_t = initial_noise
    t = torch.ones(num_samples, device=x_t.device)
    for step in tqdm(range(n_steps), "Euler-Maruyama-sampling", leave=False, disable=not verbose):
        # Use the model to get the expected noise
        pred_noises = model(t[0], x_t, mask=mask, cond=cond)

        # Use to get s_theta
        _, noise_rates = diff_sched(t.view(expanded_shape))
        s = -pred_noises / noise_rates

        # Take one step using the em method
        betas = diff_sched.get_betas(t.view(expanded_shape))
        x_t += 0.5 * betas * (x_t + 2 * s) * delta_t
        x_t += (betas * delta_t).sqrt() * torch.randn_like(x_t)
        t -= delta_t

        # Keep track of the diffusion evolution
        if keep_all:
            all_stages.append(x_t)

        # Clamp the denoised data for stability
        if clip_predictions is not None:
            x_t.clamp_(*clip_predictions)

    return x_t, all_stages
