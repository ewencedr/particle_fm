"""Generation of data with the models."""

import time
from typing import Mapping

import numpy as np
import numpy.ma as ma
import torch
from tqdm import tqdm

from src.data.components.utils import inverse_normalize_tensor

# TODO put ODE solver in config


def generate_data(
    model,
    num_jet_samples: int,
    batch_size: int = 256,
    cond: torch.Tensor = None,
    device: str = "cuda",
    variable_set_sizes: bool = False,
    mask: torch.Tensor = None,
    normalized_data: bool = False,
    normalize_sigma: int = 5,
    means=None,
    stds=None,
    shuffle_mask: bool = False,
    ode_solver: str = "dopri5_zuko",
    ode_steps: int = 100,
):
    """Generate data with a model in batches and measure time.

    Args:
        model (_type_): Model with sample method
        num_jet_samples (int): Number of jet samples to generate
        batch_size (int, optional): Batch size for generation. Defaults to 256.
        cond (torch.Tensor, optional): Conditioned data if model is conditioned. Defaults to None.
        device (str, optional): Device on which the data is generated. Defaults to "cuda".
        variable_set_sizes (bool, optional): Use variable set sizes. Defaults to False.
        mask (torch.Tensor, optional): Mask for generating variable set sizes. Defaults to None.
        normalized_data (bool, optional): Normalized data. Defaults to False.
        normalize_sigma (int, optional): Sigma for normalized data. Defaults to 5.
        means (_type_, optional): Means for normalized data. Defaults to None.
        stds (_type_, optional): Standard deviations for normalized data. Defaults to None.
        shuffle_mask (bool, optional): Shuffle mask during generation. Defaults to False.
        ode_solver (str, optional): ODE solver for sampling. Defaults to "dopri5_zuko".
        ode_steps (int, optional): Number of steps for ODE solver. Defaults to 100.

    Raises:
        ValueError: _description_

    Returns:
        np.array: sampled data of shape (num_jet_samples, num_particles, num_features) with features (eta, phi, pt)
        float: generation time
    """
    if variable_set_sizes and mask is None:
        raise ValueError("Please use mask when using variable_set_sizes=True")
    if len(mask) != num_jet_samples:
        raise ValueError(
            f"Mask should have the same length as num_jet_samples ({len(mask)} != {num_jet_samples})"
        )
    print(f"Generating data. Device: {torch.device(device)}")
    particle_data_sampled = torch.Tensor()
    start_time = 0
    for i in tqdm(range(num_jet_samples // batch_size)):
        if cond is not None:
            cond_batch = cond[i * batch_size : (i + 1) * batch_size]
        else:
            cond_batch = None
        if i == 1:
            start_time = time.time()
        if variable_set_sizes:
            if shuffle_mask:
                permutation = np.random.permutation(len(mask))
                mask = mask[permutation]
                mask_batch = mask[:batch_size]
            else:
                mask_batch = mask[i * batch_size : (i + 1) * batch_size]
        else:
            mask_batch = None
        with torch.no_grad():
            jet_samples_batch = (
                model.to(torch.device(device))
                .sample(
                    batch_size,
                    cond_batch,
                    mask_batch,
                    ode_solver=ode_solver,
                    ode_steps=ode_steps,
                )
                .cpu()
            )
        if normalized_data:
            jet_samples_batch = inverse_normalize_tensor(
                jet_samples_batch, means, stds, sigma=normalize_sigma
            )
        if variable_set_sizes:
            jet_samples_batch = jet_samples_batch * mask_batch
        particle_data_sampled = torch.cat((particle_data_sampled, jet_samples_batch))

    end_time = time.time()

    if num_jet_samples % batch_size != 0:
        remaining_samples = num_jet_samples - (num_jet_samples // batch_size * batch_size)
        if cond is not None:
            cond_batch = cond[-remaining_samples:]
        else:
            cond_batch = None
        if variable_set_sizes:
            if shuffle_mask:
                permutation = np.random.permutation(len(mask))
                mask = mask[permutation]
                mask_batch = mask[-remaining_samples:]
            else:
                mask_batch = mask[-remaining_samples:]
        else:
            mask_batch = None
        with torch.no_grad():
            jet_samples_batch = (
                model.to(torch.device(device))
                .sample(
                    remaining_samples,
                    cond_batch,
                    mask_batch,
                    ode_solver=ode_solver,
                    ode_steps=ode_steps,
                )
                .cpu()
            )
        if normalized_data:
            jet_samples_batch = inverse_normalize_tensor(
                jet_samples_batch, means, stds, sigma=normalize_sigma
            )
        if variable_set_sizes:
            jet_samples_batch = jet_samples_batch * mask_batch
        particle_data_sampled = torch.cat((particle_data_sampled, jet_samples_batch))

    particle_data_sampled = np.array(particle_data_sampled)
    generation_time = end_time - start_time
    return particle_data_sampled, generation_time
