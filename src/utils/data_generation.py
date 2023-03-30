"""Generation of data with the models."""

import time

import numpy as np
import torch
from tqdm import tqdm

from src.data.components.utils import inverse_normalize_tensor


def generate_data(
    model,
    num_jet_samples,
    batch_size,
    particles_per_jet=30,
    device="cuda",
    mgpu_model=False,
    max_particles=False,
    mask=None,
    normalised_data=False,
    means=None,
    stds=None,
    shuffle_mask=False,
):
    """Generate Data
    model: model
    mgpu_model: boolean: whether the Model is being used that is capable of being used by multiple gpus
    """

    if max_particles and mask is None:
        raise ValueError("Please use mask when using max_particles=True")
    print(f"Generating data. Device: {torch.device(device)}")
    particle_data_sampled = torch.Tensor()
    start_time = 0
    for i in tqdm(range(num_jet_samples // batch_size)):
        if i == 1:
            start_time = time.time()
        if max_particles:
            if shuffle_mask:
                permutation = np.random.permutation(len(mask))
                mask = mask[permutation]
        if mgpu_model:
            if max_particles:
                if shuffle_mask:
                    mask_batch = mask[:batch_size]
                else:
                    mask_batch = mask[i * batch_size : (i + 1) * batch_size]

                with torch.no_grad():
                    jet_samples_batch = model.to(torch.device(device)).sample(batch_size).cpu()
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
                jet_samples_batch = jet_samples_batch * mask_batch
            else:
                with torch.no_grad():
                    jet_samples_batch = model.to(torch.device(device)).sample(batch_size).cpu()
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
        else:
            if max_particles:
                if shuffle_mask:
                    mask_batch = mask[:batch_size]
                else:
                    mask_batch = mask[i * batch_size : (i + 1) * batch_size]
                with torch.no_grad():
                    jet_samples_batch = model.to(torch.device(device)).sample(batch_size).cpu()
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
                jet_samples_batch = jet_samples_batch * mask_batch
            else:
                with torch.no_grad():
                    jet_samples_batch = model.to(torch.device(device)).sample(batch_size).cpu()
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
        particle_data_sampled = torch.cat((particle_data_sampled, jet_samples_batch))

    end_time = time.time()

    if num_jet_samples % batch_size != 0:
        remaining_samples = num_jet_samples - (num_jet_samples // batch_size * batch_size)
        if max_particles:
            # rng.shuffle(mask)
            mask_batch = mask[-remaining_samples:]
            # mask = torch.reshape(mask,(remaining_samples*150,1))
        if mgpu_model:
            if max_particles:
                with torch.no_grad():
                    jet_samples_batch = (
                        model.to(torch.device(device)).sample(remaining_samples).cpu()
                    )
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
                jet_samples_batch = jet_samples_batch * mask_batch
            else:
                with torch.no_grad():
                    jet_samples_batch = (
                        model.to(torch.device(device)).sample(remaining_samples).cpu()
                    )
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
        else:
            if max_particles:
                with torch.no_grad():
                    jet_samples_batch = (
                        model.to(torch.device(device)).sample(remaining_samples).cpu()
                    )
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
                jet_samples_batch = jet_samples_batch * mask_batch
            else:
                with torch.no_grad():
                    jet_samples_batch = (
                        model.to(torch.device(device)).sample(remaining_samples).cpu()
                    )
                if normalised_data:
                    jet_samples_batch = inverse_normalize_tensor(jet_samples_batch, means, stds)
        particle_data_sampled = torch.cat((particle_data_sampled, jet_samples_batch))
    particle_data_sampled = np.array(particle_data_sampled)
    generation_time = end_time - start_time
    # if(max_particles):
    #    particle_data_sampled = np.reshape(particle_data_sampled,(-1,particles_per_jet,3))
    return particle_data_sampled, generation_time
