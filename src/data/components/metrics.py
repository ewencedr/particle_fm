from typing import Union

import numpy as np
from jetnet.evaluation import w1efp, w1m, w1p
from scipy.stats import wasserstein_distance
from torch import Tensor

rng = np.random.default_rng()


def wasserstein_distance_batched(
    data1: np.array, data2: np.array, num_eval_samples: int, num_batches: int
):
    """Calculate the Wasserstein distance between two datasets multiple times and return mean and
    std.

    Args:
        data1 (np.array): Data1 usually the real data, can be num_batches times smaller than data2
        data2 (np.array): Data2 usually the generated data, can be num_batches times larger than data1
        num_eval_samples (int): Number of samples to use for each Wasserstein distance calculation
        num_batches (int): Number of batches to split the data into

    Returns:
        float: Mean Wasserstein distance of all batches
        float: Standard deviation of the Wasserstein distances of all batches
    """
    w1 = []
    for _ in range(num_batches):
        rand1 = rng.choice(len(data1), size=num_eval_samples)
        rand2 = rng.choice(len(data2), size=num_eval_samples)
        rand_sample1 = data1[rand1]
        rand_sample2 = data2[rand2]
        w1.append(wasserstein_distance(rand_sample1, rand_sample2))
    return np.mean(w1), np.std(w1)


def calculate_wasserstein_metrics_jets(
    jet_data1: np.array,
    jet_data2: np.array,
    num_eval_samples: int = 50_000,
    num_batches: int = 40,
    **kwargs,
):
    """Calculate the Wasserstein distance for the jet coordinates (pt, eta, phi, mass)

    Args:
        data1 (np.array): Data1 usually the real data, can be num_batches times smaller than data2
        data2 (np.array): Data2 usually the generated data, can be num_batches times larger than data1
        num_eval_samples (int): Number of samples to use for each Wasserstein distance calculation
        num_batches (int): Number of batches to split the data into

    Returns:
        float: Mean Wasserstein distance of all batches
        float: Standard deviation of the Wasserstein distances of all batches
    """

    pt, pt_std = wasserstein_distance_batched(
        jet_data1[:, 0], jet_data2[:, 0], num_eval_samples, num_batches
    )
    eta, eta_std = wasserstein_distance_batched(
        jet_data1[:, 1], jet_data2[:, 1], num_eval_samples, num_batches
    )
    phi, phi_std = wasserstein_distance_batched(
        jet_data1[:, 2], jet_data2[:, 2], num_eval_samples, num_batches
    )
    mass, mass_std = wasserstein_distance_batched(
        jet_data1[:, 3], jet_data2[:, 3], num_eval_samples, num_batches
    )

    dic_jet_w1 = {
        "w1pt_jet_mean": pt,
        "w1pt_jet_std": pt_std,
        "w1eta_jet_mean": eta,
        "w1eta_jet_std": eta_std,
        "w1phi_jet_mean": phi,
        "w1phi_jet_std": phi_std,
        "w1mass_jet_mean": mass,
        "w1mass_jet_std": mass_std,
    }
    return dic_jet_w1


def calculate_all_wasserstein_metrics(
    jets1: Union[np.array, Tensor],
    jets2: Union[np.array, Tensor],
    mask1: Union[np.array, Tensor] = None,
    mask2: Union[np.array, Tensor] = None,
    num_eval_samples: int = 50_000,
    num_batches: int = 5,
    calculate_efps: bool = True,
    use_masks: bool = False,
):
    """Calculate the Wasserstein distances w1m, w1p and w1efp with standard deviations.

    Args:
        jets1 (Union[np.array, Tensor]): Jets from the real data
        jets2 (Union[np.array, Tensor]): Jets from the generated data
        mask1 (Union[np.array, Tensor]): Mask for the real data. Defaults to None.
        mask2 (Union[np.array, Tensor]): Mask for the generated data. Defaults to None.
        num_eval_samples (int, optional): Number of jets out of the total to use for W1 measurement. Defaults to 50,000.
        num_batches (int, optional): Number of different batches to average W1 scores over. Defaults to 5.
        calculate_efps (bool, optional): Calculate W1M_efps. Defaults to True.
        use_masks (bool, optional): Use mask for w1p calculation. Otherwise exclude zero padded values. Defaults to False.

    Returns:
        dict{w1m_mean, w1p_mean, w1efp_mean, w1m_std, w1p_std, w1efp_std}
    """

    jets1 = jets1[..., :3]
    jets2 = jets2[..., :3]
    if not use_masks:
        mask1 = None
        mask2 = None

    w1m_mean, w1m_std = w1m(
        jets1=jets1,
        jets2=jets2,
        num_eval_samples=num_eval_samples,
        num_batches=num_batches,
        return_std=True,
    )
    w1p_mean, w1p_std = w1p(
        jets1=jets1,
        jets2=jets2,
        mask1=mask1,
        mask2=mask2,
        exclude_zeros=True,
        num_particle_features=0,
        num_eval_samples=num_eval_samples,
        num_batches=num_batches,
        return_std=True,
    )
    w1efp_mean, w1efp_std = 0, 0
    if calculate_efps:
        w1efp_mean, w1efp_std = w1efp(
            jets1=jets1,  # real jets
            jets2=jets2,  # fake jets
            use_particle_masses=False,
            num_eval_samples=num_eval_samples,
            num_batches=num_batches,
            return_std=True,
        )

    w_dists = {
        "w1m_mean": w1m_mean,
        "w1m_std": w1m_std,
        "w1p_mean": np.mean(w1p_mean),
        "w1p_std": np.mean(w1p_std),
        "w1efp_mean": np.mean(w1efp_mean),
        "w1efp_std": np.mean(w1efp_std),
    }
    return w_dists
