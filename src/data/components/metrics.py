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
        jets1 (Union[np.array, Tensor]): Jets from the real data.
            Shape: (num_jets, num_particles, num_features)
            The first 3 features are assumed to be (eta, phi, pt)
        jets2 (Union[np.array, Tensor]): Jets from the generated data
            Shape: (num_jets, num_particles, num_features)
            The first 3 features are assumed to be (eta, phi, pt)
        mask1 (Union[np.array, Tensor]): Mask for the real data. Defaults to None.
        mask2 (Union[np.array, Tensor]): Mask for the generated data. Defaults to None.
        num_eval_samples (int, optional): Number of jets out of the total to
            use for W1 measurement. Defaults to 50,000.
        num_batches (int, optional): Number of different batches to average W1
            scores over. Defaults to 5.
        calculate_efps (bool, optional): Calculate W1M_efps. Defaults to True.
        use_masks (bool, optional): Use mask for w1p calculation. Otherwise
            exclude zero padded values. Defaults to False.

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


def kl_divergence(p, q, rescale: bool = False, verbose: bool = False):
    r"""
    Calculate the Kullback-Leibler divergence between two probability distributions.
    KLD(P || Q) = \sum_i [ P(i) * log(P(i) / Q(i)) ]
    (could have also used scipy.stats.entropy, but this implementation is more flexible
    and allows for (not) rescaling the distributions to sum to 1).
    The scipy implementation rescales by default.

    Parameters:
    - p: Target probability distribution P
    - q: Approximated probability distribution Q

    Returns:
    - kl_divergence: KL divergence between Q and P
    """
    # Ensure that the distributions sum to 1
    if not np.isclose(np.sum(p), 1):
        if verbose:
            print("Warning: Distribution p does not sum to 1")
            print(f"Sum of p: {np.sum(p)}")
        if rescale:
            if verbose:
                print("Rescaling p to sum to 1")
            p = p / np.sum(p)
    if not np.isclose(np.sum(q), 1):
        if verbose:
            print("Warning: Distribution q does not sum to 1")
            print(f"Sum of q: {np.sum(q)}")
        if rescale:
            if verbose:
                print("Rescaling q to sum to 1")
            q = q / np.sum(q)

    # Avoid division by zero and log(0) by setting the result to 0 when P(i) or Q(i) is 0
    p_or_q_zero = np.logical_or(p == 0, q == 0)
    kld = np.sum(np.where(p_or_q_zero, 0, p * np.log(p / q)))

    return kld


def histedges_equalN(x, nbin):
    """Return the edges of nbin equiprobable bins of x.

    Parameters:
    - x: Array of values to be binned
    - nbin: Number of bins

    Returns:
    - edges: Edges of the bins
    """
    number_of_points = len(x)
    # the interp below first fits a curve to the function x(i) which
    # basically has on the x-axis the indices of the array x and on the
    # y-axis the values of the (sorted) array x.

    # Then, it interpolates the curve at nbin + 1 points from 0 to number_of_points
    # which returns the corresponding (linearly interpolated) values of x.
    return np.interp(
        x=np.linspace(0, number_of_points, nbin + 1),
        xp=np.arange(number_of_points),
        fp=np.sort(x),
    )


def calc_reverse_kld(
    target,
    approx,
    nbins: int = 100,
    return_pi_qi_bins: bool = False,
    clip_approx: bool = False,
    rescale_pq: bool = False,
    verbose: bool = False,
):
    r"""
    Calculate the reverse KL divergence between two probability distributions.
    Reverse KL divergence is defined as KL(Q || P) = \sum_i [ Q(i) * log(Q(i) / P(i)) ]
    (i.e. swap P and Q compared to the normal KL divergence).

    Parameters:
    - target: Target probability distribution P
    - approx: Approximated probability distribution Q
    - nbins: Number of equiprobable bins to use for the KL divergence calculation
    - return_pi_qi_bins: Return the discrete probability distributions p_i and q_i
        and the bins used for the calculation
    - clip_approx: Clip the approximated distribution to the range of the target distribution
    - rescale_pq: Rescale the distributions to sum to 1
    - verbose: Print additional information

    Returns:
    - kld_value: Reverse KL divergence between Q and P
    - p_i: Discrete probability distribution for P (if return_pi_qi_bins is True)
    - q_i: Discrete probability distribution for Q (if return_pi_qi_bins is True)
    - bins: Bins used for the calculation (if return_pi_qi_bins is True)
    """
    # first, calculate the bins which have equal number of points
    # from the target distribution
    equiprobable_bins = histedges_equalN(target, nbins)

    # calculate discrete probability distributions for those bins
    p_i = np.histogram(target, bins=equiprobable_bins)[0] / len(target)

    # if specified, clip the approximated distribution to the range of the
    # target distribution
    if clip_approx:
        if verbose:
            print("Clipping approximated distribution to range of target distribution")
        approx = np.clip(approx, a_min=equiprobable_bins[0], a_max=equiprobable_bins[-1])
    q_i = np.histogram(approx, bins=equiprobable_bins)[0] / len(approx)

    # calculate the reverse KL divergence (reverse by swapping p and q
    # compared to the normal KL divergence)
    kld_value = kl_divergence(p=q_i, q=p_i, rescale=rescale_pq, verbose=verbose)
    if return_pi_qi_bins:
        return kld_value, p_i, q_i, equiprobable_bins
    return kld_value


def reversed_kl_divergence_batched(
    target: np.array,
    approx: np.array,
    num_eval_samples: int,
    num_batches: int,
    nbins: int = 100,
    clip_approx: bool = False,
    rescale_pq: bool = False,
    verbose: bool = False,
):
    """Calculate the reverse KL divergence between two probability distributions multiple times and
    return mean and std.

    Parameters:
    - target: Target probability distribution P
    - approx: Approximated probability distribution Q
    - num_eval_samples: Number of samples to use for each KL divergence calculation
    - num_batches: Number of times to calculate the KL divergence
    - nbins: Number of equiprobable bins to use for the KL divergence calculation
    - clip_approx: Clip the approximated distribution to the range of the target distribution
    - rescale_pq: Rescale the distributions to sum to 1
    - verbose: Print additional information

    Returns:
    - reversed_kld_mean: Mean of the reverse KL divergence over all batches
    - reversed_kld_std: Standard deviation of the reverse KL divergence over all batches
    """

    seed = 42
    rng = np.random.default_rng(seed)

    if len(target.shape) > 1:
        print("Warning: Target distribution has more than one dimension")
        print(f"target.shape: {target.shape}")
        print("---> Batches will be selected along first dimension, and then flattened")

    reversed_kld_values = []
    for _ in range(num_batches):
        # select random samples from the target and approximated distributions
        rand1 = rng.choice(len(target), size=num_eval_samples)
        rand2 = rng.choice(len(approx), size=num_eval_samples)
        rand_target = target[rand1]
        rand_approx = approx[rand2]
        # flatten the arrays if they have more than one dimension
        # (we want this to happen after the random selection, since we want to select
        # random samples from the first dimension, and then include all entries
        # from the other dimensions)
        # i.e. we want to select random jets, and then include all particles from
        # those jets
        if len(rand_target.shape) > 1:
            rand_target = rand_target.flatten()
        if len(rand_approx.shape) > 1:
            rand_approx = rand_approx.flatten()
        # calculate the reverse KL divergence with equiprobable bins
        reversed_kld, p_i, q_i, bins = calc_reverse_kld(
            target=rand_target,
            approx=rand_approx,
            nbins=nbins,
            clip_approx=clip_approx,
            return_pi_qi_bins=True,
            rescale_pq=rescale_pq,
            verbose=verbose,
        )
        if np.isnan(reversed_kld):
            print("Warning: NaN encountered in reversed KL divergence")
            print(f"p_i: {p_i}")
            print(f"q_i: {q_i}")
            print(f"frac: {p_i / q_i}")
            print(f"logfrac: {np.log(p_i / q_i)}")
        reversed_kld_values.append(reversed_kld)

    reversed_kld_mean = np.mean(reversed_kld_values)
    reversed_kld_std = np.std(reversed_kld_values)
    return reversed_kld_mean, reversed_kld_std
