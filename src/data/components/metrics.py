from typing import Union

import jetnet
import numpy as np
from scipy.stats import wasserstein_distance
from torch import Tensor


def wasserstein_distance_batched(data1: np.array, data2: np.array, num_batches: int):
    """Calculate the Wasserstein distance between two datasets multiple times and return mean and
    std.

    Args:
        data1 (np.array): Data1 usually the real data, can be num_batches times smaller than data2
        data2 (np.array): Data2 usually the generated data, can be num_batches times larger than data1
        num_batches (int): Number of batches to split the data into

    Returns:
        float: Mean Wasserstein distance of all batches
        float: Standard deviation of the Wasserstein distances of all batches
    """
    num_eval_samples = len(data1) // num_batches
    w1 = []
    i = 0
    for _ in range(num_batches):
        rand1 = [*range(0, len(data1))]  # whole real dataset
        rand2 = [*range(i, i + num_eval_samples)]  # batches of the fake dataset
        i += num_eval_samples
        rand_sample1 = data2[rand1]
        rand_sample2 = data2[rand2]
        w1.append(wasserstein_distance(rand_sample1, rand_sample2))
    return np.mean(w1), np.std(w1)


def calculate_all_wasserstein_metrics(
    particle_data1,
    particle_data2,
    mask1,
    mask2,
    num_eval_samples=10000,
    num_batches=5,
    calculate_efps=True,
    use_masks=False,
):
    """Calculate the Wasserstein distances w1m, w1p and w1efp with standard deviations.

    Args:
        particle_data1 (_type_): _description_
        particle_data2 (_type_): _description_
        mask1 (_type_): _description_
        mask2 (_type_): _description_
        num_eval_samples (int, optional): _description_. Defaults to 10000.
        num_batches (int, optional): _description_. Defaults to 5.
        calculate_efps (bool, optional): _description_. Defaults to True.
        use_masks (bool, optional): Use mask for w1m calculation. Otherwise exclude zero padded values. Defaults to False.

    Returns:
        w1m_mean, w1p_mean, w1efp_mean, w1m_std, w1p_std, w1efp_std
    """

    particle_data1 = particle_data1[..., :3]
    particle_data2 = particle_data2[..., :3]
    if not use_masks:
        mask1 = None
        mask2 = None

    w1m_mean, w1m_std = w1m(
        particle_data1,
        particle_data2,
        num_eval_samples=num_eval_samples,
        num_batches=num_batches,
        return_std=True,
    )
    # print(f"w1m mean: {w1m_mean}")
    # print(f"w1m std: {w1m_std}")
    w1p_mean, w1p_std = w1p(
        particle_data1,  # real_jets
        particle_data2,  # fake_jets
        mask1=mask1,
        mask2=mask2,
        exclude_zeros=True,
        num_particle_features=3,
        num_eval_samples=num_eval_samples,
        num_batches=num_batches,
        average_over_features=True,
        return_std=True,
    )
    # print(f"w1p_mean: {w1p_mean}")
    # print(f"w1p_std: {w1p_std}")
    w1efp_mean, w1efp_std = 0, 0
    if calculate_efps:
        w1efp_mean, w1efp_std = w1efp(
            jets1=particle_data1,  # real jets
            jets2=particle_data2,  # fake jets
            use_particle_masses=False,
            efpset_args=[("n==", 4), ("d==", 4), ("p==", 1)],
            num_eval_samples=num_eval_samples,
            num_batches=num_batches,
            average_over_efps=True,
            return_std=True,
            efp_jobs=None,
        )
    # print(f"w1efp_mean: {w1efp_mean}")
    # print(f"w1efp_std: {w1efp_std}")
    w_dists = {
        "w1m_mean": w1m_mean,
        "w1p_mean": w1p_mean,
        "w1efp_mean": w1efp_mean,
        "w1m_std": w1m_std,
        "w1p_std": w1p_std,
        "w1efp_std": w1efp_std,
    }
    return w_dists


def w1m(
    jets1: Union[Tensor, np.ndarray],
    jets2: Union[Tensor, np.ndarray],
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    return_std: bool = True,
):
    """Calculate the Wasserstein distance between two mass distributions of jets multiple times and
    return the mean.

    Args:
        jets1 (Union[Tensor, np.ndarray]): Real Jets - Particle Data (events, particles, features), features: [eta, phi, pt, (optional) mass]
        jets2 (Union[Tensor, np.ndarray]): Fake Jets - Particle Data (events, particles, features), features: [eta, phi, pt, (optional) mass]
        num_eval_samples (int, optional): Samples on which the wasserstein distance is calculated. Defaults to 10000.
        num_batches (int, optional): How often the wasserstein distance is calculated on num_eval_samples. Defaults to 5.
        return_std (bool, optional): _description_. Defaults to True.

    Returns:
        w1m_mean (float): Mean of all wasserstein distance calculations
        w1m_std (float): Standard deviaton of all wasserstein distance calculations
    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    # shuffling jets
    rand1 = np.random.permutation(len(jets1))
    rand2 = np.random.permutation(len(jets2))
    jets1 = jets1[rand1]
    jets2 = jets2[rand2]

    masses1 = jetnet.utils.jet_features(jets1)["mass"]
    masses2 = jetnet.utils.jet_features(jets2)["mass"]

    # masses1 = calculate_jet_features(jets1)[:, 3]
    # masses2 = calculate_jet_features(jets2)[:, 3]

    w1s = []
    i = 0

    for _ in range(num_batches):
        rand1 = [*range(0, len(jets1))]  # whole real dataset
        rand2 = [*range(i, i + num_eval_samples)]  # batches of the fake dataset
        i += num_eval_samples

        rand_sample1 = masses1[rand1]
        rand_sample2 = masses2[rand2]

        w1s.append(wasserstein_distance(rand_sample1, rand_sample2))

    return np.mean(w1s), np.std(w1s) if return_std else np.mean(w1s)


def w1p(
    jets1: Union[Tensor, np.ndarray],  # real_jets
    jets2: Union[Tensor, np.ndarray],  # fake_jets
    mask1: Union[Tensor, np.ndarray] = None,
    mask2: Union[Tensor, np.ndarray] = None,
    exclude_zeros: bool = True,
    num_particle_features: int = 0,
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_features: bool = True,
    return_std: bool = True,
):
    """
    adapted such that jet1 = real_jets, jet2 = fake_jets
    no more random choice, rather comparing the whole test sets to batches of fake data
    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if num_particle_features <= 0:
        num_particle_features = jets1.shape[2]

    assert (
        num_particle_features <= jets1.shape[2]
    ), "more particle features requested than were inputted"
    assert (
        num_particle_features <= jets2.shape[2]
    ), "more particle features requested than were inputted"

    if mask1 is not None:
        # TODO: should be wrapped in try catch
        mask1 = mask1.reshape(jets1.shape[0], jets1.shape[1])
        mask1 = mask1.astype(bool)

    if mask2 is not None:
        # TODO: should be wrapped in try catch
        mask2 = mask2.reshape(jets2.shape[0], jets2.shape[1])
        mask2 = mask2.astype(bool)

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    # shuffling jets
    rand1 = np.random.permutation(len(jets1))
    rand2 = np.random.permutation(len(jets2))
    jets1 = jets1[rand1]
    jets2 = jets2[rand2]
    if mask1 is not None:
        mask1 = mask1[rand1]
    if mask2 is not None:
        mask2 = mask2[rand2]

    if exclude_zeros:
        zeros1 = np.linalg.norm(jets1[:, :, :num_particle_features], axis=2) == 0
        mask1 = ~zeros1 if mask1 is None else mask1  # * ~zeros1

        zeros2 = np.linalg.norm(jets2[:, :, :num_particle_features], axis=2) == 0
        mask2 = ~zeros2 if mask2 is None else mask2  # * ~zeros2

    w1s = []
    i = 0

    for _ in range(num_batches):
        rand1 = [*range(0, len(jets1))]  # whole real dataset
        rand2 = [*range(i, i + num_eval_samples)]  # batches of the fake dataset
        i += num_eval_samples

        rand_sample1 = jets1[rand1]
        rand_sample2 = jets2[rand2]

        if mask1 is not None:
            parts1 = rand_sample1[:, :, :num_particle_features][mask1[rand1]]
        else:
            parts1 = rand_sample1[:, :, :num_particle_features].reshape(-1, num_particle_features)

        if mask2 is not None:
            parts2 = rand_sample2[:, :, :num_particle_features][mask2[rand2]]
        else:
            parts2 = rand_sample2[:, :, :num_particle_features].reshape(-1, num_particle_features)

        if parts1.shape[0] == 0 or parts2.shape[0] == 0:
            w1 = [np.inf, np.inf, np.inf]
        else:
            w1 = [
                wasserstein_distance(parts1[:, i], parts2[:, i])
                for i in range(num_particle_features)
            ]

        w1s.append(w1)

    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)

    if average_over_features:
        return np.mean(means), np.linalg.norm(stds) if return_std else np.mean(means)
    else:
        return means, stds if return_std else means


def w1efp(
    jets1: Union[Tensor, np.ndarray],  # real jets
    jets2: Union[Tensor, np.ndarray],  # fake jets
    use_particle_masses: bool = False,
    efpset_args: list = [("n==", 4), ("d==", 4), ("p==", 1)],
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_efps: bool = True,
    return_std: bool = True,
    efp_jobs: int = None,
):
    """
    adapted such that jet1 = real_jets, jet2 = fake_jets
    no more random choice, rather compairng the whole test sets to batches of fake data
    """

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    # shuffling jets
    rand1 = np.random.permutation(len(jets1))
    rand2 = np.random.permutation(len(jets2))
    jets1 = jets1[rand1]
    jets2 = jets2[rand2]

    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"
    assert (jets1.shape[2] - int(use_particle_masses) >= 3) and (
        jets1.shape[2] - int(use_particle_masses) >= 3
    ), "particle feature format is incorrect"

    efps1 = jetnet.utils.efps(
        jets1,
        use_particle_masses=use_particle_masses,
        efpset_args=efpset_args,
        efp_jobs=efp_jobs,
    )
    efps2 = jetnet.utils.efps(
        jets2,
        use_particle_masses=use_particle_masses,
        efpset_args=efpset_args,
        efp_jobs=efp_jobs,
    )
    num_efps = efps1.shape[1]

    w1s = []
    i = 0

    for _ in range(num_batches):
        rand1 = [*range(0, len(jets1))]  # whole real dataset
        rand2 = [*range(i, i + num_eval_samples)]  # batches of the fake dataset
        i += num_eval_samples

        rand_sample1 = efps1[rand1]
        rand_sample2 = efps2[rand2]

        w1 = [
            wasserstein_distance(rand_sample1[:, i], rand_sample2[:, i]) for i in range(num_efps)
        ]
        w1s.append(w1)

    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)

    if average_over_efps:
        return np.mean(means), np.linalg.norm(stds) if return_std else np.mean(means)
    else:
        return means, stds if return_std else means
