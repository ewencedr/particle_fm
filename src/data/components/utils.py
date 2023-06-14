import energyflow as ef
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(
    x: np.ndarray, categories: list = [[0, 1, 2, 3, 4]], num_other_features: int = 4
) -> np.array:
    """One hot encode the jet type and leave the rest of the features as is
        Note: The one_hot encoded value is based on the position in the categories list not the value itself,
        e.g. categories: [0,3] results in the two one_hot encoded values [1,0] and [0,1]

    Args:
        x (np.ndarray): jet data with shape (num_jets, num_features) that contains the jet type in the first column
        categories (list, optional): List with values in x that should be one hot encoded. Defaults to [[0, 1, 2, 3, 4]].
        num_other_features (int, optional): Number of features in x that are not one hot encoded. Defaults to 4.

    Returns:
        np.array: one_hot_encoded jet data (num_jets, num_features) with feature length len(categories) + 3 (pt, eta, mass)
    """
    enc = OneHotEncoder(categories=categories)
    type_encoded = enc.fit_transform(x[..., 0].reshape(-1, 1)).toarray()
    other_features = x[..., 1:].reshape(-1, num_other_features)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)


# centering


def center_jets(data):
    """Center Jets.

    Args:
        data (_type_): Particle Data (batch, particles, features); features: eta, phi, pt

    Returns:
        _type_: Particle Data (batch, particles, features); features: eta, phi, pt
    """
    data = data[:, :, [2, 0, 1]]
    etas = jet_etas(data)
    phis = jet_phis(data)
    etas = etas[:, np.newaxis].repeat(repeats=data.shape[1], axis=1)
    phis = phis[:, np.newaxis].repeat(repeats=data.shape[1], axis=1)
    mask = data[..., 0] > 0  # mask all particles with nonzero pt
    data[mask, 1] -= etas[mask]
    data[mask, 2] -= phis[mask]

    return data[:, :, [1, 2, 0]]


def jet_etas(jets_ary):
    jets_p4s = ef.p4s_from_ptyphims(jets_ary)
    etas = ef.etas_from_p4s(jets_p4s.sum(axis=1))
    return etas


def jet_phis(jets_ary):
    jets_p4s = ef.p4s_from_ptyphims(jets_ary)
    phis = ef.phis_from_p4s(jets_p4s.sum(axis=1), phi_ref=0)
    return phis


def jet_masses(jets_tensor):  # in format (jets, particles, features)
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    masses = torch_ms_from_p4s(jets_p4s.sum(axis=1))
    return masses


# p4s
def torch_p4s_from_ptyphi(ptyphi):
    # get pts, ys, phis
    # ptyphi = torch.Tensor(ptyphi).float()
    pts, ys, phis = (
        ptyphi[..., 0, np.newaxis],
        ptyphi[..., 1, np.newaxis],
        ptyphi[..., 2, np.newaxis],
    )

    # + ms**2) everything assumed massless
    Ets = torch.sqrt(pts**2)

    p4s = torch.cat(
        (
            Ets * torch.cosh(ys),
            pts * torch.cos(phis),
            pts * torch.sin(phis),
            Ets * torch.sinh(ys),
        ),
        axis=-1,
    )
    return p4s


def torch_ms_from_p4s(p4s):
    m2s = torch_m2s_from_p4s(p4s)
    return torch.sign(m2s) * torch.sqrt(torch.abs(m2s))


def torch_m2s_from_p4s(p4s):
    return p4s[..., 0] ** 2 - p4s[..., 1] ** 2 - p4s[..., 2] ** 2 - p4s[..., 3] ** 2


# mask data


def mask_data(particle_data, jet_data, num_particles, variable_jet_sizes=True):
    """Splits particle data in data and mask If variable_jet_sizes is True, the returned data only
    contains jets with num_particles constituents.

    Args:
        particle_data (_type_): Particle Data (batch, particles, features)
        jet_data (_type_): _description_
        num_particles (_type_): _description_
        variable_jet_sizes (bool, optional): _description_. Defaults to True.

    Returns:
        x (): masked particle data
        mask (): mask
        particle_data (): modified particle data with 4 features (3 + mask)
        jet_data (): masked jet data
    """
    x = None
    mask = None
    if not variable_jet_sizes:
        # take only jets with num_particles constituents
        jet_mask = np.ma.masked_where(
            np.sum(particle_data[:, :, 3], axis=1) == num_particles,
            np.sum(particle_data[:, :, 3], axis=1),
        )
        masked_particle_data = particle_data[jet_mask.mask]

        jet_data = jet_data[jet_mask.mask]

        x = torch.Tensor(masked_particle_data[:, :, :3])
        mask = torch.Tensor(masked_particle_data[:, :, 3:])
        particle_data = masked_particle_data
        # print(
        #    f"Jets with {num_particles} constituents:",
        #    np.ma.count_masked(jet_mask),
        #    f"({np.round(np.ma.count_masked(jet_mask)/(np.ma.count_masked(jet_mask)+np.ma.count(jet_mask))*100,2)}%)",
        # )
        # print(
        #    f"Jets with less than {num_particles} constituents:",
        #    np.ma.count(jet_mask),
        #    f"({np.round(np.ma.count(jet_mask)/(np.ma.count_masked(jet_mask)+np.ma.count(jet_mask))*100,2)}%)",
        # )

    elif variable_jet_sizes:
        particle_data = particle_data[:, :num_particles, :]
        x = torch.Tensor(particle_data[:, :, :3])
        mask = torch.Tensor(particle_data[:, :, 3:])

    mask[mask > 0] = 1
    mask[mask < 0] = 0

    return x, mask, particle_data, jet_data


# normalize


def normalize_tensor(tensor, mean, std, sigma=5):
    """Normalisation of every tensor feature.

        tensor[..., i] = (tensor[..., i] - mean[i]) / (std[i] / sigma)
    Args:
        tensor (_type_): (batch, particles, features)
        mean (_type_): _description_
        std (_type_): _description_
        sigma (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """

    for i in range(len(mean)):
        tensor[..., i] = (tensor[..., i] - mean[i]) / (std[i] / sigma)
    return tensor


def inverse_normalize_tensor(tensor, mean, std, sigma=5):
    """Inverse normalisation of each feature of a tensor.

        tensor[..., i] = (tensor[..., i] * (std[i] / sigma)) + mean[i]

    Args:
        tensor (_type_): _description_
        mean (_type_): _description_
        std (_type_): _description_
        sigma (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    for i in range(len(mean)):
        tensor[..., i] = (tensor[..., i] * (std[i] / sigma)) + mean[i]
    return tensor


# base distribution of flow
def get_base_distribution(x, mask, use_calculated_base_distribution=False):
    """Calculate different mean and std_dist values for Gaussian base distribution of flow based on
    data.

    Args:
        x (_type_): _description_
        mask (_type_): _description_
        use_calculated_base_distribution (bool, optional): _description_. Defaults to False.

    Returns:
        x_mean : Mean value of Gaussian calculated based on data
        x_cov : Std value of Gaussian calculated based on data
    """
    # Change base distribution of flow according to dataset
    x_mean = torch.zeros(3)
    x_cov = torch.zeros(3)
    if use_calculated_base_distribution:
        for i in range(x.shape[-1]):
            x_mean[i] = x[:, :, i].unsqueeze(-1)[mask.bool()].mean()
            x_cov[i] = x[:, :, i].unsqueeze(-1)[mask.bool()].std()
            if i == 2:
                x_cov *= 5.0
    else:
        x_mean = None
        x_cov = None

    # print(f"x_mean: {x_mean}, x_cov: {x_cov}")
    # print(f"x.shape: {x.shape}")
    return x_mean, x_cov


def get_metrics_data(path, mgpu=False):
    """Read metrics that were saved via CSV Logger.

    Args:
        path (String): Path of log file
        mgpu (bool, optional): Whether the new model with multi GPU support was used. Defaults to False.

    Returns:
        _type_: _description_
    """
    metrics_df = pd.read_csv(path)
    epochs = metrics_df["epoch"].dropna().unique()
    train_loss = metrics_df["train_loss_epoch"].dropna().to_numpy()
    if mgpu:
        val_loss = metrics_df["val_loss_epoch"].dropna().to_numpy()
    else:
        val_loss = metrics_df["val_loss"].dropna().to_numpy()
    lr = metrics_df["lr-AdamW"].dropna().to_numpy()
    if not (len(epochs) == len(train_loss) == len(val_loss) == len(lr)):
        epoch = np.min([len(epochs), len(train_loss), len(val_loss), len(lr)])
        epochs = epochs[:epoch]
        train_loss = train_loss[:epoch]
        val_loss = val_loss[:epoch]
        lr = lr[:epoch]
    return epochs, train_loss, val_loss, lr


def calculate_jet_features(particle_data):
    """Calculate the jet_features by transforming jet constituents to p4s, summing up and
    transforming back to hadrodic coordinates. Phi_ref is 0. Mask in input particle_data is
    allowed.

    Args:
        particle_data (_type_): particle data, shape: [events, particles, features], features: [eta,phi,pt,(mask)]

    Returns:
        jet_data _type_: jet data, shape: [events, features], features: [pt,y,phi,m]
    """
    particle_data = particle_data[..., [2, 0, 1]]
    p4s = ef.p4s_from_ptyphims(particle_data)
    sum_p4 = np.sum(p4s, axis=-2)
    jet_data = ef.ptyphims_from_p4s(sum_p4, phi_ref=0)
    return jet_data


def count_parameters(model):
    """Count Parameters of model.

    Args:
        model (_type_): model

    Returns:
        parameters _type_: parameters of the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pt_of_selected_particles(particle_data, selected_particles=[1, 3, 10]):
    """Return pt of selected particles.

    Args:
        particle_data (_type_): _description_
        selected_particles (list, optional): _description_. Defaults to [1, 3, 10].

    Returns:
        _type_: _description_
    """
    particle_data_sorted = np.sort(particle_data[:, :, 2])[:, ::-1]
    pt_selected_particles = []
    for selected_particle in selected_particles:
        pt_selected_particle = particle_data_sorted[:, selected_particle - 1]
        pt_selected_particles.append(pt_selected_particle)
    return np.array(pt_selected_particles)


def get_pt_of_selected_multiplicities(
    particle_data, selected_multiplicities=[10, 20, 30], num_jets=150
):
    """Return pt of jets with selected particle multiplicities.

    Args:
        particle_data (_type_): _description_
        selected_multiplicities (list, optional): _description_. Defaults to [20, 30, 40].
        num_jets (int, optional): _description_. Defaults to 150.

    Returns:
        _type_: _description_
    """
    data = {}
    for count, selected_multiplicity in enumerate(selected_multiplicities):
        particle_data_temp = particle_data[:, :selected_multiplicity, :]
        mask = np.ma.masked_where(
            np.count_nonzero(particle_data_temp[:, :, 0], axis=1) == selected_multiplicity,
            np.count_nonzero(particle_data_temp[:, :, 0], axis=1),
        )
        masked_particle_data = particle_data_temp[mask.mask]
        masked_pt = masked_particle_data[:num_jets, :, 2]
        data[f"{count}"] = masked_pt
    return data
