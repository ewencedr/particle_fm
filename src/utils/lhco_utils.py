import awkward as ak
import energyflow as ef
import matplotlib.pyplot as plt
import numpy as np
import vector
import fastjet as fj


def plot_unprocessed_data_lhco(
    sim_data: np.ndarray,
    particle_data: np.ndarray,
    num_samples: int = -1,
    labels: list = ["Gen. data"],
    sim_data_label: str = "Sim. data",
    bins: int = 100,
    plottype: str = "sim_data",
    save_fig: bool = True,
    save_folder: str = "logs/plots/",
    save_name: str = "plot",
    close_fig: bool = False,
):
    if not (len(particle_data) == len(labels)):
        raise ValueError("labels has not the same size as gen_models")
    if len(sim_data) != particle_data.shape[1]:
        raise Warning("sim_data and particle_data do not have the same size")
    plot_data_only = False
    if len(particle_data) == 0:
        plot_data_only = True

    # select only the first num_samples
    if num_samples == -1:
        num_samples = particle_data.shape[1]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axs[0]
    data1 = sim_data[:, :, 0].flatten()
    data1 = data1[data1 != 0]
    if not plot_data_only:
        data = [d[:, :, 0].flatten()[d[:, :, 0].flatten() != 0] for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_min, x_max = -0.1, 1
    ax1.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax1.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax1.set_xlabel(r"Particle $p_\mathrm{T}^\mathrm{rel}$")
    ax1.set_yscale("log")
    ax1.legend(loc="best", prop={"size": 14}, frameon=False)

    ax2 = axs[1]
    data1 = sim_data[:, :, 1].flatten()
    data1 = data1[data1 != 0]
    if not plot_data_only:
        data = [d[:, :, 1].flatten()[d[:, :, 1].flatten() != 0] for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_min, x_max = -1.7, 1.2
    if plottype == "t":
        x_min = -1
    if plottype == "q":
        x_min = -1
    if plottype == "q_max_particles":
        x_min, x_max = 0.01, 0.85
    ax2.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax2.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )

    ax2.set_xlabel(r"Particle $\eta^\mathrm{rel}$")
    ax2.set_yscale("log")

    ax3 = axs[2]
    data1 = sim_data[:, :, 2].flatten()
    data1 = data1[data1 != 0]
    if not plot_data_only:
        data = [d[:, :, 2].flatten()[d[:, :, 2].flatten() != 0] for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_min, x_max = -0.7, 0.7
    if plottype == "t":
        x_min = -1
    if plottype == "q_max_particles":
        x_min, x_max = -1.5, 1.5
    ax3.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax3.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax3.set_xlabel(r"Particle $\phi^\mathrm{rel}$")
    ax3.set_yscale("log")
    ax3.set_ylim(
        0.5,
    )

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


# define a function to sort ak.Array by pt
def sort_by_pt(data: ak.Array, ascending: bool = False, return_indices: bool = False):
    """Sort ak.Array by pt.

    Args:
        data (ak.Array): array that should be sorted by pt. It should have a pt attribute.
        ascending (bool, optional): If True, the first value in each sorted group will be smallest; if False, the order is from largest to smallest. Defaults to False.
        return_indices (bool, optional): If True, the indices of the sorted array are returned. Defaults to False.

    Returns:
        ak.Array: sorted array
        ak.Array (optional): indices of the sorted array
    """
    if isinstance(data, ak.Array):
        try:
            temppt = data.pt
        except AttributeError:
            raise AttributeError(
                "Needs either correct coordinates or embedded vector backend"
            ) from None
    tmpsort = ak.argsort(temppt, axis=-1, ascending=ascending)
    if return_indices:
        return data[tmpsort], tmpsort
    else:
        return data[tmpsort]


def get_jet_data(consts: np.ndarray) -> np.ndarray:
    """Calculate jet data from constituent data. (pt, y, phi)->(pt, y, phi, m)

    Args:
        consts (np.ndarray): constituent data. (pt, y, phi)

    Returns:
        np.ndarray: jet data. (pt, y, phi, m)
    """
    p4s = ef.p4s_from_ptyphims(consts[..., :3])
    sum_p4 = np.sum(p4s, axis=-2)
    jet_data = ef.ptyphims_from_p4s(sum_p4, phi_ref=0)
    return jet_data


def cluster_data(
    particle_data: np.ndarray, max_jets: int = 2, max_consts: int = 558, verbose: bool = False
) -> np.ndarray:
    """Cluster particle data to jets. The data is clustered with the anti-kt algorithm with R=1.0.

    Args:
        particle_data (np.ndarray): Particle data in shape (batch, num_particles, features) with features = (pt, eta, phi)
        max_jets (int, optional): Maximum number of highest pt clustered jets that will be saved. For less jets, the data will be zero-padded. Defaults to 2.
        max_consts (int, optional): Maximum number of constituents that will be saved in jet. For less constituents, the data will be zero-padded. Defaults to 558.
        verbose (bool, optional): Defaults to False.

    Returns:
        np.ndarray: Clustered constituents in shape (batch, max_jets, max_consts, features) with features = (pt, eta, phi)
    """
    # pt, eta, phi

    # make data an awkward array
    zrs = np.zeros((particle_data.shape[0], particle_data.shape[1], 1))
    data_with_mass = np.concatenate((particle_data, zrs), axis=2)
    awkward_data = ak.from_numpy(data_with_mass)

    # tell awkward that the data is in pt, eta, phi, mass format
    vector.register_awkward()
    unmasked_data = ak.zip(
        {
            "pt": awkward_data[:, :, 0],
            "eta": awkward_data[:, :, 1],
            "phi": awkward_data[:, :, 2],
            "mass": awkward_data[:, :, 3],
        },
        with_name="Momentum4D",
    )

    # remove the padded data points
    data = ak.drop_none(ak.mask(unmasked_data, unmasked_data.pt != 0))

    # define clustering algorithm
    jetdef = fj.JetDefinition(fj.antikt_algorithm, 1.0)

    # cluster the data
    cluster = fj.ClusterSequence(data, jetdef)

    # get jets and constituents
    jets_out = cluster.inclusive_jets()
    consts_out = cluster.constituents()

    # sort jets and constituents by pt
    jets_sorted, idxs = sort_by_pt(jets_out, return_indices=True)
    consts_sorted_jets = consts_out[idxs]
    consts_sorted = sort_by_pt(consts_sorted_jets)

    # only take the first max_jets highest pt jets
    consts_awk = consts_sorted[:, :max_jets]

    # get max. number of constituents in an event
    max_consts_gen = int(ak.max(ak.num(consts_awk, axis=-1)))
    if verbose:
        print(f"max. number of constituents in an event: {max_consts_gen}")

    # pad the constituents with zeros to make all jets have the same length
    zero_padding = ak.zip({"pt": 0.0, "eta": 0.0, "phi": 0.0, "mass": 0.0}, with_name="Momentum4D")
    padded_consts1 = ak.fill_none(
        ak.pad_none(consts_awk, max_consts, clip=True, axis=-1), zero_padding, axis=-1
    )
    # also pad less than nr_jets jets with zeros
    zero_padding_jet = ak.zip(
        {
            "pt": [0.0] * max_consts,
            "eta": [0.0] * max_consts,
            "phi": [0.0] * max_consts,
            "mass": [0.0] * max_consts,
        },
        with_name="Momentum4D",
    )
    padded_consts = ak.fill_none(
        ak.pad_none(padded_consts1, max_jets, clip=True, axis=1), zero_padding_jet, axis=1
    )

    # go back to numpy arrays
    pt, eta, phi, mass = ak.unzip(padded_consts)
    pt_np = ak.to_numpy(pt)
    eta_np = ak.to_numpy(eta)
    phi_np = ak.to_numpy(phi)
    consts = np.stack((pt_np, eta_np, phi_np), axis=-1)

    return consts
