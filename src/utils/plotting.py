"""Plots for analysing generated data."""
# TODO add docstrings
# TODO add comments
# TODO add type hints
# TODO add tests

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from jetnet.utils import efps
from matplotlib.gridspec import GridSpec
from scipy.stats import wasserstein_distance as w_dist
from tqdm import tqdm

from src.data.components import (
    calculate_jet_features,
    count_parameters,
    get_metrics_data,
    get_pt_of_selected_multiplicities,
    get_pt_of_selected_particles,
)

from .data_generation import generate_data


def apply_mpl_styles() -> None:
    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#B6BFC3",
            "#3B515B",
            "#0271BB",
            "#E2001A",
        ]
    )
    mpl.rcParams["font.size"] = 15
    mpl.rcParams["patch.linewidth"] = 1.25


def plot_single_jets(
    data: np.ndarray,
    color: str = "#E2001A",
    save_folder: str = "logs/",
    save_name: str = "sim_jets",
) -> plt.figure:
    """Create a plot with 16 randomly selected jets from the data.

    Args:
        data (_type_): Data to plot.
        color (str, optional): Color of plotted point cloud. Defaults to "#E2001A".
        save_folder (str, optional): Path to folder where the plot is saved. Defaults to "logs/".
        save_name (str, optional): File_name for saving the plot. Defaults to "sim_jets".
    """
    mask_data = np.ma.masked_where(
        data[:, :, 0] == 0,
        data[:, :, 0],
    )
    mask = np.expand_dims(mask_data, axis=-1)
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(4, 4)

    for i in tqdm(range(16)):

        ax = fig.add_subplot(gs[i])

        idx = np.random.randint(len(data))
        x_plot = data[idx, :, :2]  # .cpu()
        s_plot = np.abs(data[idx, :, 2])  # .cpu())
        s_plot[mask[idx, :, 0] < 0.0] = 0.0

        ax.scatter(*x_plot.T, s=5000 * s_plot, color=color, alpha=0.5)

        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\phi$")

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)

    plt.tight_layout()

    plt.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
    return fig


def plot_data(
    sim_data: np.ndarray,
    particle_data: np.array,
    jet_data_sim: np.ndarray,
    jet_data: np.ndarray,
    efps_sim: np.ndarray,
    efps_values: np.ndarray,
    labels: list[str],
    sim_data_label: str = "Sim. data",
    plot_jet_features: bool = False,
    plot_w_dists: bool = False,
    w_dist_m: float = 0,
    variable_jet_sizes_plotting: bool = True,
    selected_particles: list[int] = [1, 5, 20],
    pt_selected_particles_sim: list[float] = None,
    pt_selected_particles: list[float] = None,
    plot_selected_multiplicities: bool = False,
    selected_multiplicities: list[int] = [10, 20, 30, 40, 50, 80],
    pt_selected_multiplicities_sim: list[float] = None,
    pt_selected_multiplicities: list[float] = None,
    plottype: str = "sim_data",
    bins: int = 100,
    save_fig: bool = True,
    save_folder: str = "logs/plots/",
    save_name: str = "plot",
) -> plt.figure:
    """Create a plot of multiple histograms to compare one dataset with other datasets.

    Args:
        sim_data (np.ndarray): Reference data. Background histogram.
        particle_data (list): List of data to be plotted. Can be an empty list. shape: (num_dataset,num_samples,particles,features)
        jet_data_sim (np.ndarray): Jet data of the reference data.
        jet_data (list): Jet data of the data to be plotted.
        efps_sim (np.ndarray): EFPs of the reference data.
        efps_values (list): EFPS of the data to be plotted.
        labels (list): Labels of the plot to describe the data.
        sim_data_label (str, optional): Label of the plot for the reference data. Defaults to "Sim. data".
        plot_jet_features (bool, optional): Plot Jet Features. Defaults to False.
        plot_w_dists (bool, optional): Plot wasserstein distances inside of jet mass plot. Defaults to False.
        w_dist_m (float, optional): wasserstein distances to be plotted if plot_w_dists==True. Defaults to 0.
        variable_jet_sizes_plotting (bool, optional): Plot p_t distributions of selected jets. Count by p_t. Defaults to True.
        selected_particles (list, optional): Highest p_t particles for which the distributions are plotted if variable_jet_sizes_plotting==True. Defaults to [1, 5, 20].
        pt_selected_particles_sim (list, optional): Data from reference model for the plots if variable_jet_sizes_plotting==True. Defaults to None.
        pt_selected_particles (list, optional): Data from models for the plots if variable_jet_sizes_plotting==True. Defaults to None.
        plot_selected_multiplicities (bool, optional): Plot data of jets with selected multiplicities. Defaults to False.
        muselected_ltiplicities (list, optional): Jet multiplicities to plot if plot_selected_multiplicities==True. Defaults to [10, 20, 30, 40, 50, 80].
        pt_selected_multiplicities_sim (list, optional): Data from reference model for the plots if plot_selected_multiplicities==True. Defaults to None.
        pt_selected_multiplicities (list, optional): Data from models for the plots if plot_selected_multiplicities==True. Defaults to None.
        plottype (str, optional): Presets for setting the x_lims. "sim_data" sets the x_lims to the min and max of the reference data. Defaults to "sim_data".
        bins (int, optional): Number of bins for all histograms. Defaults to 100.
        save_fig (bool, optional): Save the fig. Defaults to True.
        save_folder (str, optional): Folder for saving the fig if save_fig==True. Defaults to "logs/plots/".
        save_name (str, optional): Filename for saving the fig if save_fig==True. Defaults to "plot".

    Returns:
        _type_: Figure
    """

    if not (len(particle_data) == len(labels)):
        raise ValueError("labels has not the same size as gen_models")
    if len(sim_data) != particle_data.shape[1]:
        raise Warning("sim_data and particle_data do not have the same size")
    plot_data_only = False
    if len(particle_data) == 0:
        plot_data_only = True

    particles_per_jet = sim_data.shape[-2]

    if plot_selected_multiplicities:
        if plot_jet_features:
            fig = plt.figure(figsize=(12, 24))
            gs = GridSpec(6, 3)
        else:
            fig = plt.figure(figsize=(12, 20))
            gs = GridSpec(5, 3)
    else:
        if plot_jet_features:
            fig = plt.figure(figsize=(12, 16))
            gs = GridSpec(4, 3)
        else:
            fig = plt.figure(figsize=(12, 12))
            gs = GridSpec(3, 3)

    gs_counter = 0
    ax1 = fig.add_subplot(gs[gs_counter])
    data1 = sim_data[:, :, 2].flatten()
    if not plot_data_only:
        data = [d[:, :, 2].flatten() for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_min, x_max = -0.1, 1
    hist1 = ax1.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax1.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax1.set_xlabel(r"Particle $p_\mathrm{T}^\mathrm{rel}$")
    ax1.set_yscale("log")
    ax1.legend(loc="best", prop={"size": 14}, frameon=False)

    ax2 = fig.add_subplot(gs[gs_counter + 1])
    data1 = sim_data[:, :, 0].flatten()
    if not plot_data_only:
        data = [d[:, :, 0].flatten() for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
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
    hist1 = ax2.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax2.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )

    ax2.set_xlabel(r"Particle $\eta^\mathrm{rel}$")
    ax2.set_yscale("log")
    ax3 = fig.add_subplot(gs[gs_counter + 2])

    data1 = sim_data[:, :, 1].flatten()
    if not plot_data_only:
        data = [d[:, :, 1].flatten() for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_min, x_max = -0.7, 0.7
    if plottype == "t":
        x_min = -1
    if plottype == "q_max_particles":
        x_min, x_max = -1.5, 1.5
    hist1 = ax3.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax3.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax3.set_xlabel(r"Particle $\phi^\mathrm{rel}$")
    ax3.set_yscale("log")

    if plot_jet_features:
        ax4 = fig.add_subplot(gs[gs_counter + 3])
        data1 = jet_data_sim[:, 0]
        if not plot_data_only:
            data = [d[:, 0] for d in jet_data]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_min, x_max = 0.5, 1.5
        if plottype == "q" or plottype == "q_max_particles":
            x_max = 1.25
        x_min, x_max = 0.5, 1.5
        hist1 = ax4.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax4.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax4.set_xlabel(r"Rel. Jet $p_\mathrm{T}$")
        ax4.set_yscale("log")

        ax5 = fig.add_subplot(gs[gs_counter + 4])
        data1 = jet_data_sim[:, 1]
        if not plot_data_only:
            data = [d[:, 1] for d in jet_data]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_min, x_max = -0.05, 0.05
        if plottype == "t":
            x_min = -0.1
        if plottype == "q" or plottype == "q_max_particles":
            x_max = 0.1
        hist1 = ax5.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        for count, model in enumerate(particle_data):
            hist = ax5.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
        ax5.set_xlabel("Jet y")
        ax5.set_yscale("log")

        ax6 = fig.add_subplot(gs[gs_counter + 5])
        data1 = jet_data_sim[:, 2]
        if not plot_data_only:
            data = [d[:, 2] for d in jet_data]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_min, x_max = -0.01, 0.01
        if (plottype == "t") or plottype == "t30":
            x_min = -0.1
        hist1 = ax6.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax6.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax6.set_xlabel(r"Jet $\phi$")
        ax6.set_yscale("log")
        gs_counter += 6
    else:
        gs_counter = 3

    ax7 = fig.add_subplot(gs[gs_counter])
    data1 = jet_data_sim[:, 3]
    if not plot_data_only:
        data = [d[:, 3] for d in jet_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.3
    if plottype == "t":
        x_max = 0.3
    elif plottype == "q" or plottype == "q_max_particles":
        x_max = 0.3
    hist1 = ax7.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax7.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
            if plot_w_dists:
                ax7.annotate(
                    f"W-Dist: {np.round(w_dist_m,5)}",
                    xy=(0.5, 0),
                    xycoords="axes fraction",
                    size=10,
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round", fc="w"),
                )

    ax7.set_xlabel("Jet Mass")
    ax7.set_yscale("log")

    ax8 = fig.add_subplot(gs[gs_counter + 1])
    data1 = np.sum(sim_data[:, :, 3], axis=1)
    if not plot_data_only:
        data = [np.count_nonzero(d[:, :, 2], axis=1) for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if variable_jet_sizes_plotting:
        binwidth = 1
        if not plot_data_only:
            bins_pm = range(x_min, x_max + binwidth, binwidth)
    else:
        bins_pm = range(0, particles_per_jet)

    hist1 = ax8.hist(
        data1,
        bins=bins_pm,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax8.hist(
                data[count],
                bins=bins_pm,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax8.set_xlabel("Particle Multiplicity")

    ax9 = fig.add_subplot(gs[gs_counter + 2])

    data1 = np.concatenate(efps_sim)
    if not plot_data_only:
        data = [np.concatenate(d) for d in efps_values]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_min, x_max = 0, 0.01
    if plottype == "q" or plottype == "q_max_particles":
        x_min, x_max = 0, 0.0002
    elif plottype == "t":
        x_min, x_max = 0, 0.01
    hist1 = ax9.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax9.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax9.set_xlabel("Jet EFPs")
    ax9.set_yscale("log")
    ax9.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax10 = fig.add_subplot(gs[gs_counter + 3])

    data1 = pt_selected_particles_sim[0]
    if not plot_data_only:
        data = pt_selected_particles[:, 0]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.5
    hist1 = ax10.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax10.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax10.set_xlabel(
        rf"$p_\mathrm{{T}}$ of ${selected_particles[0]}^{{st}}$ Highest $p_\mathrm{{T}}$ Particle"
    )
    ax10.set_yscale("log")

    ax11 = fig.add_subplot(gs[gs_counter + 4])
    data1 = pt_selected_particles_sim[1]
    if not plot_data_only:
        data = pt_selected_particles[:, 1]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.125
    hist1 = ax11.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax11.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax11.set_xlabel(
        rf"$p_\mathrm{{T}}$ of ${selected_particles[1]}^{{th}}$ Highest $p_\mathrm{{T}}$ Particle"
    )
    ax11.set_yscale("log")

    ax12 = fig.add_subplot(gs[gs_counter + 5])
    data1 = pt_selected_particles_sim[2]
    if not plot_data_only:
        data = pt_selected_particles[:, 2]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.025
    hist1 = ax12.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            hist = ax12.hist(
                data[count],
                bins=bins,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax12.set_xlabel(
        rf"$p_\mathrm{{T}}$ of ${selected_particles[2]}^{{th}}$ Highest $p_\mathrm{{T}}$ Particle"
    )
    ax12.set_yscale("log")

    if plot_selected_multiplicities:

        ax13 = fig.add_subplot(gs[gs_counter + 6])
        data1 = pt_selected_multiplicities_sim["0"].flatten()
        if not plot_data_only:
            data = [d.flatten() for d in pt_selected_multiplicities["0"]]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        hist1 = ax13.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax13.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax13.set_xlabel(
            rf"Particle $p_\mathrm{{T}}^\mathrm{{rel}}$ of Jets with {selected_multiplicities[0]} Particles"
        )
        ax13.set_yscale("log")

        ax14 = fig.add_subplot(gs[gs_counter + 7])
        data1 = pt_selected_multiplicities_sim["1"].flatten()
        if not plot_data_only:
            data = [d.flatten() for d in pt_selected_multiplicities["1"]]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        hist1 = ax14.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax14.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax14.set_xlabel(
            rf"Particle $p_\mathrm{{T}}^\mathrm{{rel}}$ of Jets with {selected_multiplicities[1]} Particles"
        )
        ax14.set_yscale("log")

        ax15 = fig.add_subplot(gs[gs_counter + 8])
        data1 = pt_selected_multiplicities_sim["2"].flatten()
        if not plot_data_only:
            data = [d.flatten() for d in pt_selected_multiplicities["2"]]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        hist1 = ax15.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax15.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax15.set_xlabel(
            rf"Particle $p_\mathrm{{T}}^\mathrm{{rel}}$ of Jets with {selected_multiplicities[2]} Particles"
        )
        ax15.set_yscale("log")

        ax16 = fig.add_subplot(gs[gs_counter + 9])
        data1 = pt_selected_multiplicities_sim["3"].flatten()
        if not plot_data_only:
            data = [d.flatten() for d in pt_selected_multiplicities["3"]]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        hist1 = ax16.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax16.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax16.set_xlabel(
            rf"Particle $p_\mathrm{{T}}^\mathrm{{rel}}$ of Jets with {selected_multiplicities[3]} Particles"
        )
        ax16.set_yscale("log")

        ax17 = fig.add_subplot(gs[gs_counter + 10])
        data1 = pt_selected_multiplicities_sim["4"].flatten()
        if not plot_data_only:
            data = [d.flatten() for d in pt_selected_multiplicities["4"]]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        hist1 = ax17.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax17.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax17.set_xlabel(
            rf"Particle $p_\mathrm{{T}}^\mathrm{{rel}}$ of Jets with {selected_multiplicities[4]} Particles"
        )
        ax17.set_yscale("log")

        ax18 = fig.add_subplot(gs[gs_counter + 11])
        data1 = pt_selected_multiplicities_sim["5"].flatten()
        if not plot_data_only:
            data = [d.flatten() for d in pt_selected_multiplicities["5"]]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        hist1 = ax18.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                hist = ax18.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax18.set_xlabel(
            rf"Particle $p_\mathrm{{T}}^\mathrm{{rel}}$ of Jets with {selected_multiplicities[5]} Particles"
        )
        ax18.set_yscale("log")

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
    return fig


def create_and_plot_data(
    sim_data: np.ndarray,
    gen_models,
    save_name: str,
    labels: list[str],
    num_jet_samples: int = 10000,
    batch_size: int = 1000,
    selected_particles: list[int] = [1, 5, 20],
    selected_multiplicities: list[int] = [10, 20, 30, 40, 50, 80],
    plottype: str = "sim_data",
    mgpu: bool = False,
    max_particles: bool = False,
    mask: np.ndarray = None,
    variable_jet_sizes_plotting: bool = True,
    save_folder: str = "./logs/plots/",
    normalised_data: list[bool] = [False],
    means: list[float] = None,
    stds: list[float] = None,
    save_fig: bool = True,
    plot_selected_multiplicities: bool = False,
    print_parameters: bool = True,
    plot_jet_features: bool = False,
    plot_w_dists: bool = False,
    bins: int = 100,
    sim_data_label: str = "JetNet",
    file_dict: dict = None,
):
    """Generate data for plotting and plot it.

    Args:
        sim_data (_type_): _description_
        gen_models (_type_): _description_
        save_name (_type_): _description_
        labels (_type_): _description_
        num_jet_samples (int, optional): _description_. Defaults to 10000.
        batch_size (int, optional): _description_. Defaults to 10000.
        selected_particles (list, optional): _description_. Defaults to [1, 5, 20].
        selected_multiplicities (list, optional): _description_. Defaults to [10, 20, 30, 40, 50, 80].
        plottype (str, optional): _description_. Defaults to "sim_data".
        mgpu (bool, optional): _description_. Defaults to False.
        max_particles (bool, optional): _description_. Defaults to False.
        mask (_type_, optional): _description_. Defaults to None.
        variable_jet_sizes_plotting (bool, optional): _description_. Defaults to True.
        save_folder (str, optional): _description_. Defaults to "/home/ewencedr/equivariant-flows".
        normalised_data (list, optional): _description_. Defaults to [False].
        means (_type_, optional): _description_. Defaults to None.
        stds (_type_, optional): _description_. Defaults to None.
        save_fig (bool, optional): _description_. Defaults to True.
        plot_selected_multiplicities (bool, optional): _description_. Defaults to False.
        print_parameters (bool, optional): _description_. Defaults to True.
        plot_jet_features (bool, optional): _description_. Defaults to True.
        plot_w_dists (bool, optional): _description_. Defaults to True.
        bins (int, optional): _description_. Defaults to 100.
        sim_data_label (str, optional): _description_. Defaults to "JetNet".
        file_dict (_type_, optional): _description_. Defaults to None.

    Raises:
        AssertionError: _description_

    Returns:
        _type_: _description_
    """
    particles_per_jet = sim_data.shape[-2]
    (
        particle_data,
        times,
        jet_data,
        efps_values,
        pt_selected_particles,
        pt_selected_multiplicities,
        w_dist_m,
        jet_data_sim,
        efps_sim,
        pt_selected_particles_sim,
        pt_selected_multiplicities_sim,
    ) = create_data_for_plotting(
        sim_data,
        gen_models,
        num_jet_samples,
        batch_size,
        particles_per_jet,
        selected_particles,
        plot_selected_multiplicities=plot_selected_multiplicities,
        selected_multiplicities=selected_multiplicities,
        mgpu=mgpu,
        max_particles=max_particles,
        mask=mask,
        normalised_data=normalised_data,
        means=means,
        stds=stds,
        file_dict=file_dict,
    )

    if print_parameters:
        for count, model in enumerate(gen_models):
            if type(model) is not str:
                print(f"Parameters {labels[count]}: {count_parameters(model)}")

    sim_data = np.append(np.array(sim_data), np.array(mask), axis=-1)

    fig = plot_data(
        particle_data=particle_data[:, :num_jet_samples, :, :],
        sim_data=sim_data[:num_jet_samples],
        jet_data_sim=jet_data_sim[:num_jet_samples],
        jet_data=jet_data[:, :num_jet_samples, :],
        efps_sim=efps_sim,
        efps_values=efps_values,
        labels=labels,
        sim_data_label=sim_data_label,
        plot_jet_features=plot_jet_features,
        plot_w_dists=plot_w_dists,
        plot_selected_multiplicities=plot_selected_multiplicities,
        selected_multiplicities=selected_multiplicities,
        selected_particles=selected_particles,
        pt_selected_particles=pt_selected_particles,
        pt_selected_multiplicities=pt_selected_multiplicities,
        pt_selected_particles_sim=pt_selected_particles_sim,
        pt_selected_multiplicities_sim=pt_selected_multiplicities_sim,
        w_dist_m=w_dist_m,
        save_folder=save_folder,
        save_name=save_name,
        plottype=plottype,
        save_fig=save_fig,
        variable_jet_sizes_plotting=variable_jet_sizes_plotting,
        bins=bins,
    )
    return fig, particle_data, times


def plot_loss_curves(
    name,
    file_paths,
    labels,
    mgpu=False,
    save_path="/home/ewencedr/equivariant-flows",
    plottype="",
):

    if not (len(file_paths) == len(labels)):
        raise ValueError("labels has not the same size as file_paths")

    loss = []
    fig, ax = plt.subplots()
    for count, file_path in enumerate(file_paths):
        print(f"file_path: {file_path}")
        if "epic" in file_path:
            continue
        # skip first color in colorcycle
        if count == 0:
            next(ax._get_lines.prop_cycler)
        epochs, train_loss, val_loss, lr = get_metrics_data(file_path, mgpu=mgpu)
        plot = ax.plot(epochs, train_loss, label=f"train loss - {labels[count]}")
        ax.plot(
            epochs,
            val_loss,
            label=f"val loss - {labels[count]}",
            color=plot[-1].get_color(),
            alpha=0.25,
        )
        loss.append(train_loss)
        loss.append(val_loss)
    y_min = np.array([d.min() for d in loss]).min()
    if not mgpu:
        ax.set_ylim(y_min - 0.01, -6)
    if plottype == "1":
        ax.set_ylim(y_min - 0.01, 6)
    elif plottype == "2":
        ax.set_ylim(y_min - 0.01, 7)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    # ax.set_yscale("log")
    ax.legend(loc="best")
    plt.savefig(f"{save_path}/plots/loss_plots_{name}.png")
    plt.show()
    plt.clf()


def do_timing_plots(
    models,
    name,
    labels,
    particles_per_jet=[10, 30, 60, 100, 150],
    jets_to_generate=1000,
    batch_sizes=[256, 256, 265, 256, 256],
    xscale_log=False,
    mgpu=False,
    max_particles=False,
    mask=None,
    save_path="/home/ewencedr/equivariant-flows",
):

    if not (len(models) == len(labels)):
        raise ValueError("labels has not the same size as models")

    times = []
    for model in models:
        if type(model) is str:
            if "epic" in model:
                times.append([12.5, 20, 45, 82])
            continue
        times_temp = []
        for i in tqdm(range(len(particles_per_jet))):
            _, time = generate_data(
                model,
                jets_to_generate,
                batch_sizes[i],
                particles_per_jet[i],
                mgpu_model=mgpu,
                max_particles=False,
                mask=None,
            )
            times_temp.append(time / jets_to_generate)
        times.append(times_temp)

    # plotting
    fig, ax = plt.subplots()
    for count, t in enumerate(times):
        # skip first color in colorcycle
        if count == 0:
            next(ax._get_lines.prop_cycler)
        ax.plot(particles_per_jet, t, label=labels[count], marker="o")
    ax.set_xlabel("Particles per Jet")
    ax.set_ylabel("Generation Time per jet in s")
    if xscale_log:
        ax.set_xscale("log")
    ax.legend(loc="best")
    # plt.title(f"Time to generate {particles_to_generate} jets")
    plt.savefig(f"{save_path}/plots/{name}.png")
    return np.array(times)


def create_data_for_plotting(
    sim_data_in: np.ndarray,
    gen_models,
    num_jet_samples: int = 10000,
    batch_size: int = 10000,
    particles_per_jet: int = 30,
    selected_particles: list[int] = [1, 3, 10],
    plot_selected_multiplicities: bool = False,
    selected_multiplicities: list[int] = [20, 30, 40],
    mgpu: bool = False,
    max_particles: bool = False,
    mask=None,
    normalised_data: list[bool] = [False],
    means: list[float] = None,
    stds: list[float] = None,
    file_dict: dict = None,
):
    data = []
    times = []
    jet_data = []
    efps_values = []
    pt_selected_particles = []
    pt_selected_multiplicities = []
    w_dist_m = []
    sim_data = sim_data_in[:num_jet_samples]
    jet_data_sim = calculate_jet_features(sim_data)
    efps_sim = efps(sim_data, efp_jobs=1)
    pt_selected_particles_sim = get_pt_of_selected_particles(sim_data, selected_particles)
    if plot_selected_multiplicities:
        pt_selected_multiplicities_sim = get_pt_of_selected_multiplicities(
            sim_data, selected_multiplicities
        )
    else:
        pt_selected_multiplicities_sim = []

    for count, model in enumerate(gen_models):

        if type(model) is str:
            print(f"Loading data for model {count+1} of {len(gen_models)}")
            data_temp = load_data_from_file(model, file_dict)[:num_jet_samples]  # pt,eta,phi
            data_temp[:, :, [0, 1, 2]] = data_temp[:, :, [1, 2, 0]]
            times_temp = 0
        else:
            print(f"Generating data for model {count+1} of {len(gen_models)}")
            data_temp, times_temp = generate_data(
                model,
                num_jet_samples,
                batch_size,
                particles_per_jet=particles_per_jet,
                mgpu_model=mgpu,
                max_particles=max_particles,
                mask=mask,
                normalised_data=normalised_data[count],
                means=means,
                stds=stds,
            )
        jet_data_temp = calculate_jet_features(data_temp)
        efps_temp = efps(data_temp)
        pt_selected_particles_temp = get_pt_of_selected_particles(data_temp, selected_particles)
        pt_selected_multiplicities_temp = get_pt_of_selected_multiplicities(
            data_temp, selected_multiplicities
        )
        w_dist_m_temp = w_dist(jet_data_sim[:, 3], jet_data_temp[:, 3])
        data.append(data_temp)
        times.append(times_temp)
        jet_data.append(jet_data_temp)
        efps_values.append(efps_temp)
        pt_selected_particles.append(pt_selected_particles_temp)
        pt_selected_multiplicities.append(pt_selected_multiplicities_temp)
        w_dist_m.append(w_dist_m_temp)

    new_dict = {}
    for count, i in enumerate(selected_multiplicities):
        new_dict[f"{count}"] = []

    for dicts in pt_selected_multiplicities:
        for count, dict_items_array in enumerate(dicts):
            new_dict[f"{count}"].append(np.array(dicts[dict_items_array]))

    for count, i in enumerate(new_dict):
        new_dict[i] = np.array(new_dict[i])
    return (
        np.array(data),
        np.array(times),
        np.array(jet_data),
        np.array(efps_values),
        np.array(pt_selected_particles),
        new_dict,
        np.array(w_dist_m),
        np.array(jet_data_sim),
        np.array(efps_sim),
        np.array(pt_selected_particles_sim),
        pt_selected_multiplicities_sim,
    )


def load_data_from_file(key, file_dict):
    """Load data from file.

    Args:
        key (str): Key for the filepath that must be in file_dict
        file_dict (dict): Dictionary matching shorter keys to file_paths

    Raises:
        ValueError: No file_dict provided
        ValueError: Key is not in file_dict

    Returns:
        np.array: loaded data
    """
    if file_dict is None:
        raise ValueError("file_dict is None. Please provide file_dict")
    if key in file_dict.keys():
        return np.load(file_dict[key])
    else:
        raise ValueError("Key not found in file_dict")