"""Plots for analysing generated data."""
# TODO add docstrings
# TODO add comments
# TODO add type hints
# TODO add tests

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
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


JETCLASS_FEATURE_LABELS = {
    "part_pt": "Particle $p_\\mathrm{T}$",
    "part_eta": "Particle $\\eta$",
    "part_phi": "Particle $\\phi$",
    "part_mass": "Particle $m$",
    "part_etarel": "Particle $\\eta^\\mathrm{rel}$",
    "part_dphi": "Particle $\\phi^\\mathrm{rel}$",
    "part_ptrel": "Particle $p_\\mathrm{T}^\\mathrm{rel}$",
    "part_d0val": "Particle $d_0$",
    "part_dzval": "Particle $d_z$",
    "part_d0err": "Particle $\\sigma_{d_0}$",
    "part_dzerr": "Particle $\\sigma_{d_z}$",
}

JET_FEATURE_LABELS = {
    "jet_pt": "Jet $p_\\mathrm{T}$",
    "jet_y": "Jet $y$",
    "jet_eta": "Jet $\\eta$",
    "jet_eta": "Jet $\\eta$",
    "jet_mrel": "Jet $m_\\mathrm{rel}$",
    "jet_m": "Jet $m$",
    "jet_phi": "Jet $\\phi$",
}


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
    num_samples: int = -1,
    labels: list[str] = ["Gen. data"],
    sim_data_label: str = "Sim. data",
    plot_xlabels: list[str] = [
        r"Particle $p_\mathrm{T}^\mathrm{rel}$",
        r"Particle $\eta^\mathrm{rel}$",
        r"Particle $\phi^\mathrm{rel}$",
        r"Rel. Jet $p_\mathrm{T}$",
    ],
    plot_jet_features: bool = False,
    plot_w_dists: bool = False,
    w_dist_m: float = 0,
    mass_linear: bool = True,
    plot_efps: bool = True,
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
    close_fig: bool = False,
) -> plt.figure:
    """Create a plot of multiple histograms to compare one dataset with other datasets.

    Args:
        sim_data (np.ndarray): Reference data. Background histogram.
        particle_data (list): List of data to be plotted.
            Can be an empty list. shape: (num_dataset,num_samples,particles,features)
        jet_data_sim (np.ndarray): Jet data of the reference data.
        jet_data (list): Jet data of the data to be plotted.
        efps_sim (np.ndarray): EFPs of the reference data.
        efps_values (list): EFPS of the data to be plotted.
        num_samples (int, optional): Number of samples to be plotted.
            Defaults to length of first dataset in particle_data.
        labels (list): Labels of the plot to describe the data.
        sim_data_label (str, optional): Label of the plot for the reference data.
            Defaults to "Sim. data".
        plot_xlabels (list, optional): Labels of the x-axis for the first four plots.
            Defaults to relative jet coordinates.
        plot_jet_features (bool, optional): Plot Jet Features. Defaults to False.
        plot_w_dists (bool, optional): Plot wasserstein distances inside of jet mass plot.
            Defaults to False.
        w_dist_m (float, optional): wasserstein distances to be plotted if plot_w_dists==True.
            Defaults to 0.
        mass_linear (bool, optional): Plot jet mass in linear scale. Defaults to True.
        plot_efps (bool, optional): Plot EFPs. Defaults to True.
        variable_jet_sizes_plotting (bool, optional): Plot p_t distributions of selected jets.
            Count by p_t. Defaults to True.
        selected_particles (list, optional): Highest p_t particles for which the distributions
            are plotted if variable_jet_sizes_plotting==True. Defaults to [1, 5, 20].
        pt_selected_particles_sim (list, optional): Data from reference model for the plots if
            variable_jet_sizes_plotting==True. Defaults to None.
        pt_selected_particles (list, optional): Data from models for the plots if
            variable_jet_sizes_plotting==True. Defaults to None.
        plot_selected_multiplicities (bool, optional): Plot data of jets with selected
            multiplicities. Defaults to False.
        selected_multiplicities (list, optional): Jet multiplicities to plot if
            plot_selected_multiplicities==True. Defaults to [10, 20, 30, 40, 50, 80].
        pt_selected_multiplicities_sim (list, optional): Data from reference model for
            the plots if plot_selected_multiplicities==True. Defaults to None.
        pt_selected_multiplicities (list, optional): Data from models for the plots if
            plot_selected_multiplicities==True. Defaults to None.
        plottype (str, optional): Presets for setting the x_lims. "sim_data" sets the
            x_lims to the min and max of the reference data. Defaults to "sim_data".
        bins (int, optional): Number of bins for all histograms. Defaults to 100.
        save_fig (bool, optional): Save the fig. Defaults to True.
        save_folder (str, optional): Folder for saving the fig if save_fig==True.
            Defaults to "logs/plots/".
        save_name (str, optional): Filename for saving the fig if save_fig==True.
            Defaults to "plot".
        close_fig (bool, optional): Close the fig after saving it. Defaults to False.

    Returns:
        _type_: Figure
    """

    if not (len(particle_data) == len(labels)):
        raise ValueError("labels has not the same size as gen_models")

    plot_data_only = False
    if len(particle_data) == 0:
        plot_data_only = True

    if not plot_data_only:
        if len(sim_data) != particle_data.shape[1]:
            raise Warning("sim_data and particle_data do not have the same size")

        # select only the first num_samples
        if num_samples == -1:
            num_samples = particle_data.shape[1]

        lengths = [sim_data.shape[0], jet_data_sim.shape[0]]
        if plot_efps:
            lengths.append(efps_sim.shape[0])
        for count, _ in enumerate(particle_data):
            lengths.append(particle_data[count].shape[0])
            lengths.append(jet_data[count].shape[0])
            if plot_efps:
                lengths.append(efps_values[count].shape[0])
        if any(np.array(lengths) < num_samples):
            raise ValueError("num_samples is larger than the smallest dataset")
        sim_data = sim_data[:num_samples]
        particle_data = particle_data[:, :num_samples]
        jet_data_sim = jet_data_sim[:num_samples]
        jet_data = jet_data[:, :num_samples]
        efps_sim = efps_sim[:num_samples]
        efps_values = efps_values[:, :num_samples]

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
    data1 = data1[data1 != 0]
    if not plot_data_only:
        data = [d[:, :, 2].flatten()[d[:, :, 2].flatten() != 0] for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
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
    ax1.set_xlabel(plot_xlabels[0])
    ax1.set_yscale("log")
    ax1.legend(loc="best", prop={"size": 14}, frameon=False)

    ax2 = fig.add_subplot(gs[gs_counter + 1])
    data1 = sim_data[:, :, 0].flatten()
    data1 = data1[data1 != 0]
    if not plot_data_only:
        data = [d[:, :, 0].flatten()[d[:, :, 0].flatten() != 0] for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
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
    ax2.set_xlabel(plot_xlabels[1])
    ax2.set_yscale("log")
    ax3 = fig.add_subplot(gs[gs_counter + 2])

    data1 = sim_data[:, :, 1].flatten()
    data1 = data1[data1 != 0]
    if not plot_data_only:
        data = [d[:, :, 1].flatten()[d[:, :, 1].flatten() != 0] for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
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
    ax3.set_xlabel(plot_xlabels[2])
    ax3.set_yscale("log")
    ax3.set_ylim(
        0.5,
    )

    if plot_jet_features:
        ax4 = fig.add_subplot(gs[gs_counter + 3])
        data1 = jet_data_sim[:, 0]
        if not plot_data_only:
            data = [d[:, 0] for d in jet_data]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
            x_min, x_max = 0.5, 1.5
        if "150" in plottype:
            x_min, x_max = 0.5, 1.5
        if plottype == "q" or plottype == "q_max_particles":
            x_max = 1.25
        ax4.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax4.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax4.set_xlabel(plot_xlabels[3])
        # ax4.set_yscale("log")

        ax5 = fig.add_subplot(gs[gs_counter + 4])
        data1 = jet_data_sim[:, 1]
        if not plot_data_only:
            data = [d[:, 1] for d in jet_data]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_min, x_max = -0.05, 0.05
        if plottype == "t":
            x_min = -0.1
        if plottype == "q" or plottype == "q_max_particles":
            x_max = 0.1
        ax5.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        for count, model in enumerate(particle_data):
            ax5.hist(
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
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_min, x_max = -0.01, 0.01
        if (plottype == "t") or plottype == "t30":
            x_min = -0.1
        ax6.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax6.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax6.set_xlabel(r"Jet $\phi$")
        ax6.set_yscale("log")
        ax6.set_ylim(
            0.5,
        )
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
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.3
    if plottype == "t":
        x_max = 0.3
    elif plottype == "q" or plottype == "q_max_particles":
        x_max = 0.3
    ax7.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax7.hist(
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
    if not mass_linear:
        ax7.set_yscale("log")

    ax8 = fig.add_subplot(gs[gs_counter + 1])
    data1 = np.count_nonzero(sim_data[:, :, 2], axis=1)
    if not plot_data_only:
        data = [np.count_nonzero(d[:, :, 2], axis=1) for d in particle_data]
        x_min, x_max = (
            np.array([d.min() for d in data]).min(),
            np.array([d.max() for d in data]).max(),
        )
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if variable_jet_sizes_plotting:
        binwidth = 1
        if not plot_data_only:
            bins_pm = range(x_min, x_max + binwidth, binwidth)
    else:
        bins_pm = range(0, particles_per_jet)

    ax8.hist(
        data1,
        bins=bins_pm,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax8.hist(
                data[count],
                bins=bins_pm,
                histtype="step",
                range=[x_min, x_max],
                label=f"{labels[count]}",
            )
    ax8.set_yscale("log")
    ax8.set_xlabel("Particle Multiplicity")

    ax9 = fig.add_subplot(gs[gs_counter + 2])
    if plot_efps:
        data1 = np.concatenate(efps_sim)
        if not plot_data_only:
            data = [np.concatenate(d) for d in efps_values]
            x_min, x_max = (
                np.array([d.min() for d in data]).min(),
                np.array([d.max() for d in data]).max(),
            )
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_min, x_max = 0, 0.01
        if plottype == "q" or plottype == "q_max_particles":
            x_min, x_max = 0, 0.0002
        elif plottype == "t":
            x_min, x_max = 0, 0.01
        ax9.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax9.hist(
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
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.5
    ax10.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax10.hist(
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
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.125
    ax11.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax11.hist(
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
        x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
    else:
        x_min, x_max = data1.min(), data1.max()
    if plottype == "sim_data":
        x_min, x_max = data1.min(), data1.max()
    if "150" in plottype:
        x_max = 0.025
    ax12.hist(
        data1,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        range=[x_min, x_max],
        label=sim_data_label,
    )
    if not plot_data_only:
        for count, model in enumerate(particle_data):
            ax12.hist(
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
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        ax13.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax13.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax13.set_xlabel(
            r"Particle $p_\mathrm{T}^\mathrm{rel}$ of Jets with"
            rf" {selected_multiplicities[0]} Particles"
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
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        ax14.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax14.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax14.set_xlabel(
            r"Particle $p_\mathrm{T}^\mathrm{rel}$ of Jets with"
            rf" {selected_multiplicities[1]} Particles"
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
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        ax15.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax15.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax15.set_xlabel(
            r"Particle $p_\mathrm{T}^\mathrm{rel}$ of Jets with"
            rf" {selected_multiplicities[2]} Particles"
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
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        ax16.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax16.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax16.set_xlabel(
            r"Particle $p_\mathrm{T}^\mathrm{rel}$ of Jets with"
            rf" {selected_multiplicities[3]} Particles"
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
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        ax17.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax17.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax17.set_xlabel(
            r"Particle $p_\mathrm{T}^\mathrm{rel}$ of Jets with"
            rf" {selected_multiplicities[4]} Particles"
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
            x_min, x_max = min(x_min, np.min(data1)), max(x_max, np.max(data1))
        else:
            x_min, x_max = data1.min(), data1.max()
        if plottype == "sim_data":
            x_min, x_max = data1.min(), data1.max()
        if "150" in plottype:
            x_max = 0.025
        ax18.hist(
            data1,
            bins=bins,
            histtype="stepfilled",
            alpha=0.5,
            range=[x_min, x_max],
            label=sim_data_label,
        )
        if not plot_data_only:
            for count, model in enumerate(particle_data):
                ax18.hist(
                    data[count],
                    bins=bins,
                    histtype="step",
                    range=[x_min, x_max],
                    label=f"{labels[count]}",
                )
        ax18.set_xlabel(
            r"Particle $p_\mathrm{T}^\mathrm{rel}$ of Jets with"
            rf" {selected_multiplicities[5]} Particles"
        )
        ax18.set_yscale("log")

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
        plt.savefig(f"{save_folder}{save_name}.pdf", bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


def create_and_plot_data(
    sim_data: np.ndarray,
    gen_models,
    cond: list[torch.Tensor] = None,
    save_name: str = "plot",
    labels: list[str] = ["Model"],
    num_jet_samples: int = 10000,
    batch_size: int = 1000,
    plot_efps: bool = False,
    selected_particles: list[int] = [1, 5, 20],
    selected_multiplicities: list[int] = [10, 20, 30, 40, 50, 80],
    plottype: str = "sim_data",
    variable_set_sizes: bool = False,
    mask: np.ndarray = None,
    variable_jet_sizes_plotting: bool = True,
    save_folder: str = "./logs/plots/",
    normalized_data: list[bool] = [False],
    normalize_sigma: int = 5,
    means: list[float] = None,
    stds: list[float] = None,
    save_fig: bool = True,
    plot_selected_multiplicities: bool = False,
    print_parameters: bool = True,
    plot_jet_features: bool = False,
    plot_w_dists: bool = False,
    mass_linear: bool = True,
    bins: int = 100,
    sim_data_label: str = "JetNet",
    file_dict: dict = None,
    close_fig: bool = False,
    ode_solver: str = "midpoint",
    ode_steps: int = 100,
):
    """Generate data for plotting and plot it.

    Args:
        sim_data (_type_): _description_
        gen_models (_type_): _description_
        cond (list[torch.Tensor], optional): Condition data in case of conditioned model.
            Defaults to None.
        save_name (_type_): _description_
        labels (_type_): _description_
        num_jet_samples (int, optional): _description_. Defaults to 10000.
        batch_size (int, optional): Batch size for generating. Defaults to 10000.
        plot_efps (bool, optional): Plot EFPs. Defaults to False.
        selected_particles (list, optional): _description_. Defaults to [1, 5, 20].
        selected_multiplicities (list, optional): _description_.
            Defaults to [10, 20, 30, 40, 50, 80].
        plottype (str, optional): _description_. Defaults to "sim_data".
        variable_set_sizes (bool, optional): _description_. Defaults to False.
        mask (_type_, optional): _description_. Defaults to None.
        variable_jet_sizes_plotting (bool, optional): _description_. Defaults to True.
        save_folder (str, optional): _description_. Defaults to "/home/ewencedr/equivariant-flows".
        normalized_data (list, optional): _description_. Defaults to [False].
        normalize_sigma (int, optional): _description_. Defaults to 5.
        means (_type_, optional): _description_. Defaults to None.
        stds (_type_, optional): _description_. Defaults to None.
        save_fig (bool, optional): _description_. Defaults to True.
        plot_selected_multiplicities (bool, optional): _description_. Defaults to False.
        print_parameters (bool, optional): _description_. Defaults to True.
        plot_jet_features (bool, optional): _description_. Defaults to True.
        plot_w_dists (bool, optional): _description_. Defaults to True.
        mass_linear (bool, optional): Plot mass distribution in linear scale. Defaults to True.
        bins (int, optional): _description_. Defaults to 100.
        sim_data_label (str, optional): _description_. Defaults to "JetNet".
        file_dict (_type_, optional): _description_. Defaults to None.
        close_fig (bool, optional): Close fig after saving. Defaults to False.
        ode_solver (str, optional): ODE solver used for sampling. Defaults to "midpoint".
        ode_steps (int, optional): Number of steps for ODE solver. Defaults to 100.

    Raises:
        AssertionError: _description_

    Returns:
        _type_: _description_
    """
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
        selected_particles,
        plot_selected_multiplicities=plot_selected_multiplicities,
        selected_multiplicities=selected_multiplicities,
        variable_set_sizes=variable_set_sizes,
        mask=mask,
        normalized_data=normalized_data,
        normalize_sigma=normalize_sigma,
        means=means,
        stds=stds,
        file_dict=file_dict,
        calculate_efps=plot_efps,
        cond=cond,
        ode_solver=ode_solver,
        ode_steps=ode_steps,
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
        plot_efps=plot_efps,
        plot_selected_multiplicities=plot_selected_multiplicities,
        selected_multiplicities=selected_multiplicities,
        selected_particles=selected_particles,
        pt_selected_particles=pt_selected_particles,
        pt_selected_multiplicities=pt_selected_multiplicities,
        pt_selected_particles_sim=pt_selected_particles_sim,
        pt_selected_multiplicities_sim=pt_selected_multiplicities_sim,
        w_dist_m=w_dist_m,
        mass_linear=mass_linear,
        save_folder=save_folder,
        save_name=save_name,
        plottype=plottype,
        save_fig=save_fig,
        variable_jet_sizes_plotting=variable_jet_sizes_plotting,
        bins=bins,
        close_fig=close_fig,
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
    plt.savefig(f"{save_path}/plots/loss_plots_{name}.pdf")
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
    variable_set_sizes=False,
    mask=None,
    save_path="/home/ewencedr/equivariant-flows",
    ode_solver: str = "midpoint",
    ode_steps: int = 100,
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
                variable_set_sizes=False,
                mask=None,
                ode_solver=ode_solver,
                ode_steps=ode_steps,
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
    plt.savefig(f"{save_path}/plots/{name}.pdf")
    return np.array(times)


def prepare_data_for_plotting(
    data: list[np.ndarray],
    calculate_efps: bool = False,
    selected_particles: list[int] = [1, 3, 10],
    selected_multiplicities: list[int] = [20, 30, 40],
):
    """Calculate the features for plotting, i.e. the jet features, the efps, the pt of selected
    particles and the pt of selected multiplicities.

    Args:
        data (list of np.ndarray): list of data where data is in the shape
            (n_jets, n_particles, n_features) with features (pt, eta, phi)
            --> this allows to process data in batches. Will be concatenated
            in the output
        calculate_efps (bool, optional): If efps should be calculated. Defaults to False.
        selected_particles (list[int], optional): Selected particles. Defaults to [1,3,10].
        selected_multiplicities (list[int], optional): Selected multiplicities.
            Defaults to [20, 30, 40].

    Returns:
        np.ndarray : jet_data, shape (len(data), n_jets, n_features)
        np.ndarray : efps, shape (len(data), n_jets, n_efps)
        np.ndarray : pt_selected_particles, shape (len(data), n_selected_particles, n_jets)
        dict : pt_selected_multiplicities
    """

    jet_data = []
    efps_values = []
    pt_selected_particles = []
    pt_selected_multiplicities = []
    for count, data_temp in enumerate(data):
        jet_data_temp = calculate_jet_features(data_temp)
        efps_temp = []
        if calculate_efps:
            efps_temp = efps(data_temp)
        pt_selected_particles_temp = get_pt_of_selected_particles(data_temp, selected_particles)
        # TODO: should probably set the number of jets in the function call below?
        pt_selected_multiplicities_temp = get_pt_of_selected_multiplicities(
            data_temp, selected_multiplicities
        )

        jet_data.append(jet_data_temp)
        efps_values.append(efps_temp)
        pt_selected_particles.append(pt_selected_particles_temp)
        pt_selected_multiplicities.append(pt_selected_multiplicities_temp)

    new_dict = {}
    for count, i in enumerate(selected_multiplicities):
        new_dict[f"{count}"] = []

    for dicts in pt_selected_multiplicities:
        for count, dict_items_array in enumerate(dicts):
            new_dict[f"{count}"].append(np.array(dicts[dict_items_array]))

    for count, i in enumerate(new_dict):
        new_dict[i] = np.array(new_dict[i])

    return np.array(jet_data), np.array(efps_values), np.array(pt_selected_particles), new_dict


def create_data_for_plotting(
    sim_data_in: np.ndarray,
    gen_models,
    num_jet_samples: int = 10000,
    batch_size: int = 10000,
    selected_particles: list[int] = [1, 3, 10],
    plot_selected_multiplicities: bool = False,
    selected_multiplicities: list[int] = [20, 30, 40],
    variable_set_sizes: bool = False,
    mask=None,
    normalized_data: list[bool] = [False],
    normalize_sigma: int = 5,
    means: list[float] = None,
    stds: list[float] = None,
    file_dict: dict = None,
    calculate_efps: bool = False,
    cond: list[torch.Tensor] = None,
    ode_solver: str = "midpoint",
    ode_steps: int = 100,
):
    data = []
    times = []
    jet_data = []
    efps_values = []
    pt_selected_particles = []
    pt_selected_multiplicities = []
    w_dist_m = []
    sim_data = sim_data_in[:num_jet_samples]
    mask = mask[:num_jet_samples]
    if cond is not None:
        for count, c in enumerate(cond):
            cond[count] = c[:num_jet_samples]
    jet_data_sim = calculate_jet_features(sim_data)
    efps_sim = []
    if calculate_efps:
        efps_sim = efps(sim_data)
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
                variable_set_sizes=variable_set_sizes,
                mask=torch.tensor(mask),
                normalized_data=normalized_data[count],
                normalize_sigma=normalize_sigma,
                means=means,
                stds=stds,
                cond=cond[count],
                ode_solver=ode_solver,
                ode_steps=ode_steps,
            )

        jet_data_temp = calculate_jet_features(data_temp)
        efps_temp = []
        if calculate_efps:
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


def plot_substructure(
    tau21: np.array,
    tau32: np.array,
    d2: np.array,
    tau21_jetnet: np.array,
    tau32_jetnet: np.array,
    d2_jetnet: np.array,
    bins: int = 100,
    save_fig: bool = True,
    close_fig: bool = True,
    save_folder: str = None,
    save_name: str = None,
    model_name: str = "Model",
    simulation_name: str = "JetNet",
) -> None:
    """Plot the tau21, tau32 and d2 distributions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    hist_tau21_jetnet = ax1.hist(
        tau21_jetnet, bins=bins, label=simulation_name, histtype="stepfilled", alpha=0.5
    )
    ax1.hist(tau21, bins=hist_tau21_jetnet[1], label=model_name, histtype="step")
    ax1.set_xlabel(r"$\tau_{21}$")
    ax1.legend(loc="best", frameon=False)

    hist_tau32_jetnet = ax2.hist(
        tau32_jetnet, bins=bins, label=simulation_name, histtype="stepfilled", alpha=0.5
    )
    ax2.hist(tau32, bins=hist_tau32_jetnet[1], label=f"{model_name}", histtype="step")
    ax2.set_xlabel(r"$\tau_{32}$")
    ax2.legend(loc="best", frameon=False)

    hist_d2_jetnet = ax3.hist(
        d2_jetnet, bins=bins, label=simulation_name, histtype="stepfilled", alpha=0.5
    )
    ax3.hist(d2, bins=hist_d2_jetnet[1], label=f"{model_name}", histtype="step")
    ax3.set_xlabel(r"$d_2$")
    ax3.legend(loc="best", frameon=False)

    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
        plt.savefig(f"{save_folder}{save_name}.pdf", bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


def plot_full_substructure(
    data_substructure: np.array,
    data_substructure_jetnet: np.array,
    keys: np.array,
    bins: int = 100,
    model_name: str = "Model",
    simulation_name: str = "JetNet",
    save_fig: bool = True,
    close_fig: bool = True,
    save_folder: str = None,
    save_name: str = None,
) -> None:
    """Plot all substructure distributions."""
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))

    for i, ax in enumerate(axs.flatten()):
        hist_jetnet = ax.hist(
            data_substructure_jetnet[i],
            bins=bins,
            label=simulation_name,
            histtype="stepfilled",
            alpha=0.5,
        )
        ax.hist(data_substructure[i], bins=hist_jetnet[1], label=f"{model_name}", histtype="step")
        ax.set_title(keys[i])
        ax.legend(loc="best", frameon=False)

    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
        plt.savefig(f"{save_folder}{save_name}.pdf", bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


def plot_particle_features(
    data_sim: np.array,
    data_gen: np.array,
    mask_sim: np.array,
    mask_gen: np.array,
    feature_names: list,
    legend_label_sim: str = "Sim. data",
    legend_label_gen: str = "Gen. data",
    plot_path: str = None,
    also_png: bool = False,
):
    """Plot the particle features.

    Args:
        data_sim (np.array): Simulated particle data of shape (n_jets, n_particles, n_features)
        data_gen (np.array): Generated particle data of shape (n_jets, n_particles, n_features)
        mask_sim (np.array): Mask for simulated particle data of shape (n_jets, n_particles, 1)
        mask_gen (np.array): Mask for generated particle data of shape (n_jets, n_particles, 1)
        feature_names (list): List of feature names (as in the file, e.g. `part_etarel`)
        legend_label_sim (str, optional): Label for the simulated data. Defaults to "Sim. data".
        legend_label_gen (str, optional): Label for the generated data. Defaults to "Gen. data".
        plot_path (str, optional): Path to save the plot. Defaults to None. Which means
            the plot is not saved.
        also_png (bool, optional): If True, also save the plot as png. Defaults to False.
    """
    # plot the generated features and compare sim. data to gen. data
    nvars = data_sim.shape[-1]
    plot_cols = 3
    plot_rows = nvars // 3 + 1 * int(bool(nvars % 3))
    fig, ax = plt.subplots(plot_rows, plot_cols, figsize=(11, 2.8 * plot_rows))
    ax = ax.flatten()
    hist_kwargs = {}
    for i in range(data_sim.shape[-1]):
        values_sim = data_sim[:, :, i][mask_sim[:, :, 0] != 0].flatten()
        values_gen = data_gen[:, :, i][mask_sim[:, :, 0] != 0].flatten()
        _, bin_edges = np.histogram(np.concatenate([values_sim, values_gen]), bins=100)
        hist_kwargs["bins"] = bin_edges
        ax[i].hist(values_sim, label=legend_label_sim, alpha=0.5, **hist_kwargs)
        ax[i].hist(
            values_gen,
            label=legend_label_gen,
            histtype="step",
            **hist_kwargs,
        )
        ax[i].set_yscale("log")
        feature_name = feature_names[i]
        ax[i].set_xlabel(JETCLASS_FEATURE_LABELS.get(feature_name, feature_name))
    ax[2].legend(frameon=False)
    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path)
        if also_png and plot_path.endswith(".pdf"):
            fig.savefig(plot_path.replace(".pdf", ".png"))

def plot_jet_features(
    jet_data_sim: np.array,
    jet_data_gen: np.array,
    jet_feature_names: list,
    legend_label_sim: str = "Sim. data",
    legend_label_gen: str = "Gen. data",
    plot_path: str = None,
    also_png: bool = False,
):
    """Plot the particle features.

    Args:
        jet_data_sim (np.array): Simulated jet data of shape (n_jets, n_features)
        jet_data_gen (np.array): Generated jet data of shape (n_jets, n_features)
        jet_feature_names (list): List of feature names (as in the file, e.g. `jet_pt`)
        legend_label_sim (str, optional): Label for the simulated data. Defaults to "Sim. data".
        legend_label_gen (str, optional): Label for the generated data. Defaults to "Gen. data".
        plot_path (str, optional): Path to save the plot. Defaults to None. Which means
            the plot is not saved.
        also_png (bool, optional): If True, also save the plot as png. Defaults to False.
    """
    # plot the generated features and compare sim. data to gen. data
    # nvars = data_sim.shape[-1]
    # plot_cols = 3
    # plot_rows = nvars // 3 + 1 * int(bool(nvars % 3))
    plot_rows = 3
    fig, ax = plt.subplots(plot_rows, 3, figsize=(11, 2.8 * plot_rows))
    ax = ax.flatten()
    hist_kwargs = {}
    for i in range(jet_data_sim.shape[-1]):
        values_sim = jet_data_sim[:, i]
        values_gen = jet_data_gen[:, i]
        _, bin_edges = np.histogram(np.concatenate([values_sim, values_gen]), bins=100)
        hist_kwargs["bins"] = bin_edges
        ax[i].hist(values_sim, label=legend_label_sim, alpha=0.5, **hist_kwargs)
        ax[i].hist(
            values_gen,
            label=legend_label_gen,
            histtype="step",
            **hist_kwargs,
        )
        ax[i].set_yscale("log")
        feature_name = jet_feature_names[i]
        ax[i].set_xlabel(JET_FEATURE_LABELS.get(feature_name, feature_name))
    ax[2].legend(frameon=False)
    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path)
        if also_png and plot_path.endswith(".pdf"):
            fig.savefig(plot_path.replace(".pdf", ".png"))

