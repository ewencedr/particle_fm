import time
import torch
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches


def generate_data_calochallenge(
    model,
    dl,
    scaler,
    verbose: bool = False,
    ode_solver: str = "dopri5_zuko",
    ode_steps: int = 100,
    hists: dict = {},
    **kwargs
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
        verbose (bool, optional): Print generation progress. Defaults to True.
        ode_solver (str, optional): ODE solver for sampling. Defaults to "dopri5_zuko".
        ode_steps (int, optional): Number of steps for ODE solver. Defaults to 100.

    Raises:
        ValueError: _description_

    Returns:
        np.array: sampled data of shape (num_jet_samples, num_particles, num_features) with features (eta, phi, pt)
        float: generation time
    """
    start_time = time.time()

    k = 0
    hists_real = hists["hists_real"]
    hists_fake = hists["hists_fake"]
    hists_real_unscaled = hists["hists_real_unscaled"]
    hists_fake_unscaled = hists["hists_fake_unscaled"]
    weighted_hists_real = hists["weighted_hists_real"]
    weighted_hists_fake = hists["weighted_hists_fake"]
    h_response_real = hists["response_real"]
    h_response_fake = hists["response_fake"]
    for i in dl:

        data, mask, cond = i[0], i[1], i[2]
        k += 1
        with torch.no_grad():
            fake = model.sample(
                data.shape[0],
                cond,
                mask,
                ode_solver=ode_solver,
                ode_steps=ode_steps,
                num_points=data.shape[1],
            ).cpu()
            for j in range(4):
                hists_real_unscaled[j].fill(data[mask.squeeze(-1)][:, j].cpu().numpy())
                hists_fake_unscaled[j].fill(
                    fake[: len(data)][mask.squeeze(-1)][:, j].cpu().numpy()
                )
            fake = scaler.inverse_transform(fake).float()
            data = scaler.inverse_transform(data).float()
            response_real = (
                (fake[: len(cond), :, 0].sum(1) / (cond[:, 0] + 10).exp())
                .cpu()
                .numpy()
                .reshape(-1)
            )
            response_fake = (
                (data[:, :, 0].sum(1) / (cond[:, 0] + 10).exp()).cpu().numpy().reshape(-1)
            )
            h_response_real.fill(response_real)
            h_response_fake.fill(response_fake)
            for j in range(4):

                if j > 0:
                    hists_real[j].fill(data[mask.squeeze(-1)][:, j].cpu().numpy().astype(int))
                    hists_fake[j].fill(
                        fake[: len(data)][mask.squeeze(-1)][:, j].cpu().numpy().astype(int)
                    )

                    weighted_hists_fake[j - 1].fill(
                        fake[: len(data)][mask.squeeze(-1)][:, j].cpu().numpy().astype(int),
                        weight=fake[: len(data)][mask.squeeze(-1)][:, 0].cpu().numpy(),
                    )
                    weighted_hists_real[j - 1].fill(
                        data[mask.squeeze(-1)][:, j].cpu().numpy().astype(int),
                        weight=data[mask.squeeze(-1)][:, 0].cpu().numpy(),
                    )
                else:
                    hists_real[j].fill(data[mask.squeeze(-1)][:, j].cpu().numpy())
                    hists_fake[j].fill(fake[: len(data)][mask.squeeze(-1)][:, j].cpu().numpy())
        if k > 10:
            break

    end_time = time.time()
    generation_time = end_time - start_time

    return hists, generation_time


class plotting_point_cloud:
    """This is a class that takes care of  plotting steps in the script,
    It is initialized with the following arguments:
    true=the simulated data, note that it needs to be scaled
    gen= Generated data , needs to be scaled
    step=The current step of the training, this is need for tensorboard
    model=the model that is trained, a bit of an overkill as it is only used to access the losses
    config=the config used for training
    logger=The logger used for tensorboard logging"""

    def __init__(self, step=None, logger=None, weight=1):

        self.step = step

        self.weight = weight
        self.fig_size1 = [6.4, 6.4]
        self.fig_size2 = [2 * 6.4, 6.4]
        self.fig_size3 = [3 * 6.4, 6.4]
        self.fig_size4 = [4 * 6.4, 6.4]
        self.alpha = 0.3
        mpl.rcParams["lines.linewidth"] = 2
        font = {"size": 18}  # "family": "normal",
        mpl.rc("font", **font)
        mpl.rc("lines", linewidth=2)
        sns.set_palette("Pastel1")
        if logger is not None:

            self.summary = logger
        else:
            self.summary = None

    def plot_calo(self, h_real, h_fake, weighted, leg=-1, unscaled=False):
        # This creates a histogram of the inclusive distributions and calculates the mass of each jet
        # and creates a histogram of that
        # if save, the histograms are logged to tensorboard otherwise they are shown
        # if quantile, this also creates a histogram of a subsample of the generated data,
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i = 0
        k = 0
        fig, ax = plt.subplots(
            2,
            4 if not weighted else 3,
            gridspec_kw={"height_ratios": [4, 1]},
            figsize=self.fig_size4,
            sharex="col",
        )

        cols = ["E", "z", "alpha", "R"]
        names = [r"$E$", r"$z$", r"$\alpha$", r"$R$"]
        if weighted:
            cols = ["z", "alpha", "R"]
            names = [r"$z$", r"$\alpha$", r"$R$"]
        for v, name in zip(cols, names):
            main_ax_artists, sublot_ax_arists = h_real[k].plot_ratio(
                h_fake[k],
                ax_dict={"main_ax": ax[0, k], "ratio_ax": ax[1, k]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0, k].set_xlabel("")
            ax[0, k].patches[1].set_fill(True)
            ax[0, k].ticklabel_format(
                axis="y", style="scientific", scilimits=(-3, 3), useMathText=True
            )
            ax[0, k].patches[1].set_fc(sns.color_palette()[1])
            ax[0, k].patches[1].set_edgecolor("black")
            ax[0, k].patches[1].set_alpha(self.alpha)
            ax[1, k].set_xlabel(name)
            ax[0, k].set_ylabel("Counts")
            ax[1, k].set_ylabel("Ratio")
            ax[0, k].patches[0].set_lw(2)
            ax[0, k].get_legend().remove()
            xticks = [int(h_real[k].axes[0].edges[-1] // 4 * i) for i in range(0, int(5))]

            ax[1, k].set_xticks(np.array(xticks), np.array(xticks))
            ax[0, k].set_xticks(np.array(xticks))

            ax[1, k].set_ylim(0.75, 1.25)
            k += 1
        if not weighted:
            ax[0, 0].set_yscale("log")
        ax[0, leg].legend(loc="best", fontsize=18)
        handles, labels = ax[0, leg].get_legend_handles_labels()
        handles[1] = mpatches.Patch(color=sns.color_palette()[1], label="The red data")
        ax[0, leg].legend(handles, labels)
        plt.tight_layout(pad=0.2)
        self.summary.log_image(
            "{}_ratio_{}".format(
                "weighted " if weighted else "unweighted ", "" if not unscaled else "unscaled"
            ),
            [fig],
            self.step,
        )
        plt.close()

    def plot_jet(self, h_real, h_fake, leg=-1):
        # This creates a histogram of the inclusive distributions and calculates the mass of each jet
        # and creates a histogram of that
        # if save, the histograms are logged to tensorboard otherwise they are shown
        # if quantile, this also creates a histogram of a subsample of the generated data,
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i = 0
        k = 0
        fig, ax = plt.subplots(2, 4, gridspec_kw={"height_ratios": [4, 1]}, figsize=self.fig_size4)
        plt.suptitle("All Particles", fontsize=18)
        for v, name in zip(
            ["eta", "phi", "pt", "m"],
            [r"$\eta^{\tt rel}$", r"$\phi^{\tt rel}$", r"$p_T^{\tt rel}$", r"$m^{\tt rel}$"],
        ):

            main_ax_artists, sublot_ax_arists = h_fake[k].plot_ratio(
                h_real[k],
                ax_dict={"main_ax": ax[0, k], "ratio_ax": ax[1, k]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar)
            )
            i += 1
            ax[0, k].set_xlabel("")
            ax[0, k].patches[1].set_fill(True)
            ax[0, k].ticklabel_format(
                axis="y", style="scientific", scilimits=(-3, 3), useMathText=True
            )
            ax[0, k].patches[1].set_fc(sns.color_palette()[1])
            ax[0, k].patches[1].set_edgecolor("black")
            ax[0, k].patches[1].set_alpha(self.alpha)
            ax[1, k].set_xlabel(name)
            ax[0, k].set_ylabel("Counts")
            ax[1, k].set_ylabel("Ratio")
            ax[0, k].patches[0].set_lw(2)
            ax[0, k].get_legend().remove()
            k += 1
        ax[0, leg].legend(loc="best", fontsize=18)
        handles, labels = ax[0, leg].get_legend_handles_labels()
        ax[0, -1].locator_params(nbins=4, axis="x")
        ax[1, -1].locator_params(nbins=4, axis="x")
        handles[1] = patches.Patch(color=sns.color_palette()[1], label="The red data")
        ax[0, leg].legend(handles, labels)
        plt.tight_layout(pad=1)
        # if not save==None:
        #     plt.savefig(save+".pdf",format="pdf")
        plt.tight_layout()
        try:
            self.summary.log_image("inclusive", [fig], self.step)
            plt.close()
        except:
            plt.show()

    def plot_scores(self, pred_real, pred_fake, train, step):
        fig, ax = plt.subplots()
        bins = 30  # np.linspace(0,1,10 if train else 100)
        ax.hist(pred_fake, label="Generated", bins=bins, histtype="step")
        if pred_real.any():
            ax.hist(
                pred_real, label="Ground Truth", bins=bins, histtype="stepfilled", alpha=self.alpha
            )
        ax.legend()
        ax.patches[0].set_lw(2)
        plt.ylabel("Counts")
        plt.xlabel("Critic Score")
        if self.summary:
            plt.tight_layout()
            if pred_real.any():
                self.summary.log_image("class_train" if train else "class_val", [fig], self.step)
            else:
                self.summary.log_image("class_gen", [fig], self.step)
            plt.close()
        else:
            plt.savefig("plots/scores_" + str(train) + ".pdf", format="pdf")
            plt.show()

    def plot_response(self, h_real, h_fake):
        fig, ax = plt.subplots(2, sharex=True)
        h_real.plot_ratio(
            h_fake,
            ax_dict={"main_ax": ax[0], "ratio_ax": ax[1]},
            rp_ylabel=r"Ratio",
            bar_="blue",
            rp_num_label="Generated",
            rp_denom_label="Ground Truth",
            rp_uncert_draw_type="line",  # line or bar
        )
        ax[0].set_xlabel("")
        plt.ylabel("Counts")
        plt.xlabel("Response")
        ax[0].legend()
        if self.summary:
            plt.tight_layout()
            self.summary.log_image("response", [fig], self.step)
            plt.close()
        else:
            plt.savefig("plots/response.pdf", format="pdf")
            plt.show()
