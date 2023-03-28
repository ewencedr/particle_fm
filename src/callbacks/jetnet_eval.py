import numpy as np
import pytorch_lightning as pl

from src.data.components import calculate_all_wasserstein_metrics
from src.data.components.utils import count_parameters
from src.utils import apply_mpl_styles, create_and_plot_data

# TODO Add dataset as input instead of test_particle_data, test_mask, means, stds


class JetNetEvaluationCallback(pl.Callback):
    def __init__(
        self,
        logger=1,
        every_n_epochs=10,
        num_jet_samples=25000,
        w_dists_batches=5,
        sampling_batch_size=1000,
        test_particle_data=None,
        test_mask=None,
        means=None,
        stds=None,
        normalised_data=True,
        image_path="/beegfs/desy/user/ewencedr/comet_logs",
        model_name="model-test",
        max_particles=True,
        mgpu=True,
        selected_particles=[1, 5, 20],
        selected_multiplicities=[10, 20, 30, 40, 50, 80],
        plot_selected_multiplicities=False,
        plottype="sim_data",
    ):
        super().__init__()
        self.logger = logger
        self.every_n_epochs = (
            every_n_epochs  # Only save those jets every N epochs to reduce logging
        )
        self.num_jet_samples = num_jet_samples  # Number of jets to generate
        self.w_dists_batches = w_dists_batches
        self.sampling_batch_size = sampling_batch_size

        # Parameters for plotting
        self.model_name = model_name
        self.test_particle_data = test_particle_data
        self.test_mask = test_mask
        self.means = means
        self.stds = stds
        self.normalised_data = normalised_data
        self.max_particles = max_particles
        self.mgpu = mgpu
        self.selected_particles = selected_particles
        self.selected_multiplicites = selected_multiplicities
        self.plot_selected_multiplicities = plot_selected_multiplicities
        self.plottype = plottype

        self.image_path = image_path
        apply_mpl_styles()

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            plot_name = f"{self.model_name}--epoch{trainer.current_epoch}"
            fig, particle_data, times = create_and_plot_data(
                self.test_particle_data,
                [pl_module],
                save_name=plot_name,
                labels=["Model"],
                normalised_data=[self.normalised_data],
                mgpu=self.mgpu,
                plottype=self.plottype,
                max_particles=self.max_particles,
                mask=self.test_mask,
                batch_size=self.sampling_batch_size,
                num_jet_samples=self.num_jet_samples,
                means=self.means,
                stds=self.stds,
                save_folder=self.image_path,
                print_parameters=False,
                selected_particles=self.selected_particles,
                selected_multiplicities=self.selected_multiplicites,
                plot_selected_multiplicities=self.plot_selected_multiplicities,
            )

            particle_data = particle_data[0]
            mask_data = np.ma.masked_where(
                particle_data[:, :, 0] == 0,
                particle_data[:, :, 0],
            )
            mask_data = np.expand_dims(mask_data, axis=-1)

            # 1 batch
            w_dists_1b = calculate_all_wasserstein_metrics(
                self.test_particle_data[..., :3],
                particle_data,
                self.test_mask,
                mask_data,
                num_eval_samples=self.num_jet_samples,
                num_batches=1,
            )

            # divide into 3 batches
            w_dists = calculate_all_wasserstein_metrics(
                self.test_particle_data[..., :3],
                particle_data,
                self.test_mask,
                mask_data,
                num_eval_samples=self.num_jet_samples // self.w_dists_batches,
                num_batches=self.w_dists_batches,
            )

            # Wasserstein Metrics
            text = f"W-Dist epoch:{trainer.current_epoch} W1m: {w_dists['w1m_mean']}+-{w_dists['w1m_std']}, W1p: {w_dists['w1p_mean']}+-{w_dists['w1p_std']}, W1efp: {w_dists['w1efp_mean']}+-{w_dists['w1efp_std']}"
            trainer.loggers[self.logger].experiment.log_text(text)
            trainer.loggers[self.logger].experiment.log_metrics(w_dists)

            text_1b = f"1 BATCH W-Dist epoch:{trainer.current_epoch} W1m: {w_dists_1b['w1m_mean']}+-{w_dists_1b['w1m_std']}, W1p: {w_dists_1b['w1p_mean']}+-{w_dists_1b['w1p_std']}, W1efp: {w_dists_1b['w1efp_mean']}+-{w_dists_1b['w1efp_std']}"
            trainer.loggers[self.logger].experiment.log_text(text_1b)
            trainer.loggers[self.logger].experiment.log_metrics(w_dists_1b)

            # Histogram Plots
            img_path = f"{self.image_path}/plots/{plot_name}.png"
            trainer.loggers[self.logger].experiment.log_image(
                img_path, name=f"epoch{trainer.current_epoch}"
            )

            # Parameters
            parameters = count_parameters(pl_module)
            trainer.loggers[self.logger].log_hyperparams({"parameters": parameters})
