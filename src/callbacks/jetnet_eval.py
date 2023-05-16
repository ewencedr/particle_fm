import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from src.data.components import calculate_all_wasserstein_metrics
from src.data.components.utils import count_parameters, jet_masses
from src.utils import apply_mpl_styles, create_and_plot_data


# TODO wandb logging video of jets, histograms, annd point clouds
# TODO wandb log interactive plot
# TODO remove parameter logging from comet
# TODO don't break if no logger is used
# TODO fix efp logging
class JetNetEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the JetNet dataset and log
    the results to loggers. Currently supported are CometLogger and WandbLogger.

    Args:
        every_n_epochs (int, optional): Log every n epochs. Defaults to 10.
        num_jet_samples (int, optional): How many jet samples to generate. Defaults to 25000.
        w_dists_batches (int, optional): How many batches to calculate Wasserstein distances. Jet samples for each batch are num_jet_samples // w_dists_batches. Defaults to 5.
        image_path (str, optional): Folder where the images are saved. Defaults to "/beegfs/desy/user/ewencedr/comet_logs".
        model_name (str, optional): Name for saving the model. Defaults to "model-test".
        calculate_efps (bool, optional): Calculate EFPs for the jets. Defaults to False.
        log_w_dists (bool, optional): Calculate and log wasserstein distances Defaults to False.
        log_num_parameters (bool, optional): Log parameters of model. Only logged in first epoch. Defaults to True.
        log_times (bool, optional): Log generation times of data. Defaults to True.
        log_epoch_zero (bool, optional): Log in first epoch. Default to False.
        mass_conditioning (bool, optional): Condition on mass. Defaults to False.
        **kwargs: Arguments for create_and_plot_data
    """

    def __init__(
        self,
        every_n_epochs: int = 10,
        num_jet_samples: int = 25000,
        w_dists_batches: int = 5,
        image_path: str = "./logs/callback_images/",
        model_name: str = "model",
        calculate_efps: bool = False,
        log_w_dists: bool = False,
        log_num_parameters: bool = True,
        log_times: bool = True,
        log_epoch_zero: bool = False,
        mass_conditioning: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_jet_samples = num_jet_samples
        self.w_dists_batches = w_dists_batches
        self.log_w_dists = log_w_dists
        self.log_num_parameters = log_num_parameters
        self.log_times = log_times
        self.log_epoch_zero = log_epoch_zero

        self.mass_conditioning = mass_conditioning

        # Parameters for plotting
        self.model_name = model_name
        self.calculate_efps = calculate_efps
        self.kwargs = kwargs

        self.image_path = image_path
        apply_mpl_styles()

        # loggers
        self.comet_logger = None
        self.wandb_logger = None

    def on_train_start(self, trainer, pl_module) -> None:
        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        log_epoch = True
        if not self.log_epoch_zero and trainer.current_epoch == 0:
            log_epoch = False
        if trainer.current_epoch % self.every_n_epochs == 0 and log_epoch:
            if self.mass_conditioning:
                cond = jet_masses(trainer.datamodule.tensor_test).unsqueeze(-1)
            else:
                cond = None
            plot_name = f"{self.model_name}--epoch{trainer.current_epoch}"
            fig, particle_data, times = create_and_plot_data(
                np.array(trainer.datamodule.tensor_test),
                [pl_module],
                cond=cond,
                save_name=plot_name,
                labels=["Model"],
                normalized_data=[trainer.datamodule.hparams.normalize],
                normalize_sigma=trainer.datamodule.hparams.normalize_sigma,
                variable_set_sizes=trainer.datamodule.hparams.variable_jet_sizes,
                mask=np.array(trainer.datamodule.mask_test),
                num_jet_samples=self.num_jet_samples,
                means=trainer.datamodule.means,
                stds=trainer.datamodule.stds,
                save_folder=self.image_path,
                print_parameters=False,
                plot_efps=self.calculate_efps,
                **self.kwargs,
            )

            particle_data = particle_data[0]
            mask_data = np.ma.masked_where(
                particle_data[:, :, 0] == 0,
                particle_data[:, :, 0],
            )
            mask_data = np.expand_dims(mask_data, axis=-1)

            if self.log_w_dists:
                # 1 batch
                w_dists_1b = calculate_all_wasserstein_metrics(
                    trainer.datamodule.tensor_test[..., :3],
                    particle_data,
                    trainer.datamodule.mask_test,
                    mask_data,
                    num_eval_samples=self.num_jet_samples,
                    num_batches=1,
                    calculate_efps=self.calculate_efps,
                )

                # divide into batches
                w_dists = calculate_all_wasserstein_metrics(
                    trainer.datamodule.tensor_test[..., :3],
                    particle_data,
                    trainer.datamodule.mask_test,
                    mask_data,
                    num_eval_samples=self.num_jet_samples // self.w_dists_batches,
                    num_batches=self.w_dists_batches,
                    calculate_efps=self.calculate_efps,
                )
                # TODO log both properly in comet
                # Wasserstein Metrics
                text = f"W-Dist epoch:{trainer.current_epoch} W1m: {w_dists['w1m_mean']}+-{w_dists['w1m_std']}, W1p: {w_dists['w1p_mean']}+-{w_dists['w1p_std']}, W1efp: {w_dists['w1efp_mean']}+-{w_dists['w1efp_std']}"
                self.comet_logger.log_text(text)
                self.comet_logger.log_metrics(w_dists)
                self.wandb_logger.log({"Wasserstein Metrics": w_dists})

                text_1b = f"1 BATCH W-Dist epoch:{trainer.current_epoch} W1m: {w_dists_1b['w1m_mean']}+-{w_dists_1b['w1m_std']}, W1p: {w_dists_1b['w1p_mean']}+-{w_dists_1b['w1p_std']}, W1efp: {w_dists_1b['w1efp_mean']}+-{w_dists_1b['w1efp_std']}"
                self.comet_logger.log_text(text_1b)
                self.comet_logger.log_metrics(w_dists_1b)
                self.wandb_logger.log({"Wasserstein Metrics": w_dists})

            # Jet genereation time
            if self.log_times:
                self.comet_logger.log_metrics({"Jet generation time": times})
                self.wandb_logger.log({"Jet generation time": times})

            # Histogram Plots
            img_path = f"{self.image_path}{plot_name}.png"
            self.comet_logger.log_image(img_path, name=f"epoch{trainer.current_epoch}")
            self.wandb_logger.log({f"epoch{trainer.current_epoch}": wandb.Image(img_path)})
            # self.wandb_logger.log({f"epoch{trainer.current_epoch}-fig": fig})
            # Parameters
            if self.log_num_parameters and trainer.current_epoch == 0:
                parameters = count_parameters(pl_module)
                self.comet_logger.log_hyperparams({"parameters": parameters})
