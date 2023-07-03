from typing import Mapping, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import yaml

from src.data.components import calculate_all_wasserstein_metrics
from src.utils.data_generation import generate_data
from src.utils.plotting import apply_mpl_styles, plot_data, prepare_data_for_plotting
from src.utils.pylogger import get_pylogger

from .ema import EMA, EMAModelCheckpoint

log = get_pylogger("JetNetFinalEvaluationCallback")


class JetNetFinalEvaluationCallback(pl.Callback):
    """Callback to do final evaluation of the model after training. Specific to JetNet dataset.

    Args:
        use_ema (bool, optional): Use exponential moving average weights for logging. Defaults to False.
        dataset (str, optional): Dataset to evaluate on. Defaults to "test".
        nr_checkpoint_callbacks (int, optional): Number of checkpoint callback that is used to select best epoch. Will only be used when ckpt_path is None. Defaults to 1.
        ckpt_path (Optional[str], optional): Path to checkpoint. If given, this ckpt will be used for evaluation. Defaults to None.
        num_jet_samples (int, optional): How many jet samples to generate. Negative values define the amount of times the whole dataset is taken, e.g. -2 would use 2*len(dataset) samples. Defaults to -1.
        fix_seed (bool, optional): Fix seed for data generation to have better reproducibility and comparability between epochs. Defaults to True.
        w_dist_config (Mapping, optional): Configuration for Wasserstein distance calculation. Defaults to {'num_jet_samples': 10_000, 'num_batches': 40}.
        generation_config (Mapping, optional): Configuration for data generation. Defaults to {"batch_size": 256, "ode_solver": "midpoint", "ode_steps": 200}.
        plot_config (Mapping, optional): Configuration for plotting. Defaults to {}.
    """

    def __init__(
        self,
        use_ema: bool = True,
        dataset: str = "test",
        nr_checkpoint_callbacks: int = 1,
        ckpt_path: Optional[str] = None,
        num_jet_samples: int = -1,
        fix_seed: bool = True,
        w_dist_config: Mapping = {
            "num_eval_samples": 10_000,
            "num_batches": 40,
        },
        generation_config: Mapping = {
            "batch_size": 256,
            "ode_solver": "midpoint",
            "ode_steps": 200,
        },
        plot_config: Mapping = {"plot_efps": False},
    ):
        super().__init__()

        apply_mpl_styles()

        self.use_ema = use_ema
        self.dataset = dataset
        self.ckpt_path = ckpt_path
        self.nr_checkpoint_callbacks = nr_checkpoint_callbacks
        self.num_jet_samples = num_jet_samples
        self.fix_seed = fix_seed
        # loggers
        self.comet_logger = None
        self.wandb_logger = None

        # configs
        self.w_dist_config = w_dist_config
        self.generation_config = generation_config
        self.plot_config = plot_config

    def on_test_start(self, trainer, pl_module) -> None:
        log.info(
            "JetNetFinalEvaluationCallback will be used for evaluating the model after training."
        )

        # set number of jet samples if negative
        if self.num_jet_samples < 0:
            self.datasets_multiplier = abs(self.num_jet_samples)
            if self.dataset == "test":
                self.num_jet_samples = len(trainer.datamodule.tensor_test) * abs(
                    self.num_jet_samples
                )
            if self.dataset == "val":
                self.num_jet_samples = len(trainer.datamodule.tensor_val) * abs(
                    self.num_jet_samples
                )
        else:
            self.datasets_multiplier = -1

        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info(f"Evaluating model on {self.dataset} dataset.")

        ckpt = self._get_checkpoint(trainer)

        log.info(f"Loading checkpoint from {ckpt}")
        model = pl_module.load_from_checkpoint(ckpt)

        if self.fix_seed:
            # fix seed for better reproducibility and comparable results
            torch.manual_seed(9999)

        # Get background data for plotting and calculating Wasserstein distances
        if self.dataset == "test":
            background_data = np.array(trainer.datamodule.tensor_test)[: self.num_jet_samples]
            background_mask = np.array(trainer.datamodule.mask_test)[: self.num_jet_samples]
            background_cond = np.array(trainer.datamodule.tensor_conditioning_test)[
                : self.num_jet_samples
            ]
        elif self.dataset == "val":
            background_data = np.array(trainer.datamodule.tensor_val)[: self.num_jet_samples]
            background_mask = np.array(trainer.datamodule.mask_val)[: self.num_jet_samples]
            background_cond = np.array(trainer.datamodule.tensor_conditioning_val)[
                : self.num_jet_samples
            ]

        # maximum number of samples to plot is the number of samples in the dataset
        num_plot_samples = len(background_data)

        if self.datasets_multiplier > 1:
            background_data = np.repeat(background_data, self.datasets_multiplier, axis=0)
            background_mask = np.repeat(background_mask, self.datasets_multiplier, axis=0)
            background_cond = np.repeat(background_cond, self.datasets_multiplier, axis=0)

        # Generate data
        data, generation_time = generate_data(
            model=model,
            num_jet_samples=len(background_data),
            cond=torch.tensor(background_cond),
            variable_set_sizes=trainer.datamodule.hparams.variable_jet_sizes,
            mask=torch.tensor(background_mask),
            normalized_data=trainer.datamodule.hparams.normalize,
            means=trainer.datamodule.means,
            stds=trainer.datamodule.stds,
            **self.generation_config,
        )

        # save generated data
        path = "/".join(ckpt.split("/")[:-2]) + "/"
        file_name = "final_generated_data.npy"
        full_path = path + file_name
        np.save(full_path, data)

        # Wasserstein distances
        w_dists = calculate_all_wasserstein_metrics(background_data, data, **self.w_dist_config)

        # Prepare Data for Plotting
        plot_prep_config = {
            "calculate_efps" if key == "plot_efps" else key: value
            for key, value in self.plot_config.items()
            if key in ["plot_efps", "selected_particles", "selected_multiplicities"]
        }

        (
            jet_data,
            efps_values,
            pt_selected_particles,
            pt_selected_multiplicities,
        ) = prepare_data_for_plotting(np.array([data]), **plot_prep_config)

        (
            jet_data_sim,
            efps_sim,
            pt_selected_particles_sim,
            pt_selected_multiplicities_sim,
        ) = prepare_data_for_plotting(
            [background_data],
            **plot_prep_config,
        )
        jet_data_sim, efps_sim, pt_selected_particles_sim = (
            jet_data_sim[0],
            efps_sim[0],
            pt_selected_particles_sim[0],
        )

        sim_data = np.concatenate([background_data, background_mask], axis=-1)

        # Plotting
        plot_name = "final_plot"
        img_path = "/".join(ckpt.split("/")[:-2]) + "/final_plots.png"
        fig = plot_data(
            particle_data=np.array([data]),
            sim_data=sim_data,
            jet_data_sim=jet_data_sim,
            jet_data=jet_data,
            efps_sim=efps_sim,
            efps_values=efps_values,
            num_samples=num_plot_samples,
            pt_selected_particles=pt_selected_particles,
            pt_selected_multiplicities=pt_selected_multiplicities,
            pt_selected_particles_sim=pt_selected_particles_sim,
            pt_selected_multiplicities_sim=pt_selected_multiplicities_sim,
            save_fig=True,
            save_folder=img_path,
            save_name=plot_name,
            close_fig=True,
            **self.plot_config,
        )

        yaml_path = "/".join(ckpt.split("/")[:-2]) + "/final_eval_metrics.yml"
        log.info(f"Writing final evaluation metrics to {yaml_path}")

        # transform numpy.float64 for better readability in yaml file
        w_dists = {k: float(v) for k, v in w_dists.items()}
        # write to yaml file
        with open(yaml_path, "w") as outfile:
            yaml.dump(w_dists, outfile, default_flow_style=False)

        # rename wasserstein distances for better distinction
        w_dists_final = {}
        for key, value in w_dists.items():
            w_dists_final[key + "_final"] = value

        # log metrics and image to loggers
        img_path = f"{img_path}{plot_name}.png"
        if self.comet_logger is not None:
            self.comet_logger.log_image(img_path, name="A_final_plot")
            self.comet_logger.log_metrics(w_dists_final)
        if self.wandb_logger is not None:
            self.wandb_logger.log({"A_final_plot": wandb.Image(img_path)})
            self.wandb_logger.log(w_dists_final)

    def _get_checkpoint(self, trainer: pl.Trainer) -> None:
        """Get checkpoint path based on the selected checkpoint callback."""
        if self.ckpt_path is None:
            if self.use_ema:
                if (
                    type(trainer.checkpoint_callbacks[self.nr_checkpoint_callbacks])
                    == EMAModelCheckpoint
                ):
                    return trainer.checkpoint_callbacks[
                        self.nr_checkpoint_callbacks
                    ].best_model_path_ema
                else:
                    raise ValueError(
                        "JetNetFinalEvaluationCallback was told to use EMA weights for evaluation but the provided checkpoint callback is not of type EMAModelCheckpoint"
                    )
            else:
                return trainer.checkpoint_callbacks[self.nr_checkpoint_callbacks].best_model_path
        else:
            return self.ckpt_path
