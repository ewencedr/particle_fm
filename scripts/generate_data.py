import os
import sys

sys.path.append("../")
import argparse

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.components import calculate_all_wasserstein_metrics
from src.utils.data_generation import generate_data
from src.utils.plotting import apply_mpl_styles


def main(params):
    load_dotenv()
    # set env variable DATA_DIR again because of hydra
    os.environ["DATA_DIR"] = os.environ.get("DATA_DIR")

    experiment = params.experiment
    # load everything from experiment config
    with hydra.initialize(version_base=None, config_path="../configs/"):
        cfg = hydra.compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
        # print(OmegaConf.to_yaml(cfg))

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    datamodule.setup()
    test_data = np.array(datamodule.tensor_test)
    test_mask = np.array(datamodule.mask_test)
    val_data = np.array(datamodule.tensor_val)
    val_mask = np.array(datamodule.mask_val)
    means = np.array(datamodule.means)
    stds = np.array(datamodule.stds)

    ckpt = params.checkpoint
    model = model.load_from_checkpoint(ckpt)

    apply_mpl_styles()

    # generate data
    mask = np.copy(val_mask)
    for _ in range(params.n_samples // len(val_mask)):
        mask = np.concatenate([mask, val_mask])
    data, generation_time = generate_data(
        model=model,
        num_jet_samples=params.n_samples,
        batch_size=params.batch_size,
        variable_set_sizes=True,
        mask=torch.tensor(mask),
        normalized_data=True,
        means=means,
        stds=stds,
        ode_solver=params.ode_solver,
        ode_steps=params.ode_steps,
    )
    particle_data = data
    mask_data = np.ma.masked_where(
        particle_data[:, :, 0] == 0,
        particle_data[:, :, 0],
    )
    mask_data = np.expand_dims(mask_data, axis=-1)
    w_dists_1b = calculate_all_wasserstein_metrics(
        test_data[..., :3],
        particle_data,
        test_mask,
        mask_data,
        num_eval_samples=len(particle_data),
        num_batches=1,
        calculate_efps=True,
    )
    w_dists = calculate_all_wasserstein_metrics(
        test_data[..., :3],
        particle_data,
        test_mask,
        mask_data,
        num_eval_samples=int(len(particle_data) / 5),
        num_batches=5,
        calculate_efps=True,
    )
    print(f"Generation time: {generation_time} s")
    print(f"Number of generated samples: {len(particle_data)}")
    print("Wasserstein distances 1 batch:")
    print(w_dists_1b)
    print("Wasserstein distances 5 batches:")
    print(w_dists)


if __name__ == "__main__":
    # ARGUMENTS
    # define parser
    parser = argparse.ArgumentParser(description="Generating Samples from a trained model.")
    parser.add_argument("--n_samples", "-n", default=125000, help="samples to generate", type=int)
    parser.add_argument(
        "--batch_size", "-bs", default=500, help="batch size for sampling", type=int
    )
    parser.add_argument(
        "--experiment", "-exp", default="fm_tops.yaml", help="experiment config to load", type=str
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        default="/beegfs/desy/user/ewencedr/deep-learning/logs/150 epoch10000 oldsetup/runs/2023-05-22_18-00-11/checkpoints/epoch_1216_loss_2.60804.ckpt",
        help="checkpoint to sample from",
        type=str,
    )
    parser.add_argument(
        "--ode_solver",
        "-ode",
        default="dopri5_zuko",
        help="ode_solver for sampling",
        type=str,
    )
    parser.add_argument(
        "--ode_steps",
        "-steps",
        default=100,
        help="steps for ode_solver",
        type=str,
    )
    params = parser.parse_args()
    main(params)
