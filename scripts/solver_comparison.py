import os
import sys

sys.path.append("../")
import argparse

import hydra
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.components import calculate_all_wasserstein_metrics
from src.utils.data_generation import generate_data
from src.utils.plotting import apply_mpl_styles, create_and_plot_data


def main(params):
    load_dotenv()
    # set env variable DATA_DIR again because of hydra
    os.environ["DATA_DIR"] = os.environ.get("DATA_DIR")

    experiment = params.experiment
    # load everything from experiment config
    with hydra.initialize(version_base=None, config_path="../configs/"):
        cfg = hydra.compose(
            config_name="train.yaml",
            overrides=[f"experiment={experiment}", "model.num_particles=150"],
        )
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

    ode_solver_adaptive = [
        "rk4",
        "euler",
        "midpoint",
        # "ieuler",
        # "alf",
        # "dopri5",
        # "dopri5_zuko",
        # "tsit5",
    ]
    steps = [20, 40, 60, 80, 100, 200]
    # steps = [100, 100, 100, 100, 100]

    solver_final = []
    steps_final = []
    generation_times_adaptive = []
    w1m_adaptive = []
    w1p_adaptive = []
    w1efp_adaptive = []
    w1m_std_adaptive = []
    w1p_std_adaptive = []
    w1efp_std_adaptive = []

    for solver in ode_solver_adaptive:
        for step in steps:
            if solver == "midpoint":
                step = step // 2
            elif solver == "rk4":
                step = step // 4

            print(f"Solver: {solver}")
            print(f"Step: {step}")
            big_mask = np.repeat(test_mask, 5, axis=0)

            data, generation_time = generate_data(
                model,
                5 * len(test_mask),
                batch_size=256,
                variable_set_sizes=True,
                mask=torch.tensor(big_mask),
                normalized_data=True,
                means=means,
                stds=stds,
                ode_solver=solver,
                ode_steps=step,
            )
            print(f"Generation time: {generation_time}")
            w_dists_big = calculate_all_wasserstein_metrics(
                test_data[..., :3],
                data,
                None,
                None,
                num_eval_samples=len(test_data),
                num_batches=5,
                calculate_efps=True,
                use_masks=False,
            )

            solver_final.append(solver)
            steps_final.append(step)
            generation_times_adaptive.append(generation_time)
            w1m_adaptive.append(w_dists_big["w1m_mean"])
            w1p_adaptive.append(w_dists_big["w1p_mean"])
            w1efp_adaptive.append(w_dists_big["w1efp_mean"])
            w1m_std_adaptive.append(w_dists_big["w1m_std"])
            w1p_std_adaptive.append(w_dists_big["w1p_std"])
            w1efp_std_adaptive.append(w_dists_big["w1efp_std"])

    dict_adaptive = {
        "Solver": solver_final,
        "Steps": steps_final,
        "Generated Jets": [len(data) for _ in range(len(ode_solver_adaptive) * len(steps))],
        "Time": generation_times_adaptive,
        "Time per Jet": np.array(generation_times_adaptive) / len(data),
        "w1m": w1m_adaptive,
        "w1p": w1p_adaptive,
        "w1efp": w1efp_adaptive,
        "w1m_std": w1m_std_adaptive,
        "w1p_std": w1p_std_adaptive,
        "w1efp_std": w1efp_std_adaptive,
    }
    df = pd.DataFrame(data=dict_adaptive)
    df.to_csv(f"{params.save_folder}/ode_solver-nfe2.csv")


if __name__ == "__main__":
    # ARGUMENTS
    # define parser
    parser = argparse.ArgumentParser(description="training of flows")
    parser.add_argument(
        "--n_samples", "-n", default=-5, help="samples to generate with each solver", type=int
    )
    parser.add_argument(
        "--batch_size", "-bs", default=500, help="batch size for sampling", type=int
    )
    parser.add_argument(
        "--experiment", "-exp", default="fm_tops.yaml", help="experiment config to load", type=str
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        default="/beegfs/desy/user/ewencedr/deep-learning/logs/150 warmuplr midpoint/runs/2023-06-02_21-46-33/checkpoints/epoch_9761_w1m_0.00064797.ckpt",
        help="checkpoint to load",
        type=str,
    )
    parser.add_argument(
        "--save_folder",
        "-sf",
        default="/beegfs/desy/user/ewencedr/deep-learning/ode_solver_comparison",
        help="folder to save csv files to",
        type=str,
    )
    params = parser.parse_args()
    main(params)
