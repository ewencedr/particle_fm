import os
import sys

sys.path.append("../")
import argparse

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.components import calculate_all_wasserstein_metrics
from src.utils.plotting import apply_mpl_styles, create_and_plot_data


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

    ode_solver_adaptive = [
        "dopri5_zuko",
        "dopri5",
        "tsit5",
        "rk4",
        "euler",
        "midpoint",
        "ieuler",
        "alf",
    ]
    steps = [20, 40, 60, 80, 100]

    solver_final = []
    steps_final = []
    generation_times_adaptive = []
    w1m_1b_adaptive = []
    w1p_1b_adaptive = []
    w1efp_1b_adaptive = []
    w1m_adaptive = []
    w1p_adaptive = []
    w1efp_adaptive = []
    w1m_std_adaptive = []
    w1p_std_adaptive = []
    w1efp_std_adaptive = []

    for solver in ode_solver_adaptive:
        for step in steps:
            print(f"Solver: {solver}")
            print(f"Step: {step}")
            fig, data, generation_times = create_and_plot_data(
                np.array(val_data),
                [model],
                cond=None,
                save_name="fm_tops_nb",
                labels=["FM"],
                mask=val_mask,
                num_jet_samples=params.n_samples,
                batch_size=params.batch_size,
                variable_set_sizes=True,
                normalized_data=[True, True],
                means=means,
                stds=stds,
                plottype="sim_data",
                plot_jet_features=True,
                plot_w_dists=False,
                plot_selected_multiplicities=False,
                selected_multiplicities=[1, 3, 5, 10, 20, 30],
                ode_solver=solver,
                ode_steps=step,
                save_fig=False,
            )
            print(f"Generation time: {generation_times}")
            particle_data = data[0]
            mask_data = np.ma.masked_where(
                particle_data[:, :, 0] == 0,
                particle_data[:, :, 0],
            )
            mask_data = np.expand_dims(mask_data, axis=-1)
            w_dists_1b_adaptive_dict = calculate_all_wasserstein_metrics(
                test_data[..., :3],
                particle_data,
                test_mask,
                mask_data,
                num_eval_samples=len(particle_data),
                num_batches=1,
                calculate_efps=True,
            )
            w_dists_adaptive_dict = calculate_all_wasserstein_metrics(
                test_data[..., :3],
                particle_data,
                test_mask,
                mask_data,
                num_eval_samples=int(len(particle_data) / 5),
                num_batches=5,
                calculate_efps=True,
            )
            solver_final.append(solver)
            steps_final.append(step)
            generation_times_adaptive.append(float(generation_times.squeeze()))
            w1m_1b_adaptive.append(w_dists_1b_adaptive_dict["w1m_mean"])
            w1p_1b_adaptive.append(w_dists_1b_adaptive_dict["w1p_mean"])
            w1efp_1b_adaptive.append(w_dists_1b_adaptive_dict["w1efp_mean"])
            w1m_adaptive.append(w_dists_adaptive_dict["w1m_mean"])
            w1p_adaptive.append(w_dists_adaptive_dict["w1p_mean"])
            w1efp_adaptive.append(w_dists_adaptive_dict["w1efp_mean"])
            w1m_std_adaptive.append(w_dists_adaptive_dict["w1m_std"])
            w1p_std_adaptive.append(w_dists_adaptive_dict["w1p_std"])
            w1efp_std_adaptive.append(w_dists_adaptive_dict["w1efp_std"])

    # print(f"Adaptive solvers: {(solver_final)}")
    # print(f"Solver: {(solver_final)}")
    # print(f"Steps: {steps_final}")
    # print(
    #     f"Generated Jets: {([len(particle_data) for _ in range(len(ode_solver_adaptive)*len(steps))])}"
    # )
    # print(f"Time: {(generation_times_adaptive)}")
    # print(f"Time per Jet: {(np.array(generation_times_adaptive) / len(particle_data))}")
    # print(f"Time per Jet: {len(np.array(generation_times_adaptive) / len(particle_data))}")
    # print(f"w1m_1b: {(w1m_1b_adaptive)}")
    # print(f"w1p_1b: {(w1p_1b_adaptive)}")
    # print(f"w1efp_1b: {(w1efp_1b_adaptive)}")
    # print(f"w1m: {(w1m_adaptive)}")
    # print(f"w1p: {(w1p_adaptive)}")
    # print(f"w1efp: {(w1efp_adaptive)}")
    # print(f"w1m_std: {(w1m_std_adaptive)}")
    # print(f"w1p_std: {(w1p_std_adaptive)}")
    # print(f"w1efp_std: {(w1efp_std_adaptive)}")
    dict_adaptive = {
        "Solver": solver_final,
        "Steps": steps_final,
        "Generated Jets": [
            len(particle_data) for _ in range(len(ode_solver_adaptive) * len(steps))
        ],
        "Time": generation_times_adaptive,
        "Time per Jet": np.array(generation_times_adaptive) / len(particle_data),
        "w1m_1b": w1m_1b_adaptive,
        "w1p_1b": w1p_1b_adaptive,
        "w1efp_1b": w1efp_1b_adaptive,
        "w1m": w1m_adaptive,
        "w1p": w1p_adaptive,
        "w1efp": w1efp_adaptive,
        "w1m_std": w1m_std_adaptive,
        "w1p_std": w1p_std_adaptive,
        "w1efp_std": w1efp_std_adaptive,
    }
    df = pd.DataFrame(data=dict_adaptive)
    df.to_csv(f"{params.save_folder}/ode_solver.csv")


if __name__ == "__main__":
    # ARGUMENTS
    # define parser
    parser = argparse.ArgumentParser(description="training of flows")
    parser.add_argument(
        "--n_samples", "-n", default=10000, help="samples to generate with each solver", type=int
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
        default="/beegfs/desy/user/ewencedr/deep-learning/logs/150 epoch10000 oldsetup/runs/2023-05-22_18-00-11/checkpoints/epoch_1216_loss_2.60804.ckpt",
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
