"""Script to generate jets as presented in the paper:

Flow Matching Beyond Kinematics: Generating Jets with Particle-ID
         and Trajectory Displacement Information

      arXiv: https://arxiv.org/abs/2312.00123
      
see also the repository of the paper: https://github.com/uhh-pd-ml/beyond_kinematics
"""
import argparse
import logging
import subprocess  # nosec
import sys
from pathlib import Path

import awkward as ak
import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import vector
from omegaconf import DictConfig, OmegaConf

sys.path.append("../../EPiC-FM")

from src.models.flow_matching_module import SetFlowMatchingLitModule
from src.utils.data_generation import generate_data

vector.register_awkward()
import os
import pickle  # nosec

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_JET_TYPES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
N_JET_TYPES = 10
MODEL_CHECKPOINT_URL = (
    "https://syncandshare.desy.de/index.php/s/ccWPzbj8qr5K9iS/download/epicfm_baseline.ckpt"
)

# fmt: off
parser = argparse.ArgumentParser(description="Script to generate jets.")
parser.add_argument("--types", nargs="+", default=ALL_JET_TYPES, help=f"Selected jet types. Valid types are: {ALL_JET_TYPES}")
parser.add_argument("--n_jets_per_type", type=int, default=100, help="Number of jets per type")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--output_dir", type=str, default="./beyond_kinematics", help="Output directory")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
# fmt: on

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


kde_features_names = {
    "jet_type": [np.linspace(-0.5, 9.5, 21), "Jet type"],
    "jet_nparticles": [np.linspace(-0.5, 128.5, 130), "Number of particles"],
    "jet_pt": [np.linspace(400, 1100, 201), "Jet $p_\\mathrm{T}$"],
    "jet_eta": [np.linspace(-2.5, 2.5, 200), "Jet $\\eta$"],
}

jet_samples_list = []


def main():
    args = parser.parse_args()

    TYPES_TO_GENERATE = args.types
    N_JETS_PER_TYPE = args.n_jets_per_type
    BATCH_SIZE = args.batch_size

    N_GENERATED_JETS = int(len(TYPES_TO_GENERATE) * N_JETS_PER_TYPE)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JET_FEATURES = (
        OUTPUT_DIR / f"beyond_kinematics_cond_features_{N_JETS_PER_TYPE:_}_pertype.h5"
    )
    OUTPUT_PARTICLE_FEATURES = (
        OUTPUT_DIR / f"beyond_kinematics_jets_{N_JETS_PER_TYPE:_}_pertype.h5"
    )
    MODEL_CHECKPOINT = OUTPUT_DIR / "model.ckpt"
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"We will generate {N_GENERATED_JETS:_} jets.")
    print(f"Selected jet types: {TYPES_TO_GENERATE} (will generate {N_JETS_PER_TYPE:_} per type).")

    # ------- DOWNLOAD MODEL --------
    # if not MODEL_CHECKPOINT.exists():
    print(f"Downloading model checkpoint to {MODEL_CHECKPOINT}")
    if not os.path.exists(MODEL_CHECKPOINT):
        subprocess.run(  # nosec
            [
                "wget",
                "-O",
                str(MODEL_CHECKPOINT),
                MODEL_CHECKPOINT_URL,
            ]
        )

    # ------- GENERATE JET FEATURES FROM KDE MODELS --------
    # generate samples from the kde models
    for i in range(N_JET_TYPES):
        # load the model and preprocessing parameters
        with open(f"scripts/models/kde_model_jet_type_{i}.pkl", "rb") as f:  # nosec
            kde_model_eta = pickle.load(f)  # nosec
        with open(f"scripts/models/preprocessing_params_jet_type_{i}.yaml") as f:  # nosec
            preprocessing_params = yaml.load(f, Loader=yaml.FullLoader)  # nosec

        # set seed for sklearn
        np.random.seed(args.seed)
        jet_samples = kde_model_eta.sample(N_GENERATED_JETS)

        # revert the standardization
        jet_samples[:, 0] = (
            jet_samples[:, 0] * preprocessing_params["jet_nparticles"]["scale"]
        ) + preprocessing_params["jet_nparticles"]["shift"]
        jet_samples[:, 1] = (
            jet_samples[:, 1] * preprocessing_params["jet_pt"]["scale"]
        ) + preprocessing_params["jet_pt"]["shift"]
        jet_samples[:, 2] = (
            jet_samples[:, 2] * preprocessing_params["jet_eta"]["scale"]
        ) + preprocessing_params["jet_eta"]["shift"]

        # --- POSTPROCESSING ---
        # round number of particles
        jet_samples[:, 0] = np.round(jet_samples[:, 0]).astype(int)
        # clip the number of particles such that each jet has at least 1 particle and max 128
        jet_samples[:, 0] = np.clip(jet_samples[:, 0], 1, 128)
        jet_samples[:, 1] = np.clip(
            jet_samples[:, 1],
            preprocessing_params["jet_pt"]["min"],
            preprocessing_params["jet_pt"]["max"],
        )
        jet_samples[:, 2] = np.clip(
            jet_samples[:, 2],
            preprocessing_params["jet_eta"]["min"],
            preprocessing_params["jet_eta"]["max"],
        )

        # add `jet_type` variable between 0 and 9 --> will be the first feature
        jet_samples = np.concatenate([i * np.ones((len(jet_samples), 1)), jet_samples], axis=1)
        jet_samples_list.append(jet_samples)

    jet_samples = np.concatenate(jet_samples_list, axis=0)
    # shuffle the samples
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(jet_samples))
    jet_samples = jet_samples[permutation]

    max_particles = 128
    # generate the masks from the particle multiplicity
    print("Generating masks")
    mask_gen = np.zeros((len(jet_samples), max_particles))
    for i in tqdm(range(len(jet_samples))):
        mask_gen[i, : int(jet_samples[i, 1])] = 1

    # save the generated data
    jet_features_gen = jet_samples
    if os.path.exists(OUTPUT_JET_FEATURES):
        print(f"File {OUTPUT_JET_FEATURES} already exists --> delete it if you want to overwrite.")
    else:
        with h5py.File(OUTPUT_JET_FEATURES, "w") as h5file:
            print(f"Writing to {OUTPUT_JET_FEATURES}")
            h5file.create_dataset("part_mask", data=mask_gen[..., np.newaxis])
            h5file.create_dataset("jet_features", data=jet_features_gen)
            h5file["jet_features"].attrs.create(
                "names_jet_features", list(kde_features_names.keys())
            )

    # ------- GENERATE CONSTITUENT FEATURES FROM EPIC-FM MODEL --------
    print("Loading model")
    model = SetFlowMatchingLitModule.load_from_checkpoint(MODEL_CHECKPOINT)
    # load means and stds from yaml
    with open("scripts/models/means_stds.yaml") as f:
        means_stds_dict = OmegaConf.load(f)
        used_means = means_stds_dict["means"]
        used_stds = means_stds_dict["stds"]
        names_part_features = means_stds_dict["names"]

    print("Loading conditioning data")
    # load conditioning data
    with h5py.File(OUTPUT_JET_FEATURES, "r") as f:
        part_mask = f["part_mask"][:]
        cond_features = f["jet_features"][:]
        names_cond_features = list(f["jet_features"].attrs["names_jet_features"][:])

    # replace jet_type with one-hot encoding
    jet_type = cond_features[:, 0]
    jet_type_onehot = torch.nn.functional.one_hot(torch.tensor(jet_type, dtype=torch.int64))
    cond_gen = np.concatenate([jet_type_onehot, cond_features[:, 2:]], axis=1)
    # update the names
    names_cond_features = [f"type_{jet_type}" for jet_type in ALL_JET_TYPES] + names_cond_features[
        2:
    ]

    # select only the jet types we want to generate
    mask_selected_types = np.isin(jet_type, [ALL_JET_TYPES.index(jt) for jt in TYPES_TO_GENERATE])

    mask_gen = part_mask
    cond_gen = cond_gen[mask_selected_types][:N_GENERATED_JETS]
    mask_gen = mask_gen[mask_selected_types][:N_GENERATED_JETS]

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_gen, generation_time = generate_data(
        model=model,
        batch_size=BATCH_SIZE,
        num_jet_samples=len(mask_gen),
        cond=torch.tensor(cond_gen, dtype=torch.float32),
        variable_set_sizes=True,
        mask=torch.tensor(mask_gen, dtype=torch.float32),
        normalized_data=True,
        means=used_means,
        normalize_sigma=5,
        stds=used_stds,
        device=device,
        ode_steps=100,
    )

    # save the generated data
    if os.path.exists(OUTPUT_PARTICLE_FEATURES):
        print(
            f"File {OUTPUT_PARTICLE_FEATURES} already exists --> delete it if you want to"
            " overwrite."
        )
    else:
        with h5py.File(OUTPUT_PARTICLE_FEATURES, "w") as h5file:
            print(f"Writing to {OUTPUT_PARTICLE_FEATURES}")
            h5file.create_dataset("particle_mask", data=mask_gen)
            h5file.create_dataset("particle_features", data=data_gen)
            h5file["particle_features"].attrs.create(
                "names_particle_features",
                names_part_features,
                dtype=h5py.special_dtype(vlen=str),
            )
            h5file.create_dataset("jet_features", data=cond_gen)
            h5file["jet_features"].attrs.create(
                "names_jet_features",
                names_cond_features,
                dtype=h5py.special_dtype(vlen=str),
            )


if __name__ == "__main__":
    main()
