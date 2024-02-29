"""script to load the JetClass dataset into numpy arrays and write to npz files. See also the repository of the paper: https://github.com/uhh-pd-ml/beyond_kinematics"""

import glob
import logging
import os
import subprocess  # nosec

import hydra
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from particle_fm.preprocessing.utils import read_file

logger = logging.getLogger("prepare_dataset")


def get_git_revision_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])  # nosec
        .decode("ascii")  # nosec
        .strip()  # nosec
    )


@hydra.main(version_base=None, config_path="../configs/preprocessing", config_name="data.yaml")
def main(cfg: DictConfig):
    if os.path.exists(cfg.output_dir):
        logger.info(f"Output directory {cfg.output_dir} already exists.")
        files_already_there = glob.glob(f"{cfg.output_dir}/*")
        if len(files_already_there) > 0:
            logger.info("Files/folders already there:")
            for filename in files_already_there:
                logger.info(filename)
            logger.error(
                f"Found {len(files_already_there)} npz files in {cfg.output_dir} --> Exiting"
            )
            logger.info(
                "Choose a different output directory or delete the files using the command below:"
            )
            logger.info(f"command: rm -rf {cfg.output_dir}")
            raise RuntimeError("Output directory already exists.")

    # get git hash for reproducibility
    cfg.git_hash = get_git_revision_hash()

    if cfg.get("num_jets_per_file", False):
        if not isinstance(cfg.num_jets_per_file, int):
            raise ValueError("num_jets_per_file must be an integer if specified.")

    # copy the config file to the output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # convert the "n_files_per_process" int to a list of ints, if it's not already
    files_indices = {}
    for train_val_test_folder in ["train_100M", "val_5M", "test_20M"]:
        if isinstance(cfg.n_files_per_process[train_val_test_folder], int):
            files_indices[train_val_test_folder] = list(
                range(cfg.n_files_per_process[train_val_test_folder])
            )
        elif isinstance(
            cfg.n_files_per_process[train_val_test_folder],
            (list, omegaconf.listconfig.ListConfig),
        ):
            files_indices[train_val_test_folder] = cfg.n_files_per_process[train_val_test_folder]
        else:
            raise ValueError(
                "n_files_per_process must be an int or a list of ints if specified."
                f"n_files_per_process for {train_val_test_folder} is "
                f"{cfg.n_files_per_process[train_val_test_folder]} "
                f"type: {type(cfg.n_files_per_process[train_val_test_folder])}"
            )

    ROOT_FILES = {
        train_val_test_folder: {
            process: sorted(
                list(glob.glob(f"{cfg.filepath_base}/{train_val_test_folder}/{process}_*.root"))
            )
            for process in cfg.processes
        }
        for train_val_test_folder in ["train_100M", "val_5M", "test_20M"]
    }

    # only keep a subset of the files specified by files_indices
    for train_val_test_folder, root_files_dict in ROOT_FILES.items():
        for process, root_files in root_files_dict.items():
            ROOT_FILES[train_val_test_folder][process] = [
                root_files[i] for i in files_indices[train_val_test_folder]
            ]

    cfg.root_files = ROOT_FILES
    with open(f"{cfg.output_dir}/data.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    for train_val_test_folder, root_files_dict in ROOT_FILES.items():
        logger.info(f"Processing {train_val_test_folder} files.")
        for jet_type, root_files in root_files_dict.items():
            logger.info(f"Processing {jet_type} files. Looping over {len(root_files)} files.")
            logger.info(root_files)
            x_particles_list = []
            x_jets_list = []
            y_list = []

            for root_file in tqdm(root_files):
                x_particles, x_jets, y = read_file(
                    root_file,
                    particle_features=cfg.features.names_part_features,
                    jet_features=cfg.features.names_jet_features,
                )
                x_particles_list.append(x_particles)
                x_jets_list.append(x_jets)
                y_list.append(y)

            x_particles = np.swapaxes(np.concatenate(x_particles_list), 1, 2)
            x_jets = np.concatenate(x_jets_list)
            y = np.concatenate(y_list)

            # only keep a subset of the jets if specified (mostly for debugging)
            if cfg.get("num_jets_per_file", False):
                if cfg.num_jets_per_file < len(x_particles):
                    logger.info(f"Only keeping {cfg.num_jets_per_file} jets.")
                    x_particles = x_particles[: cfg.num_jets_per_file]
                    x_jets = x_jets[: cfg.num_jets_per_file]
                    y = y[: cfg.num_jets_per_file]

            names_jet_features = cfg.features.names_jet_features
            names_part_features = cfg.features.names_part_features

            # get the mask
            index_part_pt = cfg.features.names_part_features.index("part_pt")
            part_mask = x_particles[:, :, index_part_pt] != 0

            # jet feature modifications
            if cfg.get("include_jet_type", True):
                # add jet-type to jet features
                jet_type_array = np.argmax(y, axis=1)
                x_jets = np.concatenate([jet_type_array[:, None], x_jets], axis=1)
                names_jet_features = ["jet_type"] + names_jet_features

            # particle feature modifications
            if cfg.get("include_ptrel", False):
                # add ptrel to jet features
                index_part_pt = names_part_features.index("part_pt")
                index_jet_pt = names_jet_features.index("jet_pt")
                ptrel = x_particles[:, :, index_part_pt] / np.expand_dims(
                    x_jets[:, index_jet_pt], axis=1
                )
                x_particles = np.concatenate([ptrel[:, :, None], x_particles], axis=2)
                names_part_features = ["part_ptrel"] + names_part_features

            if cfg.get("include_etarel", False):
                # add etarel to jet features
                index_part_eta = names_part_features.index("part_eta")
                index_jet_eta = names_jet_features.index("jet_eta")
                etarel = (
                    x_particles[:, :, index_part_eta]
                    - np.expand_dims(x_jets[:, index_jet_eta], axis=1) * part_mask
                )
                x_particles = np.concatenate([etarel[:, :, None], x_particles], axis=2)
                names_part_features = ["part_etarel"] + names_part_features

                # remove particles with large etadiff
                if cfg.get("remove_etarel_tails", False):
                    index_part_etarel = names_part_features.index("part_etarel")
                    etarel = x_particles[:, :, index_part_etarel]
                    mask_etarel_larger_one = np.abs(etarel) > 1
                    # Also set mask to zero for these particles
                    x_particles[mask_etarel_larger_one, :] = 0
                    # Set mask to zero for these particles
                    part_mask[mask_etarel_larger_one] = 0

                    # add new variable to jet features which states the number of
                    # particles after this removal
                    jet_nparticles_after_etarel_cut = np.sum(part_mask, axis=1)
                    x_jets = np.concatenate(
                        [
                            x_jets,
                            jet_nparticles_after_etarel_cut[:, None],
                        ],
                        axis=1,
                    )
                    names_jet_features += ["jet_nparticles_after_etarel_cut"]

            if cfg.get("include_energyrel", False):
                # add energyrel to jet features
                index_part_energy = names_part_features.index("part_energy")
                index_jet_energy = names_jet_features.index("jet_energy")
                energyrel = x_particles[:, :, index_part_energy] / np.expand_dims(
                    x_jets[:, index_jet_energy], axis=1
                )
                x_particles = np.concatenate([energyrel[:, :, None], x_particles], axis=2)
                names_part_features = ["part_energyrel"] + names_part_features

            logger.info(f"Particle features: {names_part_features}")
            logger.info(f"Shape of part_features: {x_particles.shape}")
            logger.info(f"Shape of part_mask: {part_mask.shape}")
            logger.info(f"Jet features: {names_jet_features}")
            logger.info(f"Shape of jet_features: {x_jets.shape}")
            logger.info(f"Labels features: {cfg.features.names_labels}")
            logger.info(f"Shape of labels: {y.shape}")

            filename = f"jetclass_{jet_type}_{len(x_particles):_}.npz"
            save_dir = f"{cfg.output_dir}/{train_val_test_folder}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{filename}"
            logger.info(f"Saving {jet_type} data to {save_path}\n")
            np.savez_compressed(
                save_path,
                part_features=x_particles,
                part_mask=part_mask,
                jet_features=x_jets,
                labels=y,
                # also save the names of the features and the filenames of the
                # root files for later reference
                names_part_features=names_part_features,
                names_jet_features=names_jet_features,
                names_labels=cfg.features.names_labels,
                root_files=root_files,
            )


if __name__ == "__main__":
    main()
