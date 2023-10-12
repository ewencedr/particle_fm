"""PyTorch Lightning DataModule for JetClass dataset."""
import os
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.pylogger import get_pylogger

from .components import one_hot_encode

pylogger = get_pylogger("JetClassDataModule")


def get_feat_index(names_array: np.array, name: str):
    """Helper function that returns the index of the features name in the jet_data array.

    Args:
        names_array (np.array): Array of feature names.
        name (str): Name of the feature to find.
    """
    names_list = list(names_array)
    if name not in names_list:
        raise Exception(
            f"Feature {name} not found in names array. Available features: {names_list}"
        )
    return names_list.index(name)


class JetClassDataModule(LightningDataModule):
    """LightningDataModule for JetClass dataset. If no conditioning is used, the conditioning
    tensor will be a tensor of zeros.

    Args:
        val_fraction (float, optional): Fraction of data to use for validation.
            Between 0 and 1. Defaults to 0.15.
        test_fraction (float, optional): Fraction of data to use for testing.
            Between 0 and 1. Defaults to 0.15.
        batch_size (int, optional): Batch size. Defaults to 256.
        num_workers (int, optional): Number of workers for dataloader. Defaults to 32.
        pin_memory (bool, optional): Pin memory for dataloader. Defaults to False.
        drop_last (bool, optional): Drop last batch for train and val dataloader.
            Defaults to False.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str,
        filename_dict: dict,
        additional_part_features: list = None,
        used_jet_types: list = None,
        number_of_used_jets: int = None,
        number_of_used_jets_val: int = None,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 256,
        num_workers: int = 32,
        pin_memory: bool = False,
        drop_last: bool = False,
        verbose: bool = True,
        variable_jet_sizes: bool = True,
        conditioning_pt: bool = True,
        conditioning_energy: bool = True,
        conditioning_eta: bool = True,
        conditioning_mass: bool = True,
        conditioning_num_particles: bool = True,
        conditioning_jet_type: bool = True,
        conditioning_jet_type_all: bool = False,
        num_particles: int = 128,
        # preprocessing
        normalize: bool = True,
        normalize_sigma: int = 5,
        loss_per_jettype: bool = False,
        conditioning_gen_filename: str = None,
        # spectator_jet_features: list = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.means: Optional[torch.Tensor] = None
        self.stds: Optional[torch.Tensor] = None
        self.cond_means: Optional[torch.Tensor] = None
        self.cond_stds: Optional[torch.Tensor] = None
        self.tensor_test: Optional[torch.Tensor] = None
        self.mask_test: Optional[torch.Tensor] = None
        self.tensor_val: Optional[torch.Tensor] = None
        self.mask_val: Optional[torch.Tensor] = None
        self.tensor_train: Optional[torch.Tensor] = None
        self.mask_train: Optional[torch.Tensor] = None
        self.tensor_conditioning_train: Optional[torch.Tensor] = None
        self.tensor_conditioning_val: Optional[torch.Tensor] = None
        self.tensor_conditioning_test: Optional[torch.Tensor] = None
        self.tensor_spectator_jet_train: Optional[torch.Tensor] = None
        self.tensor_spectator_jet_val: Optional[torch.Tensor] = None
        self.tensor_spectator_jet_test: Optional[torch.Tensor] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    @property
    def num_cond_features(self):
        return sum(
            [
                self.hparams.conditioning_pt,
                self.hparams.conditioning_energy,
                self.hparams.conditioning_eta,
                self.hparams.conditioning_mass,
                self.hparams.conditioning_num_particles,
                self.hparams.conditioning_jet_type,
            ]
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        if (
            "train" not in self.hparams.filename_dict
            or "val" not in self.hparams.filename_dict
            or "test" not in self.hparams.filename_dict
        ):
            raise ValueError("The filename_dict must contain the keys 'train', 'val' and 'test'.")

        if not self.data_train and not self.data_val and not self.data_test:
            arrays_dict = {}
            names_dict = {}

            for split, filename in self.hparams.filename_dict.items():
                pylogger.info(f"Loading {split} data from {filename}")
                if not os.path.isfile(filename):
                    raise FileNotFoundError(f"File {filename} does not exist.")

                with h5py.File(filename, "r") as f:
                    arrays_dict[split] = {key: np.array(f[key]) for key in f.keys()}
                    names_dict[split] = {
                        key: f[key].attrs[f"names_{key}"] for key in f.keys() if "mask" not in key
                    }

            names_particle_features = names_dict["train"]["part_features"]
            names_jet_features = names_dict["train"]["jet_features"]
            names_labels = names_dict["train"]["labels"]
            # check if the particle features are in the correct order
            index_part_etarel = get_feat_index(names_particle_features, "part_etarel")
            index_part_dphi = get_feat_index(names_particle_features, "part_dphi")
            index_part_ptrel = get_feat_index(names_particle_features, "part_ptrel")

            # NOTE: everything below here assumes that the particle features
            # array after preprocessing stores the features in the order
            # [eta_rel, phi_rel, pt_rel, <additional features>]
            indices_etaphiptrel = [index_part_etarel, index_part_dphi, index_part_ptrel]

            if self.hparams.additional_part_features is None:
                self.hparams.additional_part_features = ["part_etarel", "part_dphi", "part_ptrel"]
                pylogger.info(
                    "No `additional_part_features` specified. Using default:"
                    f" {self.hparams.additional_part_features}"
                )
                indices_part_features = indices_etaphiptrel
            else:
                pylogger.info(
                    f"Using `additional_part_features`: {self.hparams.additional_part_features}"
                )
                indices_additional_part_features = [
                    get_feat_index(names_particle_features, feat)
                    for feat in self.hparams.additional_part_features
                ]
                indices_part_features = indices_etaphiptrel + indices_additional_part_features

            names_particle_features = names_particle_features[indices_part_features]

            np.random.seed(332211)
            permutation_train = np.random.permutation(len(arrays_dict["train"]["jet_features"]))
            dataset_train = arrays_dict["train"]["part_features"][:, :, indices_part_features][
                permutation_train
            ]
            mask_train = arrays_dict["train"]["part_mask"][..., np.newaxis][permutation_train]
            jet_features_train = arrays_dict["train"]["jet_features"][permutation_train]
            labels_train = arrays_dict["train"]["labels"][permutation_train]
            part_stds_train = arrays_dict["train"]["part_stds"][indices_part_features]
            part_means_train = arrays_dict["train"]["part_means"][indices_part_features]

            np.random.seed(332211)
            permutation_val = np.random.permutation(len(arrays_dict["val"]["jet_features"]))
            dataset_val = arrays_dict["val"]["part_features"][:, :, indices_part_features][
                permutation_val
            ]
            mask_val = arrays_dict["val"]["part_mask"][..., np.newaxis][permutation_val]
            jet_features_val = arrays_dict["val"]["jet_features"][permutation_val]
            labels_val = arrays_dict["val"]["labels"][permutation_val]

            np.random.seed(332211)
            permutation_test = np.random.permutation(len(arrays_dict["test"]["jet_features"]))
            dataset_test = arrays_dict["test"]["part_features"][:, :, indices_part_features][
                permutation_test
            ]
            mask_test = arrays_dict["test"]["part_mask"][..., np.newaxis][permutation_test]
            jet_features_test = arrays_dict["test"]["jet_features"][permutation_test]
            labels_test = arrays_dict["test"]["labels"][permutation_test]

            # Select only the jet types that are used
            jet_types_mapping = {label.split("_")[-1]: i for i, label in enumerate(names_labels)}

            if self.hparams.used_jet_types is None:
                self.hparams.used_jet_types = list(jet_types_mapping.keys())
            else:
                # check if all used jet types are in the mapping
                for jet_type in self.hparams.used_jet_types:
                    if jet_type not in jet_types_mapping.keys():
                        raise ValueError(
                            f"Jet type {jet_type} not in jet_types_mapping."
                            f"Please choose from {list(jet_types_mapping.keys())}."
                        )
            pylogger.info(f"Using jet types: {self.hparams.used_jet_types}")
            # select only the jets of the used jet types
            # fmt: off
            index_jet_type = get_feat_index(names_jet_features, "jet_type")
            used_jet_types_values = [jet_types_mapping[jet_type] for jet_type in self.hparams.used_jet_types]
            jet_types_mask_train = np.isin(jet_features_train[:, index_jet_type], used_jet_types_values)  # noqa: E501
            jet_types_mask_val = np.isin(jet_features_val[:, index_jet_type], used_jet_types_values)  # noqa: E501
            jet_types_mask_test = np.isin(jet_features_test[:, index_jet_type], used_jet_types_values)  # noqa: E501
            # fmt: on

            dataset_train = dataset_train[jet_types_mask_train]
            mask_train = mask_train[jet_types_mask_train]
            jet_features_train = jet_features_train[jet_types_mask_train]
            labels_train = labels_train[jet_types_mask_train]

            dataset_val = dataset_val[jet_types_mask_val]
            mask_val = mask_val[jet_types_mask_val]
            jet_features_val = jet_features_val[jet_types_mask_val]
            labels_val = labels_val[jet_types_mask_val]

            dataset_test = dataset_test[jet_types_mask_test]
            mask_test = mask_test[jet_types_mask_test]
            jet_features_test = jet_features_test[jet_types_mask_test]
            labels_test = labels_test[jet_types_mask_test]

            if self.hparams.number_of_used_jets is not None:
                dataset_train = dataset_train[: self.hparams.number_of_used_jets]
                dataset_test = dataset_test[: self.hparams.number_of_used_jets]
                mask_train = mask_train[: self.hparams.number_of_used_jets]
                mask_test = mask_test[: self.hparams.number_of_used_jets]
                jet_features_train = jet_features_train[: self.hparams.number_of_used_jets]
                jet_features_test = jet_features_test[: self.hparams.number_of_used_jets]
                labels_train = labels_train[: self.hparams.number_of_used_jets]
                labels_test = labels_test[: self.hparams.number_of_used_jets]

            if self.hparams.number_of_used_jets_val is not None:
                dataset_val = dataset_val[: self.hparams.number_of_used_jets_val]
                mask_val = mask_val[: self.hparams.number_of_used_jets_val]
                jet_features_val = jet_features_val[: self.hparams.number_of_used_jets_val]
                labels_val = labels_val[: self.hparams.number_of_used_jets_val]

            if self.num_cond_features == 0:
                self.tensor_conditioning_train = torch.zeros(len(dataset_train))
                self.tensor_conditioning_val = torch.zeros(len(dataset_val))
                self.tensor_conditioning_test = torch.zeros(len(dataset_test))
                self.names_conditioning = None
            else:
                conditioning_train, self.names_conditioning = self._handle_conditioning(
                    jet_features_train, names_jet_features, names_labels
                )
                conditioning_val, _ = self._handle_conditioning(
                    jet_features_val, names_jet_features, names_labels
                )
                conditioning_test, _ = self._handle_conditioning(
                    jet_features_test, names_jet_features, names_labels
                )
                # fmt: off
                self.tensor_conditioning_train = torch.tensor(conditioning_train, dtype=torch.float32)  # noqa: E501
                self.tensor_conditioning_val = torch.tensor(conditioning_val, dtype=torch.float32)
                self.tensor_conditioning_test = torch.tensor(conditioning_test, dtype=torch.float32)  # noqa: E501

                if self.hparams.conditioning_gen_filename is not None:
                    pylogger.info("Using conditioning data from generator.")
                    pylogger.info("From file: %s", self.hparams.conditioning_gen_filename)
                    with h5py.File(self.hparams.conditioning_gen_filename, "r") as f:
                        jet_features_gen = f["jet_features"][:]
                        part_mask_gen = f["part_mask"][:]
                        index_jet_type = get_feat_index(f["jet_features"].attrs["names_jet_features"], "jet_type")
                        pylogger.info(
                            "Filtering conditioning data for generation for jet types: %s",
                            self.hparams.used_jet_types
                        )
                        jet_types_mask = np.isin(jet_features_gen[:, index_jet_type], used_jet_types_values)  # noqa: E501
                        conditioning_gen, _ = self._handle_conditioning(
                            jet_features_gen[jet_types_mask],
                            f["jet_features"].attrs["names_jet_features"],
                            names_labels
                        )
                        pylogger.info("Number of jets in conditioning gen data: %s", len(conditioning_gen))
                        self.mask_gen = torch.tensor(part_mask_gen[jet_types_mask], dtype=torch.float32)
                        self.tensor_conditioning_gen = torch.tensor(conditioning_gen, dtype=torch.float32)
                        pylogger.info("Done processing the conditioning data.")
                else:
                    pylogger.warning("No conditioning data from generator used. Will used the truth conditioning data!.")
                    self.mask_gen = None
                    self.tensor_conditioning_gen = None

                # fmt: on

            # invert the masks from the masked arrays (numpy ma masks are True for masked values)
            self.mask_train = torch.tensor(mask_train, dtype=torch.float32)
            self.mask_test = torch.tensor(mask_test, dtype=torch.float32)
            self.mask_val = torch.tensor(mask_val, dtype=torch.float32)
            tensor_train = torch.tensor(dataset_train, dtype=torch.float32)
            tensor_test = torch.tensor(dataset_test, dtype=torch.float32)
            tensor_val = torch.tensor(dataset_val, dtype=torch.float32)
            self.labels_train = torch.tensor(labels_train, dtype=torch.float32)
            self.labels_test = torch.tensor(labels_test, dtype=torch.float32)
            self.labels_val = torch.tensor(labels_val, dtype=torch.float32)
            self.names_particle_features = names_particle_features
            self.names_jet_features = names_jet_features
            self.names_labels = names_labels

            # reverse standardization for those tensors
            # The "standard tensors" (i.e. self.tensor_train, ...) are not standardized
            # (only the ones with the "_dl" suffix are standardized)
            self.tensor_train = torch.clone(tensor_train)
            self.tensor_test = torch.clone(tensor_test)
            self.tensor_val = torch.clone(tensor_val)

            # revert standardization for those tensors
            self.min_max_train_dict = {}
            for i in range(len(indices_part_features)):
                self.tensor_train[:, :, i] = (
                    (self.tensor_train[:, :, i] * part_stds_train[i]) + part_means_train[i]
                ) * mask_train[..., 0]
                # calculate min and max for each feature in the training data
                # (to shift generated data back to the original range later on)
                self.min_max_train_dict[names_particle_features[i]] = {
                    "min": self.tensor_train[:, :, i][mask_train[..., 0] != 0].min(),
                    "max": self.tensor_train[:, :, i][mask_train[..., 0] != 0].max(),
                }
                self.tensor_test[:, :, i] = (
                    (self.tensor_test[:, :, i] * part_stds_train[i]) + part_means_train[i]
                ) * mask_test[..., 0]
                self.tensor_val[:, :, i] = (
                    (self.tensor_val[:, :, i] * part_stds_train[i]) + part_means_train[i]
                ) * mask_val[..., 0]

            # if no standardization is used, just use the non-standardized tensors
            # from above
            if not self.hparams.normalize:
                self.tensor_train_dl = self.tensor_train
                self.tensor_test_dl = self.tensor_test
                self.tensor_val_dl = self.tensor_val
            else:
                sigma = 1
                if isinstance(self.hparams.normalize_sigma, (int, float)):
                    pylogger.info("Scaling data with sigma = %s", self.hparams.normalize_sigma)
                    sigma = self.hparams.normalize_sigma

                self.tensor_train_dl = tensor_train * sigma
                self.tensor_test_dl = tensor_test * sigma
                self.tensor_val_dl = tensor_val * sigma
                self.means = part_means_train
                self.stds = part_stds_train

            # TODO: add here the corresponding part in case we standardize the
            # conditioning data
            self.tensor_conditioning_train_dl = self.tensor_conditioning_train
            self.tensor_conditioning_val_dl = self.tensor_conditioning_val
            self.tensor_conditioning_test_dl = self.tensor_conditioning_test

            self.data_train = TensorDataset(
                self.tensor_train_dl,
                self.mask_train,
                self.tensor_conditioning_train_dl,
            )
            self.data_val = TensorDataset(
                self.tensor_val_dl,
                self.mask_val,
                self.tensor_conditioning_val_dl,
            )
            self.data_test = TensorDataset(
                self.tensor_test_dl,
                self.mask_test,
                self.tensor_conditioning_test_dl,
            )

            # ---------------------------------------------------------------
            # Perform some checks on the data
            pylogger.info("Checking for NaNs in the data.")
            if (
                torch.isnan(self.tensor_train_dl).any()
                or torch.isnan(self.tensor_val_dl).any()
                or torch.isnan(self.tensor_test_dl).any()
            ):
                raise ValueError("NaNs found in particle data!")

            # check if conditioning data contains nan values
            if (
                torch.isnan(self.tensor_conditioning_train_dl).any()
                or torch.isnan(self.tensor_conditioning_val_dl).any()
                or torch.isnan(self.tensor_conditioning_test_dl).any()
            ):
                raise ValueError("NaNs found in conditioning data!")

            pylogger.info("Checking that there are no jets without any constituents.")
            n_jets_no_particles_train = np.sum(np.sum(mask_train, axis=1) == 0)
            n_jets_no_particles_val = np.sum(np.sum(mask_val, axis=1) == 0)
            n_jets_no_particles_test = np.sum(np.sum(mask_test, axis=1) == 0)

            if (
                n_jets_no_particles_train > 0
                or n_jets_no_particles_val > 0
                or n_jets_no_particles_test > 0
            ):
                raise NotImplementedError(
                    "There are jets without particles in the dataset. This"
                    "is not allowed, since the model cannot handle this case."
                )

            pylogger.info("--- Done setting up the dataloader. ---")
            pylogger.info("Particle features: %s", self.names_particle_features)
            pylogger.info("Conditioning features: %s", self.names_conditioning)

            pylogger.info("--- Shape of the training data: ---")
            pylogger.info("particle features: %s", self.tensor_train_dl.shape)
            pylogger.info("mask: %s", self.mask_train.shape)
            pylogger.info("conditioning features: %s", self.tensor_conditioning_train_dl.shape)

            pylogger.info("--- Shape of the validation data: ---")
            pylogger.info("particle features: %s", self.tensor_val_dl.shape)
            pylogger.info("mask: %s", self.mask_val.shape)
            pylogger.info("conditioning features: %s", self.tensor_conditioning_val_dl.shape)

            pylogger.info("--- Shape of the test data: ---")
            pylogger.info("particle features: %s", self.tensor_test_dl.shape)
            pylogger.info("mask: %s", self.mask_test.shape)
            pylogger.info("conditioning features: %s", self.tensor_conditioning_test_dl.shape)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def _handle_conditioning(
        self,
        jet_data: np.array,
        names_jet_data: np.array,
        names_labels: np.array,
    ):
        """Select the conditioning variables.

        Args:
            jet_data: np.array of shape (n_jets, n_features) the first jet features
                is expected to be the jet-type
            names_jet_data: np.array of shape (n_features,) which contains the names of
                the features
            names_labels: np.array of shape (n_jet_types,) which contains the names of
                the jet-types (e.g. if there are three jet types: ['q', 'g', 't'], then
                a label 0 would correspond to 'q', 1 to 'g' and 2 to 't')
        Returns:
            conditioning_data: np.array of shape (n_jets, n_conditioning_features)
            names_conditioning_data: np.array of shape (n_conditioning_features,) which
                contains the names of the conditioning features
        """
        if self.hparams.conditioning_jet_type_all:
            # use all jet types as conditioning variables
            # for JetClass this should lead to 10 one-hot encoded values
            categories = np.arange(len(names_labels))
        else:
            categories = np.unique(jet_data[:, 0])

        pylogger.info("Using one-hot encoding for jet types: %s", categories)
        jet_data_one_hot = one_hot_encode(
            jet_data, categories=[categories], num_other_features=jet_data.shape[1] - 1
        )

        one_hot_len = len(categories)
        if (
            not self.hparams.conditioning_pt
            and not self.hparams.conditioning_energy
            and not self.hparams.conditioning_eta
            and not self.hparams.conditioning_mass
            and not self.hparams.conditioning_num_particles
            and not self.hparams.conditioning_jet_type
        ):
            return None

        # select the columns which correspond to the conditioning variables that should be used

        keep_col = []
        names_conditioning_data = []

        if self.hparams.conditioning_jet_type:
            keep_col += list(np.arange(one_hot_len))
            names_conditioning_data += [f"jet_type_{names_labels[int(i)]}" for i in categories]
        if self.hparams.conditioning_pt:
            keep_col.append(get_feat_index(names_jet_data, "jet_pt") + one_hot_len - 1)
            names_conditioning_data.append("jet_pt")
        if self.hparams.conditioning_energy:
            keep_col.append(get_feat_index(names_jet_data, "jet_energy") + one_hot_len - 1)
            names_conditioning_data.append("jet_energy")
        if self.hparams.conditioning_eta:
            keep_col.append(get_feat_index(names_jet_data, "jet_eta") + one_hot_len - 1)
            names_conditioning_data.append("jet_eta")
        if self.hparams.conditioning_mass:
            keep_col.append(get_feat_index(names_jet_data, "jet_sdmass") + one_hot_len - 1)
            names_conditioning_data.append("jet_sdmass")
        if self.hparams.conditioning_num_particles:
            keep_col.append(get_feat_index(names_jet_data, "jet_nparticles") + one_hot_len - 1)
            names_conditioning_data.append("jet_nparticles")

        return jet_data_one_hot[:, keep_col], names_conditioning_data


if __name__ == "__main__":
    _ = JetClassDataModule()
