"""PyTorch Lightning DataModule for JetClass dataset."""
import os
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.pylogger import get_pylogger

from .components import normalize_tensor, one_hot_encode

pylogger = get_pylogger("JetClassDataModule")


def get_feat_index(names_array: np.array, name: str):
    """Helper function that returns the index of the features name in the jet_data array.

    Args:
        names_array (np.array): Array of feature names.
        name (str): Name of the feature to find.
    """
    return np.argwhere(names_array == name)[0][0]


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
        jet_types: dict = None,
        number_of_used_jets: int = None,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 256,
        num_workers: int = 32,
        pin_memory: bool = False,
        drop_last: bool = False,
        verbose: bool = True,
        variable_jet_sizes: bool = True,
        conditioning_pt: bool = True,
        conditioning_eta: bool = True,
        conditioning_mass: bool = True,
        conditioning_num_particles: bool = True,
        conditioning_jet_type: bool = True,
        num_particles: int = 128,
        # preprocessing
        normalize: bool = True,
        normalize_sigma: int = 5,
        use_custom_eta_centering: bool = True,
        remove_etadiff_tails: bool = True,
        spectator_jet_features: list = None,
        # centering: bool = False,
        # use_calculated_base_distribution: bool = True,
    ):
        super().__init__()

        # TODO: this doesn't work yet...
        self.hparams["jet_types_list"] = list(jet_types.keys())
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
                if not os.path.isfile(filename):
                    raise FileNotFoundError(f"File {filename} does not exist.")

                with h5py.File(filename, "r") as f:
                    arrays_dict[split] = {key: np.array(f[key]) for key in f.keys()}
                    names_dict[split] = {
                        key: f[key].attrs[f"names_{key}"] for key in f.keys() if "mask" not in key
                    }

            particle_features = arrays_dict["train"]["part_features"]
            particles_mask = arrays_dict["train"]["part_mask"]
            jet_features = arrays_dict["train"]["jet_features"]
            labels_one_hot = arrays_dict["train"]["labels"]
            names_particle_features = names_dict["train"]["part_features"]
            names_jet_features = names_dict["train"]["jet_features"]
            names_labels = names_dict["train"]["labels"]

            labels = np.argmax(labels_one_hot, axis=1)

            # shuffle data
            np.random.seed(42)
            permutation = np.random.permutation(len(labels))
            particle_features = particle_features[permutation]
            particles_mask = particles_mask[permutation]
            jet_features = jet_features[permutation]
            labels = labels[permutation]

            pylogger.info("Loaded data.")
            pylogger.info("Shapes of arrays as available in files:")
            pylogger.info(f"particle_features names = {names_particle_features}")
            pylogger.info(f"particle_features shape = {particle_features.shape}")
            pylogger.info(f"jet_features names = {names_jet_features}")
            pylogger.info(f"jet_features.shape = {jet_features.shape}")
            pylogger.info(f"labels names = {names_labels}")
            pylogger.info(f"labels.shape = {labels.shape}")
            pylogger.info("Now processing data...")

            if self.hparams.number_of_used_jets is not None:
                if self.hparams.number_of_used_jets < len(jet_features):
                    pylogger.info(
                        f"Using only {self.hparams.number_of_used_jets} jets "
                        f"out of {len(jet_features)}."
                    )
                    particle_features = particle_features[: self.hparams.number_of_used_jets]
                    particles_mask = particles_mask[: self.hparams.number_of_used_jets]
                    jet_features = jet_features[: self.hparams.number_of_used_jets]
                    labels = labels[: self.hparams.number_of_used_jets]
                else:
                    pylogger.warning(
                        f"More jets requested ({self.hparams.number_of_used_jets:_}) than "
                        f"available ({len(jet_features):_})."
                        "--> Using all available jets."
                    )

            # NOTE: everything below here assumes that the particle features
            # array after preprocessing stores the features [eta_rel, phi_rel, pt_rel]

            pylogger.info("Using eta_rel, phi_rel, pt_rel as particle features.")
            # check if the particle features are in the correct order
            index_part_deta = get_feat_index(names_particle_features, "part_deta")
            assert index_part_deta == 0, "part_deta is not the first feature"
            index_part_dphi = get_feat_index(names_particle_features, "part_dphi")
            assert index_part_dphi == 1, "part_dphi is not the second feature"
            index_part_pt = get_feat_index(names_particle_features, "part_pt")
            assert index_part_pt == 2, "part_pt is not the third feature"

            # divide particle pt by jet pt
            index_jet_pt = get_feat_index(names_jet_features, "jet_pt")
            particle_features[..., index_part_pt] /= np.expand_dims(
                jet_features[:, index_jet_pt], axis=1
            )

            # instead of using the part_deta variable, use part_eta - jet_eta
            if self.hparams.use_custom_eta_centering:
                pylogger.info("Using custom eta centering -> calculating particle_eta - jet_eta")
                if "part_eta" not in names_particle_features:
                    raise ValueError(
                        "`use_custom_eta_centering` is True, but `part_eta` is not in "
                        "in the dataset --> check the dataset"
                    )
                if "jet_eta" not in names_jet_features:
                    raise ValueError(
                        "`use_custom_eta_centering` is True, but `jet_eta` is not in "
                        "in the dataset --> check the dataset"
                    )
                index_jet_eta = get_feat_index(names_jet_features, "jet_eta")
                jet_eta_repeat = jet_features[:, index_jet_eta][:, np.newaxis].repeat(
                    particle_features.shape[1], 1
                )
                index_part_eta = get_feat_index(names_particle_features, "part_eta")
                particle_eta_minus_jet_eta = (
                    particle_features[:, :, index_part_eta] - jet_eta_repeat
                )
                mask = (particles_mask[:, :] != 0).astype(int)
                particle_features[:, :, 0] = particle_eta_minus_jet_eta * mask

            if self.hparams.remove_etadiff_tails:
                pylogger.info("Removing eta tails -> removing particles with |eta_rel| > 1")
                # remove/zero-padd particles with |eta - jet_eta| > 1
                mask_etadiff_larger_1 = np.abs(particle_features[:, :, 0]) > 1
                particle_features[:, :, :][mask_etadiff_larger_1] = 0
                assert (
                    np.sum(np.abs(particle_features[mask_etadiff_larger_1]).flatten()) == 0
                ), "There are still particles with |eta - jet_eta| > 1 that are not zero-padded."

            # from here on only use the first three features (eta_rel, phi_rel, pt_rel)
            particle_features = particle_features[:, :, :3]

            # convert to masked array (more convenient for normalization later on, because
            # the mask is unaffected)
            # Note: numpy masks are True for masked values
            # Important: use pt_rel for masking, because eta_rel and phi_rel can be zero
            # even though it is a valid track
            particle_mask_zero_entries = (particles_mask == 0)[..., np.newaxis]
            ma_particle_features = np.ma.masked_array(
                particle_features,
                mask=np.repeat(
                    particle_mask_zero_entries, repeats=particle_features.shape[2], axis=2
                ),
            )
            pylogger.info("Checking that there are no jets without any constituents.")
            n_jets_without_particles = np.sum(np.sum(~particle_mask_zero_entries, axis=1) == 0)
            if n_jets_without_particles > 0:
                raise NotImplementedError(
                    f"There are {n_jets_without_particles} jets without particles in "
                    "the dataset. This is not allowed, since the model cannot handle this case."
                )

            # data splitting
            n_samples_val = int(self.hparams.val_fraction * len(particle_features))
            n_samples_test = int(self.hparams.test_fraction * len(particle_features))
            dataset_train, dataset_val, dataset_test = np.split(
                ma_particle_features,
                [
                    len(ma_particle_features) - (n_samples_val + n_samples_test),
                    len(ma_particle_features) - n_samples_test,
                ],
            )
            labels_train, labels_val, labels_test = np.split(
                labels,
                [
                    len(labels) - (n_samples_val + n_samples_test),
                    len(labels) - n_samples_test,
                ],
            )

            if self.hparams.spectator_jet_features is not None:
                # initialize and fill array
                spectator_jet_features = np.zeros(
                    (len(jet_features), len(self.hparams.spectator_jet_features))
                )
                for i, feat in enumerate(self.hparams.spectator_jet_features):
                    index = get_feat_index(names_jet_features, feat)
                    spectator_jet_features[:, i] = jet_features[:, index]
            else:
                spectator_jet_features = np.zeros(len(jet_features))

            (
                spectator_jet_features_train,
                spectator_jet_features_val,
                spectator_jet_features_test,
            ) = np.split(
                spectator_jet_features,
                [
                    len(spectator_jet_features) - (n_samples_val + n_samples_test),
                    len(spectator_jet_features) - n_samples_test,
                ],
            )

            if self.num_cond_features == 0:
                self.tensor_conditioning_train = torch.zeros(len(dataset_train))
                self.tensor_conditioning_val = torch.zeros(len(dataset_val))
                self.tensor_conditioning_test = torch.zeros(len(dataset_test))
                self.names_conditioning = None
            else:
                conditioning_features, self.names_conditioning = self._handle_conditioning(
                    jet_features, names_jet_features, labels, names_labels
                )
                (conditioning_train, conditioning_val, conditioning_test) = np.split(
                    conditioning_features,
                    [
                        len(conditioning_features) - (n_samples_val + n_samples_test),
                        len(conditioning_features) - n_samples_test,
                    ],
                )
                self.tensor_conditioning_train = torch.tensor(
                    conditioning_train, dtype=torch.float32
                )
                self.tensor_conditioning_val = torch.tensor(conditioning_val, dtype=torch.float32)
                self.tensor_conditioning_test = torch.tensor(
                    conditioning_test, dtype=torch.float32
                )
                # nan-fine until here

            # invert the masks from the masked arrays (numpy ma masks are True for masked values)
            self.mask_train = torch.tensor(~dataset_train.mask[:, :, :1], dtype=torch.float32)
            self.mask_test = torch.tensor(~dataset_test.mask[:, :, :1], dtype=torch.float32)
            self.mask_val = torch.tensor(~dataset_val.mask[:, :, :1], dtype=torch.float32)
            self.tensor_train = torch.tensor(dataset_train[:, :, :3], dtype=torch.float32)
            self.tensor_test = torch.tensor(dataset_test[:, :, :3], dtype=torch.float32)
            self.tensor_val = torch.tensor(dataset_val[:, :, :3], dtype=torch.float32)
            self.labels_train = torch.tensor(labels_train, dtype=torch.float32)
            self.labels_test = torch.tensor(labels_test, dtype=torch.float32)
            self.labels_val = torch.tensor(labels_val, dtype=torch.float32)
            self.names_particle_features = names_particle_features
            self.names_jet_features = names_jet_features
            self.names_labels = names_labels
            self.tensor_spectator_train = torch.tensor(
                spectator_jet_features_train, dtype=torch.float32
            )
            self.tensor_spectator_test = torch.tensor(
                spectator_jet_features_test, dtype=torch.float32
            )
            self.tensor_spectator_val = torch.tensor(
                spectator_jet_features_val, dtype=torch.float32
            )

            if self.hparams.normalize:
                pylogger.info("Standardizing the particle features.")
                # calculate means and stds only based on the training data
                self.means = np.ma.mean(dataset_train, axis=(0, 1))
                self.stds = np.ma.std(dataset_train, axis=(0, 1))
                norm_kwargs = {
                    "mean": self.means,
                    "std": self.stds,
                    "sigma": self.hparams.normalize_sigma,
                }

                # normalize the data
                norm_dataset_train = normalize_tensor(np.ma.copy(dataset_train), **norm_kwargs)
                norm_dataset_val = normalize_tensor(np.ma.copy(dataset_val), **norm_kwargs)
                norm_dataset_test = normalize_tensor(np.ma.copy(dataset_test), **norm_kwargs)

                self.tensor_train_dl = torch.tensor(
                    norm_dataset_train[:, :, :3], dtype=torch.float32
                )
                self.tensor_val_dl = torch.tensor(norm_dataset_val[:, :, :3], dtype=torch.float32)
                self.tensor_test_dl = torch.tensor(
                    norm_dataset_test[:, :, :3], dtype=torch.float32
                )
                # if self.num_cond_features > 0:
                #     means_cond = torch.mean(self.tensor_conditioning_train, axis=0)
                #     stds_cond = torch.std(self.tensor_conditioning_train, axis=0)
                #     # Train
                #     self.tensor_conditioning_train_dl = normalize_tensor(
                #         self.tensor_conditioning_train,
                #         means_cond,
                #         stds_cond,
                #         sigma=self.hparams.normalize_sigma,
                #     )

                #     # Validation
                #     self.tensor_conditioning_val_dl = normalize_tensor(
                #         self.tensor_conditioning_val,
                #         means_cond,
                #         stds_cond,
                #         sigma=self.hparams.normalize_sigma,
                #     )

                #     # Test
                #     self.tensor_conditioning_test_dl = normalize_tensor(
                #         self.tensor_conditioning_test,
                #         means_cond,
                #         stds_cond,
                #         sigma=self.hparams.normalize_sigma,
                #     )

            else:
                self.tensor_train_dl = torch.tensor(dataset_train[:, :, :3], dtype=torch.float32)
                self.tensor_test_dl = torch.tensor(dataset_test[:, :, :3], dtype=torch.float32)
                self.tensor_val_dl = torch.tensor(dataset_val[:, :, :3], dtype=torch.float32)

            self.tensor_conditioning_train_dl = self.tensor_conditioning_train
            self.tensor_conditioning_val_dl = self.tensor_conditioning_val
            self.tensor_conditioning_test_dl = self.tensor_conditioning_test

            pylogger.info("Checking for NaNs in the data.")
            # check if particle data contains nan values
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

            pylogger.info("--- Done setting up the dataloader. ---")
            pylogger.info("Particle features: eta_rel, phi_rel, pT_rel")
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
        labels: np.array,
        names_labels: np.array,
    ):
        """Select the conditioning variables.

        Args:
            jet_data: np.array of shape (n_jets, n_features)
            names_jet_data: np.array of shape (n_features,) which contains the names of
                the features
            labels: np.array of shape (n_jets,) which contains the labels / jet-types
            names_labels: np.array of shape (n_jet_types,) which contains the names of
                the jet-types (e.g. if there are three jet types: ['q', 'g', 't'], then
                a label 0 would correspond to 'q', 1 to 'g' and 2 to 't')
        Returns:
            conditioning_data: np.array of shape (n_jets, n_conditioning_features)
            names_conditioning_data: np.array of shape (n_conditioning_features,) which
                contains the names of the conditioning features
        """
        categories = np.unique(jet_data[:, 0])
        jet_data_one_hot = one_hot_encode(
            jet_data, categories=[categories], num_other_features=jet_data.shape[1] - 1
        )

        one_hot_len = len(categories)
        if (
            not self.hparams.conditioning_pt
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
