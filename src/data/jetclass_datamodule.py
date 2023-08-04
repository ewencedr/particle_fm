"""PyTorch Lightning DataModule for JetClass dataset."""
from typing import Any, Dict, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.pylogger import get_pylogger

from .components import normalize_tensor

log = get_pylogger("JetClassDataModule")


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
        # conditioning_jet_type: bool = True,
        num_particles: int = 128,
        # preprocessing
        normalize: bool = True,
        normalize_sigma: int = 5,
        use_custom_eta_centering: bool = True,
        remove_etadiff_tails: bool = True,
        # centering: bool = False,
        # use_calculated_base_distribution: bool = True,
    ):
        super().__init__()

        if jet_types is None:
            raise ValueError("`jet_types` must be specified in the datamodule.")

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
                # self.hparams.conditioning_jet_type,
            ]
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            particle_features_list = []
            jet_features_list = []
            labels_list = []

            names_particle_features_previous = None
            names_jet_features_previous = None
            names_labels_previous = None
            filename_previous = None

            print(self.hparams.jet_types)
            # data loading
            for jet_type, jet_type_dict in self.hparams.jet_types.items():
                for filename in jet_type_dict["files"]:
                    print(f"Loading {filename}")
                    npfile = np.load(filename, allow_pickle=True)
                    particle_features_list.append(npfile["part_features"])
                    jet_features_list.append(npfile["jet_features"])
                    labels_list.append(npfile["labels"])

                    # Check that the labels are in the same order for all files
                    names_particle_features = npfile["names_part_features"]
                    names_jet_features = npfile["names_jet_features"]
                    names_labels = npfile["names_labels"]

                    if (
                        names_particle_features_previous is None
                        and names_jet_features_previous is None
                        and names_labels_previous is None
                    ):
                        # first file
                        pass
                    else:
                        if not np.all(names_particle_features == names_particle_features_previous):
                            raise ValueError(
                                "Names of particle features are not the same for all files."
                                f"\n{filename_previous}: {names_particle_features_previous}"
                                f"\n{filename}: {names_particle_features}"
                            )
                        if not np.all(names_jet_features == names_jet_features_previous):
                            raise ValueError(
                                "Names of jet features are not the same for all files."
                                f"\n{filename_previous}: {names_jet_features_previous}"
                                f"\n{filename}: {names_jet_features}"
                            )
                        if not np.all(names_labels == names_labels_previous):
                            raise ValueError(
                                "Names of labels are not the same for all files."
                                f"\n{filename_previous}: {names_labels_previous}"
                                f"\n{filename}: {names_labels}"
                            )
                    names_particle_features_previous = names_particle_features
                    names_jet_features_previous = names_jet_features
                    names_labels_previous = names_labels
                    filename_previous = filename

            particle_features = np.concatenate(particle_features_list)
            jet_features = np.concatenate(jet_features_list)
            labels_one_hot = np.concatenate(labels_list)

            labels = np.argmax(labels_one_hot, axis=1)

            # shuffle data
            np.random.seed(42)
            permutation = np.random.permutation(len(labels))
            particle_features = particle_features[permutation]
            jet_features = jet_features[permutation]
            labels = labels[permutation]

            print("Loaded data.")
            print(f"particle_features.shape = {particle_features.shape}")
            print(f"jet_features.shape = {jet_features.shape}")
            print(f"labels.shape = {labels.shape}")

            if self.hparams.number_of_used_jets is not None:
                if self.hparams.number_of_used_jets < len(jet_features):
                    particle_features = particle_features[: self.hparams.number_of_used_jets]
                    jet_features = jet_features[: self.hparams.number_of_used_jets]
                    labels = labels[: self.hparams.number_of_used_jets]

            # NOTE: everything below here assumes that the particle features
            # array after preprocessing stores the features [eta_rel, phi_rel, pt_rel]

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
                mask = (particle_features[:, :, 0] != 0).astype(int)
                particle_features[:, :, 0] = particle_eta_minus_jet_eta * mask

            if self.hparams.remove_etadiff_tails:
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
            ma_particle_features = np.ma.masked_array(
                particle_features,
                mask=np.ma.make_mask(particle_features == 0),
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
            if self.num_cond_features == 0:
                self.tensor_conditioning_train = torch.zeros(len(dataset_train))
                self.tensor_conditioning_val = torch.zeros(len(dataset_val))
                self.tensor_conditioning_test = torch.zeros(len(dataset_test))
            else:
                jet_features = self._handle_conditioning(jet_features, names_jet_features)
                (conditioning_train, conditioning_val, conditioning_test) = np.split(
                    jet_features,
                    [
                        len(jet_features) - (n_samples_val + n_samples_test),
                        len(jet_features) - n_samples_test,
                    ],
                )
                self.tensor_conditioning_train = torch.tensor(
                    conditioning_train, dtype=torch.float32
                )
                self.tensor_conditioning_val = torch.tensor(conditioning_val, dtype=torch.float32)
                self.tensor_conditioning_test = torch.tensor(
                    conditioning_test, dtype=torch.float32
                )

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

            if self.hparams.normalize:
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

            else:
                self.tensor_train_dl = torch.tensor(dataset_train[:, :, :3], dtype=torch.float32)
                self.tensor_test_dl = torch.tensor(dataset_test[:, :, :3], dtype=torch.float32)
                self.tensor_val_dl = torch.tensor(dataset_val[:, :, :3], dtype=torch.float32)

            self.data_train = TensorDataset(
                self.tensor_train_dl,
                self.mask_train,
                self.tensor_conditioning_train,
            )
            self.data_val = TensorDataset(
                self.tensor_val_dl,
                self.mask_val,
                self.tensor_conditioning_val,
            )
            self.data_test = TensorDataset(
                self.tensor_test_dl,
                self.mask_test,
                self.tensor_conditioning_test,
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

    def _handle_conditioning(self, jet_data: np.array, names_jet_data: np.array):
        """Select the conditioning variables.

        jet_data: np.array of shape (n_jets, n_features)
        names_jet_data: np.array of shape (n_features,) which contains the names of
            the features
        """

        if (
            not self.hparams.conditioning_pt
            and not self.hparams.conditioning_eta
            and not self.hparams.conditioning_mass
            and not self.hparams.conditioning_num_particles
        ):
            return None

        # select the columns which correspond to the conditioning variables that should be used

        keep_col = []
        if self.hparams.conditioning_pt:
            keep_col.append(get_feat_index(names_jet_data, "jet_pt"))
        if self.hparams.conditioning_eta:
            keep_col.append(get_feat_index(names_jet_data, "jet_eta"))
        if self.hparams.conditioning_mass:
            keep_col.append(get_feat_index(names_jet_data, "jet_sdmass"))
        if self.hparams.conditioning_num_particles:
            keep_col.append(get_feat_index(names_jet_data, "jet_nparticles"))

        return jet_data[:, keep_col]


if __name__ == "__main__":
    _ = JetClassDataModule()
