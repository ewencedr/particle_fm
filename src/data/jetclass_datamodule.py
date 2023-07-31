"""PyTorch Lightning DataModule for JetClass dataset."""
from typing import Any, Dict, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.pylogger import get_pylogger

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
        val_fraction (float, optional): Fraction of data to use for validation. Between 0 and 1. Defaults to 0.15.
        test_fraction (float, optional): Fraction of data to use for testing. Between 0 and 1. Defaults to 0.15.
        batch_size (int, optional): Batch size. Defaults to 256.
        num_workers (int, optional): Number of workers for dataloader. Defaults to 32.
        pin_memory (bool, optional): Pin memory for dataloader. Defaults to False.
        drop_last (bool, optional): Drop last batch for train and val dataloader. Defaults to False.

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
        data_dir: str = "data/",
        data_filename: str = "jetclass.npz",
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
        # preprocessing
        centering: bool = False,
        normalize: bool = False,
        normalize_sigma: int = 5,
        use_calculated_base_distribution: bool = True,
        use_custom_eta_centering: bool = True,
        remove_etadiff_tails: bool = True,
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
            ]
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # data loading
            path = f"{self.hparams.data_dir}/{self.hparams.data_filename}"
            npfile = np.load(path, allow_pickle=True)

            particle_features = npfile["part_features"]
            jet_features = npfile["jet_features"]
            if self.hparams.number_of_used_jets is not None:
                if self.hparams.number_of_used_jets < len(jet_features):
                    particle_features = particle_features[: self.hparams.number_of_used_jets]
                    jet_features = jet_features[: self.hparams.number_of_used_jets]

            names_part_features = npfile["names_part_features"]
            names_jet_features = npfile["names_jet_features"]
            # TODO: anything to do with labels?
            # labels = npfile["labels"]

            # divide particle pt by jet pt
            jet_pt_index = get_feat_index(names_jet_features, "jet_pt")
            part_pt_index = get_feat_index(names_part_features, "part_pt")
            particle_features[..., part_pt_index] /= np.expand_dims(
                jet_features[:, jet_pt_index], axis=1
            )

            # instead of using the part_deta variable, use part_eta - jet_eta
            if self.hparams.use_custom_eta_centering:
                jet_eta_repeat = jet_features[:, get_feat_index(names_jet_features, "jet_eta")][
                    :, np.newaxis
                ].repeat(particle_features.shape[1], 1)
                particle_eta_minus_jet_eta = (
                    particle_features[:, :, get_feat_index(names_part_features, "part_eta")]
                    - jet_eta_repeat
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

            # data splitting
            n_samples_val = int(self.hparams.val_fraction * len(particle_features))
            n_samples_test = int(self.hparams.test_fraction * len(particle_features))
            dataset_train, dataset_val, dataset_test = np.split(
                particle_features,
                [
                    len(particle_features) - (n_samples_val + n_samples_test),
                    len(particle_features) - n_samples_test,
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

            self.tensor_train = torch.tensor(dataset_train[:, :, :3], dtype=torch.float32)
            self.mask_train = torch.tensor(
                np.expand_dims(dataset_train[:, :, 3] > 0, axis=-1), dtype=torch.float32
            )
            self.tensor_test = torch.tensor(dataset_test[:, :, :3], dtype=torch.float32)
            self.mask_test = torch.tensor(
                np.expand_dims(dataset_test[:, :, 3] > 0, axis=-1), dtype=torch.float32
            )
            self.tensor_val = torch.tensor(dataset_val[:, :, :3], dtype=torch.float32)
            self.mask_val = torch.tensor(
                np.expand_dims(dataset_val[:, :, 3] > 0, axis=-1), dtype=torch.float32
            )

            self.data_train = TensorDataset(
                self.tensor_train,
                self.mask_train,
                self.tensor_conditioning_train,
            )
            self.data_val = TensorDataset(
                self.tensor_val,
                self.mask_val,
                self.tensor_conditioning_val,
            )
            self.data_test = TensorDataset(
                self.tensor_test,
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
