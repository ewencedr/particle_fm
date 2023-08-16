from typing import Any, Dict, Optional

import energyflow as ef
import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.pylogger import get_pylogger

from .components import normalize_tensor

log = get_pylogger("JetNetDataModule")


# TODO Standardization
class LHCOJetFeatureDataModule(LightningDataModule):
    """LightningDataModule for JetFeatures of LHCO dataset. If no conditioning is used, the
    conditioning tensor will be a tensor of zeros.

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
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 256,
        num_workers: int = 32,
        pin_memory: bool = False,
        drop_last: bool = False,
        verbose: bool = True,
        # data
        normalize: bool = True,
        normalize_sigma: int = 5,
        window_left: float = 3.3e3,
        window_right: float = 3.7e3,
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

        self.tensor_train_sr: Optional[torch.Tensor] = None
        self.tensor_val_sr: Optional[torch.Tensor] = None
        self.tensor_test_sr: Optional[torch.Tensor] = None
        self.tensor_conditioning_train_sr: Optional[torch.Tensor] = None
        self.tensor_conditioning_val_sr: Optional[torch.Tensor] = None
        self.tensor_conditioning_test_sr: Optional[torch.Tensor] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # data loading
            path = f"{self.hparams.data_dir}/lhco/final_data/processed_data_background_rel.h5"
            with h5py.File(path, "r") as f:
                jet_data = f["jet_data"][:]
                mask = f["mask"][:]

            # get particle multilicities
            n_particles = np.sum(mask, axis=-2)

            p4_jets = ef.p4s_from_ptyphims(jet_data)

            # get mjj from p4_jets
            sum_p4 = p4_jets[:, 0] + p4_jets[:, 1]
            mjj = ef.ms_from_p4s(sum_p4)

            jet_data2 = jet_data.copy()
            n_particles2 = n_particles.copy()

            # cut window
            args_to_keep = ((mjj < 3300) & (mjj > 2300)) | ((mjj > 3700) & (mjj < 5000))
            conditioning_cut = mjj[args_to_keep].reshape(-1, 1)

            args_to_keep_sr = (mjj > 3300) & (mjj < 3700)
            conditioning_sr = mjj.copy()[args_to_keep_sr].reshape(-1, 1)

            jet_data_cut = jet_data[args_to_keep]
            n_particles_cut = n_particles[args_to_keep]
            jet_data_sr = jet_data2[args_to_keep_sr]
            n_particles_sr = n_particles2[args_to_keep_sr]

            jet_data_n_particles_cut = np.concatenate([jet_data_cut, n_particles_cut], axis=-1)
            jet_data_n_particles_sr = np.concatenate([jet_data_sr, n_particles_sr], axis=-1)

            data = np.reshape(jet_data_n_particles_cut, (jet_data_n_particles_cut.shape[0], -1))
            data_sr = np.reshape(jet_data_n_particles_sr, (jet_data_n_particles_sr.shape[0], -1))

            # data splitting
            n_samples_val = int(self.hparams.val_fraction * len(data))
            n_samples_test = int(self.hparams.test_fraction * len(data))
            n_samples_val_sr = int(self.hparams.val_fraction * len(data_sr))
            n_samples_test_sr = int(self.hparams.test_fraction * len(data_sr))

            dataset_train, dataset_val, dataset_test = np.split(
                data,
                [
                    len(data) - (n_samples_val + n_samples_test),
                    len(data) - n_samples_test,
                ],
            )
            dataset_train_sr, dataset_val_sr, dataset_test_sr = np.split(
                data_sr,
                [
                    len(data_sr) - (n_samples_val_sr + n_samples_test_sr),
                    len(data_sr) - n_samples_test_sr,
                ],
            )

            conditioning_train, conditioning_val, conditioning_test = np.split(
                conditioning_cut,
                [
                    len(conditioning_cut) - (n_samples_val + n_samples_test),
                    len(conditioning_cut) - n_samples_test,
                ],
            )
            conditioning_train_sr, conditioning_val_sr, conditioning_test_sr = np.split(
                conditioning_sr,
                [
                    len(conditioning_sr) - (n_samples_val_sr + n_samples_test_sr),
                    len(conditioning_sr) - n_samples_test_sr,
                ],
            )

            tensor_conditioning_train = torch.tensor(conditioning_train, dtype=torch.float)
            tensor_conditioning_val = torch.tensor(conditioning_val, dtype=torch.float)
            tensor_conditioning_test = torch.tensor(conditioning_test, dtype=torch.float)
            tensor_conditioning_train_sr = torch.tensor(conditioning_train_sr, dtype=torch.float)
            tensor_conditioning_val_sr = torch.tensor(conditioning_val_sr, dtype=torch.float)
            tensor_conditioning_test_sr = torch.tensor(conditioning_test_sr, dtype=torch.float)
            if self.hparams.normalize:
                means = np.mean(dataset_train, axis=0)
                stds = np.std(dataset_train, axis=0)
                means_cond = torch.mean(tensor_conditioning_train, axis=0)
                stds_cond = torch.std(tensor_conditioning_train, axis=0)

                # Training
                normalized_dataset_train = normalize_tensor(
                    np.copy(dataset_train), means, stds, sigma=self.hparams.normalize_sigma
                )
                tensor_train = torch.tensor(normalized_dataset_train, dtype=torch.float)

                tensor_conditioning_train = normalize_tensor(
                    tensor_conditioning_train,
                    means_cond,
                    stds_cond,
                    sigma=self.hparams.normalize_sigma,
                )

                normalized_dataset_train_sr = normalize_tensor(
                    np.copy(dataset_train_sr), means, stds, sigma=self.hparams.normalize_sigma
                )
                tensor_train_sr = torch.tensor(normalized_dataset_train_sr, dtype=torch.float)
                tensor_conditioning_train_sr = normalize_tensor(
                    tensor_conditioning_train_sr,
                    means_cond,
                    stds_cond,
                    sigma=self.hparams.normalize_sigma,
                )

                # Validation
                normalized_dataset_val = normalize_tensor(
                    np.copy(dataset_val),
                    means,
                    stds,
                    sigma=self.hparams.normalize_sigma,
                )
                tensor_val = torch.tensor(normalized_dataset_val, dtype=torch.float)

                tensor_conditioning_val = normalize_tensor(
                    tensor_conditioning_val,
                    means_cond,
                    stds_cond,
                    sigma=self.hparams.normalize_sigma,
                )

                normalized_dataset_val_sr = normalize_tensor(
                    np.copy(dataset_val_sr),
                    means,
                    stds,
                    sigma=self.hparams.normalize_sigma,
                )
                tensor_val_sr = torch.tensor(normalized_dataset_val_sr, dtype=torch.float)
                tensor_conditioning_val_sr = normalize_tensor(
                    tensor_conditioning_val_sr,
                    means_cond,
                    stds_cond,
                    sigma=self.hparams.normalize_sigma,
                )

                # Test
                tensor_conditioning_test = normalize_tensor(
                    tensor_conditioning_test,
                    means_cond,
                    stds_cond,
                    sigma=self.hparams.normalize_sigma,
                )

                tensor_conditioning_test_sr = normalize_tensor(
                    tensor_conditioning_test_sr,
                    means_cond,
                    stds_cond,
                    sigma=self.hparams.normalize_sigma,
                )

            unnormalized_tensor_train = torch.tensor(dataset_train, dtype=torch.float)
            unnormalized_tensor_val = torch.tensor(dataset_val, dtype=torch.float)

            unnormalized_tensor_train_sr = torch.tensor(dataset_train_sr, dtype=torch.float)
            unnormalized_tensor_val_sr = torch.tensor(dataset_val_sr, dtype=torch.float)

            tensor_test = torch.tensor(dataset_test, dtype=torch.float)
            tensor_test_sr = torch.tensor(dataset_test_sr, dtype=torch.float)

            if self.hparams.normalize:
                self.data_train = TensorDataset(tensor_train, tensor_conditioning_train)
                self.data_val = TensorDataset(tensor_val, tensor_conditioning_val)
                self.data_test = TensorDataset(tensor_test, tensor_conditioning_test)
                self.means = torch.tensor(means)
                self.stds = torch.tensor(stds)
                self.cond_means = means_cond
                self.cond_stds = stds_cond
            else:
                self.data_train = TensorDataset(
                    unnormalized_tensor_train, tensor_conditioning_train
                )
                self.data_val = TensorDataset(unnormalized_tensor_val, tensor_conditioning_val)
                self.data_test = TensorDataset(tensor_test, tensor_conditioning_test)

            if self.hparams.verbose:
                print(f"Window: {self.hparams.window_left} - {self.hparams.window_right}")
                print(f"{len(p4_jets) - len(data)} events are removed due to the window cut.")
                print("Train dataset size:", len(self.data_train))
                print("Validation dataset size:", len(self.data_val))
                print("Test dataset size:", len(self.data_test))

            self.tensor_train = tensor_train
            self.tensor_val = unnormalized_tensor_val
            self.tensor_test = tensor_test
            self.tensor_conditioning_train = tensor_conditioning_train
            self.tensor_conditioning_val = tensor_conditioning_val
            self.tensor_conditioning_test = tensor_conditioning_test

            self.tensor_train_sr = tensor_train_sr
            self.tensor_val_sr = unnormalized_tensor_val_sr
            self.tensor_test_sr = tensor_test_sr
            self.tensor_conditioning_train_sr = tensor_conditioning_train_sr
            self.tensor_conditioning_val_sr = tensor_conditioning_val_sr
            self.tensor_conditioning_test_sr = tensor_conditioning_test_sr

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


if __name__ == "__main__":
    _ = LHCOJetFeatureDataModule()
