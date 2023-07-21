from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from src.utils.pylogger import get_pylogger

log = get_pylogger("JetNetDataModule")


class LHCODataModule(LightningDataModule):
    """LightningDataModule for JetNet dataset. If no conditioning is used, the conditioning tensor
    will be a tensor of zeros.

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
        variable_jet_sizes: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

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
            path = "/beegfs/desy/user/ewencedr/data/lhco/events_anomalydetection_v2_processed.h5"
            with h5py.File(path, "r") as f:
                data = f["data"][:]
                mask = f["mask"][:]
            data = data[:, :, :3]
            data = np.concatenate([data, np.expand_dims(mask, axis=-1)], axis=-1)
            # data splitting
            n_samples_val = int(self.hparams.val_fraction * len(data))
            n_samples_test = int(self.hparams.test_fraction * len(data))
            dataset_train, dataset_val, dataset_test = np.split(
                data,
                [
                    len(data) - 1 - (n_samples_val + n_samples_test),
                    len(data) - 1 - n_samples_test,
                ],
            )
            tensor_conditioning_train = torch.zeros(len(dataset_train))
            tensor_conditioning_val = torch.zeros(len(dataset_val))
            tensor_conditioning_test = torch.zeros(len(dataset_test))
            self.data_train = TensorDataset(
                torch.tensor(dataset_train[:, :, :3], dtype=torch.float32),
                torch.tensor(np.expand_dims(dataset_train[:, :, 3], axis=-1), dtype=torch.float32),
                tensor_conditioning_train,
            )
            self.data_val = TensorDataset(
                torch.tensor(dataset_val[:, :, :3], dtype=torch.float32),
                torch.tensor(np.expand_dims(dataset_val[:, :, 3], axis=-1), dtype=torch.float32),
                tensor_conditioning_val,
            )
            self.data_test = TensorDataset(
                torch.tensor(dataset_test[:, :, :3], dtype=torch.float32),
                torch.tensor(np.expand_dims(dataset_test[:, :, 3], axis=-1), dtype=torch.float32),
                tensor_conditioning_test,
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


if __name__ == "__main__":
    _ = LHCODataModule()
