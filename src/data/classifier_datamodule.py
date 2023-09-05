from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import h5py
import energyflow as ef
import numpy as np


class ClassifierDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            path_bckg = f"{self.hparams.data_dir}/lhco/final_data/processed_data_background_rel.h5"
            path_sgnl = f"{self.hparams.data_dir}/lhco/final_data/processed_data_signal_rel.h5"

            with h5py.File(path_bckg, "r") as f:
                jet_data_bckg = f["jet_data"][:]
                particle_data_bckg = f["constituents"][:]
                mask_bckg = f["mask"][:]

            with h5py.File(path_sgnl, "r") as f:
                jet_data_sgnl = f["jet_data"][:]
                particle_data_sgnl = f["constituents"][:]
                mask_sgnl = f["mask"][:]

            # cut mjj window
            p4_jets_bckg = ef.p4s_from_ptyphims(jet_data_bckg)
            # get mjj from p4_jets
            sum_p4_bckg = p4_jets_bckg[:, 0] + p4_jets_bckg[:, 1]
            mjj_bckg = ef.ms_from_p4s(sum_p4_bckg)

            p4_jets_sgnl = ef.p4s_from_ptyphims(jet_data_sgnl)
            # get mjj from p4_jets
            sum_p4_sgnl = p4_jets_sgnl[:, 0] + p4_jets_sgnl[:, 1]
            mjj_sgnl = ef.ms_from_p4s(sum_p4_sgnl)

            args_to_keep_bckg = (mjj_bckg > 3300) & (mjj_bckg < 3700)
            args_to_keep_sgnl = (mjj_sgnl > 3300) & (mjj_sgnl < 3700)

            jet_data_bckg = jet_data_bckg[args_to_keep_bckg]
            particle_data_bckg = particle_data_bckg[args_to_keep_bckg]
            mask_bckg = mask_bckg[args_to_keep_bckg]

            jet_data_sgnl = jet_data_sgnl[args_to_keep_sgnl]
            particle_data_sgnl = particle_data_sgnl[args_to_keep_sgnl]
            mask_sgnl = mask_sgnl[args_to_keep_sgnl]

            print(f"Number of background events: {len(jet_data_bckg)}")
            print(f"Number of signal events: {len(jet_data_sgnl)}")
            n_signal = 2000
            n_background = 100_000

            jet_data_mixed = np.concatenate(
                [jet_data_bckg[:n_background], jet_data_sgnl[:n_signal]]
            )
            particle_data_mixed = np.concatenate(
                [particle_data_bckg[:n_background], particle_data_sgnl[:n_signal]]
            )
            mask_mixed = np.concatenate([mask_bckg[:n_background], mask_sgnl[:n_signal]])

            # shuffle
            np.random_permutation = np.random.permutation(len(jet_data_mixed))
            jet_data_mixed = jet_data_mixed[np.random_permutation]
            particle_data_mixed = particle_data_mixed[np.random_permutation]
            mask_mixed = mask_mixed[np.random_permutation]

            labels_mixed = np.ones(len(jet_data_mixed))

            # Load generated data for test

            # TODO

            # if idealized:

            jet_data_background = jet_data_bckg[:n_background]
            particle_data_background = particle_data_bckg[:n_background]
            mask_background = mask_bckg[:n_background]

            labels_background = np.zeros(len(jet_data_background))

            input_data = np.concatenate([particle_data_mixed, particle_data_background])
            input_mask = np.concatenate([mask_mixed, mask_background])
            input_labels = np.concatenate([labels_mixed, labels_background])

            data_train, data_val = np.split(
                input_data,
                [int(0.8 * len(input_data))],
            )
            mask_train, mask_val = np.split(
                input_mask,
                [int(0.8 * len(input_data))],
            )
            labels_train, labels_val = np.split(
                input_labels,
                [int(0.8 * len(input_data))],
            )

            print(f"data_train.shape: {data_train.shape}")
            print(f"mask_train.shape: {mask_train.shape}")
            print(f"labels_train.shape: {labels_train.shape}")
            print(f"data_val.shape: {data_val.shape}")
            print(f"mask_val.shape: {mask_val.shape}")
            print(f"labels_val.shape: {labels_val.shape}")

            mean = np.mean(data_train, axis=(0, 1))
            std = np.std(data_train, axis=(0, 1))

            normalize_sigma = 5

            data_train = (data_train - mean) / (normalize_sigma * std)
            data_val = (data_val - mean) / (normalize_sigma * std)

            self.data_train = torch.utils.data.TensorDataset(
                torch.from_numpy(data_train).float(),
                torch.from_numpy(mask_train).float(),
                torch.from_numpy(labels_train).float(),
            )
            self.data_val = torch.utils.data.TensorDataset(
                torch.from_numpy(data_val).float(),
                torch.from_numpy(mask_val).float(),
                torch.from_numpy(labels_val).float(),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ClassifierDataModule()
