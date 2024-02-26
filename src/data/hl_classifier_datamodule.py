from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components import normalize_tensor


class HLClassifierDataModule(LightningDataModule):
    """`LightningDataModule` for High Level Classifier of LHCO data.

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
        file_name1: str = "data1",
        file_name2: str = "data2",
        train_val_test_split: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `HLClassifierDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split in percent. Defaults to `(0.70, 0.15, 0.15)`.
        :param batch_size: The batch size. Defaults to `128`.
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

        self.means = None
        self.stds = None

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
            save_path_true = self.hparams.data_dir + self.hparams.file_name1 + ".npy"
            path_data = self.hparams.data_dir + self.hparams.file_name2 + ".npy"

            data = np.load(path_data)
            data_truth = np.load(save_path_true)

            data = data[:, :4]
            data_truth = data_truth[:, :4]

            data = data[: len(data_truth)]

            labels_true = np.expand_dims(np.ones(len(data_truth)), axis=-1)
            labels_false = np.expand_dims(np.zeros(len(data)), axis=-1)

            print("data_truth.shape", data_truth.shape)
            print("data.shape", data.shape)

            data = np.concatenate([data_truth, data])
            labels = np.concatenate([labels_true, labels_false])

            random_permutation = np.random.permutation(len(data))
            data = data[random_permutation]
            labels = labels[random_permutation]

            n_samples_val = int(self.hparams.train_val_test_split[1] * len(data))
            n_samples_test = int(self.hparams.train_val_test_split[2] * len(data))

            data_train, data_val, data_test = np.split(
                data,
                [
                    len(data) - n_samples_val - n_samples_test,
                    len(data) - n_samples_test,
                ],
            )

            labels_train, labels_val, labels_test = np.split(
                labels,
                [
                    len(labels) - n_samples_val - n_samples_test,
                    len(labels) - n_samples_test,
                ],
            )

            # preprocess data
            mean = np.mean(data_train, axis=0)
            std = np.std(data_train, axis=0)

            self.means = mean
            self.stds = std

            data_train = normalize_tensor(
                torch.tensor(data_train).clone(), torch.tensor(mean), torch.tensor(std)
            )
            data_val = normalize_tensor(
                torch.tensor(data_val).clone(), torch.tensor(mean), torch.tensor(std)
            )
            data_test = normalize_tensor(
                torch.tensor(data_test).clone(), torch.tensor(mean), torch.tensor(std)
            )

            self.data_train = torch.utils.data.TensorDataset(
                data_train.float(),
                torch.from_numpy(labels_train).float(),
            )
            self.data_val = torch.utils.data.TensorDataset(
                data_val.float(),
                torch.from_numpy(labels_val).float(),
            )
            self.data_test = torch.utils.data.TensorDataset(
                data_test.float(),
                torch.from_numpy(labels_test).float(),
            )

            print("data_train.shape", data_train.shape)
            print("data_val.shape", data_val.shape)
            print("data_test.shape", data_test.shape)
            print("labels_train.shape", labels_train.shape)
            print("labels_val.shape", labels_val.shape)
            print("labels_test.shape", labels_test.shape)

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
    _ = HLClassifierDataModule()
