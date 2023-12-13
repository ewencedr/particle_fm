from typing import Any, Dict, Optional

import energyflow as ef
import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.preprocessing import LogitScaler
from src.utils.pylogger import get_pylogger

from .components import normalize_tensor

log = get_pylogger("GenChallengeDataModule")


class GenChallengeDataModule(LightningDataModule):
    """LightningDataModule for Generative Challenge 2023. If no conditioning is used, the
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
        val_sets: list = [3],
        test_sets: list = [4],
        batch_size: int = 256,
        num_workers: int = 32,
        pin_memory: bool = False,
        drop_last: bool = False,
        verbose: bool = True,
        # data
        normalize: bool = True,
        normalize_sigma: int = 5,
        set_data: bool = False,
        variable_jet_sizes: bool = False,
        logit_transform: bool = False,
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

        self.preprocessing_pipeline: Optional[preprocessing.Pipeline] = None
        self.preprocessing_pipeline_cond: Optional[preprocessing.Pipeline] = None

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

            data_load0 = np.load(f"{self.hparams.data_dir}/outerdata_kfold_0.npy")
            data_load1 = np.load(f"{self.hparams.data_dir}/outerdata_kfold_1.npy")
            data_load2 = np.load(f"{self.hparams.data_dir}/outerdata_kfold_2.npy")
            data_load3 = np.load(f"{self.hparams.data_dir}/outerdata_kfold_3.npy")
            data_load4 = np.load(f"{self.hparams.data_dir}/outerdata_kfold_4.npy")

            data_load = np.concatenate(
                (data_load0, data_load1, data_load2, data_load3, data_load4), axis=0
            )

            data_load0_sr = np.load(f"{self.hparams.data_dir}/innerdata_kfold_0.npy")
            data_load1_sr = np.load(f"{self.hparams.data_dir}/innerdata_kfold_1.npy")
            data_load2_sr = np.load(f"{self.hparams.data_dir}/innerdata_kfold_2.npy")
            data_load3_sr = np.load(f"{self.hparams.data_dir}/innerdata_kfold_3.npy")
            data_load4_sr = np.load(f"{self.hparams.data_dir}/innerdata_kfold_4.npy")

            data_load_list = [data_load0, data_load1, data_load2, data_load3, data_load4]
            data_load_list_sr = [
                data_load0_sr,
                data_load1_sr,
                data_load2_sr,
                data_load3_sr,
                data_load4_sr,
            ]

            # data_load_sr = np.concatenate(
            #    (data_load0_sr, data_load1_sr, data_load2_sr, data_load3_sr, data_load4_sr), axis=0
            # )
            #
            # conditioning = np.expand_dims(data_load[:, 0], -1)  # mjj
            # data = data_load[:, 1:]  # features
            #
            # conditioning_sr = np.expand_dims(data_load_sr[:, 0], -1)  # mjj
            # data_sr = data_load_sr[:, 1:]  # features

            data_train = np.concatenate(
                [
                    data_load_list[i]
                    for i in range(5)
                    if i not in self.hparams.val_sets + self.hparams.test_sets
                ],
                axis=0,
            )
            data_val = np.concatenate([data_load_list[i] for i in self.hparams.val_sets], axis=0)
            data_test = np.concatenate([data_load_list[i] for i in self.hparams.test_sets], axis=0)
            data_train_sr = np.concatenate(
                [
                    data_load_list_sr[i]
                    for i in range(5)
                    if i not in self.hparams.val_sets + self.hparams.test_sets
                ],
                axis=0,
            )
            data_val_sr = np.concatenate(
                [data_load_list_sr[i] for i in self.hparams.val_sets], axis=0
            )
            data_test_sr = np.concatenate(
                [data_load_list_sr[i] for i in self.hparams.test_sets], axis=0
            )

            dataset_train = data_train[:, 1:]
            dataset_val = data_val[:, 1:]
            dataset_test = data_test[:, 1:]
            dataset_train_sr = data_train_sr[:, 1:]
            dataset_val_sr = data_val_sr[:, 1:]
            dataset_test_sr = data_test_sr[:, 1:]

            # mjj conditioning
            conditioning_train = np.expand_dims(data_train[:, 0], -1)
            conditioning_val = np.expand_dims(data_val[:, 0], -1)
            conditioning_test = np.expand_dims(data_test[:, 0], -1)
            conditioning_train_sr = np.expand_dims(data_train_sr[:, 0], -1)
            conditioning_val_sr = np.expand_dims(data_val_sr[:, 0], -1)
            conditioning_test_sr = np.expand_dims(data_test_sr[:, 0], -1)

            # data splitting
            # n_samples_val = int(self.hparams.val_fraction * len(data))
            # n_samples_test = int(self.hparams.test_fraction * len(data))
            # n_samples_val_sr = int(self.hparams.val_fraction * len(data_sr))
            # n_samples_test_sr = int(self.hparams.test_fraction * len(data_sr))

            # dataset_train, dataset_val, dataset_test = np.split(
            #    data,
            #    [
            #        len(data) - (n_samples_val + n_samples_test),
            #        len(data) - n_samples_test,
            #    ],
            # )
            # dataset_train_sr, dataset_val_sr, dataset_test_sr = np.split(
            #    data_sr,
            #    [
            #        len(data_sr) - (n_samples_val_sr + n_samples_test_sr),
            #        len(data_sr) - n_samples_test_sr,
            #    ],
            # )
            #
            # conditioning_train, conditioning_val, conditioning_test = np.split(
            #    conditioning,
            #    [
            #        len(conditioning) - (n_samples_val + n_samples_test),
            #        len(conditioning) - n_samples_test,
            #    ],
            # )
            # conditioning_train_sr, conditioning_val_sr, conditioning_test_sr = np.split(
            #    conditioning_sr,
            #    [
            #        len(conditioning_sr) - (n_samples_val_sr + n_samples_test_sr),
            #        len(conditioning_sr) - n_samples_test_sr,
            #    ],
            # )

            tensor_conditioning_train = torch.tensor(conditioning_train, dtype=torch.float)
            tensor_conditioning_val = torch.tensor(conditioning_val, dtype=torch.float)
            tensor_conditioning_test = torch.tensor(conditioning_test, dtype=torch.float)
            tensor_conditioning_train_sr = torch.tensor(conditioning_train_sr, dtype=torch.float)
            tensor_conditioning_val_sr = torch.tensor(conditioning_val_sr, dtype=torch.float)
            tensor_conditioning_test_sr = torch.tensor(conditioning_test_sr, dtype=torch.float)

            if self.hparams.normalize:
                if self.hparams.logit_transform:
                    pipeline = make_pipeline(LogitScaler(), preprocessing.StandardScaler()).fit(
                        dataset_train
                    )
                else:
                    pipeline = make_pipeline(preprocessing.StandardScaler()).fit(dataset_train)

                # means = pipeline.mean_
                # stds = pipeline.scale_

                # if self.hparams.set_data:
                #    means = np.mean(dataset_train, axis=(0, 1))
                #    stds = np.std(dataset_train, axis=(0, 1))
                # else:
                #    means = np.mean(dataset_train, axis=0)
                #    stds = np.std(dataset_train, axis=0)

                means_cond = torch.mean(tensor_conditioning_train, axis=0)
                stds_cond = torch.std(tensor_conditioning_train, axis=0)

                pipeline_cond = make_pipeline(preprocessing.StandardScaler()).fit(
                    tensor_conditioning_train
                )

                self.preprocessing_pipeline = pipeline
                self.preprocessing_pipeline_cond = pipeline_cond
                # Training
                # normalized_dataset_train = normalize_tensor(
                #    np.copy(dataset_train), means, stds, sigma=self.hparams.normalize_sigma
                # )

                normalized_dataset_train = pipeline.transform(dataset_train)

                tensor_train = torch.tensor(normalized_dataset_train, dtype=torch.float)

                # tensor_conditioning_train = normalize_tensor(
                #    tensor_conditioning_train,
                #    means_cond,
                #    stds_cond,
                #    sigma=self.hparams.normalize_sigma,
                # )
                tensor_conditioning_train = torch.tensor(
                    pipeline_cond.transform(tensor_conditioning_train), dtype=torch.float
                )

                # normalized_dataset_train_sr = normalize_tensor(
                #    np.copy(dataset_train_sr), means, stds, sigma=self.hparams.normalize_sigma
                # )
                normalized_dataset_train_sr = pipeline.transform(dataset_train_sr)
                tensor_train_sr = torch.tensor(normalized_dataset_train_sr, dtype=torch.float)
                # tensor_conditioning_train_sr = normalize_tensor(
                #    tensor_conditioning_train_sr,
                #    means_cond,
                #    stds_cond,
                #    sigma=self.hparams.normalize_sigma,
                # )
                tensor_conditioning_train_sr = torch.tensor(
                    pipeline_cond.transform(tensor_conditioning_train_sr), dtype=torch.float
                )

                # Validation
                # normalized_dataset_val = normalize_tensor(
                #    np.copy(dataset_val),
                #    means,
                #    stds,
                #    sigma=self.hparams.normalize_sigma,
                # )
                normalized_dataset_val = pipeline.transform(dataset_val)
                tensor_val = torch.tensor(normalized_dataset_val, dtype=torch.float)

                # tensor_conditioning_val = normalize_tensor(
                #    tensor_conditioning_val,
                #    means_cond,
                #    stds_cond,
                #    sigma=self.hparams.normalize_sigma,
                # )
                tensor_conditioning_val = torch.tensor(
                    pipeline_cond.transform(tensor_conditioning_val), dtype=torch.float
                )

                # normalized_dataset_val_sr = normalize_tensor(
                #    np.copy(dataset_val_sr),
                #    means,
                #    stds,
                #    sigma=self.hparams.normalize_sigma,
                # )
                normalized_dataset_val_sr = pipeline.transform(dataset_val_sr)
                tensor_val_sr = torch.tensor(normalized_dataset_val_sr, dtype=torch.float)
                # tensor_conditioning_val_sr = normalize_tensor(
                #    tensor_conditioning_val_sr,
                #    means_cond,
                #    stds_cond,
                #    sigma=self.hparams.normalize_sigma,
                # )
                tensor_conditioning_val_sr = torch.tensor(
                    pipeline_cond.transform(tensor_conditioning_val_sr), dtype=torch.float
                )

                # Test
                # tensor_conditioning_test = normalize_tensor(
                #    tensor_conditioning_test,
                #    means_cond,
                #    stds_cond,
                #    sigma=self.hparams.normalize_sigma,
                # )
                tensor_conditioning_test = torch.tensor(
                    pipeline_cond.transform(tensor_conditioning_test), dtype=torch.float
                )

                # tensor_conditioning_test_sr = normalize_tensor(
                #    tensor_conditioning_test_sr,
                #    means_cond,
                #    stds_cond,
                #    sigma=self.hparams.normalize_sigma,
                # )
                tensor_conditioning_test_sr = torch.tensor(
                    pipeline_cond.transform(tensor_conditioning_test_sr), dtype=torch.float
                )

            unnormalized_tensor_train = torch.tensor(dataset_train, dtype=torch.float)
            unnormalized_tensor_val = torch.tensor(dataset_val, dtype=torch.float)

            unnormalized_tensor_train_sr = torch.tensor(dataset_train_sr, dtype=torch.float)
            unnormalized_tensor_val_sr = torch.tensor(dataset_val_sr, dtype=torch.float)

            tensor_test = torch.tensor(dataset_test, dtype=torch.float)
            tensor_test_sr = torch.tensor(dataset_test_sr, dtype=torch.float)

            # mask not needed, so just create a tensor of ones
            mask_train = torch.ones_like(tensor_train[..., 0]).unsqueeze(-1)
            mask_val = torch.ones_like(tensor_val[..., 0]).unsqueeze(-1)
            mask_test = torch.ones_like(tensor_test[..., 0]).unsqueeze(-1)

            if self.hparams.normalize:
                print(f"tensor train: {tensor_train.shape}")
                print(f"mask train: {mask_train.shape}")
                print(f"tensor conditioning train: {tensor_conditioning_train.shape}")
                print(f"tensor train dtype {tensor_train.dtype}")
                print(f"mask train dtype {mask_train.dtype}")
                print(f"tensor conditioning train dtype {tensor_conditioning_train.dtype}")

                print(f"Tensor train: {np.count_nonzero(np.isnan(np.array(tensor_train)))}")
                print(f"Mask train: {np.count_nonzero(np.isnan(np.array(mask_train)))}")
                print(
                    "Tensor conditioning train:"
                    f" {np.count_nonzero(np.isnan(np.array(tensor_conditioning_train)))}"
                )
                self.data_train = TensorDataset(
                    tensor_train, mask_train, tensor_conditioning_train
                )
                self.data_val = TensorDataset(tensor_val, mask_val, tensor_conditioning_val)
                self.data_test = TensorDataset(tensor_test, mask_test, tensor_conditioning_test)
                # self.means = torch.tensor(means)
                # self.stds = torch.tensor(stds)
                self.cond_means = means_cond
                self.cond_stds = stds_cond
            else:
                self.data_train = TensorDataset(
                    unnormalized_tensor_train, mask_train, tensor_conditioning_train
                )
                self.data_val = TensorDataset(
                    unnormalized_tensor_val, mask_val, tensor_conditioning_val
                )
                self.data_test = TensorDataset(tensor_test, mask_test, tensor_conditioning_test)

            if self.hparams.verbose:
                # print(f"{len(p4_jets) - len(data)} events are removed due to the window cut.")
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
    _ = GenChallengeDataModule()
