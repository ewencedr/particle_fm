from typing import Any, Dict, Optional

import energyflow as ef
import h5py
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from src.utils.pylogger import get_pylogger

from .components import (
    center_jets,
    get_base_distribution,
    mask_data,
    normalize_tensor,
    one_hot_encode,
)

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
        # data
        file_suffix_processed_data: str = "",
        num_particles: int = 279,
        variable_jet_sizes: bool = True,
        conditioning: bool = False,
        relative_coords: bool = True,
        jet_type: str = "x",
        use_all_data: bool = False,
        shuffle_data: bool = True,
        window_left: float = 3.3e3,
        window_right: float = 3.7e3,
        # preprocessing
        centering: bool = False,
        normalize: bool = False,
        normalize_sigma: int = 5,
        use_calculated_base_distribution: bool = True,
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
        self.mask_train_sr: Optional[torch.Tensor] = None
        self.tensor_test_sr: Optional[torch.Tensor] = None
        self.mask_test_sr: Optional[torch.Tensor] = None
        self.tensor_val_sr: Optional[torch.Tensor] = None
        self.mask_val_sr: Optional[torch.Tensor] = None
        self.tensor_conditioning_train_sr: Optional[torch.Tensor] = None
        self.tensor_conditioning_val_sr: Optional[torch.Tensor] = None
        self.tensor_conditioning_test_sr: Optional[torch.Tensor] = None

        self.jet_data_sr_raw: Optional[np.array] = None
        self.particle_data_sr_raw: Optional[np.array] = None
        self.mask_sr_raw: Optional[np.array] = None

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
            if self.hparams.use_all_data:
                path = f"{self.hparams.data_dir}/lhco/events_anomalydetection_v2.h5"
                df = pd.read_hdf(path)
                df_np = np.array(df)
                background = df_np[df_np[:, 2100] == 0]
                particle_data = background[:, :2100].reshape(-1, 700, 3)[:, :560, :]
                mask = np.expand_dims((particle_data[..., 0] > 0).astype(int), axis=-1)

                jet_data = None
            else:
                if self.hparams.relative_coords:
                    path = f"{self.hparams.data_dir}/lhco/final_data/processed_data_background_rel{self.hparams.file_suffix_processed_data}.h5"
                else:
                    path = f"{self.hparams.data_dir}/lhco/final_data/processed_data_background{self.hparams.file_suffix_processed_data}.h5"
                with h5py.File(path, "r") as f:
                    jet_data = f["jet_data"][:]
                    particle_data = f["constituents"][:]
                    mask = f["mask"][:]

                # cut mjj window
                p4_jets = ef.p4s_from_ptyphims(jet_data)
                # get mjj from p4_jets
                sum_p4 = p4_jets[:, 0] + p4_jets[:, 1]
                mjj = ef.ms_from_p4s(sum_p4)

                # args_to_remove = (mjj >= self.hparams.window_left) & (
                #    mjj <= self.hparams.window_right
                # )
                # args_to_remove = (mjj > 5000) | (mjj < 2300) | ((mjj > 3300) & (mjj < 3700))

                jet_data2 = jet_data.copy()
                particle_data2 = particle_data.copy()
                mask2 = mask.copy()

                args_to_keep = ((mjj < 3300) & (mjj > 2300)) | ((mjj > 3700) & (mjj < 5000))
                jet_data = jet_data[args_to_keep]
                particle_data = particle_data[args_to_keep]
                mask = mask[args_to_keep]

                # sr
                args_to_keep_sr = (mjj > 3300) & (mjj < 3700)
                jet_data_sr = jet_data2[args_to_keep_sr]
                particle_data_sr = particle_data2[args_to_keep_sr]
                mask_sr = mask2[args_to_keep_sr]

                if self.hparams.jet_type == "all_one_pc":
                    particle_data = particle_data.reshape(
                        particle_data.shape[0], -1, particle_data.shape[-1]
                    )
                    mask = mask.reshape(mask.shape[0], -1, mask.shape[-1])
                    if self.hparams.conditioning:
                        raise ValueError("Conditioning does not make sense for one_pc")
                elif self.hparams.jet_type == "all":
                    jet_data = np.reshape(jet_data, (-1, jet_data.shape[-1]), order="F")
                    particle_data = np.reshape(
                        particle_data,
                        (-1, particle_data.shape[-2], particle_data.shape[-1]),
                        order="F",
                    )
                    mask = np.reshape(mask, (-1, mask.shape[-2], mask.shape[-1]), order="F")

                    self.jet_data_sr_raw = jet_data_sr.copy()
                    self.particle_data_sr_raw = particle_data_sr.copy()
                    self.mask_sr_raw = mask_sr.copy()

                    jet_data_sr = np.reshape(jet_data_sr, (-1, jet_data_sr.shape[-1]), order="F")
                    particle_data_sr = np.reshape(
                        particle_data_sr,
                        (-1, particle_data_sr.shape[-2], particle_data_sr.shape[-1]),
                        order="F",
                    )
                    mask_sr = np.reshape(
                        mask_sr, (-1, mask_sr.shape[-2], mask_sr.shape[-1]), order="F"
                    )
                elif self.hparams.jet_type == "x":
                    particle_data = particle_data[:, 0]
                    mask = mask[:, 0]
                    jet_data = jet_data[:, 0]
                    # sr
                    particle_data_sr = particle_data_sr[:, 0]
                    mask_sr = mask_sr[:, 0]
                    jet_data_sr = jet_data_sr[:, 0]
                elif self.hparams.jet_type == "y":
                    particle_data = particle_data[:, 1]
                    mask = mask[:, 1]
                    jet_data = jet_data[:, 1]
                    # sr
                    particle_data_sr = particle_data_sr[:, 1]
                    mask_sr = mask_sr[:, 1]
                    jet_data_sr = jet_data_sr[:, 1]
                else:
                    raise ValueError("Unknown jet type")

            # reorder to eta, phi, pt to match the order of jetnet
            particle_data = particle_data[:, :, [1, 2, 0]]
            particle_data = np.concatenate([particle_data, mask], axis=-1)
            particle_data_sr = particle_data_sr[:, :, [1, 2, 0]]
            particle_data_sr = np.concatenate([particle_data_sr, mask_sr], axis=-1)

            # shuffle data
            if self.hparams.shuffle_data:
                perm = np.random.permutation(len(particle_data))
                if jet_data is not None and len(jet_data) == len(particle_data):
                    jet_data = jet_data[perm]
                particle_data = particle_data[perm]

                perm_sr = np.random.permutation(len(particle_data_sr))
                if jet_data_sr is not None and len(jet_data_sr) == len(particle_data_sr):
                    jet_data_sr = jet_data_sr[perm_sr]
                particle_data_sr = particle_data_sr[perm_sr]

            # mask and select number of particles, mainly relevant for smaller jet sizes
            x, mask, masked_particle_data, masked_jet_data = mask_data(
                particle_data,
                jet_data,
                num_particles=self.hparams.num_particles,
                variable_jet_sizes=self.hparams.variable_jet_sizes,
            )
            x_sr, mask_sr, masked_particle_data_sr, masked_jet_data_sr = mask_data(
                particle_data_sr,
                jet_data_sr,
                num_particles=self.hparams.num_particles,
                variable_jet_sizes=self.hparams.variable_jet_sizes,
            )

            # data splitting
            n_samples_val = int(self.hparams.val_fraction * len(x))
            n_samples_test = int(self.hparams.test_fraction * len(x))
            n_samples_val_sr = int(self.hparams.val_fraction * len(x_sr))
            n_samples_test_sr = int(self.hparams.test_fraction * len(x_sr))

            full_mask = np.repeat(mask, repeats=3, axis=-1) == 0
            full_mask = np.ma.make_mask(full_mask, shrink=False)
            x_ma = np.ma.masked_array(x, full_mask)
            dataset_train, dataset_val, dataset_test = np.split(
                x_ma,
                [
                    len(x_ma) - (n_samples_val + n_samples_test),
                    len(x_ma) - n_samples_test,
                ],
            )
            full_mask_sr = np.repeat(mask_sr, repeats=3, axis=-1) == 0
            full_mask_sr = np.ma.make_mask(full_mask_sr, shrink=False)
            x_ma_sr = np.ma.masked_array(x_sr, full_mask_sr)
            dataset_train_sr, dataset_val_sr, dataset_test_sr = np.split(
                x_ma_sr,
                [
                    len(x_ma_sr) - (n_samples_val_sr + n_samples_test_sr),
                    len(x_ma_sr) - n_samples_test_sr,
                ],
            )

            # conditioning
            conditioning_data = self._handle_conditioning(jet_data)
            conditioning_data_sr = jet_data_sr
            if conditioning_data is not None:
                tensor_conditioning = torch.tensor(conditioning_data, dtype=torch.float32)
                conditioning_train, conditioning_val, conditioning_test = np.split(
                    conditioning_data,
                    [
                        len(conditioning_data) - (n_samples_val + n_samples_test),
                        len(conditioning_data) - n_samples_test,
                    ],
                )
                tensor_conditioning_train = torch.tensor(conditioning_train, dtype=torch.float32)
                tensor_conditioning_val = torch.tensor(conditioning_val, dtype=torch.float32)
                tensor_conditioning_test = torch.tensor(conditioning_test, dtype=torch.float32)
                if len(tensor_conditioning) != len(x_ma):
                    raise ValueError("Conditioning tensor and data tensor must have same length.")

                tensor_conditioning_sr = torch.tensor(conditioning_data_sr, dtype=torch.float32)
                conditioning_train_sr, conditioning_val_sr, conditioning_test_sr = np.split(
                    conditioning_data_sr,
                    [
                        len(conditioning_data_sr) - (n_samples_val_sr + n_samples_test_sr),
                        len(conditioning_data_sr) - n_samples_test_sr,
                    ],
                )
                tensor_conditioning_train_sr = torch.tensor(
                    conditioning_train_sr, dtype=torch.float32
                )
                tensor_conditioning_val_sr = torch.tensor(conditioning_val_sr, dtype=torch.float32)
                tensor_conditioning_test_sr = torch.tensor(
                    conditioning_test_sr, dtype=torch.float32
                )

            else:
                tensor_conditioning_train = torch.zeros(len(dataset_train))
                tensor_conditioning_val = torch.zeros(len(dataset_val))
                tensor_conditioning_test = torch.zeros(len(dataset_test))

                tensor_conditioning_train_sr = torch.zeros(len(dataset_train_sr))
                tensor_conditioning_val_sr = torch.zeros(len(dataset_val_sr))
                tensor_conditioning_test_sr = torch.zeros(len(dataset_test_sr))

            if self.hparams.normalize:
                means = np.ma.mean(dataset_train, axis=(0, 1))
                stds = np.ma.std(dataset_train, axis=(0, 1))

                normalized_dataset_train = normalize_tensor(
                    np.ma.copy(dataset_train), means, stds, sigma=self.hparams.normalize_sigma
                )
                mask_train = np.ma.getmask(normalized_dataset_train) == 0
                mask_train = mask_train.astype(int)
                mask_train = torch.tensor(np.expand_dims(mask_train[..., 0], axis=-1))
                tensor_train = torch.tensor(normalized_dataset_train)

                normalized_dataset_train_sr = normalize_tensor(
                    np.ma.copy(dataset_train_sr), means, stds, sigma=self.hparams.normalize_sigma
                )
                mask_train_sr = np.ma.getmask(normalized_dataset_train_sr) == 0
                mask_train_sr = mask_train_sr.astype(int)
                mask_train_sr = torch.tensor(np.expand_dims(mask_train_sr[..., 0], axis=-1))
                tensor_train_sr = torch.tensor(normalized_dataset_train_sr)

                # Validation
                normalized_dataset_val = normalize_tensor(
                    np.ma.copy(dataset_val),
                    means,
                    stds,
                    sigma=self.hparams.normalize_sigma,
                )
                mask_val = np.ma.getmask(normalized_dataset_val) == 0
                mask_val = mask_val.astype(int)
                mask_val = torch.tensor(np.expand_dims(mask_val[..., 0], axis=-1))
                tensor_val = torch.tensor(normalized_dataset_val)

                normalized_dataset_val_sr = normalize_tensor(
                    np.ma.copy(dataset_val_sr),
                    means,
                    stds,
                    sigma=self.hparams.normalize_sigma,
                )
                mask_val_sr = np.ma.getmask(normalized_dataset_val_sr) == 0
                mask_val_sr = mask_val_sr.astype(int)
                mask_val_sr = torch.tensor(np.expand_dims(mask_val_sr[..., 0], axis=-1))
                tensor_val_sr = torch.tensor(normalized_dataset_val_sr)

                if conditioning_data is not None:
                    means_cond = torch.mean(tensor_conditioning_train, axis=0)
                    stds_cond = torch.std(tensor_conditioning_train, axis=0)

                    # Train
                    tensor_conditioning_train = normalize_tensor(
                        tensor_conditioning_train,
                        means_cond,
                        stds_cond,
                        sigma=self.hparams.normalize_sigma,
                    )
                    tensor_conditioning_train_sr = normalize_tensor(
                        tensor_conditioning_train_sr,
                        means_cond,
                        stds_cond,
                        sigma=self.hparams.normalize_sigma,
                    )

                    # Validation
                    tensor_conditioning_val = normalize_tensor(
                        tensor_conditioning_val,
                        means_cond,
                        stds_cond,
                        sigma=self.hparams.normalize_sigma,
                    )
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

            # Train without normalization
            unnormalized_tensor_train = torch.tensor(dataset_train)
            unnormalized_mask_train = np.ma.getmask(dataset_train) == 0
            unnormalized_mask_train = unnormalized_mask_train.astype(int)
            unnormalized_mask_train = torch.tensor(
                np.expand_dims(unnormalized_mask_train[..., 0], axis=-1)
            )
            unnormalized_tensor_train_sr = torch.tensor(dataset_train_sr)
            unnormalized_mask_train_sr = np.ma.getmask(dataset_train_sr) == 0
            unnormalized_mask_train_sr = unnormalized_mask_train_sr.astype(int)
            unnormalized_mask_train_sr = torch.tensor(
                np.expand_dims(unnormalized_mask_train_sr[..., 0], axis=-1)
            )

            # Validation without normalization
            unnormalized_tensor_val = torch.tensor(dataset_val)
            unnormalized_mask_val = np.ma.getmask(dataset_val) == 0
            unnormalized_mask_val = unnormalized_mask_val.astype(int)
            unnormalized_mask_val = torch.tensor(
                np.expand_dims(unnormalized_mask_val[..., 0], axis=-1)
            )
            unnormalized_tensor_val_sr = torch.tensor(dataset_val_sr)
            unnormalized_mask_val_sr = np.ma.getmask(dataset_val_sr) == 0
            unnormalized_mask_val_sr = unnormalized_mask_val_sr.astype(int)
            unnormalized_mask_val_sr = torch.tensor(
                np.expand_dims(unnormalized_mask_val_sr[..., 0], axis=-1)
            )

            # Test
            tensor_test = torch.tensor(dataset_test)
            mask_test = np.ma.getmask(dataset_test) == 0
            mask_test = mask_test.astype(int)
            mask_test = torch.tensor(np.expand_dims(mask_test[..., 0], axis=-1))
            tensor_test_sr = torch.tensor(dataset_test_sr)
            mask_test_sr = np.ma.getmask(dataset_test_sr) == 0
            mask_test_sr = mask_test_sr.astype(int)
            mask_test_sr = torch.tensor(np.expand_dims(mask_test_sr[..., 0], axis=-1))

            if self.hparams.normalize:
                self.data_train = TensorDataset(
                    tensor_train, mask_train, tensor_conditioning_train
                )
                self.data_val = TensorDataset(tensor_val, mask_val, tensor_conditioning_val)
                self.data_test = TensorDataset(tensor_test, mask_test, tensor_conditioning_test)

                self.means = torch.tensor(means)
                self.stds = torch.tensor(stds)

                if conditioning_data is not None:
                    self.cond_means = means_cond
                    self.cond_stds = stds_cond
            else:
                self.data_train = TensorDataset(
                    unnormalized_tensor_train, unnormalized_mask_train, tensor_conditioning_train
                )
                self.data_val = TensorDataset(
                    unnormalized_tensor_val, unnormalized_mask_val, tensor_conditioning_val
                )
                self.data_test = TensorDataset(tensor_test, mask_test, tensor_conditioning_test)

                self.means = None
                self.stds = None

            if self.hparams.verbose:
                log.info(f"Data of jet types {self.hparams.jet_type} loaded.")
                # log.info(
                #    f"Conditioning on {tensor_conditioning_train.shape[-1] if len(tensor_conditioning_train.shape)==2 else 0} variables, consisting of jet_type: {len(self.hparams.jet_type) if self.hparams.conditioning_type else 0}, pt: {1 if self.hparams.conditioning_pt else 0}, eta: {1 if self.hparams.conditioning_eta else 0}, mass: {1 if self.hparams.conditioning_mass else 0}, num_particles: {1 if self.hparams.conditioning_num_particles else 0}"
                # )
                if self.hparams.conditioning:
                    log.info(
                        f"Conditioning on {tensor_conditioning_train.shape[-1]} jet variables (pt,"
                        " eta, phi, mass)"
                    )
                if self.hparams.normalize:
                    log.info(f"{'Training data shape:':<23} {tensor_train.shape}")
                    log.info(f"{'Validation data shape:':<23} {tensor_val.shape}")
                    log.info(f"{'Test data shape:':<23} {tensor_test.shape}")
                    log.info(f"Normalizing data with sigma = {self.hparams.normalize_sigma}")
                if self.hparams.centering:
                    log.info("Centering data")

            self.tensor_train = unnormalized_tensor_train
            self.mask_train = unnormalized_mask_train
            self.tensor_test = tensor_test
            self.mask_test = mask_test
            self.tensor_val = unnormalized_tensor_val
            self.mask_val = unnormalized_mask_val
            self.tensor_conditioning_train = tensor_conditioning_train
            self.tensor_conditioning_val = tensor_conditioning_val
            self.tensor_conditioning_test = tensor_conditioning_test

            self.tensor_train_sr = unnormalized_tensor_train_sr
            self.mask_train_sr = unnormalized_mask_train_sr
            self.tensor_test_sr = tensor_test_sr
            self.mask_test_sr = mask_test_sr
            self.tensor_val_sr = unnormalized_tensor_val_sr
            self.mask_val_sr = unnormalized_mask_val_sr
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

    def _handle_conditioning(self, jet_data: np.array):
        """Select the conditioning variables and one-hot encode the type conditioning of jets."""

        if self.hparams.conditioning:
            return jet_data
        else:
            return None


if __name__ == "__main__":
    _ = LHCODataModule()
