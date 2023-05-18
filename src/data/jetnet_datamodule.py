from typing import Any, Dict, Optional

import numpy as np
import torch
from jetnet.datasets import JetNet
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from .components import center_jets, get_base_distribution, mask_data, normalize_tensor


class JetNetDataModule(LightningDataModule):
    """LightningDataModule for JetNet dataset.

    Args:
        data_dir (str, optional): Path to data directory. Defaults to "data/".
        val_fraction (float, optional): Fraction of data to use for validation. Between 0 and 1. Defaults to 0.15.
        test_fraction (float, optional): Fraction of data to use for testing. Between 0 and 1. Defaults to 0.15.
        batch_size (int, optional): Batch size. Defaults to 256.
        num_workers (int, optional): Number of workers for dataloader. Defaults to 32.
        pin_memory (bool, optional): Pin memory for dataloader. Defaults to False.
        jet_type (str, optional): Type of jets. Options: g, q, t, w, z. Defaults to "t".
        num_particles (number, optional): Number of particles to use (max 150). Defaults to 150.
        variable_jet_sizes (bool, optional): Use variable jet sizes, jets with lesser constituents than num_particles will be zero padded and masked.
        centering (bool, optional): Center the data. Defaults to True.
        normalize (bool, optional): Standardise each feature to have zero mean and normalize_sisgma std deviation. Defaults to True.
        normalize_sigma (int, optional): Number of std deviations to use for normalization. Defaults to 5.
        use_calculated_base_distribution (bool, optional): Calculate mean and covariance of base distribution from data. Defaults to True.

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
        # data
        jet_type: str = "t",
        num_particles: int = 150,
        variable_jet_sizes: bool = True,
        # preprocessing
        centering=True,
        normalize=True,
        normalize_sigma=5,
        use_calculated_base_distribution=True,
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
        self.tensor_test: Optional[torch.Tensor] = None
        self.mask_test: Optional[torch.Tensor] = None
        self.tensor_val: Optional[torch.Tensor] = None
        self.mask_val: Optional[torch.Tensor] = None
        self.x_mean: Optional[torch.Tensor] = None
        self.x_cov: Optional[torch.Tensor] = None

    @property
    def num_classes(self):
        pass

    def get_data_args(self) -> Dict[str, Any]:
        if self.hparams.num_particles != 30 and self.hparams.num_particles != 150:
            if self.hparams.num_particles > 150:
                raise NotImplementedError
            else:
                load_num_particles = 150
        else:
            load_num_particles = self.hparams.num_particles
        data_args = {
            "jet_type": [self.hparams.jet_type],
            "data_dir": f"{self.hparams.data_dir}/jetnet",
            "particle_features": ["etarel", "phirel", "ptrel", "mask"],
            "num_particles": load_num_particles,
            "jet_features": ["type", "pt", "eta", "mass", "num_particles"],
        }
        return data_args

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        data_args = self.get_data_args()
        JetNet.getData(**data_args)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # data loading
            data_args = self.get_data_args()
            particle_data, jet_data = JetNet.getData(**data_args)

            # centering
            if self.hparams.centering:
                mask = particle_data[..., 3]
                particle_data = center_jets(particle_data[..., :3])
                particle_data = np.append(particle_data, np.expand_dims(mask, axis=-1), axis=-1)

            x, mask, masked_particle_data, masked_jet_data = mask_data(
                particle_data,
                jet_data,
                num_particles=self.hparams.num_particles,
                variable_jet_sizes=self.hparams.variable_jet_sizes,
            )

            x_mean, x_cov = get_base_distribution(
                x,
                mask,
                use_calculated_base_distribution=self.hparams.use_calculated_base_distribution,
            )

            n_samples_val = int(self.hparams.val_fraction * len(x))
            n_samples_test = int(self.hparams.test_fraction * len(x))
            full_mask = np.repeat(mask, repeats=3, axis=-1) == 0
            full_mask = np.ma.make_mask(full_mask, shrink=False)
            x_ma = np.ma.masked_array(x, full_mask)
            dataset_train, dataset_val, dataset_test = np.split(
                x_ma,
                [
                    len(x_ma) - (n_samples_val + n_samples_test),
                    len(x_ma) - n_samples_val,
                ],
            )

            if self.hparams.normalize:
                means = np.ma.mean(dataset_train, axis=(0, 1))
                stds = np.ma.std(dataset_train, axis=(0, 1))

                normalized_dataset_train = normalize_tensor(
                    dataset_train, means, stds, sigma=self.hparams.normalize_sigma
                )
                mask_train = np.ma.getmask(normalized_dataset_train) == 0
                mask_train = mask_train.astype(int)
                mask_train = torch.tensor(np.expand_dims(mask_train[..., 0], axis=-1))
                tensor_train = torch.tensor(normalized_dataset_train)

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

            # Validation without normalization
            unnormalized_tensor_val = torch.tensor(dataset_val)
            unnormalized_mask_val = np.ma.getmask(dataset_val) == 0
            unnormalized_mask_val = unnormalized_mask_val.astype(int)
            unnormalized_mask_val = torch.tensor(
                np.expand_dims(unnormalized_mask_val[..., 0], axis=-1)
            )

            # Test
            tensor_test = torch.tensor(dataset_test)
            mask_test = np.ma.getmask(dataset_test) == 0
            mask_test = mask_test.astype(int)
            mask_test = torch.tensor(np.expand_dims(mask_test[..., 0], axis=-1))

            if self.hparams.normalize:
                self.data_train = TensorDataset(tensor_train, mask_train)
                self.data_val = TensorDataset(tensor_val, mask_val)
                self.data_test = TensorDataset(tensor_test, mask_test)

                self.means = torch.tensor(means)
                self.stds = torch.tensor(stds)
            else:
                dataset = TensorDataset(x, mask)
                (self.data_train, self.data_val, self.data_test,) = random_split(
                    dataset,
                    [
                        len(x) - (n_samples_val + n_samples_test),
                        n_samples_val,
                        n_samples_test,
                    ],
                )

                self.means = None
                self.stds = None

            self.tensor_test = tensor_test
            self.mask_test = mask_test
            self.tensor_val = unnormalized_tensor_val
            self.mask_val = unnormalized_mask_val
            self.x_mean = x_mean
            self.x_cov = x_cov

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
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
    _ = JetNetDataModule()
