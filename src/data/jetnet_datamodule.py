from typing import Any, Dict, Optional

import numpy as np
import torch
from jetnet.datasets import JetNet
from particle_fm.utils.pylogger import get_pylogger
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .components import (
    center_jets,
    get_base_distribution,
    mask_data,
    normalize_tensor,
    one_hot_encode,
)

log = get_pylogger("JetNetDataModule")


class JetNetDataModule(LightningDataModule):
    """LightningDataModule for JetNet dataset. If no conditioning is used, the conditioning tensor
    will be a tensor of zeros.

    Args:
        data_dir (str, optional): Path to data directory. Defaults to "data/".
        val_fraction (float, optional): Fraction of data to use for validation. Between 0 and 1. Defaults to 0.15.
        test_fraction (float, optional): Fraction of data to use for testing. Between 0 and 1. Defaults to 0.15.
        batch_size (int, optional): Batch size. Defaults to 256.
        num_workers (int, optional): Number of workers for dataloader. Defaults to 32.
        pin_memory (bool, optional): Pin memory for dataloader. Defaults to False.
        drop_last (bool, optional): Drop last batch for train and val dataloader. Defaults to False.
        verbose (bool, optional): Verbose. Defaults to False.
        jet_type (str | list[str],  optional): Type of jets. Options: g, q, t, w, z. Defaults to "t".
        num_particles (number, optional): Number of particles to use (max 150). Defaults to 150.
        variable_jet_sizes (bool, optional): Use variable jet sizes, jets with lesser constituents than num_particles will be zero padded and masked.
        conditioning_type (bool, optional): Condition on jet type. Defaults to True.
        conditioning_pt (bool, optional): Condition on jet pt. Defaults to True.
        conditioning_eta (bool, optional): Condition on jet eta. Defaults to True.
        conditioning_mass (bool, optional): Condition on jet mass. Defaults to True.
        conditioning_num_particles (bool, optional): Condition on jet num_particles. Defaults to True.
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
        drop_last: bool = False,
        verbose: bool = True,
        # data
        jet_type: str | list[str] = "t",
        num_particles: int = 150,
        variable_jet_sizes: bool = True,
        conditioning_type: bool = True,
        conditioning_pt: bool = True,
        conditioning_eta: bool = True,
        conditioning_mass: bool = True,
        conditioning_num_particles: bool = True,
        # preprocessing
        centering: bool = True,
        normalize: bool = True,
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
        self.x_mean: Optional[torch.Tensor] = None
        self.x_cov: Optional[torch.Tensor] = None
        self.tensor_conditioning_train: Optional[torch.Tensor] = None
        self.tensor_conditioning_val: Optional[torch.Tensor] = None
        self.tensor_conditioning_test: Optional[torch.Tensor] = None

    @property
    def num_classes(self):
        return len(self.hparams.jet_type)

    @property
    def jet_types(self):
        return self.hparams.jet_type

    @property
    def num_cond_features(self):
        return sum(
            [
                self.hparams.conditioning_pt,
                self.hparams.conditioning_eta,
                self.hparams.conditioning_mass,
                self.hparams.conditioning_num_particles,
            ]
            + [self.hparams.conditioning_type] * self.num_classes
        )

    def get_data_args(self) -> Dict[str, Any]:
        if self.hparams.num_particles != 30 and self.hparams.num_particles != 150:
            if self.hparams.num_particles > 150:
                raise NotImplementedError
            else:
                load_num_particles = 150
        else:
            load_num_particles = self.hparams.num_particles
        data_args = {
            "jet_type": self.hparams.jet_type,
            "data_dir": f"{self.hparams.data_dir}/jetnet",
            "particle_features": ["etarel", "phirel", "ptrel", "mask"],
            "num_particles": load_num_particles,
            "jet_features": ["type", "pt", "eta", "mass", "num_particles"],
            "split": "all",
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

            # mask and select number of particles, mainly relevant for smaller jet sizes
            x, mask, masked_particle_data, masked_jet_data = mask_data(
                particle_data,
                jet_data,
                num_particles=self.hparams.num_particles,
                variable_jet_sizes=self.hparams.variable_jet_sizes,
            )

            # get mean and covariance of noise distribution
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
                    len(x_ma) - 1 - (n_samples_val + n_samples_test),
                    len(x_ma) - 1 - n_samples_test,
                ],
            )

            # conditioning
            conditioning_data = self._handle_conditioning(jet_data)
            if conditioning_data is not None:
                tensor_conditioning = torch.tensor(conditioning_data, dtype=torch.float32)
                conditioning_train, conditioning_val, conditioning_test = np.split(
                    conditioning_data,
                    [
                        len(conditioning_data) - 1 - (n_samples_val + n_samples_test),
                        len(conditioning_data) - 1 - n_samples_test,
                    ],
                )
                tensor_conditioning_train = torch.tensor(conditioning_train, dtype=torch.float32)
                tensor_conditioning_val = torch.tensor(conditioning_val, dtype=torch.float32)
                tensor_conditioning_test = torch.tensor(conditioning_test, dtype=torch.float32)

                if len(tensor_conditioning) != len(x_ma):
                    raise ValueError("Conditioning tensor and data tensor must have same length.")
            else:
                tensor_conditioning_train = torch.zeros(len(dataset_train))
                tensor_conditioning_val = torch.zeros(len(dataset_val))
                tensor_conditioning_test = torch.zeros(len(dataset_test))

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

                    # Validation
                    tensor_conditioning_val = normalize_tensor(
                        tensor_conditioning_val,
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

            # Train without normalization
            unnormalized_tensor_train = torch.tensor(dataset_train)
            unnormalized_mask_train = np.ma.getmask(dataset_train) == 0
            unnormalized_mask_train = unnormalized_mask_train.astype(int)
            unnormalized_mask_train = torch.tensor(
                np.expand_dims(unnormalized_mask_train[..., 0], axis=-1)
            )

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
                log.info(
                    "Conditioning on"
                    f" {tensor_conditioning_train.shape[-1] if len(tensor_conditioning_train.shape)==2 else 0} variables,"
                    " consisting of jet_type:"
                    f" {len(self.hparams.jet_type) if self.hparams.conditioning_type else 0}, pt:"
                    f" {1 if self.hparams.conditioning_pt else 0}, eta:"
                    f" {1 if self.hparams.conditioning_eta else 0}, mass:"
                    f" {1 if self.hparams.conditioning_mass else 0}, num_particles:"
                    f" {1 if self.hparams.conditioning_num_particles else 0}"
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
            self.x_mean = x_mean
            self.x_cov = x_cov
            self.tensor_conditioning_train = tensor_conditioning_train
            self.tensor_conditioning_val = tensor_conditioning_val
            self.tensor_conditioning_test = tensor_conditioning_test

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

        # get the integer categories which classify the jets into different types
        categories = np.unique(jet_data[:, 0])
        jet_data_one_hot = one_hot_encode(
            jet_data, categories=[categories], num_other_features=jet_data.shape[1] - 1
        )

        if (
            not self.hparams.conditioning_type
            and not self.hparams.conditioning_pt
            and not self.hparams.conditioning_eta
            and not self.hparams.conditioning_mass
            and not self.hparams.conditioning_num_particles
        ):
            return None

        # select the columns which correspond to the conditioning variables that should be used
        one_hot_len = len(categories)
        keep_col = []
        if self.hparams.conditioning_type:
            keep_col.append(np.arange(one_hot_len))
        if self.hparams.conditioning_pt:
            keep_col.append(np.arange(one_hot_len, one_hot_len + 1))
        if self.hparams.conditioning_eta:
            keep_col.append(np.arange(one_hot_len + 1, one_hot_len + 2))
        if self.hparams.conditioning_mass:
            keep_col.append(np.arange(one_hot_len + 2, one_hot_len + 3))
        if self.hparams.conditioning_num_particles:
            keep_col.append(np.arange(one_hot_len + 3, one_hot_len + 4))
        keep_col = np.concatenate(keep_col)

        return jet_data_one_hot[:, keep_col]


if __name__ == "__main__":
    _ = JetNetDataModule()
