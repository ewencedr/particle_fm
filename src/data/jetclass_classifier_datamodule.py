from typing import Any, Dict, Optional, Tuple

import energyflow as ef
import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset


class JetClassClassifierDataModule(LightningDataModule):
    """Data module for jet classification task (classifier test)."""

    def __init__(
        self,
        data_dir: str = "data/",
        data_file: str = None,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.6, 0.2, 0.2),
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()
        if data_file is None:
            raise ValueError("data_file must be specified")
        if np.sum(train_val_test_split) != 1:
            raise ValueError("train_val_test_split must sum to 1")

    def prepare_data(self) -> None:
        """Prepare the data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`, so be
        careful not to execute things like random split twice! Also, it is
        called after `self.prepare_data()` and there is a barrier in between
        which ensures that all the processes proceed to `self.setup()` once the
        data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`,
        `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Load data from the data_file (that's the output from `eval_ckpt.py`)
        with h5py.File(self.hparams.data_file, "r") as h5file:
            data_gen = h5file["part_data_gen"][:]
            mask_gen = h5file["part_mask_gen"][:]
            label_gen = np.ones(len(data_gen))
            # cond_gen = h5file["cond_data_gen"][:]
            data_sim = h5file["part_data_sim"][:]
            mask_sim = h5file["part_mask_sim"][:]
            label_sim = np.zeros(len(data_sim))
            # cond_sim = h5file["cond_data_sim"][:]
            x_features = np.concatenate([data_gen, data_sim])
            # use the first three particle features as coordinates (etarel, phirel, ptrel)
            x_coords = x_features[:, :, :3]
            x_mask = np.concatenate([mask_gen, mask_sim])
            y = np.concatenate([label_gen, label_sim])

            # shuffle data
            permutation = np.random.permutation(len(x_features))
            x_features = x_features[permutation]
            x_coords = x_coords[permutation]
            x_mask = x_mask[permutation]
            y = y[permutation]

        # Swap indices to get required shape for ParticleNet: (batch, features, particles)
        x_coords = np.swapaxes(x_coords, 1, 2)
        x_features = np.swapaxes(x_features, 1, 2)
        x_mask = np.swapaxes(x_mask, 1, 2)

        # Split data into train, val, test
        fractions = self.hparams.train_val_test_split
        total_length = len(x_features)
        split_indices = [
            int(fractions[0] * total_length),
            int((fractions[0] + fractions[1]) * total_length),
        ]
        print(f"Splitting data into {split_indices} for train, val, test")

        x_features_train, x_features_val, x_features_test = np.split(x_features, split_indices)
        x_coords_train, x_coords_val, x_coords_test = np.split(x_coords, split_indices)
        x_mask_train, x_mask_val, x_mask_test = np.split(x_mask, split_indices)
        y_train, y_val, y_test = np.split(y, split_indices)

        # Create datasets
        self.data_train = TensorDataset(
            torch.tensor(x_features_train, dtype=torch.float32),
            torch.tensor(x_coords_train, dtype=torch.float32),
            torch.tensor(x_mask_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        self.data_val = TensorDataset(
            torch.tensor(x_features_val, dtype=torch.float32),
            torch.tensor(x_coords_val, dtype=torch.float32),
            torch.tensor(x_mask_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        self.data_test = TensorDataset(
            torch.tensor(x_features_test, dtype=torch.float32),
            torch.tensor(x_coords_test, dtype=torch.float32),
            torch.tensor(x_mask_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

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
