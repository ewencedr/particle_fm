from typing import Any, Dict, Optional, Tuple

import awkward as ak
import energyflow as ef
import h5py
import numpy as np
import torch
import vector
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

        # Features needed for ParT
        # part_etarel
        # part_phirel
        # log ( part_pt )
        # log ( part_energy )
        # log ( part_ptrel )
        # log ( part_energyrel )
        # part_deltaR
        # part_charge
        # part_isElectron
        # part_isMuon
        # part_isPhoton
        # part_isChargedHadron
        # part_isNeutralHadron
        # tanh ( part_d0val)
        # tanh ( part_dzval)
        # part_d0err
        # part_dzerr

        # Load data from the data_file (that's the output from `eval_ckpt.py`)
        with h5py.File(self.hparams.data_file, "r") as h5file:
            part_names = list(h5file["part_data_sim"].attrs["names"])
            cond_names = list(h5file["cond_data_sim"].attrs["names"])
            # jet_names = list(h5file["jet_data_sim"].attrs["names"])

            def idx_part(feat_name):
                return part_names.index(feat_name)

            def idx_cond(feat_name):
                return cond_names.index(feat_name)

            print("part_names:", part_names)
            print("cond_names:", cond_names)

            print("part index (part_etarel):", idx_part("part_etarel"))
            print("cond index (jet_type_label_Tbqq):", idx_cond("jet_type_label_Tbqq"))

            data_gen = h5file["part_data_gen"][:]
            mask_gen = h5file["part_mask_gen"][:]
            cond_gen = h5file["cond_data_gen"][:]
            label_gen = np.ones(len(data_gen))

            data_sim = h5file["part_data_sim"][:]
            mask_sim = h5file["part_mask_sim"][:]
            cond_sim = h5file["cond_data_sim"][:]
            label_sim = np.zeros(len(data_sim))

            x_features = np.concatenate([data_gen, data_sim])
            cond_features = np.concatenate([cond_gen, cond_sim])
            # use the first three particle features as coordinates (etarel, phirel, ptrel)
            # TODO: probably not needed for ParT (only for ParticleNet)
            pf_points = x_features[:, :, :3]
            pf_mask = np.concatenate([mask_gen, mask_sim])
            y = np.concatenate([label_gen, label_sim])

            # create the features array as needed by ParT
            pf_features = np.concatenate(
                [
                    # log ( part_pt )
                    (
                        (
                            np.log(
                                x_features[:, :, idx_part("part_ptrel")]
                                * cond_features[:, idx_cond("jet_pt")][:, None]
                            )
                        )[..., None]
                        - 1.7
                    )
                    * 0.7,
                    # log ( part_energy )
                    (
                        (
                            np.log(
                                x_features[:, :, idx_part("part_energyrel")]
                                * cond_features[:, idx_cond("jet_energy")][:, None]
                            )
                        )[..., None]
                        - 2.0
                    )
                    * 0.7,
                    # log ( part_ptrel )
                    (np.log(x_features[:, :, idx_part("part_ptrel")])[..., None] + 4.7) * 0.7,
                    # log ( part_energyrel )
                    (np.log(x_features[:, :, idx_part("part_energyrel")])[..., None] + 4.7) * 0.7,
                    # part_deltaR
                    np.clip(
                        (
                            np.hypot(
                                x_features[:, :, idx_part("part_etarel")],
                                x_features[:, :, idx_part("part_dphi")],
                            )[..., None]
                            - 0.2
                        )
                        * 4.0,
                        -5,
                        5,
                    )
                    * pf_mask[:, :, 0][..., None],
                    x_features[:, :, idx_part("part_charge")][..., None],
                    x_features[:, :, idx_part("part_isChargedHadron")][..., None],
                    x_features[:, :, idx_part("part_isNeutralHadron")][..., None],
                    x_features[:, :, idx_part("part_isPhoton")][..., None],
                    x_features[:, :, idx_part("part_isElectron")][..., None],
                    x_features[:, :, idx_part("part_isMuon")][..., None],
                    np.tanh(x_features[:, :, idx_part("part_d0val")])[..., None],
                    np.clip(x_features[:, :, idx_part("part_d0err")][..., None], 0, 1),
                    np.tanh(x_features[:, :, idx_part("part_dzval")])[..., None],
                    np.clip(x_features[:, :, idx_part("part_dzerr")][..., None], 0, 1),
                    x_features[:, :, idx_part("part_etarel")][..., None],
                    x_features[:, :, idx_part("part_dphi")][..., None],
                ],
                axis=-1,
            )
            self.names_pf_features = [
                "log_part_pt",
                "log_part_energy",
                "log_part_ptrel",
                "log_part_energyrel",
                "part_deltaR",
                "part_charge",
                "part_isChargedHadron",
                "part_isNeutralHadron",
                "part_isPhoton",
                "part_isElectron",
                "part_isMuon",
                "tanh_part_d0val",
                "part_d0err",
                "tanh_part_dzval",
                "part_dzerr",
                "part_etarel",
                "part_dphi",
            ]

            # TODO: add shuffling
            # shuffle data
            permutation = np.random.permutation(len(x_features))
            x_features = x_features[permutation]
            pf_features = pf_features[permutation]
            pf_points = pf_points[permutation]
            pf_mask = pf_mask[permutation]
            y = y[permutation]
            cond_features = cond_features[permutation]

            # remove inf and nan values
            # TODO: check if this is ok with JetClass...
            pf_features = np.nan_to_num(pf_features, nan=0.0, posinf=0.0, neginf=0.0)
            vector.register_awkward()

            # --- define x_lorentz as px, py, pz, energy using eta, phi, pt to calculate px, py, pz
            pt = (
                x_features[:, :, idx_part("part_ptrel")]
                * cond_features[:, idx_cond("jet_pt")][:, None]
                * pf_mask[:, :, 0]
            )
            eta = (
                x_features[:, :, idx_part("part_etarel")]
                + cond_features[:, idx_cond("jet_eta")][:, None] * pf_mask[:, :, 0]
            )
            rng = np.random.default_rng(1234)
            phi = (
                x_features[:, :, idx_part("part_dphi")]
                + rng.uniform(0, 2 * np.pi, size=(len(pf_features), 1)) * pf_mask[:, :, 0]
            )
            px = pt * np.cos(phi) * pf_mask[:, :, 0]
            py = pt * np.sin(phi) * pf_mask[:, :, 0]
            pz = pt * np.sinh(eta) * pf_mask[:, :, 0]
            energy = (
                x_features[:, :, idx_part("part_energyrel")]
                * cond_features[:, idx_cond("jet_energy")][:, None]
                * pf_mask[:, :, 0]
            )
            pf_vectors = np.concatenate(
                [
                    px[..., None],
                    py[..., None],
                    pz[..., None],
                    energy[..., None],
                ],
                axis=-1,
            )

        # Swap indices of particle-level arrays to get required shape for
        # ParT/ParticleNet: (n_jets, n_features, n_particles)
        x_features = np.swapaxes(x_features, 1, 2)
        pf_features = np.swapaxes(pf_features, 1, 2)
        pf_vectors = np.swapaxes(pf_vectors, 1, 2)
        pf_mask = np.swapaxes(pf_mask, 1, 2)
        pf_points = np.swapaxes(pf_points, 1, 2)

        self.cond = cond_features
        self.names_cond = cond_names
        self.pf_features = pf_features
        self.pf_vectors = pf_vectors
        self.pf_mask = pf_mask
        self.pf_points = pf_points
        self.y = y
        # self.jet_data = jet_data

        return

        # Split data into train, val, test
        fractions = self.hparams.train_val_test_split
        total_length = len(x_features)
        split_indices = [
            int(fractions[0] * total_length),
            int((fractions[0] + fractions[1]) * total_length),
        ]
        print(f"Splitting data into {split_indices} for train, val, test")

        x_features_train, x_features_val, x_features_test = np.split(x_features, split_indices)
        x_coords_train, x_coords_val, x_coords_test = np.split(pf_points, split_indices)
        x_mask_train, x_mask_val, x_mask_test = np.split(pf_mask, split_indices)
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
