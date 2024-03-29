import os
from typing import Any, Dict, Optional, Tuple

import awkward as ak
import energyflow as ef
import h5py
import numpy as np
import torch
import vector
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from particle_fm.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


class JetClassClassifierDataModule(LightningDataModule):
    """Data module for jet classification task (classifier test)."""

    def __init__(
        self,
        data_dir: str = "data/",
        data_file: str = None,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.5, 0.2, 0.2),
        kin_only: bool = False,
        used_flavor: Tuple[str, ...] = None,
        debug_sim_only: bool = False,
        debug_sim_gen_fraction: float = None,
        number_of_jets: int = None,
        use_weaver_axes_convention: bool = True,
        pf_features_list: list = None,
        hl_features_list: list = None,
        set_energy_equal_to_p: bool = False,
        set_neutral_particle_ips_zero: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            data_dir: Path to the data directory (not used but there for backwards compatibility).
            data_file: Path to the data file.
            batch_size: Batch size.
            num_workers: Number of workers for the data loaders.
            pin_memory: Whether to pin memory for the data loaders.
            train_val_test_split: Fraction of data to use for train, val, test.
            kin_only: Whether to use only kinematic features.
            used_flavor: Which flavor to use. If None, use all flavors.
            debug_sim_only: Whether to use only sim data for debugging (half of the
            debug_sim_gen_fraction: Fraction of gen data to mix in for debugging.
                This can only be used in combination with `debug_sim_only`.
                The resulting dataset will have "fake"-labelled jets that only have
                `debug_sim_gen_fraction`% actual fake jets. The remaining part will
                be real jets that are labelled as fake.
            real data will be labelled as fake).
            kwargs: Additional arguments.
        """
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
        logger.info(f"Loading data from {self.hparams.data_file}...")
        with h5py.File(self.hparams.data_file, "r") as h5file:
            part_names = list(h5file["part_data_sim"].attrs["names"])
            cond_names = list(h5file["cond_data_sim"].attrs["names"])
            # jet_names = list(h5file["jet_data_sim"].attrs["names"])

            def idx_part(feat_name):
                return part_names.index(feat_name)

            def idx_cond(feat_name):
                return cond_names.index(feat_name)

            logger.info(f"part_names: {part_names}")
            logger.info(f"cond_names: {cond_names}")

            data_gen = h5file["part_data_gen"][: self.hparams.number_of_jets]
            mask_gen = h5file["part_mask_gen"][: self.hparams.number_of_jets]
            cond_gen = h5file["cond_data_gen"][: self.hparams.number_of_jets]
            label_gen = np.ones(len(data_gen))

            data_sim = h5file["part_data_sim"][: self.hparams.number_of_jets]
            mask_sim = h5file["part_mask_sim"][: self.hparams.number_of_jets]
            cond_sim = h5file["cond_data_sim"][: self.hparams.number_of_jets]
            label_sim = np.zeros(len(data_sim))

            logger.info(f"Shape of data_gen: {data_gen.shape}")
            logger.info(f"Shape of data_sim: {data_sim.shape}")
            logger.info(f"Shape of cond_gen: {cond_gen.shape}")
            logger.info(f"Shape of cond_sim: {cond_sim.shape}")
            logger.info(f"Shape of mask_gen: {mask_gen.shape}")
            logger.info(f"Shape of mask_sim: {mask_sim.shape}")

        # fmt: off
        # load the substructure features / high-level features
        subs_filename = self.hparams.data_file.replace(".h5", "_substructure.h5")  # noqa E501
        self.names_hl_features_all = [ "d2", "tau1", "tau2", "tau21", "tau3", "tau32"] # noqa E501

        # fmt: on
        if os.path.isfile(subs_filename) and os.path.isfile(subs_filename):
            with h5py.File(subs_filename) as h5file:
                hl_features_gen = np.concatenate(
                    [
                        h5file[hl_name + "_gen"][: self.hparams.number_of_jets][..., None]
                        for hl_name in self.names_hl_features_all
                    ],
                    axis=-1,
                )
                hl_features_sim = np.concatenate(
                    [
                        h5file[hl_name + "_sim"][: self.hparams.number_of_jets][..., None]
                        for hl_name in self.names_hl_features_all
                    ],
                    axis=-1,
                )
        else:
            hl_features_gen = np.zeros((len(data_gen), len(self.names_hl_features_all)))
            hl_features_sim = np.zeros((len(data_sim), len(self.names_hl_features_all)))

        x_features = np.concatenate([data_gen, data_sim])
        cond_features = np.concatenate([cond_gen, cond_sim])
        hl_features_all = np.concatenate([hl_features_gen, hl_features_sim])
        # use the first three particle features as coordinates (etarel, phirel)
        # TODO: probably not needed for ParT (only for ParticleNet)
        self.names_pf_points = ["part_etarel", "part_dphi"]
        pf_points_indices = [idx_part(name) for name in self.names_pf_points]
        pf_points = x_features[:, :, pf_points_indices]
        pf_mask = np.concatenate([mask_gen, mask_sim])
        y = np.concatenate([label_gen, label_sim])

        if self.hparams.debug_sim_only:
            # use only sim for debugging
            logger.warning("Using only sim data for debugging.")
            logger.warning("HALF OF THE REAL DATA WILL BE LABELLED AS FAKE.")
            x_features = data_sim
            cond_features = cond_sim
            pf_points = x_features[:, :, pf_points_indices]
            pf_mask = mask_sim
            y = label_sim
            y[len(y) // 2 :] = 1
            if self.hparams.debug_sim_gen_fraction is not None:
                # mix in some gen data in the second half of the arrays
                logger.warning(
                    f"MIXING IN {self.hparams.debug_sim_gen_fraction*100}% GEN DATA FOR DEBUGGING."
                )
                len_half = len(y) // 2
                len_mix = int(len_half * self.hparams.debug_sim_gen_fraction)
                x_features[-len_mix:] = data_gen[:len_mix]
                hl_features_all[-len_mix:] = hl_features_all[:len_mix]
                cond_features[-len_mix:] = cond_gen[:len_mix]
                pf_points[-len_mix:] = x_features[-len_mix:, :, pf_points_indices]
                pf_mask[-len_mix:] = mask_gen[:len_mix]
                y[-len_mix:] = label_gen[:len_mix]

        if self.hparams.used_flavor is not None:
            logger.info(f"Using only the following flavor: {self.hparams.used_flavor}")
            idx = idx_cond(f"jet_type_label_{self.hparams.used_flavor}")
            mask_sel_jet = cond_features[:, idx] == 1
            x_features = x_features[mask_sel_jet]
            cond_features = cond_features[mask_sel_jet]
            pf_points = pf_points[mask_sel_jet]
            pf_mask = pf_mask[mask_sel_jet]
            hl_features_all = hl_features_all[mask_sel_jet]
            y = y[mask_sel_jet]

        # --- define x_lorentz as px, py, pz, energy using eta, phi, pt to calculate px, py, pz
        pt = (
            x_features[:, :, idx_part("part_ptrel")]
            * cond_features[:, idx_cond("jet_pt")][:, None]
            * pf_mask[:, :, 0]
        )
        eta = (
            x_features[:, :, idx_part("part_etarel")]
            + cond_features[:, idx_cond("jet_eta")][:, None]
        ) * pf_mask[:, :, 0]
        rng = np.random.default_rng(1234)
        phi = (
            x_features[:, :, idx_part("part_dphi")]
            + rng.uniform(0, 2 * np.pi, size=(len(pf_mask), 1))
        ) * pf_mask[:, :, 0]
        px = pt * np.cos(phi) * pf_mask[:, :, 0]
        py = pt * np.sin(phi) * pf_mask[:, :, 0]
        pz = pt * np.sinh(eta) * pf_mask[:, :, 0]

        p = np.sqrt(px**2 + py**2 + pz**2)

        # ensure that energy >= momentum
        # check if part_energyrel and jet_energy is available
        if "part_energyrel" in part_names and "jet_energy" in cond_names:
            logger.warning("Using energy equal to momentum.")
            part_energyrel = x_features[:, :, idx_part("part_energyrel")]
            jet_energy = cond_features[:, idx_cond("jet_energy")]
            part_energy = part_energyrel * jet_energy[:, None]
        else:
            logger.warning("part_energyrel and/or jet_energy not available.")
            self.hparams.set_energy_equal_to_p = True

        if self.hparams.set_energy_equal_to_p:
            logger.warning("Setting energy equal to momentum.")
            part_energy = p
            jet_energy = np.sum(p, axis=1)
            part_energyrel = p / np.sum(p, axis=1)[:, None]

        self.pt_energy_inspect = np.concatenate(
            [
                p[..., None],
                pt[..., None],
                part_energy[..., None],
                part_energy[..., None],
                part_energyrel[..., None],
                x_features[:, :, idx_part("part_ptrel")][..., None],
            ],
            axis=-1,
        )
        self.y_inspect = y

        pf_vectors = np.concatenate(
            [
                px[..., None],
                py[..., None],
                pz[..., None],
                part_energy[..., None],
            ],
            axis=-1,
        )

        if self.hparams.set_neutral_particle_ips_zero:
            logger.warning("Setting neutral particle IP and IP uncertainty to zero.")
            # set d0val, dzval, d0err, dzerr to zero if particle is not charged
            not_charged_hadron = x_features[:, :, idx_part("part_isChargedHadron")] == 0
            not_electron = x_features[:, :, idx_part("part_isElectron")] == 0
            not_muon = x_features[:, :, idx_part("part_isMuon")] == 0

            not_charged = not_charged_hadron & not_electron & not_muon

            x_features[:, :, idx_part("part_d0val")][not_charged] = 0
            x_features[:, :, idx_part("part_dzval")][not_charged] = 0
            x_features[:, :, idx_part("part_d0err")][not_charged] = 0
            x_features[:, :, idx_part("part_dzerr")][not_charged] = 0

        # for compatibility with kin-only trainings:
        non_kin_feature_names = [
            "part_isChargedHadron",
            "part_isNeutralHadron",
            "part_isPhoton",
            "part_isElectron",
            "part_isMuon",
            "part_charge",
            "part_d0val",
            "part_d0err",
            "part_dzval",
            "part_dzerr",
        ]
        for name in non_kin_feature_names:
            if name not in part_names:
                logger.warning(f"Adding {name} as zeros to x_features.")
                x_features = np.concatenate(
                    [x_features, np.zeros((len(x_features), len(x_features[0]), 1))], axis=-1
                )
                part_names.append(name)

        self.names_x_features = part_names

        # create the features array as needed by ParT
        pf_features = np.concatenate(
            [
                # log ( part_pt )
                (np.log(pt)[..., None] - 1.7) * 0.7,
                # log ( part_energy )
                (np.log(part_energy)[..., None] - 2.0) * 0.7,
                # log ( part_ptrel )
                (np.log(x_features[:, :, idx_part("part_ptrel")])[..., None] + 4.7) * 0.7,
                # log ( part_energyrel )
                (np.log(part_energyrel)[..., None] + 4.7) * 0.7,
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

        if self.hparams.kin_only:
            logger.info("Using only kinematic pf_features.")
            names_pf_features_kin = [
                "log_part_pt",
                "log_part_energy",
                "log_part_ptrel",
                "log_part_energyrel",
                "part_deltaR",
                "part_etarel",
                "part_dphi",
            ]
            index_pf_features_kin = [
                self.names_pf_features.index(name) for name in names_pf_features_kin
            ]
            self.names_pf_features = names_pf_features_kin
            pf_features = pf_features[:, :, index_pf_features_kin]
        elif self.hparams.pf_features_list is not None:
            logger.info(f"Using only the following pf_features: {self.hparams.pf_features_list}")
            index_pf_features_kin = [
                self.names_pf_features.index(name) for name in self.hparams.pf_features_list
            ]
            self.names_pf_features = self.hparams.pf_features_list
            pf_features = pf_features[:, :, index_pf_features_kin]
        else:
            logger.info("Using all pf_features.")

        if self.hparams.hl_features_list is not None:
            logger.info(f"Using only the following hl_features: {self.hparams.hl_features_list}")
            index_hl_features = [
                self.names_hl_features_all.index(name) for name in self.hparams.hl_features_list
            ]
            self.names_hl_features = self.hparams.hl_features_list
            hl_features = hl_features_all[:, index_hl_features]
        else:
            self.names_hl_features = self.names_hl_features_all
            hl_features = hl_features_all

        logger.info(f"pf_features:   shape={pf_features.shape}, {self.names_pf_features}")
        logger.info(f"pf_points:     shape={pf_points.shape}, {self.names_pf_points}")
        logger.info(f"pf_mask:       shape={pf_mask.shape}")
        logger.info(f"cond_features: shape={cond_features.shape}, {cond_names}")
        logger.info(f"hl_features:   shape={hl_features_all.shape}, {self.names_hl_features_all}")

        rng = np.random.default_rng(1234)
        permutation = rng.permutation(len(x_features))
        x_features = x_features[permutation]
        pf_features = pf_features[permutation]
        pf_vectors = pf_vectors[permutation]
        pf_points = pf_points[permutation]
        pf_mask = pf_mask[permutation]
        y = y[permutation]
        hl_features_all = hl_features_all[permutation]
        hl_features = hl_features[permutation]
        cond_features = cond_features[permutation]

        # remove inf and nan values
        # TODO: check if this is ok with JetClass...
        pf_features = np.nan_to_num(pf_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Swap indices of particle-level arrays to get required shape for
        # ParT/ParticleNet: (n_jets, n_features, n_particles)
        if self.hparams.use_weaver_axes_convention:
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
        self.hl_features = hl_features_all
        self.hl_features_all = hl_features_all
        # self.jet_data = jet_data

        # Split data into train, val, test
        fractions = self.hparams.train_val_test_split
        total_length = len(x_features)
        split_indices = [
            int(fractions[0] * total_length),
            int((fractions[0] + fractions[1]) * total_length),
        ]
        logger.info(
            f"Splitting data into fractions {fractions} for train, val, test "
            f"(at indices {split_indices})."
        )

        self.pf_features_train, self.pf_features_val, self.pf_features_test = np.split(
            pf_features, split_indices
        )
        self.pf_vectors_train, self.pf_vectors_val, self.pf_vectors_test = np.split(
            pf_vectors, split_indices
        )
        self.pf_points_train, self.pf_points_val, self.pf_points_test = np.split(
            pf_points, split_indices
        )
        self.pf_mask_train, self.pf_mask_val, self.pf_mask_test = np.split(pf_mask, split_indices)
        self.cond_train, self.cond_val, self.cond_test = np.split(cond_features, split_indices)
        self.y_train, self.y_val, self.y_test = np.split(y, split_indices)
        self.hl_features_train, self.hl_features_val, self.hl_features_test = np.split(
            hl_features, split_indices
        )
        self.hl_features_all_train, self.hl_features_all_val, self.hl_features_all_test = np.split(
            hl_features_all, split_indices
        )
        self.x_features_train, self.x_features_val, self.x_features_test = np.split(
            x_features, split_indices
        )

        # Create datasets
        self.data_train = TensorDataset(
            torch.tensor(self.pf_points_train, dtype=torch.float32),
            torch.tensor(self.pf_features_train, dtype=torch.float32),
            torch.tensor(self.pf_vectors_train, dtype=torch.float32),
            torch.tensor(self.pf_mask_train, dtype=torch.float32),
            torch.tensor(self.cond_train, dtype=torch.float32),
            torch.tensor(self.hl_features_train, dtype=torch.float32),
            torch.nn.functional.one_hot(torch.tensor(self.y_train, dtype=torch.int64)),
        )
        self.data_val = TensorDataset(
            torch.tensor(self.pf_points_val, dtype=torch.float32),
            torch.tensor(self.pf_features_val, dtype=torch.float32),
            torch.tensor(self.pf_vectors_val, dtype=torch.float32),
            torch.tensor(self.pf_mask_val, dtype=torch.float32),
            torch.tensor(self.cond_val, dtype=torch.float32),
            torch.tensor(self.hl_features_val, dtype=torch.float32),
            torch.nn.functional.one_hot(torch.tensor(self.y_val, dtype=torch.int64)),
        )
        self.data_test = TensorDataset(
            torch.tensor(self.pf_points_test, dtype=torch.float32),
            torch.tensor(self.pf_features_test, dtype=torch.float32),
            torch.tensor(self.pf_vectors_test, dtype=torch.float32),
            torch.tensor(self.pf_mask_test, dtype=torch.float32),
            torch.tensor(self.cond_test, dtype=torch.float32),
            torch.tensor(self.hl_features_test, dtype=torch.float32),
            torch.nn.functional.one_hot(torch.tensor(self.y_test, dtype=torch.int64)),
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
