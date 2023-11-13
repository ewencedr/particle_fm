from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import h5py
import energyflow as ef
import numpy as np

from src.data.components import (
    normalize_tensor,
)
from src.data.components.utils import (
    get_mjj,
    get_jet_data,
    get_nonrel_consts,
    sort_consts,
    sort_jets,
)


class ClassifierDataModule(LightningDataModule):
    """`LightningDataModule` for the LHCO dataset.

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
        train_val_test_split: Tuple[float, float, float] = (0.70, 0.20, 0.10),
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        gendatafile: str = "idealized_LHCO",
        idealized: bool = False,
        gen_jet: str = "both",
        ref_jet: str = "both",
        use_shuffled_data: bool = False,
        use_nonrel_data: bool = False,
        n_signal: int = 0,
        n_background: int = 100_000,
    ) -> None:
        """Initialize a `ClassifierDataModule`.

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

        if "both" in self.hparams.ref_jet and "both" in self.hparams.gen_jet:
            self.jets_to_use = "both"
        else:
            self.jets_to_use = ""

        if self.hparams.gen_jet not in ["first", "second", "both", "both_first", "both_second"]:
            raise ValueError(
                "gen_jet must be one of 'first' or 'second' or 'both', or 'both_first', or 'both_second'"
            )

        if self.hparams.ref_jet not in ["first", "second", "both", "both_first", "both_second"]:
            raise ValueError(
                "ref_jet must be one of 'first' or 'second', or 'both', or 'both_first', or 'both_second'"
            )

        if self.hparams.gen_jet == "both" and self.hparams.ref_jet != "both":
            raise ValueError("gen_jet must be 'both' if ref_jet is 'both'")
        if self.hparams.gen_jet != "both" and self.hparams.ref_jet == "both":
            raise ValueError("ref_jet must be 'both' if gen_jet is 'both'")

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.use_shuffled_data:
                path_bckg = f"{self.hparams.data_dir}/lhco/final_data/processed_data_background_rel_shuffled.h5"
                path_sgnl = f"{self.hparams.data_dir}/lhco/final_data/processed_data_signal_rel_shuffled.h5"
            else:
                path_bckg = (
                    f"{self.hparams.data_dir}/lhco/final_data/processed_data_background_rel.h5"
                )
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

            jet_data_mixed = np.concatenate(
                [
                    jet_data_bckg[: self.hparams.n_background],
                    jet_data_sgnl[: self.hparams.n_signal],
                ]
            )
            particle_data_mixed = np.concatenate(
                [
                    particle_data_bckg[: self.hparams.n_background],
                    particle_data_sgnl[: self.hparams.n_signal],
                ]
            )
            mask_mixed = np.concatenate(
                [mask_bckg[: self.hparams.n_background], mask_sgnl[: self.hparams.n_signal]]
            )

            # shuffle
            random_permutation = np.random.permutation(len(jet_data_mixed))
            jet_data_mixed = jet_data_mixed[random_permutation]
            particle_data_mixed = particle_data_mixed[random_permutation]
            mask_mixed = mask_mixed[random_permutation]

            labels_mixed = np.ones(len(jet_data_mixed))

            # Load generated data
            path_gen = f"{self.hparams.data_dir}/lhco/generated/{self.hparams.gendatafile}.h5"
            print(f"Loading generated data from {path_gen}")

            if self.hparams.use_shuffled_data:
                with h5py.File(path_gen, "r") as f:
                    jet_data_gen = f["jet_features"][:]
                    particle_data_gen = f["particle_features"][:]  # pt, eta, phi
                    # particle_data_gen_raw = f["data_raw"][:]  # pt, eta, phi
            else:
                with h5py.File(path_gen, "r") as f:
                    jet_data_gen = f["jet_features"][:]
                    particle_data_gen = f["particle_features"][:]  # pt, eta, phi
                    particle_data_gen_raw = f["data_raw"][:]  # pt, eta, phi

                print(f"particle data shape: {particle_data_gen.shape}")
                # if self.hparams.jets_to_use != "both":
                #    particle_data_gen = particle_data_gen_raw
                particle_data_gen = particle_data_gen_raw

            jet_data_gen = jet_data_gen[..., :4]

            mask_gen = np.expand_dims(np.array(particle_data_gen[..., 0] != 0), axis=-1).astype(
                int
            )

            # TODO length of both classes should not need to be the same
            if self.hparams.idealized:
                print(f"Using background as background")
                jet_data_background = jet_data_bckg[: len(particle_data_mixed)]
                particle_data_background = particle_data_bckg[: len(particle_data_mixed)]
                mask_background = mask_bckg[: len(particle_data_mixed)]
            else:
                print("Using generated data as background")
                jet_data_background = jet_data_gen[: len(particle_data_mixed)]
                particle_data_background = particle_data_gen[: len(particle_data_mixed)]
                mask_background = mask_gen[: len(particle_data_mixed)]

            labels_background = np.zeros(len(jet_data_background))

            if self.hparams.use_nonrel_data:
                particle_data_background = get_nonrel_consts(
                    jet_data_background, particle_data_background
                )
                particle_data_mixed = get_nonrel_consts(jet_data_mixed, particle_data_mixed)

            if self.hparams.gen_jet == "first":
                particle_data_background = particle_data_background[:, 0]
                jet_data_background = jet_data_background[:, 0]
                mask_background = mask_background[:, 0]
            elif self.hparams.gen_jet == "second":
                particle_data_background = particle_data_background[:, 1]
                jet_data_background = jet_data_background[:, 1]
                mask_background = mask_background[:, 1]
            elif self.hparams.gen_jet == "both_first":
                particle_data_bckg_temp = particle_data_background[:, 0]
                jet_data_bck_temp = jet_data_background[:, 0]
                mask_bckg_temp = mask_background[:, 0]
                particle_data_background = np.stack(
                    (particle_data_bckg_temp, particle_data_bckg_temp), axis=1
                )
                jet_data_background = np.stack((jet_data_bck_temp, jet_data_bck_temp), axis=1)
                mask_background = np.stack((mask_bckg_temp, mask_bckg_temp), axis=1)
            elif self.hparams.gen_jet == "both_second":
                particle_data_bckg_temp = particle_data_background[:, 1]
                jet_data_bck_temp = jet_data_background[:, 1]
                mask_bckg_temp = mask_background[:, 1]
                particle_data_background = np.stack(
                    (particle_data_bckg_temp, particle_data_bckg_temp), axis=1
                )
                jet_data_background = np.stack((jet_data_bck_temp, jet_data_bck_temp), axis=1)
                mask_background = np.stack((mask_bckg_temp, mask_bckg_temp), axis=1)

            if self.hparams.ref_jet == "first":
                particle_data_mixed = particle_data_mixed[:, 0]
                jet_data_mixed = jet_data_mixed[:, 0]
                mask_mixed = mask_mixed[:, 0]
            elif self.hparams.ref_jet == "second":
                particle_data_mixed = particle_data_mixed[:, 1]
                jet_data_mixed = jet_data_mixed[:, 1]
                mask_mixed = mask_mixed[:, 1]
            elif self.hparams.ref_jet == "both_first":
                particle_data_mixed_temp = particle_data_mixed[:, 0]
                jet_data_mixed_temp = jet_data_mixed[:, 0]
                mask_mixed_temp = mask_mixed[:, 0]
                particle_data_mixed = np.stack(
                    (particle_data_mixed_temp, particle_data_mixed_temp), axis=1
                )
                jet_data_mixed = np.stack((jet_data_mixed_temp, jet_data_mixed_temp), axis=1)
                mask_mixed = np.stack((mask_mixed_temp, mask_mixed_temp), axis=1)
            elif self.hparams.ref_jet == "both_second":
                particle_data_mixed_temp = particle_data_mixed[:, 1]
                jet_data_mixed_temp = jet_data_mixed[:, 1]
                mask_mixed_temp = mask_mixed[:, 1]
                particle_data_mixed = np.stack(
                    (particle_data_mixed_temp, particle_data_mixed_temp), axis=1
                )
                jet_data_mixed = np.stack((jet_data_mixed_temp, jet_data_mixed_temp), axis=1)
                mask_mixed = np.stack((mask_mixed_temp, mask_mixed_temp), axis=1)

            # concatenate both classes
            print(f"particle_data_mixed.shape: {particle_data_mixed.shape}")
            print(f"particle_data_background.shape: {particle_data_background.shape}")
            input_data = np.concatenate([particle_data_mixed, particle_data_background])
            input_mask = np.concatenate([mask_mixed, mask_background])
            input_labels = np.concatenate([labels_mixed, labels_background])

            # shuffle data
            # needed because data is ordered by class
            perm = np.random.permutation(len(input_data))
            input_data = input_data[perm]
            input_mask = input_mask[perm]
            input_labels = input_labels[perm]

            print(f"input_data.shape: {input_data.shape}")
            print(f"input_mask.shape: {input_mask.shape}")
            print(f"input_labels.shape: {input_labels.shape}")

            if len(input_data) != len(input_mask) or len(input_data) != len(input_labels):
                raise ValueError("Data, mask and labels must have the same length.")

            n_samples_val = int(self.hparams.train_val_test_split[1] * len(input_data))
            n_samples_test = int(self.hparams.train_val_test_split[2] * len(input_data))

            data_train, data_val, data_test = np.split(
                input_data,
                [
                    len(input_data) - n_samples_val - n_samples_test,
                    len(input_data) - n_samples_test,
                ],
            )
            mask_train, mask_val, mask_test = np.split(
                input_mask,
                [
                    len(input_mask) - n_samples_val - n_samples_test,
                    len(input_mask) - n_samples_test,
                ],
            )
            labels_train, labels_val, labels_test = np.split(
                input_labels,
                [
                    len(input_labels) - n_samples_val - n_samples_test,
                    len(input_labels) - n_samples_test,
                ],
            )

            print(f"data_train.shape: {data_train.shape}")
            print(f"mask_train.shape: {mask_train.shape}")
            print(f"labels_train.shape: {labels_train.shape}")
            print(f"data_val.shape: {data_val.shape}")
            print(f"mask_val.shape: {mask_val.shape}")
            print(f"labels_val.shape: {labels_val.shape}")
            print(f"data_test.shape: {data_test.shape}")
            print(f"mask_test.shape: {mask_test.shape}")
            print(f"labels_test.shape: {labels_test.shape}")

            # preprocess data
            full_mask_train = np.repeat(mask_train, repeats=3, axis=-1) == 0
            full_mask_train = np.ma.make_mask(full_mask_train, shrink=False)
            masked_data_train = np.ma.masked_array(data_train, full_mask_train)

            if self.jets_to_use != "both":
                mean = np.ma.mean(masked_data_train, axis=(0, 1))
                std = np.ma.std(masked_data_train, axis=(0, 1))
            else:
                mean = np.ma.mean(masked_data_train, axis=(0, 1, 2))
                std = np.ma.std(masked_data_train, axis=(0, 1, 2))

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
                torch.from_numpy(mask_train),
                torch.from_numpy(labels_train).float(),
            )
            self.data_val = torch.utils.data.TensorDataset(
                data_val.float(),
                torch.from_numpy(mask_val),
                torch.from_numpy(labels_val).float(),
            )
            self.data_test = torch.utils.data.TensorDataset(
                data_test.float(),
                torch.from_numpy(mask_test),
                torch.from_numpy(labels_test).float(),
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
