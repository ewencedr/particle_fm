from copy import deepcopy
from typing import Mapping

import numpy as np
from jetnet.datasets import JetNet
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def numpy_locals_to_mass_and_pt(
    csts: np.ndarray,
    mask: np.ndarray,
    pt_logged=False,
) -> np.ndarray:
    """Calculate the overall jet pt and mass from the constituents. The constituents are expected
    to be expressed as:

    - del_eta
    - del_phi
    - log_pt or just pt depending on pt_logged
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = np.exp(csts[..., 2]) * mask if pt_logged else csts[..., 2]

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(np.clip(jet_px**2 + jet_py**2, 0, None))
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None))

    return np.vstack([jet_pt, jet_m]).T


def log_squash(data: np.ndarray) -> np.ndarray:
    """Apply a log squashing function for distributions with high tails."""
    return np.sign(data) * np.log(np.abs(data) + 1)


class JetNetData(Dataset):
    """Wrapper for the JetNet dataset so it works with our models with different inputs."""

    def __init__(self, **kwargs) -> None:
        # Extra arguments used here
        self.log_squash_pt = kwargs.pop("log_squash_pt", False)
        self.high_as_context = kwargs.pop("high_as_context", True)
        self.recalc_high = kwargs.pop("recalculate_jet_from_pc", True)
        self.n_jets = kwargs.pop("n_jets", None)

        # All other arguments passed to the jetnet dataset constructor
        self.csts, self.high = JetNet.getData(**kwargs)
        self.csts = self.csts.astype(np.float32)
        self.high = self.high.astype(np.float32)

        # Trim the data based on the requested number of jets (None does nothing)
        self.csts = self.csts[: self.n_jets].astype(np.float32)
        self.high = self.high[: self.n_jets].astype(np.float32)

        # Manually calculate the mask by looking for zero padding
        self.mask = ~np.all(self.csts == 0, axis=-1)

        # Change the constituent information from pt-fraction to pure pt
        csts = self.csts.copy()
        csts[..., -1] = csts[..., -1] * self.high[..., 0:1]

        # Recalculate the jet mass and pt using the point cloud
        if self.recalc_high:
            self.high = numpy_locals_to_mass_and_pt(csts, self.mask)

        # Change the pt fraction to log_squash(pt)
        if self.log_squash_pt:
            self.csts[..., -1] = log_squash(csts[..., -1]) * self.mask

    def __getitem__(self, idx) -> tuple:
        csts = self.csts[idx]
        high = self.high[idx] if self.high_as_context else np.empty(0, dtype="f")
        mask = self.mask[idx]
        return csts, mask, high

    def __len__(self) -> int:
        return len(self.high)


class JetNetJediDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        data_conf: Mapping,
        loader_kwargs: Mapping,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Get the dimensions of the data from the config file
        self.dim = len(data_conf["particle_features"])
        self.n_nodes = data_conf["num_particles"]
        if data_conf["high_as_context"]:
            self.ctxt_dim = len(data_conf["jet_features"])
        else:
            self.ctxt_dim = 0

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""

        if stage == "fit":
            self.train_set = JetNetData(**self.hparams.data_conf, split="train")
            self.valid_set = JetNetData(**self.hparams.data_conf, split="valid")

            self.test_set = JetNetData(**self.hparams.data_conf, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)
