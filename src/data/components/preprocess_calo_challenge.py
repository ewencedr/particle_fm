import os
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from tqdm import tqdm

data_dir = "/beegfs/desy/user/kaechben/calochallenge/"
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# Custom transformer for logit transformation
class LogitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.log(X / (1 - X))

    def inverse_transform(self, X, y=None):
        return 1 / (1 + np.exp(-X))

    def check_inverse(self, X):
        assert np.allclose(self.transform(self.inverse_transform(X)), X)


# Custom transformer for inverse logit transformation
class DQ(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        X = X + np.random.rand(*X.shape)
        return X

    def inverse_transform(self, X, y=None):
        X = np.floor(X)
        return X


class ScalerBase:
    def __init__(self, transfs, name, featurenames: list[str]) -> None:
        self.transfs = transfs
        self.featurenames = featurenames
        self.n_features = len(self.transfs)

        self.scalerpath = Path(data_dir) / f"scaler_{name}.gz"
        if self.scalerpath.is_file():
            self.transfs = joblib.load(self.scalerpath)

    def save_scalar(self, pcs: torch.Tensor):
        # The features need to be converted to numpy immediately
        # otherwise the queuflow afterwards does not work
        assert pcs.dim() == 2
        assert self.n_features == pcs.shape[1]
        pcs = pcs.detach().cpu().numpy()
        self.plot_scaling(pcs)

        assert pcs.shape[1] == self.n_features
        pcs = np.hstack(
            [transf.fit_transform(arr.reshape(-1, 1)) for arr, transf in zip(pcs.T, self.transfs)]
        )
        self.plot_scaling(pcs, True)

        joblib.dump(self.transfs, self.scalerpath)

    def transform(self, pcs: np.ndarray):
        assert len(pcs.shape) == 2
        pcs = pcs.astype(np.float64)
        assert pcs.shape[1] == self.n_features
        return torch.from_numpy(
            np.hstack(
                [transf.transform(arr.reshape(-1, 1)) for arr, transf in zip(pcs.T, self.transfs)]
            )
        ).float()

    def inverse_transform(self, pcs: torch.Tensor):
        assert pcs.shape[-1] == self.n_features

        orgshape = pcs.shape
        dev = pcs.device
        pcs = pcs.to("cpu").detach().reshape(-1, self.n_features).numpy()
        pcs = pcs.astype(np.float64)
        t_stacked = np.hstack(
            [
                transf.inverse_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )
        return torch.from_numpy(t_stacked.reshape(*orgshape)).float().to(dev)

    def plot_scaling(self, pcs, post=False):
        for k, v in zip(self.featurenames, pcs.T):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(v, bins=500)
            fig.savefig(
                Path(data_dir) / f"{k}_post.png" if post else Path(data_dir) / f"{k}_pre.png"
            )
            plt.close(fig)
