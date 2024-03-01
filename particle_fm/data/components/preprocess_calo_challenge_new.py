import os
from pathlib import Path

import h5py
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rv_continuous
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from tqdm import tqdm

# Custom transformer for logit transformation
matplotlib.use("Agg")


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


class SqrtTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        X[:, 2] = (X[:, 2]) ** 0.5
        return X

    def inverse_transform(self, X, y=None):
        X[:, 2] = (X[:, 2]) ** 2
        return X

    def check_inverse(self, X):
        assert np.allclose(self.transform(self.inverse_transform(X)), X)


# Custom transformer for inverse logit transformation


class Cart(BaseEstimator, TransformerMixin):
    # Transform (z,alpha,R) to (x,y,z)
    def __init__(self, num_alpha):
        self.num_alpha = num_alpha

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        x = X[:, 2] * np.cos(X[:, 1] / self.num_alpha * (2 * np.pi))
        y = X[:, 2] * np.sin(X[:, 1] / self.num_alpha * (2 * np.pi))
        X[:, 2] = X[:, 0]
        X[:, 1] = y
        X[:, 0] = x
        return X

    def inverse_transform(self, X, y=None):
        a = (np.arctan2(X[:, 1], X[:, 0]) + np.pi) * 16 / (2 * np.pi)
        R = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        X[:, 0] = X[:, 2]
        X[:, 1] = a
        X[:, 2] = R
        return X

    def check_inverse(self, X):
        assert np.allclose(self.transform(self.inverse_transform(X)), X)


class DQLinear(BaseEstimator, TransformerMixin):
    # Linearly interpolate between bins, note that the forward transformation is very slow
    def __init__(self, name):
        self.name = name
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fig, ax = plt.subplots(3, 2, figsize=(10, 5))
        names = ["x", "y", "z"]
        for i in [0, 1, 2]:  # or replace 3 with X.shape[1] if number of features varies
            data = X[:, i]
            ni, _, _ = ax[i, 0].hist(data, bins=100)
            if i != 1:

                unique_values, counts = np.unique(data, return_counts=True)

                for j in range(len(unique_values)):
                    # Select data points between this value and the next
                    value, count = unique_values[j], counts[j]
                    if j < len(unique_values) - 1:
                        nvalue, ncounts = unique_values[j + 1], counts[j + 1]

                        lid = LinearInterpolatedDistribution(x0=count, x1=ncounts)

                    mask = (X[:, i] >= value) & (X[:, i] < value + 1)
                    samples = lid.rvs(sum(mask))
                    data = X[mask, i] + samples
                    ax[i, 1].hist(data, bins=100)
                    X[mask, i] = data
            else:
                X[:, i] = X[:, i] + np.random.rand(*X[:, i].shape)
                ax[i, 1].hist(data, bins=100)
            ax[i, 0].set_xlabel(names[i])
            ax[i, 1].set_xlabel(names[i])
            ax[i, 0].set_ylabel("Counts")
            ax[i, 1].set_ylabel("Counts")
            nf, _, _ = ax[i, 0].hist(np.floor(X[:, i]), bins=100, histtype="step", color="red")
            print("difference initial and final", ni - nf)
        plt.tight_layout()
        plt.savefig(f"{self.name}_DQ.png")
        plt.close()
        return X

    def inverse_transform(self, X, y=None):
        X[:, 0:] = np.floor(X[:, 0:])
        X[:, 1:] = np.floor(X[:, 1:])
        X[:, 2:] = np.floor(X[:, 2:])
        return X


class LinearInterpolatedDistribution(rv_continuous):
    def __init__(self, x0, x1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x0 = x0
        self.x1 = x1
        self.s = self.x1 - self.x0
        self.a = 0
        self.b = 1
        self._compute_normalization_constant()

    def _pdf(self, x):
        return 1 / self.k * (self.x0 + self.s * x)

    def _compute_normalization_constant(self):
        self.k = (self.x1 + self.x0) * 0.50

    def _cdf(self, x):
        return 1 / self.k * (self.x0 * x + 0.5 * (self.s) * x**2)

    def _ppf(self, q):

        a = 0.5 * (self.s)
        b = self.x0
        c = -self.k * q
        discriminant = np.sqrt(b**2 - 4 * a * c)
        root1 = (discriminant - b) / (2 * a + 1e-5)
        root2 = (-discriminant - b) / (2 * a + 1e-5)
        # Use the root that falls within [0, 1]
        # results=np.where((root1 >= 0) & (root1 <= 1), root1, root2)
        return root1  # if (root1 >= 0 and root1 <= 1) else root2

    def rvs(self, size=None):
        u = np.random.uniform(size=size)
        return self._ppf(u)


class ScalerBaseNew:
    def __init__(
        self,
        transfs,
        name,
        featurenames,
        overwrite=False,
        data_dir="/beegfs/desy/user/kaechben/calochallenge/",
    ):
        self.transfs = transfs
        self.featurenames = featurenames
        self.n_features = len(featurenames)
        self.data_dir = data_dir
        self.scalerpath = Path(data_dir) / f"scaler_{name}.gz"
        self.name = name
        if self.scalerpath.is_file() and not overwrite:
            self.transfs = joblib.load(self.scalerpath)

    def save_scalar(self, pcs, save=False):
        # The features need to be converted to numpy immediately
        # otherwise the queuflow afterwards does not work
        print("pre scaling")
        pcs = pcs.astype(np.float64)
        self.plot_scaling(pcs)
        pcs = np.hstack(
            [self.transfs[0].fit_transform(pcs[:, :1]), self.transfs[1].fit_transform(pcs[:, 1:])]
        )
        print("post scaling")
        self.plot_scaling(pcs, True)
        pcs = np.hstack(
            [
                self.transfs[0].inverse_transform(pcs[:, :1]),
                self.transfs[1].inverse_transform(pcs[:, 1:]),
            ]
        )
        print("sanity test scaling")
        self.plot_scaling(pcs, False, True)
        if save:
            joblib.dump(self.transfs, self.scalerpath)

    def transform(self, pcs):
        # assert len(pcs.shape) == 2
        orgshape = pcs.shape
        dev = pcs.device
        pcs = pcs.cpu().numpy().astype(np.float64).reshape(-1, self.n_features)
        return (
            torch.from_numpy(
                np.hstack(
                    [self.transfs[0].transform(pcs[:, :1]), self.transfs[1].transform(pcs[:, 1:])]
                ).reshape(*orgshape)
            )
            .to(dev)
            .float()
        )

    def inverse_transform(self, pcs: torch.Tensor):

        orgshape = pcs.shape
        dev = pcs.device
        pcs = pcs.to("cpu").detach().reshape(-1, self.n_features).numpy().astype(np.float64)
        t_stacked = np.hstack(
            [
                self.transfs[0].inverse_transform(pcs[:, :1]),
                self.transfs[1].inverse_transform(pcs[:, 1:]),
            ]
        )
        return torch.from_numpy(t_stacked.reshape(*orgshape)).to(dev).float()

    def plot_scaling(self, pcs, post=False, re=False):
        fig, ax = plt.subplots(1, len(self.featurenames), figsize=(20, 5))
        i = 0
        for k, v in zip(self.featurenames, pcs.T):
            bins = min(500, len(np.unique(v)))
            ax[i].hist(v, bins=bins)
            i += 1
            if post:
                savename = self.data_dir + f"{self.name}_post.png"
            elif not re:
                savename = self.data_dir + f"{self.name}_pre.png"
            else:
                savename = self.data_dir + f"{self.name}_id.png"
        fig.savefig(savename)
        plt.close(fig)
