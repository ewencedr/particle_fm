import torch
from torch import nn


class SWD(nn.Module):
    """Sliced Wasserstein Distance Compute the Wasserstein distance between two point clouds.
    Inspired by https://github.com/apple/ml-cvpr2019-swd/blob/master/swd.py#L45.

    Args:
        data (torch.Tensor) [batch, n_points, feats]: Ground truth.
        preds (torch.Tensor) [batch, n_points, feats]: Predictions.
        n_proj (int, optional): number of random 1d projections. Defaults to 1024.

    Returns:
        wdist (torch.Tensor) [1]: Wasserstein distance
    """

    def __init__(self, n_proj: int = 1024):
        super().__init__()
        self.n_proj = n_proj

    def forward(self, data, preds):
        n_proj = self.n_proj  # no of random 1-d projections
        b, p, f = data.shape  # [batch,points,feats]
        data, preds = data.float(), preds.float()
        proj = torch.randn(f, n_proj, device=data.device)  # [feats, l]
        proj *= torch.rsqrt(torch.sum(torch.square(proj), 0, keepdim=True))
        proj = proj.view(1, f, n_proj).expand(b, -1, -1)  # first add dim, then expand to batch dim
        p1 = torch.matmul(data, proj)  # shape: [batch, n_points, l]
        p2 = torch.matmul(preds, proj)  # shape: [batch, n_points, l]
        p1, _ = torch.sort(p1, 1, descending=True)  # point wise sorting
        p2, _ = torch.sort(p2, 1, descending=True)
        wdist = torch.mean(torch.square(p1 - p2))  # MSE
        return wdist


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (
                self.get_bandwidth(L2_distances).to(X.device)
                * self.bandwidth_multipliers.to(X.device)
            )[:, None, None]
        ).sum(dim=0)


class MMDLoss(nn.Module):
    """MMD Loss based on https://github.com/yiftachbeer/mmd_loss_pytorch."""

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y])).to(X.device)

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
