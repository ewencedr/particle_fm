import torch


def swd(data: torch.Tensor, preds: torch.Tensor, n_proj: int = 1024) -> torch.Tensor:
    """Sliced Wassersteini Distance Compute the Wasserstein distance between two point clouds.
    Inspired by https://github.com/apple/ml-cvpr2019-swd/blob/master/swd.py#L45.

    Args:
        data (torch.Tensor) [batch, n_points, feats]: Ground truth.
        preds (torch.Tensor) [batch, n_points, feats]: Predictions.
        n_proj (int, optional): number of random 1d projections. Defaults to 1024.

    Returns:
        wdist (torch.Tensor) [1]: Wasserstein distance
    """

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
