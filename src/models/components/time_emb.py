"""Functions for time embeddings."""

import math

import torch
import torch.nn as nn


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.

    Inspired by https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[..., None] * self.W[None, ...] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class CosineEncoding:
    """Cosine encoding of a tensor.

    adapted from: https://github.com/rodem-hep/PC-JeDi/blob/main/src/models/modules.py
    """

    def __init__(
        self,
        outp_dim: int = 32,
        min_value: float = 0.0,
        max_value: float = 1.0,
        frequency_scaling: str = "exponential",
    ) -> None:
        self.outp_dim = outp_dim
        self.min_value = min_value
        self.max_value = max_value
        self.frequency_scaling = frequency_scaling

    def __call__(self, inpt: torch.Tensor) -> torch.Tensor:
        return cosine_encoding(
            inpt, self.outp_dim, self.min_value, self.max_value, self.frequency_scaling
        )


def cosine_encoding(
    x: torch.Tensor,
    outp_dim: int = 32,
    min_value: float = 0.0,
    max_value: float = 1.0,
    frequency_scaling: str = "exponential",
) -> torch.Tensor:
    """Computes a positional cosine encodings with an increasing series of frequencies.

    The frequencies either increase linearly or exponentially (default).
    The latter is good for when max_value is large and extremely high sensitivity to the
    input is required.
    If inputs greater than the max value are provided, the outputs become degenerate.
    If inputs smaller than the min value are provided, the inputs the the cosine will
    be both positive and negative, which may lead degenerate outputs.

    Always make sure that the min and max bounds are not exceeded!

    Args:
        x: The input, the final dimension is encoded. If 1D then it will be unqueezed
        out_dim: The dimension of the output encoding
        min_value: Added to x (and max) as cosine embedding works with positive inputs
        max_value: The maximum expected value, sets the scale of the lowest frequency
        frequency_scaling: Either 'linear' or 'exponential'

    Returns:
        The cosine embeddings of the input using (out_dim) many frequencies
    """

    # Unsqueeze if final dimension is flat
    if x.shape[-1] != 1 or x.dim() == 1:
        x = x.unsqueeze(-1)

    # Check the the bounds are obeyed
    if torch.any(x > max_value):
        print("Warning! Passing values to cosine_encoding encoding that exceed max!")
    if torch.any(x < min_value):
        print("Warning! Passing values to cosine_encoding encoding below min!")

    # Calculate the various frequencies
    if frequency_scaling == "exponential":
        freqs = torch.arange(outp_dim, device=x.device).exp()
    elif frequency_scaling == "linear":
        freqs = torch.arange(1, outp_dim + 1, device=x.device)
    else:
        raise RuntimeError(f"Unrecognised frequency scaling: {frequency_scaling}")

    return torch.cos((x + min_value) * freqs * math.pi / (max_value + min_value))
