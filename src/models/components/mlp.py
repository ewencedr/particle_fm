import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple multi-layer neural network.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        activation (str, optional): Activation function from `torch.nn`.
            Default: 'Tanh'
        final_activation (str, optional): Last activation. Default: None
        wrapper_func (callable, optional): Wrapper function for `nn.Linear`,
            e.g. st.util.spectral_norm. Default: None
    """

    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        activation="Tanh",
        final_activation=None,
        wrapper_func=None,
        **kwargs
    ):
        super().__init__()

        if not wrapper_func:

            def wrapper_func(x):
                return x

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)
        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(getattr(nn, activation)())
            layers.append(wrapper_func(nn.Linear(hidden_dims[i], hidden_dims[i + 1])))
        layers[-1].bias.data.fill_(0.0)

        if final_activation is not None:
            layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        """
        Args:
            x (tensor): Input with shape (..., in_dim)

        Returns:
            y (tensor): Output with shape (..., out_dim)
        """

        return self.net(x)


class DiffeqConcat(nn.Module):
    """Differential equation that concatenates the input and time.

    Args:
        net (Type[nn.Module]): Neural network that concatenates
            input `x`, time `t` and `latent` (optional) and outputs
            the derivative of the same size as `x`.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, x, latent=None, **kwargs):
        """
        Args:
            t (tensor): Time with shape (..., 1)
            x (tensor): Input with shape (..., dim)
            latent (tensor, optional): Latent vector with shape (..., latent_dim).
                Default: None

        Returns:
            dx (tensor): Derivative in `x` with shape (..., dim)
        """
        t = torch.ones_like(x[..., :1]) * t
        input = torch.cat([t, x], -1)
        if latent is not None:
            input = torch.cat([input, latent], -1)
        return self.net(input, **kwargs)


class DiffeqMLP(DiffeqConcat):
    """Differential equation defined with MLP.

    Args:
        Same as in `MLP`
    """

    def __init__(
        self, in_dim, hidden_dims, out_dim, activation="Tanh", final_activation=None, **kwargs
    ):
        super().__init__(MLP(in_dim, hidden_dims, out_dim, activation, final_activation))
