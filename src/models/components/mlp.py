import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int] = [64, 64],
        activation: str = "ELU",
    ):
        layers = []

        for a, b in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([nn.Linear(a, b), getattr(nn, activation)()])

        super().__init__(*layers[:-1])


class small_cond_MLP_model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "ELU",
        dim_t: int = 6,
        dim_cond: int = 1,
    ):
        super().__init__()
        self.mlp1 = MLP(
            in_features + dim_t + dim_cond,
            out_features=64,
            hidden_features=[64, 64],
            activation=activation,
        )
        self.mlp2 = MLP(
            64 + dim_t + dim_cond,
            out_features=256,
            hidden_features=[256, 256],
            activation=activation,
        )
        self.mlp3 = MLP(
            256 + dim_t + dim_cond,
            out_features=256,
            hidden_features=[256, 256],
            activation=activation,
        )
        self.mlp4 = MLP(
            256 + dim_t + dim_cond,
            out_features=out_features,
            hidden_features=[64, 64],
            activation=activation,
        )

    def forward(self, t, x, cond):
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp1(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp2(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp4(x)
        return x


class very_small_cond_MLP_model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "ELU",
        dim_t: int = 6,
        dim_cond: int = 1,
    ):
        super().__init__()
        self.mlp1 = MLP(
            in_features + dim_t + dim_cond,
            out_features=out_features,
            hidden_features=[64, 64],
            activation=activation,
        )

    def forward(self, t, x, cond):
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp1(x)
        return x
