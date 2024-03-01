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


class resnetBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=64):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = nn.LeakyReLU(self.linear(x))
        x = nn.LeakyReLU(self.linear2(x) + x)
        return x


class small_cond_ResNet_model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim_t: int = 6,
        dim_cond: int = 1,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.mlp1 = resnetBlock(
            in_features + dim_t + dim_cond,
            out_features=64,
            hidden_features=64,
        )
        self.mlp2 = resnetBlock(
            64 + dim_t + dim_cond,
            out_features=256,
            hidden_features=256,
        )
        self.mlp3 = resnetBlock(
            256 + dim_t + dim_cond,
            out_features=256,
            hidden_features=256,
        )
        self.mlp4 = resnetBlock(
            256 + dim_t + dim_cond,
            out_features=256,
            hidden_features=256,
        )
        self.mlp5 = resnetBlock(
            256 + dim_t + dim_cond,
            out_features=out_features,
            hidden_features=64,
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
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp5(x)
        return x


class cathode_classifier(nn.Module):
    def __init__(
        self,
        features: int = 4,
        layers: int = [64, 64, 64],
    ):
        super().__init__()
        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(features, nodes))
            self.layers.append(nn.ReLU())
            features = nodes
        self.layers.append(nn.Linear(features, 1))
        # self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)