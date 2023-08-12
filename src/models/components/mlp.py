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
