from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int,
        non_linearity: Callable[[], nn.Module],
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        assert n_layers >= 0
        if n_layers == 0:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), non_linearity()]
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(non_linearity())
            layers.append(nn.Linear(hidden_dim, out_dim))

        super().__init__(*layers)
