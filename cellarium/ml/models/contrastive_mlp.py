# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.models.nt_xent import NT_Xent

import pdb


class ContrastiveMLP(CellariumModel, PredictMixin):
    def __init__(
        self,
        g_genes: int,
        hidden_size: Sequence[int],
        embed_dim: int,
        world_size: int,
        temperature: float = 1.0,
    ):
        super(ContrastiveMLP, self).__init__()

        layer_list = []
        layer_list.append(nn.Linear(g_genes, hidden_size[0]))
        layer_list.append(nn.BatchNorm1d(hidden_size[0]))
        layer_list.append(nn.ReLU())
        for size_i, size_j in zip(hidden_size[:-1], hidden_size[1:]):
            layer_list.append(nn.Linear(size_i, size_j))
            layer_list.append(nn.BatchNorm1d(size_j))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_size[-1], embed_dim))

        self.layers = nn.Sequential(*layer_list)

        self.world_size = world_size

        self.Xent_loss = NT_Xent(world_size, temperature)

        self.reset_parameters()

    
    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        # compute deep embeddings
        z = F.normalize(self.layers(x_ng))

        # pdb.set_trace()
        
        # split input into augmented halves
        z1, z2 = torch.chunk(z, 2)

        # SimCLR loss
        loss = self.Xent_loss(z1, z2)
        return {'loss': loss}

    def predict(self, x_ng: torch.Tensor, **kwargs: Any):
        with torch.no_grad():
            z = F.normalize(self.layers(x_ng))
        return torch.chunk(z, 2)[0]

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
