# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.models.nt_xent import NT_Xent


class ContrastiveMLP(CellariumModel, PredictMixin):
    """
    Multilayer perceptron trained with contrastive learning.

    Args:
        g_genes:
            Number of genes in each entry (network input size).
        hidden_size:
            Dimensionality of the fully-connected hidden layers.
        embed_dim:
            Size of embedding (network output size).
        world_size:
            Number of devices used in training.
        temperature:
            Parameter governing Normalized Temperature-scaled cross-entropy (NT-Xent) loss.
    """

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

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng:
                Gene counts matrix.
        Returns:
            A dictionary with the loss value.
        """
        # compute deep embeddings
        z = F.normalize(self.layers(x_ng))

        # split input into augmented halves
        z1, z2 = torch.chunk(z, 2)

        # SimCLR loss
        loss = self.Xent_loss(z1, z2)
        return {"loss": loss}

    def predict(self, x_ng: torch.Tensor, **kwargs: Any):
        """
        Send (transformed) data through the model and return outputs.

        Args:
            x_ng:
                Gene counts matrix.
        Returns:
            A dictionary with the embedding matrix.
        """
        with torch.no_grad():
            z = F.normalize(self.layers(x_ng))
        return {"x_ng": z}
