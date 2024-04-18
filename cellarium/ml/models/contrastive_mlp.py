# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.models.nt_xent import NT_Xent
from cellarium.ml.transforms import Log1p, NormalizeTotal, Randomize, ZScore
from cellarium.ml.utilities.data import get_rank_and_num_replicas

from lightly.loss import NTXentLoss

import logging

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger()


class ContrastiveMLP(CellariumModel, PredictMixin):
    def __init__(
        self,
        g_genes: int,
        hidden_size: Sequence[int],
        embed_dim: int,
        batch_size: int,
        world_size: int,
        temperature: float = 1.0,
    ):
        super(ContrastiveMLP, self).__init__()

        layer_list = []
        layer_list.append(nn.Linear(g_genes, hidden_size[0]))
        # layer_list.append(nn.BatchNorm1d(hidden_size[0]))
        # layer_list.append(nn.LayerNorm(hidden_size[0]))
        layer_list.append(nn.ReLU())
        for size_i, size_j in zip(hidden_size[:-1], hidden_size[1:]):
            layer_list.append(nn.Linear(size_i, size_j))
            # layer_list.append(nn.BatchNorm1d(size_j))
            # layer_list.append(nn.LayerNorm(size_j))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_size[-1], embed_dim))

        self.layers = nn.Sequential(*layer_list)

        self.world_size = world_size
        self.Xent_loss = NTXentLoss(temperature=1, gather_distributed=True)
            
        # self.Xent_loss = NT_Xent(batch_size, world_size, temperature)
        
        self.toy_ds = np.load('/home/jupyter/bw-bican-data/toy_ds_40.npy')
        # logger.debug('RAW DATA')
        # logger.debug(self.toy_ds)

    
    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        rank, num_replicas = get_rank_and_num_replicas()
#         logger.debug(f'RANK {rank}, N_REPLICA {num_replicas}')
#         logger.debug(f'WORLD {self.world_size}')
        
        # split input into augmented halves
        x_aug1, x_aug2 = torch.chunk(x_ng, 2)
        
        # np.save(f'/home/jupyter/bw-bican-data/toy_ds_40_aug_gpu-{num_replicas}_rank-{rank}.npy', x_aug1.cpu().numpy())
        
        # logger.debug('mini-batch dupe')
        # logger.debug([
        #     np.where((self.toy_ds == row).all(axis=1))[0].item() for row in x_ng.cpu().numpy()
        # ])

        # compute deep embeddings
        z1 = F.normalize(self.layers(x_aug1))
        z2 = F.normalize(self.layers(x_aug2))

        # SimCLR loss
        loss = self.Xent_loss(z1, z2)
        return {'loss': loss}

    def predict(self, x_ng: torch.Tensor, **kwargs: Any):
        with torch.no_grad():
            z = F.normalize(self.layers(x_ng))

        return z
