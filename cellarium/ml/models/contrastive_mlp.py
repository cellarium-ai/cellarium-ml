# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from cellarium.ml.models.gather import GatherLayer
from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.models.nt_xent import NT_Xent
from cellarium.ml.transforms import Log1p, NormalizeTotal, Randomize, ZScore
from cellarium.ml.utilities.data import get_rank_and_num_replicas

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
        augment: nn.Module | Sequence[nn.Module | str],
        stats_path: str | None = None,
        temperature: float = 1.0,
        target_count: int = 10_000,
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

        stats = None if stats_path is None else torch.load(stats_path)
        
        self.transform = nn.Sequential(NormalizeTotal(target_count), Log1p())
        
        if isinstance(augment, Sequence):
            augment[augment.index('transform')] = self.transform
            if stats is not None:
                augment[augment.index('zscore')] = ZScore(stats["mean_g"], stats["std_g"], None)
            augment = nn.Sequential(*augment)
        self.augment = augment

        self.world_size = world_size
        self.Xent_loss = NT_Xent(batch_size, world_size, temperature)
        
        self.toy_ds = np.load('/home/jupyter/bw-bican-data/toy_ds.npy')

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng):
#         rank, num_replicas = get_rank_and_num_replicas()
#         logger.debug(f'RANK {rank}, N_REPLICA {num_replicas}')
#         logger.debug(f'WORLD {self.world_size}')
        
#         x_ng_full = torch.cat(GatherLayer.apply(x_ng), dim=0)
#         x_ng_full = x_ng
#         logger.debug('mini-batch')
#         logger.debug([
#             np.where((self.toy_ds == row).all(axis=1))[0].item() for row in x_ng.cpu().numpy()
#         ])
#         logger.debug('full batch')
#         logger.debug([
#             np.where((self.toy_ds == row).all(axis=1))[0].item() for row in x_ng_full.cpu().numpy()
#         ])
        
        # data augmentation for contrastive learning
        x_ng_twice = x_ng.repeat((2, 1))
        x_aug = x_ng_twice if self.augment is None else self.augment(x_ng_twice)
        x_aug1, x_aug2 = torch.chunk(x_aug, 2)

        # compute deep embeddings
        z1 = F.normalize(self.layers(x_aug1))
        z2 = F.normalize(self.layers(x_aug2))

        # SimCLR loss
        loss = self.Xent_loss(z1, z2)
        return loss

    def predict(self, x_ng: torch.Tensor, **kwargs: Any):
        with torch.no_grad():
            if self.transform is not None:
                x_ng = self.transform(x_ng)
            z = F.normalize(self.layers(x_ng))

        return z
