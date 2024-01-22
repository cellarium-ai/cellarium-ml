# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.models.nt_xent import NT_Xent
from cellarium.ml.transforms import Log1p, NormalizeTotal, Randomize, ZScore


class ContrastiveMLP(CellariumModel, PredictMixin):
    def __init__(
        self,
        g_genes: int,
        hidden_size: List[int],
        embed_dim: int,
        stats_path: str,
        batch_size: int,
        world_size: int,
        augment: nn.Module | List[nn.Module],
        temperature: float = 1.0,
        target_count: int = 10_000,
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

        stats = torch.load(stats_path)
        
        self.transform = nn.Sequential(NormalizeTotal(target_count), Log1p())
        
        if isinstance(augment, List):
            augment[augment.index('transform')] = self.transform
            augment[augment.index('ZScore')] = ZScore(stats["mean_g"], stats["std_g"], None)
            augment = nn.Sequential(*augment)
        self.augment = augment

        self.world_size = world_size
        self.Xent_loss = NT_Xent(batch_size, world_size, temperature)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x = tensor_dict["x_ng"]
        return (x,), {}

    def forward(self, x_ng):
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
