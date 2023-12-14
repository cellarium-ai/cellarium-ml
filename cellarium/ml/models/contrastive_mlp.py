# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn
import torch.nn.functional as F

from typing import Any, List

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.models.nt_xent import NT_Xent
from cellarium.ml.transforms import Log1p, NormalizeTotal


class ContrastiveMLP(CellariumModel, PredictMixin):

    def __init__(
            self,
            input_size: int,
            hidden_size: List[int],
            output_size: int,
            stats_path: str,
            batch_size: int,
            world_size: int,
            temperature: float = 1.0):
        super(ContrastiveMLP, self).__init__()

        layer_list = []
        layer_list.append(nn.Linear(input_size, hidden_size[0]))
        layer_list.append(nn.ReLU())
        for size_i, size_j in zip(hidden_size[:-1], hidden_size[1:]):
            layer_list.append(nn.Linear(size_i, size_j))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.BatchNorm1d(size_j))
        layer_list.append(nn.Linear(hidden_size[-1], output_size))
        
        self.layers = nn.Sequential(*layer_list)
        
        stats = torch.load(stats_path)
        
        self.transform = nn.Sequential(
            NormalizeTotal(),
            Log1p())
        self.augment = nn.Sequential(
            Randomize(BinomialResample(0.8, 1), 0.7),
            self.transform,
            ZScore(stats['mean_g'], stats['std_g'], None),
            Randomize(Dropout(0, 0.2), 1),
            Randomize(GaussianNoise(0.01, 0.2), 0.7))
        
        self.world_size = world_size
        self.Xent_loss = NT_Xent(batch_size, world_size, temperature)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x = tensor_dict['X']
        return (x,), {}
    
    def forward(self, x_ng):
        # data augmentation for contrastive learning
        x_aug_all = self.augment(x_ng.repeat((2, 1)))
        x_aug1, x_aug2 = torch.chunk(x_aug_all, 2)
        
        # compute deep embeddings
        z1 = F.normalize(self.layers(x_aug1))
        z2 = F.normalize(self.layers(x_aug2))
        
        # SimCLR loss
        loss = self.Xent_loss(z1, z2)
        return loss

    def predict(self, x_ng: torch.Tensor, **kwargs: Any):
        with torch.no_grad():
            x_norm = self.transform(x_ng)
            z = F.normalize(self.layers(x_norm))
        
        return z
