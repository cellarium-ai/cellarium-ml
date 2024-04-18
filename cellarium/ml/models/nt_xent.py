# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn

from cellarium.ml.models.gather import GatherLayer
from cellarium.ml.utilities.data import get_rank_and_num_replicas

import logging

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger()


class NT_Xent(nn.Module):
    """
    Normalized Temperature-scaled cross-entropy loss.

    Args:
        batch_size:
            Expected batch size per distributed process.
        world_size:
            Number of distributed processes.
        temperature:
            Logit scaling coefficient. A higher temperature reduces
            the scale of the output logits, resulting in a more volatile
            update step.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        temperature: float = 1.0,
    ):
        super(NT_Xent, self).__init__()

        self.batch_size = batch_size
        self.world_size = world_size

        self.temperature = temperature

        self.negative_mask_full = ~torch.eye(self.world_size * self.batch_size, dtype=bool).repeat((1, 2))
        
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # equivalent to CosineSimilarity on normalized inputs
        self.similarity_f = lambda x1, x2: torch.einsum("nc,mc->nm", x1, x2)

    def _slice_negative_mask(self, rank: int) -> torch.Tensor:
        """
        Returns row slice of full negative mask corresponding to the segment
        of the full batch held by the specified device.

        Args:
            rank:
                The rank of the specified device.
        """

        mask = torch.chunk(self.negative_mask_full, self.world_size, dim=0)[rank]
        return mask

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Gathers all inputs, then computes NT-Xent loss averaged over all
        2n augmented samples. Each sample's corresponding pair is used as
        its positive class, while the remaining (2n - 2) samples are its
        negative classes.
        """

        # gather embeddings from distributed processing
        if self.world_size > 1:
            z_i_full = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j_full = torch.cat(GatherLayer.apply(z_j), dim=0)
        else:
            z_i_full = z_i
            z_j_full = z_j
        
        assert len(z_i_full) == self.batch_size * self.world_size

        rank, _ = get_rank_and_num_replicas()
        negative_mask = self._slice_negative_mask(rank)
        
        # logger.debug(rank)

        z_both_full = torch.cat((z_i_full, z_j_full), dim=0)

        sim_i = self.similarity_f(z_i, z_both_full) / self.temperature
        sim_j = self.similarity_f(z_j, z_both_full) / self.temperature

        pos_i = torch.diag(sim_i, (self.world_size + rank) * self.batch_size)
        pos_j = torch.diag(sim_j, rank * self.batch_size)
        
        # logger.debug(sim_i)
        # logger.debug(pos_i)
        # logger.debug(sim_j)
        # logger.debug(pos_j)

        positive_samples = torch.cat((pos_i, pos_j))
        negative_samples = torch.cat([
            sim_i[negative_mask].reshape(self.batch_size, -1),
            sim_j[negative_mask].reshape(self.batch_size, -1)])
        
#         logger.debug('positive_samples')
#         logger.debug(positive_samples.shape)
        
#         logger.debug('negative_samples')
#         logger.debug(negative_samples.shape)

        labels = torch.zeros_like(positive_samples).long()
        logits = torch.cat((positive_samples.unsqueeze(1), negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        return loss
