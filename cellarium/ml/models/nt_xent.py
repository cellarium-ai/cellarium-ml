# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn

from cellarium.ml.distributed.gather import GatherLayer
from cellarium.ml.utilities.data import get_rank_and_num_replicas


class NT_Xent(nn.Module):
    """
    Normalized Temperature-scaled cross-entropy loss.

    **References:**

    1. `A simple framework for contrastive learning of visual representations
       (Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.)
       <https://arxiv.org/abs/2002.05709>`_.

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
        world_size: int,
        temperature: float = 1.0,
    ):
        super(NT_Xent, self).__init__()

        self.world_size = world_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _slice_negative_mask(self, size: int, rank: int) -> torch.Tensor:
        """
        Returns row slice of full negative mask corresponding to the segment
        of the full batch held by the specified device.

        Args:
            rank:
                The rank of the specified device.
        """
        negative_mask_full = ~torch.eye(size).bool().repeat((1, 2))
        mask = torch.chunk(negative_mask_full, self.world_size, dim=0)[rank]
        return mask

    @staticmethod
    def _similarity_fn(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between normalized vectors,
        which is equivalent to a standard inner product.
        """
        return torch.einsum("nc,mc->nm", z1, z2)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Gathers all inputs, then computes NT-Xent loss averaged over all
        2n augmented samples.
        """
        # gather embeddings from distributed forward pass
        if self.world_size > 1:
            z_i_full = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j_full = torch.cat(GatherLayer.apply(z_j), dim=0)
        else:
            z_i_full = z_i
            z_j_full = z_j

        assert (
            len(z_i_full) % self.world_size == 0
        ), "Expected batch to evenly divide across devices (set drop_last to True)."

        batch_size = len(z_i_full) // self.world_size
        rank, _ = get_rank_and_num_replicas()
        negative_mask = self._slice_negative_mask(len(z_i_full), rank)

        z_both_full = torch.cat((z_i_full, z_j_full), dim=0)

        # normalized similarity logits between device minibatch and full batch embeddings
        sim_i = NT_Xent._similarity_fn(z_i, z_both_full) / self.temperature
        sim_j = NT_Xent._similarity_fn(z_j, z_both_full) / self.temperature

        pos_i = torch.diag(sim_i, (self.world_size + rank) * batch_size)
        pos_j = torch.diag(sim_j, rank * batch_size)

        positive_samples = torch.cat((pos_i, pos_j))
        negative_samples = torch.cat(
            [sim_i[negative_mask].reshape(batch_size, -1), sim_j[negative_mask].reshape(batch_size, -1)]
        )

        labels = torch.zeros_like(positive_samples).long()
        logits = torch.cat((positive_samples.unsqueeze(1), negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        return loss
