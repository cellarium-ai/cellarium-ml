# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from cellarium.ml.models.gather import GatherLayer

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
            temperature: float = 1.0):
        super(NT_Xent, self).__init__()
        
        self.batch_size = batch_size
        self.world_size = world_size
        
        self.temperature = temperature
        
        self.negative_mask = self._get_negative_mask(self.world_size * self.batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        # equivalent to CosineSimilarity on normalized inputs
        self.similarity_f = lambda x1, x2: torch.einsum('nc,mc->nm', x1, x2)

    def _get_negative_mask(self, n_batch: int) -> torch.Tensor:
        """
        Computes a (2n x 2n) boolean mask, where element m_ij == 1
        iff elements i and j correspond to different identities.
        
        Args:
            n_batch:
                The value of n.
        """
        
        ones = torch.ones(n_batch, dtype=bool)
        
        mask = torch.eye(2 * n_batch, dtype=bool)
        mask |= torch.diag(ones, n_batch)
        mask |= torch.diag(ones, -n_batch)
        
        return ~mask
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Gathers all inputs, then computes NT-Xent loss averaged over all
        2n augmented samples. Each sample's corresponding pair is used as
        its positive class, while the remaining (2n - 2) samples are its
        negative classes.
        """
        
        # gather embeddings from distributed processing
        if self.world_size > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        
        full_batch_size = z_i.shape[0]
        negative_mask = (
            self.negative_mask if full_batch_size == self.batch_size * self.world_size
            else self._get_negative_mask(full_batch_size))
        
        z_all = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z_all, z_all) / self.temperature

        sim_ij = torch.diag(sim, full_batch_size)
        sim_ji = torch.diag(sim, -full_batch_size)

        positive_samples = torch.cat((sim_ij, sim_ji))
        negative_samples = sim[negative_mask].reshape(2 * full_batch_size, -1)

        labels = torch.zeros_like(positive_samples).long()
        logits = torch.cat((positive_samples.unsqueeze(1), negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        
        return loss
