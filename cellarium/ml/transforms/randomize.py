# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn


class Randomize(nn.Module):
    """
    Randomizely applies transform with probability p;
    otherwise returns original gene counts.
    
    Args:
        transform:
            Transform to apply.
        p_apply:
            Probability that transform is applied.
    """

    def __init__(self, transform: nn.Module, p_apply: float):
        super().__init__()
        
        self.transform = transform
        self.p_apply = p_apply
    
    def forward(self, x_ng):
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Gene counts with randomly applied transform.
        """
        return self.transform(x_ng) if torch.rand(1) < self.p_apply else x_ng
