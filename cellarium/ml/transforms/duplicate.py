# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn


class Duplicate(nn.Module):
    """
    Duplicates every row of the input tensor,
    used for contrastive augmentations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Duplicated counts.
        """
        return {'x_ng': x_ng.repeat((2, 1))}
