# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn


class Duplicate(nn.Module):
    """
    Duplicates every row of the input tensor,
    used for contrastive augmentations.
    """

    def __init__(self, enabled=True):
        """
        Args:
            enabled:
                If True, performs duplication; otherwise does nothing.
                Set False when performing model inference so the
                transformation pipeline remains consistent with training.
        """
        super().__init__()
        self.enabled = enabled

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Duplicated counts.
        """
        if self.enabled:
            x_ng = x_ng.repeat((2, 1))
        return {"x_ng": x_ng}
