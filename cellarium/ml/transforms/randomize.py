# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn


class Randomize(nn.Module):
    """
    Randomly selects between the augmented and original data
    for each sample according to probability p_apply.

    Args:
        p_apply:
            Probability of selecting augmentation for each sample.
    """

    def __init__(self, p_apply):
        super().__init__()

        self.p_apply = p_apply

    def forward(self, x_aug: torch.Tensor, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_aug: Augmented gene counts.
            x_ng: Gene counts.

        Returns:
            Randomized augmented gene counts.
        """
        p_apply_n1 = torch.Tensor([self.p_apply]).expand(x_ng.shape[0], 1).type_as(x_ng)
        apply_mask_n1 = torch.bernoulli(p_apply_n1).bool()

        x_ng = torch.where(apply_mask_n1, x_aug, x_ng)
        return x_ng
