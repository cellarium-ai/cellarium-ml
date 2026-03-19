# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn

from .randomize import Randomize


class Dropout(nn.Module):
    """
    Applies random dropout to gene counts.

    For each count, the dropout parameter is independently
    and uniformly sampled according to the bounding parameters,
    yielding the parameter matrix p_ng.

    .. math::

        y_{ng} = x_{ng} * (1 - Bernoulli(p_ng))

    Args:
        p_dropout_min:
            Lower bound on dropout parameter.
        p_dropout_max:
            Upper bound on dropout parameter.
        p_apply:
            Probability of applying transform to each sample.
    """

    def __init__(self, p_dropout_min, p_dropout_max, p_apply):
        super().__init__()

        self.p_dropout_min = p_dropout_min
        self.p_dropout_max = p_dropout_max
        self.randomize = Randomize(p_apply)

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Gene counts with random dropout.
        """
        p_dropout_ng = torch.empty_like(x_ng).uniform_(self.p_dropout_min, self.p_dropout_max)
        x_aug = torch.where(torch.bernoulli(p_dropout_ng).bool(), 0, x_ng)

        x_ng = self.randomize(x_aug, x_ng)
        return {"x_ng": x_ng}
