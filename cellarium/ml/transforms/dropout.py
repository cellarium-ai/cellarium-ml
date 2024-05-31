# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn
from torch.distributions import Bernoulli, Uniform


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
    """

    def __init__(self, p_dropout_min, p_dropout_max, p_apply):
        super().__init__()

        self.p_dropout_min = p_dropout_min
        self.p_dropout_max = p_dropout_max
        self.p_apply = p_apply

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Gene counts with random dropout.
        """
        p_dropout_ng = Uniform(self.p_dropout_min, self.p_dropout_max).sample(x_ng.shape).type_as(x_ng)
        p_apply_n = Bernoulli(probs=self.p_apply).sample(x_ng.shape[:1]).type_as(x_ng).bool()

        x_aug = torch.clone(x_ng)
        x_aug[Bernoulli(probs=p_dropout_ng).sample().bool()] = 0

        x_ng = torch.where(p_apply_n.unsqueeze(1), x_ng, x_aug)
        return {"x_ng": x_ng}
