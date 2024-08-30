# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn
from torch.distributions import Binomial

from .randomize import Randomize


class BinomialResample(nn.Module):
    """
    Binomial resampling of gene counts.

    For each count, the parameter to the binomial distribution
    is independently and uniformly sampled according to the
    bounding parameters, yielding the parameter matrix p_ng.

    .. math::

        y_{ng} = Binomial(n=x_{ng}, p=p_{ng})

    Args:
        p_binom_min:
            Lower bound on binomial distribution parameter.
        p_binom_max:
            Upper bound on binomial distribution parameter.
        p_apply:
            Probability of applying transform to each sample.
    """

    def __init__(self, p_binom_min: float, p_binom_max: float, p_apply: float):
        super().__init__()

        self.p_binom_min = p_binom_min
        self.p_binom_max = p_binom_max
        self.randomize = Randomize(p_apply)

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Binomially resampled gene counts.
        """
        p_binom_ng = torch.empty_like(x_ng.shape).uniform_(self.p_binom_min, self.p_binom_max)
        x_aug = Binomial(total_count=x_ng, probs=p_binom_ng).sample()

        x_ng = self.randomize(x_aug, x_ng)
        return {"x_ng": x_ng}
