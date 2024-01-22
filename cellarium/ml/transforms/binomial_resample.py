# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn
from torch.distributions import Binomial, Uniform


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
    """

    def __init__(self, p_binom_min, p_binom_max):
        super().__init__()

        self.p_binom_min = p_binom_min
        self.p_binom_max = p_binom_max

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Binomially resampled gene counts.
        """
        p_binom_ng = Uniform(self.p_binom_min, self.p_binom_max).sample(x_ng.shape).type_as(x_ng)

        x_aug = Binomial(total_count=x_ng, probs=p_binom_ng).sample()
        return x_aug
