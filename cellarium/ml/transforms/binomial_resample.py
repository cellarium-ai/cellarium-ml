# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn
from torch.distributions import Bernoulli, Binomial, Uniform


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

    def __init__(self, p_binom_min, p_binom_max, p_apply):
        super().__init__()

        self.p_binom_min = p_binom_min
        self.p_binom_max = p_binom_max
        self.p_apply = p_apply

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Binomially resampled gene counts.
        """
        p_binom_ng = Uniform(self.p_binom_min, self.p_binom_max).sample(x_ng.shape).type_as(x_ng)
        p_apply_n = Bernoulli(probs=self.p_apply).sample(x_ng.shape[:1]).type_as(x_ng).bool()

        x_aug = Binomial(total_count=x_ng, probs=p_binom_ng).sample()

        x_ng = torch.where(p_apply_n.unsqueeze(1), x_ng, x_aug)
        return {"x_ng": x_ng}
