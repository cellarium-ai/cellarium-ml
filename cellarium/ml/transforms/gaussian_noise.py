# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn
from torch.distributions import Bernoulli, Normal, Uniform


class GaussianNoise(nn.Module):
    """
    Adds Gaussian noise to gene counts.

    For each count, Gaussian sigma is independently
    and uniformly sampled according to the bounding parameters,
    yielding the sigma matrix sigma_ng.

    .. math::

        y_{ng} = x_{ng} + N(0, \\sigma_{ng})

    Args:
        sigma_min:
            Lower bound on Gaussian sigma parameter.
        sigma_max:
            Upper bound on Gaussian sigma parameter.
    """

    def __init__(self, sigma_min, sigma_max, p_apply):
        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_apply = p_apply

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Gene counts with added Gaussian noise.
        """
        sigma_min = torch.Tensor([self.sigma_min]).type_as(x_ng)
        sigma_max = torch.Tensor([self.sigma_max]).type_as(x_ng)
        p_apply = torch.Tensor([self.p_apply]).type_as(x_ng)

        sigma_ng = Uniform(sigma_min, sigma_max).sample(x_ng.shape).squeeze(-1)
        p_apply_n1 = Bernoulli(probs=p_apply).sample(x_ng.shape[:1]).bool()

        x_aug = x_ng + Normal(0, sigma_ng).sample()

        x_ng = torch.where(p_apply_n1, x_aug, x_ng)
        return {"x_ng": x_ng}
