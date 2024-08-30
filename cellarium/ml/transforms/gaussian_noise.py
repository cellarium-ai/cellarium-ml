# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn

from .randomize import Randomize


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
        p_apply:
            Probability of applying transform to each sample.
    """

    def __init__(self, sigma_min, sigma_max, p_apply):
        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.randomize = Randomize(p_apply)

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts (log-transformed).

        Returns:
            Gene counts with added Gaussian noise.
        """
        sigma_ng = torch.empty_like(x_ng.shape).uniform_(self.sigma_min, self.sigma_max)
        x_aug = x_ng + torch.normal(std=sigma_ng)

        x_ng = self.randomize(x_aug, x_ng)
        return {"x_ng": x_ng}
