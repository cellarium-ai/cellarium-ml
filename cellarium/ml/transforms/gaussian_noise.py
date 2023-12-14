# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn
from torch.distributions import Normal, Uniform


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

    def __init__(self, sigma_min, sigma_max):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Gene counts with added Gaussian noise.
        """
        sigma_ng = Uniform(self.sigma_min, self.sigma_max).sample(x_ng.shape).type_as(x_ng)
        
        return x_ng + Normal(0, sigma_ng).sample()
