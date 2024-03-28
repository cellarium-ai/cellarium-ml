# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Probability distributions"""

import torch
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all
from numbers import Number


class NegativeBinomial(Distribution):
    """Negative binomial distribution.

    Args:
        mu:
            Mean of the distribution.
        theta:
            Inverse dispersion.
    """

    arg_constraints = {"mu": constraints.greater_than_eq(0), "theta": constraints.greater_than_eq(0)}
    support = constraints.nonnegative_integer

    def __init__(self, mu: torch.Tensor, theta: torch.Tensor, validate_args: bool = False) -> None:
        self.mu, self.theta = broadcast_all(mu, theta)
        if isinstance(mu, Number) and isinstance(theta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.mu + (self.mu**2) / self.theta

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        # Original implementation from scVI:
        #
        #   log_theta_mu_eps = torch.log(self.theta + self.mu + self.eps)
        #   return (
        #       self.theta * (torch.log(self.theta + self.eps) - log_theta_mu_eps)
        #       + value * (torch.log(self.mu + self.eps) - log_theta_mu_eps)
        #       + torch.lgamma(value + self.theta)
        #       - torch.lgamma(self.theta)
        #       - torch.lgamma(value + 1)
        #   )
        delta = torch.where(
            (value / self.theta < 1e-2) & (self.theta > 1e2),
            (value + self.theta - 0.5) * torch.log1p(value / self.theta) - value,
            (value + self.theta).lgamma() - self.theta.lgamma() - torch.xlogy(value, self.theta),
        )
        return (
            delta
            - (value + self.theta) * torch.log1p(self.mu / self.theta)
            - (value + 1).lgamma()
            + torch.xlogy(value, self.mu)
        )
