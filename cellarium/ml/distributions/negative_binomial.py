# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Number

import torch
from pyro.distributions import TorchDistribution, constraints
from torch.distributions.utils import broadcast_all, lazy_property


class NegativeBinomial(TorchDistribution):
    """Negative binomial distribution.

    Args:
        mu:
            Mean of the distribution.
        theta:
            Inverse dispersion.
    """

    THETA_THRESHOLD_STIRLING_SWITCH = 200

    arg_constraints = {"mu": constraints.greater_than_eq(0), "theta": constraints.greater_than_eq(0)}
    support = constraints.nonnegative_integer

    def __init__(self, mu: torch.Tensor, theta: torch.Tensor, validate_args: bool | None = None) -> None:
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
        return (self.mu + (self.mu**2) / self.theta).masked_fill(self.theta == 0, 0)

    @lazy_property
    def _gamma(self) -> torch.distributions.Gamma:
        # Note we avoid validating because self.theta can be zero.
        return torch.distributions.Gamma(
            concentration=self.theta,
            rate=(self.theta / self.mu).masked_fill(self.theta == 0, 1.0),
            validate_args=False,
        )

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
            self.theta > self.THETA_THRESHOLD_STIRLING_SWITCH,
            (value + self.theta - 0.5) * torch.log1p(value / self.theta) - value,
            (value + self.theta).lgamma() - self.theta.lgamma() - torch.xlogy(value, self.theta),
        )
        # The case self.theta == 0 and value == 0 has probability 1.
        # The case self.theta == 0 and value != 0 has probability 0.
        return (
            (delta - (value + self.theta) * torch.log1p(self.mu / self.theta)).masked_fill(self.theta == 0, 0)
            - (value + 1).lgamma()
            + torch.xlogy(value, self.mu)
        )
