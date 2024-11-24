# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints


class CustomPyroBernoulli(TorchDistribution):
    arg_constraints = {"log_prob_tensor": constraints.real, "log1m_prob_tensor": constraints.real}
    support = constraints.boolean
    has_rsample = False

    def __init__(self, log_prob_tensor, log1m_prob_tensor, validate_args=None):
        """
        Custom Bernoulli-like distribution where log probabilities are directly provided.

        Args:
            log_prob_tensor (torch.Tensor): Logarithm of probabilities for the event happening.
            log1m_prob_tensor (torch.Tensor): Logarithm of the complement of probabilities.
            validate_args (bool): Whether to validate input arguments.
        """
        self.log_prob_tensor = log_prob_tensor
        self.log1m_prob_tensor = log1m_prob_tensor
        self.logits = log_prob_tensor - log1m_prob_tensor  # For compatibility, not used directly
        batch_shape = torch.broadcast_shapes(log_prob_tensor.shape, log1m_prob_tensor.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # Use log_prob_tensor for 1 and log1m_prob_tensor for 0
        temp = torch.where(value == 1, torch.clamp(self.log_prob_tensor,min=-100,max=100),
                           torch.clamp(self.log1m_prob_tensor,min=-100,max=100))
        if torch.isnan(temp).any() or torch.isinf(temp).any():
            print("NAN OR INF IN TEMP")
        return torch.where(value == 1, torch.clamp(self.log_prob_tensor,min=-100,max=100),
                           torch.clamp(self.log1m_prob_tensor,min=-100,max=100))

    def entropy(self):
        # Compute probabilities from log_prob_tensor and log1m_prob_tensor
        p = self.log_prob_tensor.exp()  # Probability for event = 1
        p1m = self.log1m_prob_tensor.exp()  # Probability for event = 0 (1-p)
        # Calculate entropy: H = -p * log(p) - (1-p) * log(1-p)
        return -(p * self.log_prob_tensor + p1m * self.log1m_prob_tensor)

    @property
    def mean(self):
        return self.log_prob_tensor.exp()

    @property
    def variance(self):
        mean = self.mean
        return mean * (1 - mean)
