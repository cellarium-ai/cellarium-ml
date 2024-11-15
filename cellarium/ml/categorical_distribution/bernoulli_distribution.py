# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

# # Copyright Contributors to the Cellarium project.
# # SPDX-License-Identifier: BSD-3-Clause

# import torch
# from pyro.distributions import Bernoulli as PyroBernoulli
# from torch.distributions.utils import broadcast_all


# class CustomPyroBernoulli(PyroBernoulli):
#     def log_prob(self, value):
#         # Validate inputs (optional, inherited behavior)
#         if self._validate_args:
#             self._validate_sample(value)

#         # Broadcast logits and value to compatible shapes
#         logits, value = broadcast_all(self.logits, value)

#         # Custom log_prob computation
#         log_probs = value * logits + (1 - value) * (-logits - torch.logsumexp(torch.stack([logits * 0, -logits], dim=0), dim=0))
#         return log_probs

import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import Bernoulli
from torch.distributions.utils import logits_to_probs


class CustomPyroBernoulli(TorchDistribution):
    arg_constraints = {"logits": torch.distributions.constraints.real}
    support = torch.distributions.constraints.boolean
    has_rsample = False

    def __init__(self, logits):
        self.logits = logits
        self._bernoulli = Bernoulli(logits=logits)
        super().__init__(self._bernoulli.batch_shape, self._bernoulli.event_shape)

    def sample(self, sample_shape=torch.Size()):
        return self._bernoulli.sample(sample_shape)

    def log_prob(self, value):
        logits = self.logits
        logsumexp_term = torch.logsumexp(torch.stack([torch.zeros_like(logits), -logits], dim=0), dim=0)
        return value * logits + (1 - value) * (-logits - logsumexp_term)

    def to_event(self, n=1):
        return CustomPyroBernoulli(self.logits).expand(self._bernoulli.batch_shape).to_event(n)

    @property
    def mean(self):
        return logits_to_probs(self.logits)

    @property
    def variance(self):
        probs = logits_to_probs(self.logits)
        return probs * (1 - probs)
