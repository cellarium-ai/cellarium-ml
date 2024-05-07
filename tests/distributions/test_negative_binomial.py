# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pyro.distributions as dist
import pytest
import torch

from cellarium.ml.distributions import NegativeBinomial


@pytest.mark.parametrize("logits_shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("total_counts_shape", [(), (2,), (3, 2)])
def test_negative_binomial(logits_shape: torch.Size, total_counts_shape: torch.Size) -> None:
    logits = torch.randn(logits_shape)
    total_counts = torch.rand(total_counts_shape) * 10

    if len(total_counts_shape) == 2:
        total_counts[0] = 0

    pyro_dist = dist.NegativeBinomial(total_counts, logits=logits)  # type: ignore[attr-defined]

    mu = torch.exp(logits) * total_counts
    theta = total_counts
    cellarium_nb = NegativeBinomial(mu, theta)

    # shape
    assert cellarium_nb.batch_shape == pyro_dist.batch_shape

    # mean
    np.testing.assert_allclose(cellarium_nb.mean, pyro_dist.mean, rtol=1e-5)

    # variance
    np.testing.assert_allclose(cellarium_nb.variance, pyro_dist.variance, rtol=1e-5)

    # log_prob
    value = torch.randint(20, size=(3, 2))
    if len(total_counts_shape) == 2:
        value[0, 0] = 0
        value[0, 1] = 2.0
    pyro_log_prob = pyro_dist.log_prob(value)
    cellarium_log_prob = cellarium_nb.log_prob(value)
    np.testing.assert_allclose(pyro_log_prob, cellarium_log_prob, rtol=1e-5)

    # sample
    samples = cellarium_nb.sample(torch.Size([50_000]))

    expected_mean = cellarium_nb.mean
    actual_mean = samples.mean(0)
    np.testing.assert_allclose(actual_mean, expected_mean, atol=0.02, rtol=0.05)

    expected_var = cellarium_nb.variance
    actual_var = samples.var(0)
    np.testing.assert_allclose(actual_var, expected_var, atol=0.02, rtol=0.05)


@pytest.mark.parametrize("mu", torch.logspace(-4, 3, 8))
@pytest.mark.parametrize("theta", torch.logspace(-2, 6, 9))
def test_total_probability(mu: torch.Tensor, theta: torch.Tensor) -> None:
    values = torch.arange(0, 2 + int(mu * 1e3))
    log_probs = NegativeBinomial(mu, theta).log_prob(values)
    expected = torch.tensor(0.0)
    actual = log_probs.logsumexp(0)
    assert torch.allclose(actual, expected, atol=5e-4)
