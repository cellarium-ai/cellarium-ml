# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from cellarium.ml.distributions.unnormalized_categorical import PyroCategorical, TorchCategorical


def test_initialization_with_probs_1d():
    """Test initialization with 1D probability tensor."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = TorchCategorical(probs=probs)

    # Check that probs are normalized
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0))
    assert dist._num_events == 4
    assert dist.batch_shape == torch.Size([])


def test_initialization_with_probs_2d():
    """Test initialization with 2D (batched) probability tensor."""
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.0, 0.0]])
    dist = TorchCategorical(probs=probs)

    # Check that probs are normalized per batch
    assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(2))
    assert dist._num_events == 4
    assert dist.batch_shape == torch.Size([2])


def test_initialization_with_unnormalized_probs():
    """Test that unnormalized probabilities are normalized."""
    # Unnormalized probabilities
    probs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    dist = TorchCategorical(probs=probs)

    # Should be normalized to sum to 1
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0))
    expected_probs = probs / probs.sum()
    assert torch.allclose(dist.probs, expected_probs)


def test_initialization_with_logits_1d():
    """Test initialization with 1D logits tensor."""
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    dist = TorchCategorical(logits=logits)

    # Check that logits are normalized (logsumexp)
    assert torch.allclose(dist.logits.logsumexp(dim=-1), torch.tensor(0.0), atol=1e-6)
    assert dist._num_events == 4
    assert dist.batch_shape == torch.Size([])


def test_initialization_with_logits_2d():
    """Test initialization with 2D (batched) logits tensor."""
    logits = torch.randn(3, 5)
    dist = TorchCategorical(logits=logits)

    # Check that logits are normalized per batch
    assert torch.allclose(dist.logits.logsumexp(dim=-1), torch.zeros(3), atol=1e-6)
    assert dist._num_events == 5
    assert dist.batch_shape == torch.Size([3])


def test_initialization_validation_both_params():
    """Test that providing both probs and logits raises ValueError."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Either `probs` or `logits` must be specified, but not both"):
        TorchCategorical(probs=probs, logits=logits)


def test_initialization_validation_no_params():
    """Test that providing neither probs nor logits raises ValueError."""
    with pytest.raises(ValueError, match="Either `probs` or `logits` must be specified, but not both"):
        TorchCategorical()


def test_initialization_validation_0d_probs():
    """Test that 0-dimensional probs raises ValueError."""
    probs = torch.tensor(0.5)

    with pytest.raises(ValueError, match="`probs` parameter must be at least one-dimensional"):
        TorchCategorical(probs=probs)


def test_initialization_validation_0d_logits():
    """Test that 0-dimensional logits raises ValueError."""
    logits = torch.tensor(1.0)

    with pytest.raises(ValueError, match="`logits` parameter must be at least one-dimensional"):
        TorchCategorical(logits=logits)


def test_sample_validity():
    """Test that samples are valid category indices."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = TorchCategorical(probs=probs)

    samples = dist.sample(torch.Size([100]))

    # All samples should be integers in [0, 3]
    assert samples.dtype == torch.long
    assert torch.all(samples >= 0)
    assert torch.all(samples < 4)


def test_sample_batched():
    """Test sampling from batched distribution."""
    probs = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
    dist = TorchCategorical(probs=probs)

    samples = dist.sample(torch.Size([50]))

    # Shape should be [50, 2]
    assert samples.shape == torch.Size([50, 2])
    assert torch.all(samples >= 0)
    assert torch.all(samples < 2)


def test_sample_distribution():
    """Test that samples follow the expected distribution."""
    # Heavily biased distribution
    probs = torch.tensor([0.99, 0.01])
    dist = TorchCategorical(probs=probs)

    samples = dist.sample(torch.Size([10000]))

    # Most samples should be 0
    empirical_prob_0 = (samples == 0).float().mean()
    assert empirical_prob_0 > 0.95  # Should be very close to 0.99


def test_log_prob_basic():
    """Test log_prob returns correct values."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = TorchCategorical(probs=probs)

    values = torch.tensor([0, 1, 2, 3])
    log_probs = dist.log_prob(values)

    # Should match log of normalized probabilities
    expected = torch.log(probs / probs.sum())
    assert torch.allclose(log_probs, expected, atol=1e-6)


def test_log_prob_batched():
    """Test log_prob with batched distribution."""
    probs = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
    dist = TorchCategorical(probs=probs)

    values = torch.tensor([0, 1])
    log_probs = dist.log_prob(values)

    assert log_probs.shape == torch.Size([2])
    # First batch, value=0: log(0.5)
    # Second batch, value=1: log(0.1)
    expected = torch.tensor([torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.1))])
    assert torch.allclose(log_probs, expected, atol=1e-6)


def test_log_prob_with_logits_init():
    """Test log_prob when distribution is initialized with logits."""
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    dist = TorchCategorical(logits=logits)

    values = torch.tensor([0, 1, 2, 3])
    log_probs = dist.log_prob(values)

    # Log probs should sum to 1 when exponentiated
    assert torch.allclose(log_probs.exp().sum(), torch.tensor(1.0))


def test_property_mean():
    """Test that mean property returns NaN (as expected for categorical)."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    dist = TorchCategorical(probs=probs)

    mean = dist.mean
    assert torch.isnan(mean)


def test_property_mode():
    """Test that mode property returns argmax of probabilities."""
    probs = torch.tensor([0.1, 0.2, 0.5, 0.2])
    dist = TorchCategorical(probs=probs)

    mode = dist.mode
    assert mode == 2  # Index of max probability


def test_property_mode_batched():
    """Test mode property with batched distribution."""
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    dist = TorchCategorical(probs=probs)

    mode = dist.mode
    expected = torch.tensor([0, 1, 2])
    assert torch.equal(mode, expected)


def test_property_variance():
    """Test that variance property returns NaN (as expected for categorical)."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    dist = TorchCategorical(probs=probs)

    variance = dist.variance
    assert torch.isnan(variance)


def test_property_support():
    """Test that support property returns correct range."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = TorchCategorical(probs=probs)

    support = dist.support
    # Support should be integers from 0 to 3
    assert support.lower_bound == 0
    assert support.upper_bound == 3


def test_lazy_property_probs_from_logits():
    """Test lazy conversion from logits to probs."""
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    dist = TorchCategorical(logits=logits)

    # Initially, only logits should be in __dict__
    assert "logits" in dist.__dict__
    assert "probs" not in dist.__dict__

    # Accessing probs should trigger lazy computation
    probs = dist.probs
    assert "probs" in dist.__dict__

    # Probs should sum to 1
    assert torch.allclose(probs.sum(), torch.tensor(1.0))

    # Converting back to logits should be consistent
    logits_from_probs = torch.log(probs)
    # The normalized logits should match the computed log probs
    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    assert torch.allclose(logits_from_probs, normalized_logits, atol=1e-6)


def test_lazy_property_logits_from_probs():
    """Test lazy conversion from probs to logits."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = TorchCategorical(probs=probs)

    # Initially, only probs should be in __dict__
    assert "probs" in dist.__dict__
    assert "logits" not in dist.__dict__

    # Accessing logits should trigger lazy computation
    logits = dist.logits
    assert "logits" in dist.__dict__

    # Converting back to probs should be consistent
    probs_from_logits = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs_from_logits, dist.probs, atol=1e-6)


def test_conversion_consistency():
    """Test that probs<->logits conversions are consistent."""
    original_probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

    # Start with probs
    dist1 = TorchCategorical(probs=original_probs)
    logits = dist1.logits

    # Create new distribution with those logits
    dist2 = TorchCategorical(logits=logits)
    probs = dist2.probs

    # Should get back normalized original probs
    expected_probs = original_probs / original_probs.sum()
    assert torch.allclose(probs, expected_probs, atol=1e-6)


def test_basic_functionality():
    """Test that PyroCategorical works for basic operations."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    dist = PyroCategorical(probs=probs)

    # Should work like TorchCategorical for basic ops
    assert dist._num_events == 4
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0))

    samples = dist.sample(torch.Size([10]))
    assert samples.shape == torch.Size([10])


def test_log_prob_normal_case():
    """Test log_prob for normal case (without enumeration optimization)."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = PyroCategorical(probs=probs)

    values = torch.tensor([0, 1, 2, 3])
    log_probs = dist.log_prob(values)

    # Should match TorchCategorical
    torch_dist = TorchCategorical(probs=probs)
    torch_log_probs = torch_dist.log_prob(values)
    assert torch.allclose(log_probs, torch_log_probs)


def test_log_prob_enumerated_support():
    """Test log_prob optimization with enumerated support."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = PyroCategorical(probs=probs)

    # Use enumerate_support to get optimized values
    values = dist.enumerate_support(expand=False)

    # Check that the special attribute is set
    assert hasattr(values, "_pyro_categorical_support")
    assert values._pyro_categorical_support == id(dist)

    # Compute log_prob with optimized path
    log_probs = dist.log_prob(values)

    # Should match the logits (since we're evaluating all categories)
    assert torch.allclose(log_probs, dist.logits)


def test_log_prob_enumerated_vs_normal():
    """Test that enumerated and normal log_prob give same results."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist = PyroCategorical(probs=probs)

    # Normal path
    normal_values = torch.arange(4)
    normal_log_probs = dist.log_prob(normal_values)

    # Enumerated path
    enum_values = dist.enumerate_support(expand=False)
    enum_log_probs = dist.log_prob(enum_values)

    # Should give the same results
    assert torch.allclose(normal_log_probs, enum_log_probs)


def test_batched_log_prob_with_enumeration():
    """Test batched distributions with enumeration."""
    probs = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    dist = PyroCategorical(probs=probs)

    # Enumerate support
    values = dist.enumerate_support(expand=False)
    log_probs = dist.log_prob(values)

    # Should have shape [2, 2] (num_events, batch_size)
    assert log_probs.shape == torch.Size([2, 2])

    # Each column should match the logits for that batch
    assert torch.allclose(log_probs[:, 0], dist.logits[0])
    assert torch.allclose(log_probs[:, 1], dist.logits[1])


@pytest.mark.parametrize("distribution_class", [TorchCategorical, PyroCategorical])
def test_param_shape(distribution_class):
    """Test param_shape property."""
    probs = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
    dist = distribution_class(probs=probs)

    assert dist.param_shape == torch.Size([2, 2])


@pytest.mark.parametrize("distribution_class", [TorchCategorical, PyroCategorical])
def test_entropy(distribution_class):
    """Test entropy computation."""
    # Uniform distribution should have maximum entropy
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    dist = distribution_class(probs=probs)

    entropy = dist.entropy()

    # For uniform distribution over 4 categories: H = log(4)
    expected = torch.log(torch.tensor(4.0))
    assert torch.allclose(entropy, expected, atol=1e-5)


@pytest.mark.parametrize("distribution_class", [TorchCategorical, PyroCategorical])
def test_entropy_deterministic(distribution_class):
    """Test entropy for deterministic distribution."""
    # Deterministic distribution should have zero entropy
    probs = torch.tensor([1.0, 0.0, 0.0, 0.0])
    dist = distribution_class(probs=probs)

    entropy = dist.entropy()

    # Should be close to 0
    assert torch.allclose(entropy, torch.tensor(0.0), atol=1e-5)
