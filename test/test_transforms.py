import numpy as np
import pytest
import torch

from scvid.transforms import LogNormalize

n, g, C = 100, 3, 10_000


@pytest.fixture
def x_ng():
    rng = torch.Generator()
    rng.manual_seed(1465)
    rates = torch.rand((n, g), generator=rng) * 50
    x_ng = torch.poisson(rates.float(), generator=rng)
    return x_ng


@pytest.fixture
def log_normalize(x_ng):
    l_n1 = x_ng.sum(axis=-1, keepdim=True)
    y_ng = torch.log1p(C * x_ng / l_n1)
    mean_g = y_ng.mean(axis=0)
    std_g = y_ng.std(axis=0)
    transform = LogNormalize(mean_g, std_g, C)
    return transform


def test_log_normalize_shape(x_ng, log_normalize):
    new_x_ng = log_normalize(x_ng)
    assert x_ng.shape == new_x_ng.shape


def test_log_normalize_mean_std(x_ng, log_normalize):
    new_x_ng = log_normalize(x_ng)

    actual_mean = new_x_ng.mean(axis=0)
    actual_std = new_x_ng.std(axis=0)

    np.testing.assert_allclose(0, actual_mean, atol=1e-5)
    np.testing.assert_allclose(1, actual_std, atol=1e-5)
