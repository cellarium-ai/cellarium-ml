# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from cellarium.ml.transforms import ZScoreLog1pNormalize

n, g, target_count = 100, 3, 10_000


@pytest.fixture
def x_ng():
    rng = torch.Generator()
    rng.manual_seed(1465)
    rates = torch.rand((n, g), generator=rng) * 50
    x_ng = torch.poisson(rates.float(), generator=rng)
    return x_ng


@pytest.fixture
def log_normalize(x_ng: torch.Tensor):
    l_n1 = x_ng.sum(dim=-1, keepdim=True)
    y_ng = torch.log1p(target_count * x_ng / l_n1)
    mean_g = y_ng.mean(dim=0)
    std_g = y_ng.std(dim=0)
    transform = ZScoreLog1pNormalize(mean_g, std_g, True, target_count)
    return transform


def test_log_normalize_shape(x_ng: torch.Tensor, log_normalize: ZScoreLog1pNormalize):
    new_x_ng = log_normalize(x_ng)
    assert x_ng.shape == new_x_ng.shape


def test_log_normalize_mean_std(x_ng: torch.Tensor, log_normalize: ZScoreLog1pNormalize):
    new_x_ng = log_normalize(x_ng)

    actual_mean = new_x_ng.mean(dim=0)
    actual_std = new_x_ng.std(dim=0)

    np.testing.assert_allclose(0, actual_mean, atol=1e-5)
    np.testing.assert_allclose(1, actual_std, atol=1e-5)
