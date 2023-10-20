# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from cellarium.ml.transforms import Filter, Log1p, NormalizeTotal, ZScore

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
    feature_schema = [f"gene_{i}" for i in range(g)]
    transform = torch.nn.Sequential(
        NormalizeTotal(target_count),
        Log1p(),
        ZScore(mean_g, std_g, feature_schema),
    )
    return transform


def test_log_normalize_shape(x_ng: torch.Tensor, log_normalize: torch.nn.Sequential):
    new_x_ng = log_normalize(x_ng)
    assert x_ng.shape == new_x_ng.shape


def test_log_normalize_mean_std(x_ng: torch.Tensor, log_normalize: torch.nn.Sequential):
    new_x_ng = log_normalize(x_ng)

    actual_mean = new_x_ng.mean(dim=0)
    actual_std = new_x_ng.std(dim=0)

    np.testing.assert_allclose(0, actual_mean, atol=1e-5)
    np.testing.assert_allclose(1, actual_std, atol=1e-5)


@pytest.mark.parametrize(
    "filter_list",
    [["gene_0"], ["gene_0", "gene_1"], ["gene_0", "gene_2"]],
)
def test_filter(x_ng: torch.Tensor, filter_list: list[str]):
    transform = Filter(filter_list)
    feature_g = np.array([f"gene_{i}" for i in range(g)])
    new_x_ng = transform(x_ng, feature_g)
    assert new_x_ng.shape[1] == len(filter_list)
    assert new_x_ng.shape[0] == x_ng.shape[0]


def test_filter_cache():
    filter_list = ["gene_0", "gene_1"]
    transform = Filter(filter_list)
    transform.filter.cache_clear()

    m = 4
    for g in range(1, 1 + m):
        feature_g = np.array([f"gene_{i}" for i in range(g)])
        x_ng = torch.zeros((2, g))
        for _ in range(g):
            transform(x_ng, feature_g)

    assert transform.filter.cache_info().currsize == m
    assert transform.filter.cache_info().misses == m
    assert transform.filter.cache_info().hits == m * (m - 1) / 2
