# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from cellarium.ml import CellariumPipeline
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
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    transform = CellariumPipeline(
        [
            NormalizeTotal(target_count),
            Log1p(),
            ZScore(mean_g, std_g, var_names_g),
        ]
    )
    return transform


def test_log_normalize_shape(x_ng: torch.Tensor, log_normalize: CellariumPipeline):
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    batch = {"x_ng": x_ng, "var_names_g": var_names_g}
    new_x_ng = log_normalize(batch)["x_ng"]
    assert x_ng.shape == new_x_ng.shape


def test_log_normalize_mean_std(x_ng: torch.Tensor, log_normalize: CellariumPipeline):
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    batch = {"x_ng": x_ng, "var_names_g": var_names_g}
    new_x_ng = log_normalize(batch)["x_ng"]

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
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    new_x_ng = transform(x_ng, var_names_g)["x_ng"]
    assert new_x_ng.shape[1] == len(filter_list)
    assert new_x_ng.shape[0] == x_ng.shape[0]


def test_filter_ordering_reorders(x_ng: torch.Tensor):
    # filter_list order differs from input order; output should follow filter_list
    var_names_g = np.array(["gene_0", "gene_1", "gene_2"])
    filter_list = ["gene_2", "gene_0"]
    transform = Filter(filter_list, ordering=True)
    result = transform(x_ng, var_names_g)
    assert list(result["var_names_g"]) == filter_list
    # gene_2 is column 2 of original; gene_0 is column 0
    np.testing.assert_array_equal(result["x_ng"][:, 0].numpy(), x_ng[:, 2].numpy())
    np.testing.assert_array_equal(result["x_ng"][:, 1].numpy(), x_ng[:, 0].numpy())


def test_filter_ordering_false_preserves_input_order(x_ng: torch.Tensor):
    # ordering=False: output follows input order, not filter_list order
    var_names_g = np.array(["gene_0", "gene_1", "gene_2"])
    filter_list = ["gene_2", "gene_0"]
    transform = Filter(filter_list, ordering=False)
    result = transform(x_ng, var_names_g)
    # gene_0 appears before gene_2 in input, so output order is gene_0, gene_2
    assert list(result["var_names_g"]) == ["gene_0", "gene_2"]
    np.testing.assert_array_equal(result["x_ng"][:, 0].numpy(), x_ng[:, 0].numpy())
    np.testing.assert_array_equal(result["x_ng"][:, 1].numpy(), x_ng[:, 2].numpy())


def test_filter_allow_missing_fills_zeros():
    n_cells = 5
    # input has gene_0 and gene_1; filter_list also wants gene_X which is absent
    var_names_g = np.array(["gene_0", "gene_1"])
    filter_list = ["gene_1", "gene_X", "gene_0"]
    x_ng = torch.ones(n_cells, 2) * torch.tensor([[10.0, 20.0]])
    transform = Filter(filter_list, ordering=True, allow_missing=True)
    result = transform(x_ng, var_names_g)

    assert list(result["var_names_g"]) == filter_list
    assert result["x_ng"].shape == (n_cells, 3)
    # column 0 -> gene_1 (value 20), column 1 -> gene_X (zero), column 2 -> gene_0 (value 10)
    np.testing.assert_array_equal(result["x_ng"][:, 0].numpy(), np.full(n_cells, 20.0))
    np.testing.assert_array_equal(result["x_ng"][:, 1].numpy(), np.zeros(n_cells))
    np.testing.assert_array_equal(result["x_ng"][:, 2].numpy(), np.full(n_cells, 10.0))


def test_filter_allow_missing_requires_ordering():
    with pytest.raises(ValueError, match="requires `ordering=True`"):
        Filter(["gene_0"], ordering=False, allow_missing=True)


def test_filter_missing_genes_raises():
    var_names_g = np.array(["gene_0", "gene_1"])
    x_ng = torch.zeros(2, 2)
    transform = Filter(["gene_0", "gene_missing"], ordering=True, allow_missing=False)
    with pytest.raises(AssertionError, match="gene_missing"):
        transform(x_ng, var_names_g)


def test_filter_cache():
    filter_list = ["gene_0", "gene_1"]
    # Use ordering=False so partial var_names_g (missing some filter_list genes) is allowed,
    # matching the original cache-mechanics test intent.
    transform = Filter(filter_list, ordering=False)
    transform.filter.cache_clear()

    m = 4
    for g in range(1, 1 + m):
        var_names_g = np.array([f"gene_{i}" for i in range(g)])
        x_ng = torch.zeros((2, g))
        for _ in range(g):
            transform(x_ng, var_names_g)

    assert transform.filter.cache_info().currsize == m
    assert transform.filter.cache_info().misses == m
    assert transform.filter.cache_info().hits == m * (m - 1) / 2
