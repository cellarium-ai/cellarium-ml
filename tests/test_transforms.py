# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from cellarium.ml import CellariumPipeline
from cellarium.ml.transforms import DivideByScale, Filter, Log1p, NormalizeTotal, ZScore

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


# ---------------------------------------------------------------------------
# ZScore flexible gene-subset tests
# ---------------------------------------------------------------------------

G_FULL = 5
_full_var_names = np.array([f"gene_{i}" for i in range(G_FULL)])


@pytest.fixture
def zscore_full():
    """ZScore initialized on the full G_FULL-gene space."""
    rng = torch.Generator()
    rng.manual_seed(42)
    mean_g = torch.rand(G_FULL, generator=rng)
    std_g = torch.rand(G_FULL, generator=rng) + 0.1
    return ZScore(mean_g, std_g, _full_var_names.copy())


def test_zscore_full_space_unchanged(zscore_full: ZScore):
    """Full schema input still works as before."""
    x = torch.ones(3, G_FULL)
    out = zscore_full(x, _full_var_names)
    assert out["x_ng"].shape == (3, G_FULL)


def test_zscore_subset_correct_stats(zscore_full: ZScore):
    """Input is a strict subset; only the matching per-gene stats are applied."""
    subset_names = np.array(["gene_1", "gene_3"])
    x = torch.ones(4, 2)
    out = zscore_full(x, subset_names)
    assert out["x_ng"].shape == (4, 2)

    # Verify values: z = (1 - mean_g[i]) / (std_g[i] + eps)
    idx = np.array([1, 3])
    expected = (x - zscore_full.mean_g[idx]) / (zscore_full.std_g[idx] + zscore_full.eps)
    np.testing.assert_allclose(out["x_ng"].numpy(), expected.numpy(), rtol=1e-5)


def test_zscore_reordered_correct_stats(zscore_full: ZScore):
    """Input genes are a reordering of the full schema; stats follow input order."""
    reordered = np.array(["gene_4", "gene_0", "gene_2", "gene_1", "gene_3"])
    x = torch.ones(2, G_FULL)
    out = zscore_full(x, reordered)
    assert out["x_ng"].shape == (2, G_FULL)

    idx = np.array([4, 0, 2, 1, 3])
    expected = (x - zscore_full.mean_g[idx]) / (zscore_full.std_g[idx] + zscore_full.eps)
    np.testing.assert_allclose(out["x_ng"].numpy(), expected.numpy(), rtol=1e-5)


def test_zscore_unknown_gene_raises(zscore_full: ZScore):
    """Gene absent from stored schema raises ValueError."""
    bad_names = np.array(["gene_0", "gene_unknown"])
    x = torch.ones(2, 2)
    with pytest.raises(ValueError, match="gene_unknown"):
        zscore_full(x, bad_names)


# ---------------------------------------------------------------------------
# DivideByScale flexible gene-subset tests
# ---------------------------------------------------------------------------


@pytest.fixture
def divide_by_scale_full():
    """DivideByScale initialized on the full G_FULL-gene space."""
    rng = torch.Generator()
    rng.manual_seed(7)
    scale_g = torch.rand(G_FULL, generator=rng) + 0.5
    return DivideByScale(scale_g, _full_var_names.copy())


def test_divide_by_scale_subset_correct_stats(divide_by_scale_full: DivideByScale):
    """Input is a strict subset; only matching per-gene scales are applied."""
    subset_names = np.array(["gene_0", "gene_2", "gene_4"])
    x = torch.ones(3, 3) * 2.0
    out = divide_by_scale_full(x, subset_names)
    assert out["x_ng"].shape == (3, 3)

    idx = np.array([0, 2, 4])
    expected = x / (divide_by_scale_full.scale_g[idx] + divide_by_scale_full.eps)
    np.testing.assert_allclose(out["x_ng"].numpy(), expected.numpy(), rtol=1e-5)


def test_divide_by_scale_unknown_gene_raises(divide_by_scale_full: DivideByScale):
    """Gene absent from stored schema raises ValueError."""
    bad_names = np.array(["gene_0", "gene_missing"])
    x = torch.ones(2, 2)
    with pytest.raises(ValueError, match="gene_missing"):
        divide_by_scale_full(x, bad_names)


# ---------------------------------------------------------------------------
# End-to-end pipeline: Filter → NormalizeTotal → Log1p → ZScore
# ---------------------------------------------------------------------------


def test_filter_then_zscore_pipeline():
    """
    ZScore is initialized on the full gene space.  A Filter upstream reduces
    the batch to a subset before ZScore sees it — this must work without error
    and produce numerically correct results.
    """
    n_cells = 20
    G_PIPE = 6
    full_names = np.array([f"gene_{i}" for i in range(G_PIPE)])
    keep = ["gene_1", "gene_3", "gene_5"]

    rng = torch.Generator()
    rng.manual_seed(99)
    x_ng = torch.poisson(torch.rand(n_cells, G_PIPE, generator=rng) * 30 + 1)

    # Compute reference stats on the filtered+log-normalized data
    keep_idx = np.array([1, 3, 5])
    x_sub = x_ng[:, keep_idx]
    normed = target_count * x_sub / x_sub.sum(dim=-1, keepdim=True)
    logged = torch.log1p(normed)
    mean_g = logged.mean(dim=0)
    std_g = logged.std(dim=0)

    # Build ZScore on the FULL gene space by padding with dummy stats for unused genes
    mean_full = torch.zeros(G_PIPE)
    std_full = torch.ones(G_PIPE)
    mean_full[keep_idx] = mean_g
    std_full[keep_idx] = std_g

    pipeline = CellariumPipeline(
        [
            Filter(keep),
            NormalizeTotal(target_count),
            Log1p(),
            ZScore(mean_full, std_full, full_names),
        ]
    )

    batch = {"x_ng": x_ng, "var_names_g": full_names}
    result = pipeline(batch)["x_ng"]

    assert result.shape == (n_cells, len(keep))
    np.testing.assert_allclose(result.mean(dim=0).numpy(), np.zeros(len(keep)), atol=1e-5)
    np.testing.assert_allclose(result.std(dim=0).numpy(), np.ones(len(keep)), atol=1e-5)
