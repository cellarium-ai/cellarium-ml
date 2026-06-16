# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipy.sparse
import torch

from cellarium.ml import CellariumPipeline
from cellarium.ml.transforms import (
    CenterPerCell,
    Densify,
    DivideByScale,
    Filter,
    Log1p,
    NormalizeTotal,
    PFlogPF,
    ZScore,
)
from cellarium.ml.utilities.data import to_torch_sparse_csr

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
# Filter — sparse (scipy CSR) input path
# ---------------------------------------------------------------------------

_SPARSE_DENSE = np.array(
    [
        [1.0, 0.0, 3.0],
        [0.0, 2.0, 0.0],
        [4.0, 0.0, 5.0],
        [0.0, 6.0, 0.0],
        [7.0, 0.0, 8.0],
    ],
    dtype=np.float32,
)
_SPARSE_VAR_NAMES = np.array(["gene_0", "gene_1", "gene_2"])


@pytest.fixture
def x_ng_sparse() -> scipy.sparse.csr_matrix:
    return scipy.sparse.csr_matrix(_SPARSE_DENSE)


def test_filter_sparse_ordering_true_reorders_columns(x_ng_sparse: scipy.sparse.csr_matrix):
    """ordering=True: output columns follow filter_list order, not input order."""
    filter_list = ["gene_2", "gene_0"]
    transform = Filter(filter_list, ordering=True)
    result = transform(x_ng_sparse, _SPARSE_VAR_NAMES)

    assert result["x_ng"].is_sparse_csr
    assert list(result["var_names_g"]) == filter_list

    dense_out = result["x_ng"].to_dense().numpy()
    # column 0 of output should be gene_2 (column 2 of input)
    np.testing.assert_array_equal(dense_out[:, 0], _SPARSE_DENSE[:, 2])
    # column 1 of output should be gene_0 (column 0 of input)
    np.testing.assert_array_equal(dense_out[:, 1], _SPARSE_DENSE[:, 0])


def test_filter_sparse_ordering_false_preserves_input_order(x_ng_sparse: scipy.sparse.csr_matrix):
    """ordering=False: output columns follow input order, not filter_list order."""
    filter_list = ["gene_2", "gene_0"]
    transform = Filter(filter_list, ordering=False)
    result = transform(x_ng_sparse, _SPARSE_VAR_NAMES)

    assert result["x_ng"].is_sparse_csr
    # gene_0 comes before gene_2 in the input, so output order is gene_0, gene_2
    assert list(result["var_names_g"]) == ["gene_0", "gene_2"]

    dense_out = result["x_ng"].to_dense().numpy()
    np.testing.assert_array_equal(dense_out[:, 0], _SPARSE_DENSE[:, 0])
    np.testing.assert_array_equal(dense_out[:, 1], _SPARSE_DENSE[:, 2])


def test_filter_sparse_ordering_true_vs_false_differ(x_ng_sparse: scipy.sparse.csr_matrix):
    """ordering=True and ordering=False must produce different column layouts when
    filter_list order differs from input order."""
    filter_list = ["gene_2", "gene_0"]

    result_ordered = Filter(filter_list, ordering=True)(x_ng_sparse, _SPARSE_VAR_NAMES)
    result_unordered = Filter(filter_list, ordering=False)(x_ng_sparse, _SPARSE_VAR_NAMES)

    ordered_col0 = result_ordered["x_ng"].to_dense().numpy()[:, 0]
    unordered_col0 = result_unordered["x_ng"].to_dense().numpy()[:, 0]

    # ordering=True → col 0 is gene_2; ordering=False → col 0 is gene_0
    assert not np.array_equal(ordered_col0, unordered_col0)
    np.testing.assert_array_equal(ordered_col0, _SPARSE_DENSE[:, 2])
    np.testing.assert_array_equal(unordered_col0, _SPARSE_DENSE[:, 0])


def test_filter_sparse_allow_missing_fills_zeros(x_ng_sparse: scipy.sparse.csr_matrix):
    """allow_missing=True: absent genes produce zero-filled columns in a dense output."""
    filter_list = ["gene_1", "gene_X", "gene_0"]  # gene_X not in input
    transform = Filter(filter_list, ordering=True, allow_missing=True)
    result = transform(x_ng_sparse, _SPARSE_VAR_NAMES)

    # allow_missing path always returns a dense tensor
    assert not result["x_ng"].is_sparse
    assert result["x_ng"].shape == (5, 3)
    assert list(result["var_names_g"]) == filter_list

    out = result["x_ng"].numpy()
    np.testing.assert_array_equal(out[:, 0], _SPARSE_DENSE[:, 1])  # gene_1
    np.testing.assert_array_equal(out[:, 1], np.zeros(5))  # gene_X (missing)
    np.testing.assert_array_equal(out[:, 2], _SPARSE_DENSE[:, 0])  # gene_0


def test_filter_sparse_column_mismatch_raises(x_ng_sparse: scipy.sparse.csr_matrix):
    """x_ng columns != len(var_names_g) raises ValueError."""
    bad_var_names = np.array(["gene_0", "gene_1"])  # 2 names but x_ng has 3 columns
    transform = Filter(["gene_0"], ordering=True)
    with pytest.raises(ValueError, match="must match"):
        transform(x_ng_sparse, bad_var_names)


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
# Densify tests
# ---------------------------------------------------------------------------


def test_densify_dense_passthrough():
    """Dense tensor passes through unchanged (values and shape preserved)."""
    x = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
    out = Densify()(x)["x_ng"]
    assert not out.is_sparse
    torch.testing.assert_close(out, x)


def test_densify_sparse_csr_to_dense():
    """torch.sparse_csr_tensor is converted to a dense tensor with identical values."""
    dense = torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]], dtype=torch.float32)
    x_sparse = to_torch_sparse_csr(scipy.sparse.csr_matrix(dense.numpy()))
    assert x_sparse.is_sparse_csr
    out = Densify()(x_sparse)["x_ng"]
    assert not out.is_sparse
    torch.testing.assert_close(out, dense)


def test_densify_scipy_sparse_raises():
    """scipy sparse matrices must have been converted before reaching Densify."""
    x = scipy.sparse.csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(TypeError, match="scipy sparse matrix"):
        Densify()(x)


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


# ---------------------------------------------------------------------------
# CenterPerCell
# ---------------------------------------------------------------------------


def test_center_per_cell():
    x_ng = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    out = CenterPerCell()(x_ng)["x_ng"]
    # Each row must have zero mean
    np.testing.assert_allclose(out.mean(dim=-1).numpy(), np.zeros(2), atol=1e-6)
    # Values: row0 mean=2 → [-1, 0, 1]; row1 mean=4 → [0, 0, 0]
    np.testing.assert_allclose(out.numpy(), [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], atol=1e-6)


# ---------------------------------------------------------------------------
# PFlogPF
# ---------------------------------------------------------------------------


def test_pflogpf_output_has_zero_row_mean():
    """Each cell must have zero mean after PFlogPF (CLR property)."""
    rng = torch.Generator()
    rng.manual_seed(42)
    x_ng = torch.poisson(torch.rand(10, 5, generator=rng) * 50 + 1)
    out = PFlogPF()(x_ng)["x_ng"]
    np.testing.assert_allclose(out.mean(dim=-1).numpy(), np.zeros(10), atol=1e-5)


def test_pflogpf_matches_manual_pipeline():
    """PFlogPF result must equal NormalizeTotal → Log1p → CenterPerCell applied manually."""
    x_ng = torch.tensor([[1.0, 3.0, 6.0], [2.0, 2.0, 6.0]])
    target: int = 10_000

    out_wrapper = PFlogPF(target_count=target)(x_ng)["x_ng"]

    x = NormalizeTotal(target_count=target)(x_ng)["x_ng"]
    x = Log1p()(x)["x_ng"]
    x = CenterPerCell()(x)["x_ng"]

    np.testing.assert_allclose(out_wrapper.numpy(), x.numpy(), atol=1e-6)
