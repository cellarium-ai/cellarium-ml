# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipy.sparse as sp

from cellarium.ml.api import _sparse_rows
from cellarium.ml.api._sparse_rows import SparseRowCollector


def _collect(matrix: sp.csr_matrix, permutation: np.ndarray, piece_sizes: list[int]) -> SparseRowCollector:
    collector = SparseRowCollector()
    start = 0
    for size in piece_sizes:
        rows = permutation[start : start + size]
        collector.append(matrix[rows], rows)
        start += size
    return collector


@pytest.mark.parametrize("slab_entries", [7, 50, 10_000_000])
def test_finish_orders_rows_like_vstack(monkeypatch: pytest.MonkeyPatch, slab_entries: int) -> None:
    monkeypatch.setattr(_sparse_rows, "_SLAB_ENTRIES", slab_entries)
    rng = np.random.default_rng(0)
    matrix = sp.random(40, 12, density=0.3, format="csr", dtype=np.float32, random_state=1)
    matrix[3] = 0  # empty rows
    matrix.eliminate_zeros()
    rows = np.sort(rng.choice(40, size=25, replace=False))
    permutation = rng.permutation(rows)
    collector = _collect(matrix, permutation, [1, 10, 4, 10])
    assert len(collector) == 25

    result, order = collector.finish()

    np.testing.assert_array_equal(permutation[order], rows)
    assert result.dtype == np.float32
    assert result.format == "csr"
    np.testing.assert_array_equal(result.toarray(), matrix[rows].toarray())
    assert len(collector) == 0


def test_inconsistent_input_raises() -> None:
    collector = SparseRowCollector()
    with pytest.raises(ValueError, match="positions"):
        collector.append(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), np.array([0]))
    collector.append(sp.csr_matrix(np.ones((1, 3), dtype=np.float32)), np.array([0]))
    with pytest.raises(ValueError, match="columns"):
        collector.append(sp.csr_matrix(np.ones((1, 4), dtype=np.float32)), np.array([1]))
    with pytest.raises(ValueError, match="No rows"):
        SparseRowCollector().finish()
