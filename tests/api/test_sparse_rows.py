# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import tempfile

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


def _make_pieces(tmp_path, monkeypatch: pytest.MonkeyPatch) -> tuple[SparseRowCollector, sp.csr_matrix, np.ndarray]:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    matrix = sp.random(20, 6, density=0.4, format="csr", dtype=np.float32, random_state=2)
    permutation = np.random.default_rng(1).permutation(20)
    return _collect(matrix, permutation, [7, 6, 7]), matrix, permutation


def test_the_pieces_are_kept_in_files_that_finish_deletes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, matrix, permutation = _make_pieces(tmp_path, monkeypatch)
    directories = list(tmp_path.glob("cellarium_rows_*"))
    assert len(directories) == 1
    assert sorted(path.name for path in directories[0].iterdir()) == ["columns.i32", "values.f32"]
    assert (directories[0] / "values.f32").stat().st_size == 4 * matrix.nnz

    result, order = collector.finish()

    np.testing.assert_array_equal(result.toarray(), matrix[permutation[order]].toarray())
    assert list(tmp_path.glob("cellarium_rows_*")) == []


def test_close_deletes_the_files_and_the_collector_can_be_used_again(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, _, _ = _make_pieces(tmp_path, monkeypatch)
    collector.close()
    assert list(tmp_path.glob("cellarium_rows_*")) == [] and len(collector) == 0
    with pytest.raises(ValueError, match="No rows"):
        collector.finish()

    matrix = sp.csr_matrix(np.arange(6, dtype=np.float32).reshape(2, 3))
    collector.append(matrix, np.array([1, 0]))
    result, order = collector.finish()
    np.testing.assert_array_equal(result.toarray(), matrix.toarray()[[1, 0]])
    assert order.tolist() == [1, 0]


def test_the_files_are_deleted_if_the_collector_is_dropped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, _, _ = _make_pieces(tmp_path, monkeypatch)
    assert len(list(tmp_path.glob("cellarium_rows_*"))) == 1
    del collector
    gc.collect()
    assert list(tmp_path.glob("cellarium_rows_*")) == []


def test_rows_without_nonzero_counts_are_collected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    collector = SparseRowCollector()
    collector.append(sp.csr_matrix((3, 4), dtype=np.float32), np.array([2, 0, 1]))
    result, order = collector.finish()
    assert result.shape == (3, 4) and result.nnz == 0 and order.tolist() == [1, 2, 0]
    assert list(tmp_path.glob("cellarium_rows_*")) == []


def test_inconsistent_input_raises() -> None:
    collector = SparseRowCollector()
    with pytest.raises(ValueError, match="positions"):
        collector.append(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), np.array([0]))
    collector.append(sp.csr_matrix(np.ones((1, 3), dtype=np.float32)), np.array([0]))
    with pytest.raises(ValueError, match="columns"):
        collector.append(sp.csr_matrix(np.ones((1, 4), dtype=np.float32)), np.array([1]))
    with pytest.raises(ValueError, match="No rows"):
        SparseRowCollector().finish()
