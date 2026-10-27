# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import os

import numpy as np
import pandas as pd
import pytest
import torch

from cellarium.ml.api._obsm_storage import ObsmWriter, RowSubset, subset_rows


def make_writer(values: np.ndarray, max_bytes: int | None, dtype: type = np.float32) -> ObsmWriter:
    n_rows, n_cols = values.shape
    return ObsmWriter(n_rows, n_cols, max_bytes=max_bytes, dtype=dtype)


def directory_of(array: np.ndarray) -> str:
    """The directory of the file behind a memory-mapped array."""
    assert isinstance(array, np.memmap) and array.filename is not None
    return os.path.dirname(array.filename)


def on_disk_array(values: np.ndarray) -> np.memmap:
    """``values`` as a finished, read-only, on-disk array."""
    with make_writer(values, max_bytes=0, dtype=values.dtype.type) as writer:
        writer.array[:] = values
    result = writer.result
    assert isinstance(result, np.memmap)
    return result


@pytest.fixture
def values() -> np.ndarray:
    return np.arange(60, dtype=np.float32).reshape(20, 3)


# ---- ObsmWriter ----


def test_writer_keeps_a_small_array_in_memory(values):
    with make_writer(values, max_bytes=values.nbytes) as writer:
        assert not writer.on_disk
        writer.array[:] = values
    assert type(writer.result) is np.ndarray
    assert np.array_equal(writer.result, values)


def test_writer_with_no_limit_stays_in_memory(values):
    with make_writer(values, max_bytes=None) as writer:
        assert not writer.on_disk
    assert type(writer.result) is np.ndarray


def test_writer_puts_a_big_array_in_a_read_only_file(values):
    with make_writer(values, max_bytes=values.nbytes - 1) as writer:
        assert writer.on_disk
        writer.array[:10] = values[:10]
        writer.array[10:] = values[10:]
    result = writer.result
    assert isinstance(result, np.memmap)
    assert not result.flags.writeable
    assert np.array_equal(result, values)
    with pytest.raises(ValueError):
        result[0, 0] = 1.0


def test_the_file_is_deleted_when_the_array_is_no_longer_referenced(values):
    result = on_disk_array(values)
    directory = directory_of(result)
    assert os.path.exists(directory)

    rows = RowSubset(result, np.array([1, 2]))
    del result
    gc.collect()
    assert os.path.exists(directory)  # the RowSubset still refers to it
    assert np.array_equal(np.asarray(rows), values[[1, 2]])

    del rows
    gc.collect()
    assert not os.path.exists(directory)


def test_the_file_is_deleted_if_filling_it_fails(values):
    writer = make_writer(values, max_bytes=0)
    with pytest.raises(RuntimeError, match="boom"):
        with writer:
            directory = directory_of(writer.array)
            assert os.path.exists(directory)
            raise RuntimeError("boom")
    assert not os.path.exists(directory)
    with pytest.raises(RuntimeError, match="after the `with` block"):
        writer.result


def test_the_array_is_only_available_inside_the_block(values):
    writer = make_writer(values, max_bytes=None)
    with pytest.raises(RuntimeError, match="inside the `with` block"):
        writer.array


# ---- RowSubset ----


def test_row_subset_has_the_shape_and_dtype_of_its_rows(values):
    rows = RowSubset(on_disk_array(values), np.array([4, 5, 9]))
    assert rows.shape == (3, 3)
    assert rows.ndim == 2
    assert rows.dtype == np.float32
    assert len(rows) == 3
    assert rows.nbytes == 3 * 3 * 4
    assert "RowSubset" in repr(rows)


@pytest.mark.parametrize(
    "key",
    [
        0,
        -1,
        slice(None),
        slice(1, 3),
        slice(None, None, 2),
        [2, 0, 1],
        np.array([1, 1, 3]),
        np.array([True, False, True, True, False, False, True, True]),
        (slice(None), 1),
        (slice(None), slice(0, 2)),
        (np.array([0, 2]), [0, 2]),
        (3, 2),
        (2, slice(None)),
        ([1, 2], 0),
    ],
)
def test_row_subset_indexes_like_the_ndarray_of_its_rows(values, key):
    positions = np.array([3, 4, 8, 9, 10, 15, 19, 0])
    rows = RowSubset(on_disk_array(values), positions)
    expected = values[positions][key]
    assert np.array_equal(rows[key], expected)
    assert type(rows[key]) is type(expected)


def test_row_subset_results_do_not_share_the_file(values):
    rows = RowSubset(on_disk_array(values), np.array([1, 2, 3]))
    row = rows[0]
    assert isinstance(row, np.ndarray) and not isinstance(row, np.memmap)


def test_row_subset_is_read_only(values):
    rows = RowSubset(on_disk_array(values), np.array([1, 2, 3]))
    with pytest.raises(TypeError, match="read-only"):
        rows[0] = 0.0


def test_row_subset_rejects_ellipsis_and_newaxis(values):
    rows = RowSubset(on_disk_array(values), np.array([1, 2, 3]))
    with pytest.raises(IndexError, match="not supported"):
        rows[..., 0]
    with pytest.raises(IndexError, match="not supported"):
        rows[None]


def test_row_subset_rejects_bad_positions(values):
    array = on_disk_array(values)
    with pytest.raises(IndexError):
        RowSubset(array, np.array([0, 20]))
    with pytest.raises(IndexError):
        RowSubset(array, np.array([-1]))
    with pytest.raises(ValueError, match="1-dimensional"):
        RowSubset(array, np.array([[0, 1]]))


def test_row_subset_converts_to_numpy_and_other_array_likes(values):
    positions = np.array([5, 6, 18])
    rows = RowSubset(on_disk_array(values), positions)
    assert np.array_equal(np.asarray(rows), values[positions])
    assert np.array_equal(rows.to_numpy(), values[positions])
    assert np.asarray(rows, dtype=np.float64).dtype == np.float64
    assert np.array_equal(pd.DataFrame(rows).to_numpy(), values[positions])
    assert np.array_equal(torch.as_tensor(rows).numpy(), values[positions])
    assert np.array_equal(list(rows), list(values[positions]))
    with pytest.raises(ValueError, match="needs a copy"):
        rows.__array__(copy=False)


def test_row_subset_of_a_row_subset_refers_to_the_same_file(values):
    array = on_disk_array(values)
    outer = RowSubset(array, np.array([2, 4, 6, 8, 10]))
    inner = outer.take(np.array([4, 0, 2]))
    assert inner._base is array  # composed, not nested
    assert np.array_equal(np.asarray(inner), values[[10, 2, 6]])


# ---- subset_rows ----


def test_subset_rows_reads_into_memory_if_the_rows_fit(values):
    array = on_disk_array(values)
    positions = np.array([1, 5, 7])
    result = subset_rows(array, positions, max_bytes=3 * 3 * 4)
    assert type(result) is np.ndarray
    assert np.array_equal(result, values[positions])
    assert type(subset_rows(array, positions, max_bytes=None)) is np.ndarray


def test_subset_rows_stays_on_disk_if_the_rows_do_not_fit(values):
    array = on_disk_array(values)
    positions = np.array([1, 5, 7])
    result = subset_rows(array, positions, max_bytes=3 * 3 * 4 - 1)
    assert isinstance(result, RowSubset)
    assert np.array_equal(np.asarray(result), values[positions])

    # and a subset of that, once it fits, is read into memory
    again = subset_rows(result, np.array([0, 2]), max_bytes=2 * 3 * 4)
    assert type(again) is np.ndarray
    assert np.array_equal(again, values[[1, 7]])


# ---- in CellariumData ----


@pytest.fixture
def loadings(h5ad_cdata):
    """40 cells with a (40, 3) loadings array of random numbers on disk in `obsm["X_test"]`."""
    values = np.random.default_rng(0).random((h5ad_cdata.n_obs, 3)).astype(np.float32)
    h5ad_cdata.obsm["X_test"] = on_disk_array(values)
    return values


def test_a_view_reads_the_rows_that_fit_into_memory(h5ad_cdata, loadings):
    h5ad_cdata.max_gb = 1.0
    positions = np.array([0, 5, 21, 39])
    view = h5ad_cdata[positions]
    assert type(view.obsm["X_test"]) is np.ndarray
    assert np.array_equal(view.obsm["X_test"], loadings[positions])


def test_a_view_leaves_the_rows_that_do_not_fit_on_disk(h5ad_cdata, loadings):
    h5ad_cdata.max_gb = 1e-9  # a byte
    positions = np.array([0, 5, 21, 39])
    view = h5ad_cdata[positions]
    entry = view.obsm["X_test"]
    assert isinstance(entry, RowSubset)
    assert len(entry) == 4
    assert np.array_equal(np.asarray(entry), loadings[positions])


def test_a_view_of_a_view_composes_on_disk(h5ad_cdata, loadings):
    h5ad_cdata.max_gb = 1e-9
    view = h5ad_cdata[np.arange(0, 40, 2)]
    view_of_view = view[np.array([1, 3, 5])]
    entry = view_of_view.obsm["X_test"]
    assert isinstance(entry, RowSubset)
    assert entry._base is h5ad_cdata.obsm["X_test"]
    assert np.array_equal(np.asarray(entry), loadings[[2, 6, 10]])


def test_views_inherit_max_gb(h5ad_cdata):
    h5ad_cdata.max_gb = 0.5
    view = h5ad_cdata[np.array([0, 1, 2])]
    assert view.max_gb == 0.5
    assert view[np.array([0, 1])].max_gb == 0.5
    view.max_gb = 2.0  # changing a view does not change its parent
    assert h5ad_cdata.max_gb == 0.5


def test_max_gb_defaults_to_half_the_available_memory_and_must_be_positive(h5ad_cdata):
    from cellarium.ml.api._view_collections import available_memory_bytes

    assert h5ad_cdata.max_gb is None
    assert abs(h5ad_cdata._max_bytes() - available_memory_bytes() // 2) < 2**30
    assert h5ad_cdata._max_bytes(2.0) == 2 * 2**30
    h5ad_cdata.max_gb = 3.0
    assert h5ad_cdata._max_bytes() == 3 * 2**30
    assert h5ad_cdata._max_bytes(1.0) == 2**30
    with pytest.raises(ValueError, match="positive"):
        h5ad_cdata.max_gb = 0
    with pytest.raises(ValueError, match="positive"):
        h5ad_cdata.max_gb = -1.0


def test_to_anndata_exports_rows_of_an_entry_that_stays_on_disk(h5ad_cdata, loadings):
    h5ad_cdata.max_gb = 1e-9
    positions = np.array([3, 4, 30])
    view = h5ad_cdata[positions]
    assert isinstance(view.obsm["X_test"], RowSubset)

    adata = view.to_anndata(collect_embeddings=False, max_gb=1.0)
    assert type(adata.obsm["X_test"]) is np.ndarray
    assert np.array_equal(adata.obsm["X_test"], loadings[positions])


def test_the_obsm_mapping_accepts_an_entry_that_stays_on_disk(h5ad_cdata, loadings):
    h5ad_cdata.max_gb = 1e-9
    view = h5ad_cdata[np.array([0, 1, 2])]
    view.obsm["again"] = view.obsm["X_test"]  # the length check works with len()
    with pytest.raises(ValueError, match="expected 3"):
        view.obsm["wrong"] = h5ad_cdata.obsm["X_test"]
