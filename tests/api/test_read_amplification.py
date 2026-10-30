# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings

import numpy as np
import pytest

from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import pca

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`

SCATTERED = np.array([0, 20])  # one cell in each of the shards: 2 of the 40 cells, in all of the shards (2 files)


@pytest.fixture
def warns_for_small_data(monkeypatch):
    """Make the warning apply to the few shards of the test data."""
    module = sys.modules["cellarium.ml.api.cellariumdata"]
    monkeypatch.setattr(module, "_READ_AMPLIFICATION_WARNING", 2.0)
    monkeypatch.setattr(module, "_MIN_SHARDS_FOR_READ_AMPLIFICATION_WARNING", 1)


def _nudges(record) -> list[str]:
    return [str(w.message) for w in record if "to_deltacells" in str(w.message)]


def test_read_amplification_is_how_much_more_of_the_data_a_pass_reads_than_the_cells_need(cdata):
    assert cdata.read_amplification() is None  # not a view
    scattered = cdata[SCATTERED]
    # every shard is read for 5% of the cells
    assert scattered.read_amplification() > 5
    # the view of everything reads exactly what it needs
    assert cdata[np.arange(40)].read_amplification() == pytest.approx(1.0)
    # a view of a view is measured against the data, not against its parent: the same cells, the same value
    assert scattered[[0]].read_amplification() == cdata[[0]].read_amplification()


def test_a_view_that_was_written_as_a_dataset_of_its_own_does_not_amplify(cdata, tmp_path, deltacells_kwargs):
    pytest.importorskip("deltacells._core")
    view = cdata[SCATTERED]
    assert view.read_amplification() > 5
    compact = view.to_deltacells(str(tmp_path / "compact"), tile_size=4, level=3, deltacells_kwargs=deltacells_kwargs)
    assert compact.read_amplification() is None
    assert view.to_ram().read_amplification() is None


def test_a_pass_over_a_scattered_view_warns_once_and_says_what_to_do(cdata, warns_for_small_data):
    view = cdata[SCATTERED]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        list(view.predict_batches(batch_size=1))
    messages = _nudges(record)
    assert len(messages) == 1
    assert "to_deltacells" in messages[0] and "of the data" in messages[0] and "2 cells of this view" in messages[0]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        list(view.predict_batches(batch_size=1))
    assert _nudges(record) == []  # once for each view


def test_training_and_sketching_over_a_scattered_view_warn_too(cdata, warns_for_small_data):
    view = cdata[np.array([0, 1, 20, 21, 22, 39])]
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    view.hvg = cdata.hvg
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        pca(view, n_components=2, zscore=False, accelerator="cpu")
    assert len(_nudges(record)) == 1

    other = cdata[np.array([0, 1, 20, 21, 22, 39])]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        other.to_anndata("geometric_sketch", 3, geometric_sketch_kwargs={"accelerator": "cpu"})
    assert len(_nudges(record)) == 1


def test_the_data_itself_and_views_of_whole_shards_do_not_warn(cdata, warns_for_small_data):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        list(cdata.predict_batches(batch_size=10))
        list(cdata[np.arange(40)].predict_batches(batch_size=10))
    assert _nudges(record) == []


def test_the_default_thresholds_leave_small_data_alone(cdata):
    view = cdata[SCATTERED]  # amplified, but of 2 shards: nothing to save
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        list(view.predict_batches(batch_size=1))
    assert _nudges(record) == []
