# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from cellarium.ml.api._datamodule_context import (
    batch_index_field,
    check_batch_key,
    temporary_batch_keys,
    temporary_val_split,
)
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes

# --- temporary_batch_keys() -------------------------------------------------------------------


def test_temporary_batch_keys_injects_and_restores_single_entry(h5ad_cdata):
    datamodule = h5ad_cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_obs_columns_to_validate = datamodule.dadc.obs_columns_to_validate
    assert "batch_index_n" not in original_batch_keys

    with temporary_batch_keys(
        datamodule, {"batch_index_n": AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes)}
    ):
        assert "batch_index_n" in datamodule.batch_keys
        assert datamodule.dadc.obs_columns_to_validate == ["cell_type"]
        assert datamodule.dadc.schema.obs_columns_to_validate == ["cell_type"]

    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert datamodule.dadc.obs_columns_to_validate == original_obs_columns_to_validate
    assert datamodule.dadc.schema.obs_columns_to_validate == original_obs_columns_to_validate


def test_temporary_batch_keys_injects_and_restores_multiple_entries(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())

    extra_batch_keys = {
        "batch_index_n": AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes),
        "other_batch_index_n": AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes),
    }
    with temporary_batch_keys(datamodule, extra_batch_keys):
        assert "batch_index_n" in datamodule.batch_keys
        assert "other_batch_index_n" in datamodule.batch_keys
        assert getattr(datamodule.dadc, "obs_columns_to_validate", ["cell_type"]) == ["cell_type"]  # h5ad only

    assert set(datamodule.batch_keys.keys()) == original_batch_keys


def test_temporary_batch_keys_makes_new_obs_columns_local_for_deltacells(cdata):
    from cellarium.ml.data import DistributedDeltaCellsCollection

    datamodule = cdata.datamodule
    if not isinstance(datamodule.dadc, DistributedDeltaCellsCollection):
        pytest.skip("deltacells only")
    obs = datamodule.dadc.dataset.obs
    assert "n_counts" not in obs.localized_columns()

    with temporary_batch_keys(datamodule, {"counts_n": AnnDataField(attr="obs", key="n_counts")}):
        assert "n_counts" in obs.localized_columns()
        assert "counts_n" in next(iter(datamodule.train_dataloader()))

    assert "counts_n" not in datamodule.batch_keys


def test_temporary_batch_keys_restores_preexisting_entry(cdata):
    datamodule = cdata.datamodule
    preexisting_field = AnnDataField(attr="obs", key="n_counts")
    datamodule.batch_keys["batch_index_n"] = preexisting_field

    try:
        with temporary_batch_keys(
            datamodule, {"batch_index_n": AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes)}
        ):
            assert datamodule.batch_keys["batch_index_n"] is not preexisting_field

        assert datamodule.batch_keys["batch_index_n"] is preexisting_field
    finally:
        datamodule.batch_keys.pop("batch_index_n", None)


def test_temporary_batch_keys_restores_on_exception(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_obs_columns_to_validate = getattr(datamodule.dadc, "obs_columns_to_validate", None)  # h5ad only

    with pytest.raises(RuntimeError):
        with temporary_batch_keys(
            datamodule, {"batch_index_n": AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes)}
        ):
            raise RuntimeError("boom")

    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert getattr(datamodule.dadc, "obs_columns_to_validate", None) == original_obs_columns_to_validate


# --- temporary_val_split() --------------------------------------------------------------------


def test_temporary_val_split_sets_and_restores(cdata):
    datamodule = cdata.datamodule
    original_n_train, original_n_val = datamodule.n_train, datamodule.n_val
    n_total = len(datamodule.dadc)

    with temporary_val_split(datamodule, val_size=0.2):
        assert datamodule.n_val > 0
        assert datamodule.n_train + datamodule.n_val == n_total

    assert datamodule.n_train == original_n_train
    assert datamodule.n_val == original_n_val


def test_temporary_val_split_restores_on_exception(cdata):
    datamodule = cdata.datamodule
    original_n_train, original_n_val = datamodule.n_train, datamodule.n_val

    with pytest.raises(RuntimeError):
        with temporary_val_split(datamodule, val_size=0.2):
            raise RuntimeError("boom")

    assert datamodule.n_train == original_n_train
    assert datamodule.n_val == original_n_val


# --- check_batch_key() and batch_index_field() ------------------------------------------------


def test_obs_is_categorical_comes_from_the_schema_of_the_data(cdata):
    for datamodule in (cdata.datamodule, cdata[[1, 5, 30]].datamodule):  # also through a view
        assert datamodule.obs_is_categorical("cell_type")
        assert not datamodule.obs_is_categorical("n_counts")
        with pytest.raises(ValueError, match="not found in obs columns"):
            datamodule.obs_is_categorical("nope")


def test_check_batch_key_fails_fast_on_a_column_that_is_not_categorical(cdata):
    check_batch_key(cdata.datamodule, "cell_type")  # fine
    with pytest.raises(ValueError, match="batch_key 'n_counts' is not a categorical obs column"):
        check_batch_key(cdata.datamodule, "n_counts")


def test_api_functions_fail_fast_on_a_batch_key_that_is_not_categorical(cdata):
    from cellarium.ml.api.preprocessing import highly_variable_genes, pseudobulk
    from cellarium.ml.api.tools import scvi

    for call in (
        lambda: highly_variable_genes(cdata, flavor="seurat_v3", batch_key="n_counts", accelerator="cpu"),
        lambda: highly_variable_genes(cdata, flavor="seurat", batch_key="n_counts", accelerator="cpu"),
        lambda: pseudobulk(cdata, batch_key="n_counts", accelerator="cpu"),
        lambda: scvi(cdata, batch_key="n_counts", max_epochs=1, accelerator="cpu"),
    ):
        with pytest.raises(ValueError, match="not a categorical obs column"):
            call()
    assert not cdata.trained_modules  # nothing was trained


def test_batch_index_field_reads_the_codes_of_the_batch_column(cdata):
    cells = cdata.datamodule.dadc[np.arange(12)]
    codes = batch_index_field("cell_type")(cells)
    categories = list(cdata.datamodule.dadc.obs_categories("cell_type"))
    expected = [categories.index(label) for label in cdata.obs["cell_type"].iloc[:12]]
    np.testing.assert_array_equal(codes, expected)
    with pytest.raises(ValueError, match="not categorical"):  # the backstop, as the cells are read
        batch_index_field("n_counts")(cells)
