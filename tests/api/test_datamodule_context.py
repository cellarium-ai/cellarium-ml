# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from cellarium.ml.api._datamodule_context import temporary_batch_keys, temporary_val_split
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
