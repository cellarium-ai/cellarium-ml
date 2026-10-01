# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellarium.ml.api.cellariumdata import CellariumData, LazyObs, ObsmMapping
from cellarium.ml.api.utils import write_obs_parquet


@pytest.fixture
def obs_parquet(tmp_path, h5ad_paths):
    output_path = str(tmp_path / "obs.parquet")
    write_obs_parquet(h5ad_paths, output_path, processes=2)
    # ground truth is what anndata actually persisted on disk, not the pre-write in-memory obs
    expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)
    return output_path, expected


# --- LazyObs -----------------------------------------------------------------------------------


def test_lazy_obs_construction_does_not_open_dataset(obs_parquet):
    path, _ = obs_parquet
    obs = LazyObs(path)
    assert obs._dataset is None


def test_lazy_obs_len_and_columns(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    assert len(obs) == len(expected)
    assert set(obs.columns) == set(expected.columns)


def test_lazy_obs_column_selection_returns_series(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    result = obs["n_counts"]
    assert isinstance(result, pd.Series)
    pd.testing.assert_series_equal(result.sort_index(), expected["n_counts"].sort_index(), check_names=False)


def test_lazy_obs_column_selection_returns_dataframe(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    result = obs[["cell_type", "n_counts"]]
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["cell_type", "n_counts"]
    assert len(result) == len(expected)


def test_lazy_obs_loc_row_subset_preserves_order(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    names = [expected.index[3], expected.index[0], expected.index[1]]
    result = obs.loc[names]
    assert list(result.index) == names
    pd.testing.assert_frame_equal(result.astype(object), expected.loc[names].astype(object), check_dtype=False)


def test_lazy_obs_loc_row_and_column_subset(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    names = list(expected.index[:2])
    result = obs.loc[names, ["n_counts"]]
    assert list(result.columns) == ["n_counts"]
    assert list(result.index) == names


def test_lazy_obs_loc_missing_label_raises_keyerror(obs_parquet):
    path, _ = obs_parquet
    obs = LazyObs(path)
    with pytest.raises(KeyError):
        obs.loc[["does_not_exist"]]


def test_lazy_obs_to_frame_matches_full_dataset(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    result = obs.to_frame()
    pd.testing.assert_frame_equal(
        result.sort_index().astype(object), expected.sort_index().astype(object), check_dtype=False
    )


def test_lazy_obs_iter_batches_covers_all_rows(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    chunks = list(obs.iter_batches(batch_size=4))
    assert len(chunks) > 1
    combined = pd.concat(chunks)
    pd.testing.assert_frame_equal(
        combined.sort_index().astype(object), expected.sort_index().astype(object), check_dtype=False
    )


def test_lazy_obs_requires_parquet_path_or_h5ad_paths():
    with pytest.raises(ValueError):
        LazyObs(None)


def test_lazy_obs_auto_builds_from_h5ad_paths_on_first_query(h5ad_paths):
    expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)

    obs = LazyObs(None, h5ad_paths=h5ad_paths)
    assert obs._parquet_path is None
    assert obs._dataset is None

    result = obs.to_frame()
    assert obs._parquet_path is not None  # built lazily, on the first real query
    pd.testing.assert_frame_equal(  # type:ignore[unreachable]
        result.sort_index().astype(object), expected.sort_index().astype(object), check_dtype=False
    )


# --- ObsmMapping ---------------------------------------------------------------------------------


def test_obsm_mapping_n_obs_not_evaluated_until_setitem(obs_parquet):
    path, _ = obs_parquet
    obs = LazyObs(path)
    obsm = ObsmMapping(n_obs=obs)
    assert obs._dataset is None

    obsm["x"] = np.zeros(len(obs))
    assert obs._dataset is not None


def test_obsm_mapping_length_mismatch_raises(obs_parquet):
    path, expected = obs_parquet
    obs = LazyObs(path)
    obsm = ObsmMapping(n_obs=obs)
    with pytest.raises(ValueError):
        obsm["x"] = np.zeros(len(expected) - 1)


# --- CellariumData.obs ---------------------------------------------------------------------------


def test_cellarium_data_obs_is_always_lazy_by_default(h5ad_paths):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    assert isinstance(cdata.obs, LazyObs)
    assert cdata.obs._dataset is None  # constructing CellariumData shouldn't build/touch obs


def test_cellarium_data_repr_does_not_force_obs_build(h5ad_paths):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    repr(cdata)
    assert cdata.obs._dataset is None


def test_cellarium_data_default_obs_is_queryable_and_matches_h5ad_files(h5ad_paths):
    expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)

    cdata = CellariumData(h5ad_paths=h5ad_paths)
    name = expected.index[0]
    result = cdata.obs.loc[[name]]
    assert list(result.index) == [name]
    pd.testing.assert_frame_equal(result.astype(object), expected.loc[[name]].astype(object), check_dtype=False)
    assert cdata.obs._dataset is not None  # opened only once actually queried


def test_cellarium_data_obs_parquet_path_is_lazy_and_queryable(tmp_path, h5ad_paths):
    output_path = str(tmp_path / "obs.parquet")
    write_obs_parquet(h5ad_paths, output_path, processes=2)
    expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)

    cdata = CellariumData(h5ad_paths=h5ad_paths, obs_parquet_path=output_path)
    assert isinstance(cdata.obs, LazyObs)
    assert cdata.obs._dataset is None  # constructing CellariumData shouldn't touch the parquet file

    name = expected.index[0]
    result = cdata.obs.loc[[name]]
    assert list(result.index) == [name]
    pd.testing.assert_frame_equal(result.astype(object), expected.loc[[name]].astype(object), check_dtype=False)
    assert cdata.obs._dataset is not None  # opened only once actually queried


# --- CellariumData.var ----------------------------------------------------------------------------


def test_cellarium_data_var_matches_source_h5ad(h5ad_paths):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    expected_var = ad.read_h5ad(h5ad_paths[0], backed="r").var.copy()
    pd.testing.assert_frame_equal(cdata.var.astype(object), expected_var.astype(object), check_dtype=False)


# --- CellariumData.hvg ----------------------------------------------------------------------------


@pytest.fixture
def cdata(h5ad_paths):
    return CellariumData(h5ad_paths=h5ad_paths)


def test_cellarium_data_hvg_defaults_to_none(cdata):
    assert cdata.hvg is None


def test_cellarium_data_hvg_set_bool_array(cdata):
    var_names_g = cdata.datamodule.var_names_g
    mask = np.zeros(len(var_names_g), dtype=bool)
    mask[:2] = True
    cdata.hvg = mask
    pd.testing.assert_series_equal(cdata.hvg, pd.Series(mask, index=var_names_g), check_names=False)


def test_cellarium_data_hvg_set_bool_array_wrong_length_raises(cdata):
    var_names_g = cdata.datamodule.var_names_g
    with pytest.raises(ValueError):
        cdata.hvg = np.zeros(len(var_names_g) - 1, dtype=bool)


def test_cellarium_data_hvg_set_gene_name_list(cdata):
    var_names_g = cdata.datamodule.var_names_g
    chosen = list(var_names_g[:2])
    cdata.hvg = chosen
    assert cdata.hvg.loc[chosen].all()
    assert cdata.hvg.sum() == len(chosen)


def test_cellarium_data_hvg_set_unknown_gene_name_raises(cdata):
    with pytest.raises(ValueError):
        cdata.hvg = ["not_a_real_gene"]


def test_cellarium_data_hvg_set_series_missing_entries_raises(cdata):
    var_names_g = cdata.datamodule.var_names_g
    partial = pd.Series([True], index=[var_names_g[0]])
    with pytest.raises(ValueError):
        cdata.hvg = partial


def test_cellarium_data_hvg_set_none_clears(cdata):
    var_names_g = cdata.datamodule.var_names_g
    cdata.hvg = [var_names_g[0]]
    cdata.hvg = None
    assert cdata.hvg is None
