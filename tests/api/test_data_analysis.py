# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellarium.ml.api.data_analysis import CellariumData, LazyObs, ObsmMapping
from cellarium.ml.api.utils import write_obs_parquet


def _make_h5ad_files(tmp_path, n_files: int = 3, cells_per_file: int = 5) -> list[str]:
    rng = np.random.default_rng(0)
    cell_types = ["T cell", "B cell", "NK cell"]
    h5ad_paths = []
    for i in range(n_files):
        obs = pd.DataFrame(
            {
                "cell_type": pd.Categorical(rng.choice(cell_types, size=cells_per_file), categories=cell_types),
                "n_counts": rng.integers(100, 10000, size=cells_per_file).astype(np.int64),
            },
            index=[f"file{i}_cell{j}" for j in range(cells_per_file)],
        )
        obs.index.name = "barcode"
        var = pd.DataFrame(index=[f"gene{k}" for k in range(4)])
        X = rng.normal(size=(cells_per_file, 4)).astype(np.float32)
        adata = ad.AnnData(X=X, obs=obs, var=var)
        path = tmp_path / f"data_{i}.h5ad"
        adata.write_h5ad(path)
        h5ad_paths.append(str(path))
    return h5ad_paths


@pytest.fixture
def obs_parquet(tmp_path):
    h5ad_paths = _make_h5ad_files(tmp_path)
    output_path = str(tmp_path / "obs.parquet")
    write_obs_parquet(h5ad_paths, output_path, processes=2)
    # ground truth is what anndata actually persisted on disk, not the pre-write in-memory obs
    expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)
    return output_path, expected


class TestLazyObs:
    def test_construction_does_not_open_dataset(self, obs_parquet):
        path, _ = obs_parquet
        obs = LazyObs(path)
        assert obs._dataset is None

    def test_len_and_columns(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        assert len(obs) == len(expected)
        assert set(obs.columns) == set(expected.columns)

    def test_column_selection_returns_series(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        result = obs["n_counts"]
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result.sort_index(), expected["n_counts"].sort_index(), check_names=False)

    def test_column_selection_returns_dataframe(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        result = obs[["cell_type", "n_counts"]]
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["cell_type", "n_counts"]
        assert len(result) == len(expected)

    def test_loc_row_subset_preserves_order(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        names = [expected.index[3], expected.index[0], expected.index[1]]
        result = obs.loc[names]
        assert list(result.index) == names
        pd.testing.assert_frame_equal(result.astype(object), expected.loc[names].astype(object), check_dtype=False)

    def test_loc_row_and_column_subset(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        names = list(expected.index[:2])
        result = obs.loc[names, ["n_counts"]]
        assert list(result.columns) == ["n_counts"]
        assert list(result.index) == names

    def test_loc_missing_label_raises_keyerror(self, obs_parquet):
        path, _ = obs_parquet
        obs = LazyObs(path)
        with pytest.raises(KeyError):
            obs.loc[["does_not_exist"]]

    def test_to_frame_matches_full_dataset(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        result = obs.to_frame()
        pd.testing.assert_frame_equal(
            result.sort_index().astype(object), expected.sort_index().astype(object), check_dtype=False
        )

    def test_iter_batches_covers_all_rows(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        chunks = list(obs.iter_batches(batch_size=4))
        assert len(chunks) > 1
        combined = pd.concat(chunks)
        pd.testing.assert_frame_equal(
            combined.sort_index().astype(object), expected.sort_index().astype(object), check_dtype=False
        )


class TestObsmMapping:
    def test_n_obs_not_evaluated_until_setitem(self, obs_parquet):
        path, _ = obs_parquet
        obs = LazyObs(path)
        obsm = ObsmMapping(n_obs=obs)
        assert obs._dataset is None

        obsm["x"] = np.zeros(len(obs))
        assert obs._dataset is not None

    def test_length_mismatch_raises(self, obs_parquet):
        path, expected = obs_parquet
        obs = LazyObs(path)
        obsm = ObsmMapping(n_obs=obs)
        with pytest.raises(ValueError):
            obsm["x"] = np.zeros(len(expected) - 1)


class TestLazyObsAutoBuild:
    def test_requires_parquet_path_or_h5ad_paths(self):
        with pytest.raises(ValueError):
            LazyObs(None)

    def test_auto_builds_from_h5ad_paths_on_first_query(self, tmp_path):
        h5ad_paths = _make_h5ad_files(tmp_path)
        expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)

        obs = LazyObs(None, h5ad_paths=h5ad_paths)
        assert obs._parquet_path is None
        assert obs._dataset is None

        result = obs.to_frame()
        assert obs._parquet_path is not None  # built lazily, on the first real query
        pd.testing.assert_frame_equal(  # type: ignore[unreachable]
            result.sort_index().astype(object), expected.sort_index().astype(object), check_dtype=False
        )


class TestCellariumDataObsParquet:
    def test_obs_is_always_lazy_by_default(self, tmp_path):
        h5ad_paths = _make_h5ad_files(tmp_path)
        cdata = CellariumData(h5ad_paths=h5ad_paths, accelerator="cpu")
        assert isinstance(cdata.obs, LazyObs)
        assert cdata.obs._dataset is None  # constructing CellariumData shouldn't build/touch obs

    def test_repr_does_not_force_obs_build(self, tmp_path):
        h5ad_paths = _make_h5ad_files(tmp_path)
        cdata = CellariumData(h5ad_paths=h5ad_paths, accelerator="cpu")
        repr(cdata)
        assert cdata.obs._dataset is None

    def test_default_obs_is_queryable_and_matches_h5ad_files(self, tmp_path):
        h5ad_paths = _make_h5ad_files(tmp_path)
        expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)

        cdata = CellariumData(h5ad_paths=h5ad_paths, accelerator="cpu")
        name = expected.index[0]
        result = cdata.obs.loc[[name]]
        assert list(result.index) == [name]
        pd.testing.assert_frame_equal(result.astype(object), expected.loc[[name]].astype(object), check_dtype=False)
        assert cdata.obs._dataset is not None  # opened only once actually queried

    def test_obs_parquet_path_is_lazy_and_queryable(self, tmp_path):
        h5ad_paths = _make_h5ad_files(tmp_path)
        output_path = str(tmp_path / "obs.parquet")
        write_obs_parquet(h5ad_paths, output_path, processes=2)
        expected = pd.concat([ad.read_h5ad(p, backed="r").obs.copy() for p in h5ad_paths], axis=0)

        cdata = CellariumData(h5ad_paths=h5ad_paths, obs_parquet_path=output_path, accelerator="cpu")
        assert isinstance(cdata.obs, LazyObs)
        assert cdata.obs._dataset is None  # constructing CellariumData shouldn't touch the parquet file

        name = expected.index[0]
        result = cdata.obs.loc[[name]]
        assert list(result.index) == [name]
        pd.testing.assert_frame_equal(result.astype(object), expected.loc[[name]].astype(object), check_dtype=False)
        assert cdata.obs._dataset is not None  # opened only once actually queried
