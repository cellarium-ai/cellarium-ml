# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellarium.ml.api.utils import write_obs_parquet


def _write_h5ad(tmp_path, name: str, obs: pd.DataFrame, n_genes: int = 2) -> str:
    var = pd.DataFrame(index=[f"gene{k}" for k in range(n_genes)])
    X = np.zeros((len(obs), n_genes), dtype=np.float32)
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(tmp_path / name)
    return str(tmp_path / name)


def test_write_obs_parquet_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    h5ad_paths = []
    expected = []
    for i in range(3):
        obs = pd.DataFrame(
            {
                "cell_type": pd.Categorical(rng.choice(["T cell", "B cell"], size=4), categories=["T cell", "B cell"]),
                "n_counts": rng.integers(100, 1000, size=4).astype(np.int64),
            },
            index=[f"file{i}_cell{j}" for j in range(4)],
        )
        obs.index.name = "barcode"
        path = _write_h5ad(tmp_path, f"data_{i}.h5ad", obs)
        h5ad_paths.append(path)
        expected.append(ad.read_h5ad(path, backed="r").obs.copy())
    expected_df = pd.concat(expected, axis=0)

    output_path = str(tmp_path / "obs.parquet")
    write_obs_parquet(h5ad_paths, output_path, processes=2)
    result = pd.read_parquet(output_path)

    assert list(result.index) == list(expected_df.index)  # row order matches h5ad_paths order
    pd.testing.assert_frame_equal(result.astype(object), expected_df.astype(object), check_dtype=False)
    assert isinstance(result["cell_type"].dtype, pd.CategoricalDtype)  # harmonized categories are preserved


def test_write_obs_parquet_falls_back_when_categories_differ_across_files(tmp_path):
    # shard 0 has few categories (narrow dictionary index width internally)
    obs0 = pd.DataFrame({"cell_type": pd.Categorical(["T cell", "B cell"])}, index=["a", "b"])
    # shard 1 has enough categories to require a wider dictionary index width
    many_cats = [f"type_{i}" for i in range(300)]
    obs1 = pd.DataFrame({"cell_type": pd.Categorical(many_cats[:3], categories=many_cats)}, index=["c", "d", "e"])

    h5ad_paths = [_write_h5ad(tmp_path, "data_0.h5ad", obs0), _write_h5ad(tmp_path, "data_1.h5ad", obs1)]
    output_path = str(tmp_path / "obs.parquet")

    write_obs_parquet(h5ad_paths, output_path, processes=2)
    result = pd.read_parquet(output_path)

    assert list(result["cell_type"]) == ["T cell", "B cell", "type_0", "type_1", "type_2"]


def test_write_obs_parquet_raises_on_empty_input():
    with pytest.raises(ValueError):
        write_obs_parquet([], "/tmp/should_not_be_created.parquet")
