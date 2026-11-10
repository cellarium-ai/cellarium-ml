# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellarium.ml.api.utils import get_h5ad_file_var_names_g, get_h5ad_files_limits, write_obs_parquet


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


def test_get_h5ad_files_limits_variable_sizes(tmp_path):
    sizes = [3, 5, 2]
    h5ad_paths = [
        _write_h5ad(tmp_path, f"data_{i}.h5ad", pd.DataFrame(index=[f"file{i}_cell{j}" for j in range(n)]))
        for i, n in enumerate(sizes)
    ]
    limits = get_h5ad_files_limits(h5ad_paths, nexus_extract_uniform_sizes=False)
    np.testing.assert_array_equal(limits, np.cumsum(sizes))


def test_get_h5ad_files_limits_uniform_sizes_assumes_first_and_last(tmp_path):
    # the middle file's true size (2) is never read when nexus_extract_uniform_sizes=True;
    # it's assumed to match the first file's size (4) instead
    sizes = [4, 2, 4]
    h5ad_paths = [
        _write_h5ad(tmp_path, f"data_{i}.h5ad", pd.DataFrame(index=[f"file{i}_cell{j}" for j in range(n)]))
        for i, n in enumerate(sizes)
    ]
    limits = get_h5ad_files_limits(h5ad_paths, nexus_extract_uniform_sizes=True)
    np.testing.assert_array_equal(limits, np.cumsum([4, 4, 4]))


def test_get_h5ad_file_var_names_g(tmp_path):
    path = _write_h5ad(tmp_path, "data.h5ad", pd.DataFrame(index=["c0", "c1"]), n_genes=3)
    var_names_g = get_h5ad_file_var_names_g(path)
    np.testing.assert_array_equal(var_names_g, np.array(["gene0", "gene1", "gene2"]))
