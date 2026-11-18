# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from cellarium.ml.api import utils
from cellarium.ml.api.utils import (
    get_h5ad_file_var_names_g,
    get_h5ad_files_limits,
    get_h5ad_files_obs_info,
    harmonize_obs_categories,
    release_memory_after,
    release_unused_memory,
    validate_h5ad_obs_schemas,
    write_obs_parquet,
)


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


def _obs(i: int, n: int = 3, /, **columns) -> pd.DataFrame:
    return pd.DataFrame(columns, index=[f"file{i}_cell{j}" for j in range(n)])


def _validate(h5ad_paths: list[str], nexus_extract_uniform_sizes: bool = False) -> None:
    _, obs_infos = get_h5ad_files_obs_info(h5ad_paths, nexus_extract_uniform_sizes=nexus_extract_uniform_sizes)
    validate_h5ad_obs_schemas(obs_infos)


def test_get_h5ad_files_obs_info_reads_sizes_and_schemas(tmp_path):
    obs = _obs(
        0,
        cat=pd.Categorical(["a", "b", "a"], categories=["b", "a"]),
        ordered=pd.Categorical(["lo", "hi", "lo"], categories=["lo", "hi"], ordered=True),
        f=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        s=["x", "y", "z"],
        nullable=pd.array([1, None, 3], dtype="Int64"),
    )
    path = _write_h5ad(tmp_path, "data.h5ad", obs)
    limits, obs_infos = get_h5ad_files_obs_info([path])
    np.testing.assert_array_equal(limits, [3])
    columns = obs_infos[path].columns
    assert list(columns) == ["cat", "ordered", "f", "s", "nullable"]
    assert columns["cat"].categories == ("b", "a") and not columns["cat"].ordered and columns["cat"].dtype == "str"
    assert columns["ordered"].ordered
    assert columns["f"].dtype == "float32" and columns["f"].is_numeric
    assert columns["s"].dtype == "str" and not columns["s"].is_numeric
    assert columns["nullable"].encoding == "nullable-integer" and columns["nullable"].is_numeric
    assert not columns["cat"].is_numeric


def test_validate_h5ad_obs_schemas_raises_on_extra_columns_naming_the_files(tmp_path):
    paths = [
        _write_h5ad(tmp_path, "binned_0.h5ad", _obs(0, count=[1.0, 2.0, 3.0])),
        _write_h5ad(tmp_path, "binned_1.h5ad", _obs(1, count=[1.0, 2.0, 3.0])),
        _write_h5ad(tmp_path, "clustered.h5ad", _obs(2, count=[1.0, 2.0, 3.0], leiden=["0", "1", "0"])),
    ]
    with pytest.raises(ValueError, match="obs columns differ") as excinfo:
        _validate(paths)
    message = str(excinfo.value)
    assert "2 files (binned_0.h5ad, binned_1.h5ad): the 1 obs columns of the first file, ['count']" in message
    assert "1 file (clustered.h5ad): with 1 more, ['leiden']" in message


def test_obs_column_order_does_not_matter(tmp_path):
    paths = [
        _write_h5ad(tmp_path, "data_0.h5ad", _obs(0, a=[1, 2, 3], b=["x", "y", "z"])),
        _write_h5ad(tmp_path, "data_1.h5ad", _obs(1, b=["u", "v", "w"], a=[4, 5, 6])),
    ]
    _validate(paths)
    output_path = str(tmp_path / "obs.parquet")
    write_obs_parquet(paths, output_path)
    result = pd.read_parquet(output_path)
    assert list(result.columns) == ["a", "b"]  # the order of the first file
    assert list(result["a"]) == [1, 2, 3, 4, 5, 6]
    assert list(result["b"]) == ["x", "y", "z", "u", "v", "w"]


def test_validate_h5ad_obs_schemas_raises_on_different_dtypes(tmp_path):
    paths = [
        _write_h5ad(tmp_path, "data_0.h5ad", _obs(0, count=np.array([1, 2, 3], dtype=np.float32))),
        _write_h5ad(tmp_path, "data_1.h5ad", _obs(1, count=np.array([1, 2, 3], dtype=np.float64))),
    ]
    with pytest.raises(ValueError, match=r"'count' has different types: float32 in 1 file .*; float64 in 1 file"):
        _validate(paths)


def test_validate_h5ad_obs_schemas_raises_on_different_categories_and_points_to_harmonize(tmp_path):
    paths = [
        _write_h5ad(tmp_path, f"data_{i}.h5ad", _obs(i, sample=pd.Categorical([f"s{i}"] * 3), n=[1, 2, 3]))
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="'sample' has different categories in 2 files") as excinfo:
        _validate(paths)
    assert "harmonize_obs_categories(h5ad_paths, columns=['sample'])" in str(excinfo.value)


def test_validate_h5ad_obs_schemas_accepts_categories_in_a_different_order(tmp_path):
    paths = [
        _write_h5ad(tmp_path, "data_0.h5ad", _obs(0, c=pd.Categorical(["a", "b", "a"], categories=["a", "b"]))),
        _write_h5ad(tmp_path, "data_1.h5ad", _obs(1, c=pd.Categorical(["a", "b", "a"], categories=["b", "a"]))),
    ]
    _validate(paths)


def test_validate_h5ad_obs_schemas_requires_the_same_order_of_an_ordered_categorical(tmp_path):
    def ordered(categories):
        return pd.Categorical(["a", "b", "a"], categories=categories, ordered=True)

    paths = [
        _write_h5ad(tmp_path, "data_0.h5ad", _obs(0, c=ordered(["a", "b"]))),
        _write_h5ad(tmp_path, "data_1.h5ad", _obs(1, c=ordered(["b", "a"]))),
    ]
    with pytest.raises(ValueError, match="different order") as excinfo:
        _validate(paths)
    assert "harmonize_obs_categories" not in str(excinfo.value)  # it cannot choose the order


def test_validate_h5ad_obs_schemas_with_nexus_sizes_reads_only_the_first_and_last_files(tmp_path):
    paths = [
        _write_h5ad(tmp_path, "data_0.h5ad", _obs(0, a=[1, 2, 3])),
        _write_h5ad(tmp_path, "data_1.h5ad", _obs(1, a=[1, 2, 3], b=[1, 2, 3])),  # never read
        _write_h5ad(tmp_path, "data_2.h5ad", _obs(2, a=[1, 2, 3])),
    ]
    _validate(paths, nexus_extract_uniform_sizes=True)
    with pytest.raises(ValueError, match="obs columns differ"):
        _validate(paths)


def test_harmonize_obs_categories_keeps_the_values_and_makes_the_files_valid(tmp_path):
    obs0 = _obs(0, sample=pd.Categorical(["s0", None, "s0"]), k=pd.Categorical(np.array([3, 1, 3], dtype=np.int32)))
    obs1 = _obs(1, sample=pd.Categorical(["s1", "s0", "s1"]), k=pd.Categorical(np.array([2, 2, 1], dtype=np.int32)))
    obs2 = _obs(2, sample=pd.Categorical(["s0", "s1", "s1"]), k=pd.Categorical(np.array([1, 2, 3], dtype=np.int32)))
    paths = [_write_h5ad(tmp_path, f"data_{i}.h5ad", obs) for i, obs in enumerate([obs0, obs1, obs2])]

    unions = harmonize_obs_categories(paths)

    assert unions == {"sample": ["s0", "s1"], "k": [1, 3, 2]}  # in the order of first appearance
    _validate(paths)
    for path, obs in zip(paths, [obs0, obs1, obs2]):
        result = ad.read_h5ad(path).obs
        assert list(result.columns) == ["sample", "k"]
        assert list(result["sample"].cat.categories) == ["s0", "s1"]
        assert result["k"].cat.categories.dtype == np.int32
        pd.testing.assert_series_equal(result["sample"].astype(object), obs["sample"].astype(object))
        pd.testing.assert_series_equal(result["k"].astype(object), obs["k"].astype(object))


def test_harmonize_obs_categories_refuses_a_column_that_is_not_categorical(tmp_path):
    paths = [
        _write_h5ad(tmp_path, "data_0.h5ad", _obs(0, c=pd.Categorical(["a", "b", "a"]))),
        _write_h5ad(tmp_path, "data_1.h5ad", _obs(1, c=["a", "b", "c"])),
    ]
    with pytest.raises(ValueError, match="not categorical in every file"):
        harmonize_obs_categories(paths, columns=["c"])


def test_get_h5ad_file_var_names_g(tmp_path):
    path = _write_h5ad(tmp_path, "data.h5ad", pd.DataFrame(index=["c0", "c1"]), n_genes=3)
    var_names_g = get_h5ad_file_var_names_g(path)
    np.testing.assert_array_equal(var_names_g, np.array(["gene0", "gene1", "gene2"]))


def test_release_memory_after_releases_when_the_function_returns_and_when_it_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "release_unused_memory", lambda: calls.append(1))

    @release_memory_after
    def double(x: int) -> int:
        """Doubles."""
        return 2 * x

    @release_memory_after
    def fail() -> None:
        raise RuntimeError("no")

    assert double(3) == 6 and calls == [1]
    assert double.__name__ == "double" and double.__doc__ == "Doubles."
    with pytest.raises(RuntimeError):
        fail()
    assert calls == [1, 1]


def test_release_unused_memory_clears_the_cublas_workspaces_before_emptying_the_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append("workspaces"), raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cache"))
    release_unused_memory()
    assert calls == ["workspaces", "cache"]

    # a version of torch without the private function is fine too
    calls.clear()
    monkeypatch.delattr(torch._C, "_cuda_clearCublasWorkspaces")
    release_unused_memory()
    assert calls == ["cache"]
