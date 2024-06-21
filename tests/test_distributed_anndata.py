# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
    read_h5ad_file,
)
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes


@pytest.fixture
def adatas_path(tmp_path: Path):
    n_cell, n_gene = (10, 5)
    # adata0.n_obs == 2, adata1.n_obs == 3, adata2.n_obs == 5
    limits = [2, 5, 10]

    rng = np.random.default_rng(1465)
    X = rng.integers(50, size=(n_cell, n_gene))
    L = rng.integers(50, size=(n_cell, n_gene))
    M = rng.normal(size=(n_cell, 2))
    V = rng.normal(size=(n_gene, 2))
    P = rng.normal(size=(n_gene, n_gene))
    obs = pd.DataFrame(
        {
            "A": rng.integers(0, 2, n_cell),
            "B": rng.integers(0, 2, n_cell),
            "C": pd.Categorical(
                np.array(["e", "f"])[np.random.randint(0, 2, n_cell)],
                categories=["e", "f"],
            ),
        },
        index=[f"ref_cell{i:03d}" for i in range(n_cell)],
    )
    var = pd.DataFrame(
        {
            "D": np.zeros(n_gene),
            "E": np.ones(n_gene),
            "F": np.arange(n_gene),
        },
        index=[f"gene{i:03d}" for i in range(n_gene)],
    )
    adata = AnnData(
        X,
        dtype=X.dtype,
        obs=obs,
        var=var,
        layers={"L": L},
        obsm={"M": M},
        varm={"V": V},
        varp={"P": P},
    )
    adata.write(os.path.join(tmp_path, "adata.h5ad"))
    for i, limit in enumerate(zip([0] + limits, limits)):
        sliced_adata = adata[slice(*limit)]
        sliced_adata.obs["C"] = sliced_adata.obs["C"].cat.set_categories(adata.obs["C"].cat.categories)
        sliced_adata.write(os.path.join(tmp_path, f"adata.00{i}.h5ad"))
    return tmp_path


@pytest.fixture
def adt(adatas_path: Path):
    # single anndata
    adt = read_h5ad_file(str(os.path.join(adatas_path, "adata.h5ad")))
    return adt


@pytest.fixture(params=[(i, j) for i in (1, 2, 3) for j in (True, False)])
def dat(adatas_path: Path, request: pytest.FixtureRequest):
    # distributed anndata
    filenames = str(os.path.join(adatas_path, "adata.{000..002}.h5ad"))
    limits = [2, 5, 10]
    max_cache_size, cache_size_strictly_enforced = request.param
    dat = DistributedAnnDataCollection(
        filenames,
        limits,
        max_cache_size=max_cache_size,
        cache_size_strictly_enforced=cache_size_strictly_enforced,
    )
    return dat


def test_init_dat(dat: DistributedAnnDataCollection):
    # check that only one anndata was loaded during initialization
    assert len(dat.cache) == 1

    for i, ladata in enumerate(dat.adatas):
        # check that calling repr doesn't load anndata
        repr(ladata)
        if i == 0:
            assert ladata.cached is True
        else:
            assert ladata.cached is False

        # validate anndata
        adata = ladata.adata
        dat.schema.validate_anndata(adata)


@pytest.mark.parametrize("num_shards", [3, 4, 10])
@pytest.mark.parametrize("last_shard_size", [1, 2, 3, None])
def test_init_shard_size(adatas_path: Path, num_shards: int, last_shard_size: int | None):
    shard_size = 2
    filenames = str(os.path.join(adatas_path, f"adata.{{000..{num_shards-1:03}}}.h5ad"))
    dadc = DistributedAnnDataCollection(
        filenames,
        shard_size=shard_size,
        last_shard_size=last_shard_size,
        max_cache_size=1,
    )

    actual_len = len(dadc)
    expected_len = num_shards * shard_size
    if last_shard_size is not None:
        expected_len = expected_len + last_shard_size - shard_size

    assert actual_len == expected_len


@pytest.mark.parametrize(
    "row_select",
    [(slice(0, 2), 1), (slice(1, 4), 2), ([1, 2, 4, 4], 2), ([6, 1, 3], 3)],
    ids=["one adata", "two adatas", "sorted two adatas", "unsorted three adatas"],
)
@pytest.mark.parametrize("vidx", [slice(0, 2), [3, 1, 0]])
def test_indexing(
    adt: AnnData,
    dat: DistributedAnnDataCollection,
    row_select: tuple[slice | list, int],
    vidx: slice | list,
):
    # compare indexing single and distributed anndata
    max_cache_size = dat.max_cache_size
    cache_size_strictly_enforced = dat.cache_size_strictly_enforced
    oidx, n_adatas = row_select

    if cache_size_strictly_enforced and (n_adatas > max_cache_size):
        with pytest.raises(ValueError, match="Expected the number of anndata files"):
            dat_view = dat[oidx, vidx]
    else:
        adt_view = adt[oidx, vidx]
        dat_view = dat[oidx, vidx]
        adt_view.obs["C"] = adt_view.obs["C"].cat.set_categories(adt.obs["C"].cat.categories)
        np.testing.assert_array_equal(adt_view.X, dat_view.X)
        np.testing.assert_array_equal(adt_view.var_names, dat_view.var_names)
        np.testing.assert_array_equal(adt_view.obs_names, dat_view.obs_names)
        np.testing.assert_array_equal(adt_view.layers["L"], dat_view.layers["L"])
        np.testing.assert_array_equal(adt_view.obsm["M"], dat_view.obsm["M"])
        assert adt_view.obs["A"].equals(dat_view.obs["A"])
        assert adt_view.obs["C"].equals(dat_view.obs["C"])


def test_pickle(dat: DistributedAnnDataCollection):
    new_dat: DistributedAnnDataCollection = pickle.loads(pickle.dumps(dat))

    assert len(new_dat.cache) == 1

    new_dat_view, dat_view = new_dat[:2], dat[:2]

    np.testing.assert_array_equal(new_dat_view.X, dat_view.X)
    np.testing.assert_array_equal(new_dat_view.var_names, dat_view.var_names)
    np.testing.assert_array_equal(new_dat_view.obs_names, dat_view.obs_names)
    np.testing.assert_array_equal(new_dat_view.layers["L"], dat_view.layers["L"])
    np.testing.assert_array_equal(new_dat_view.obsm["M"], dat_view.obsm["M"])
    assert new_dat_view.obs["A"].equals(dat_view.obs["A"])
    assert new_dat_view.obs["C"].equals(dat_view.obs["C"])


@pytest.mark.parametrize(
    "row_select",
    [(slice(0, 2), 1), (slice(1, 4), 2), ([1, 2, 4, 4], 2), ([6, 1, 3], 3)],
    ids=["one adata", "two adatas", "sorted two adatas", "unsorted three adatas"],
)
def test_indexing_dataset(
    adt: AnnData,
    dat: DistributedAnnDataCollection,
    row_select: tuple[slice | list, int],
):
    # compare indexing single anndata and distributed anndata dataset
    max_cache_size = dat.max_cache_size
    cache_size_strictly_enforced = dat.cache_size_strictly_enforced
    oidx, n_adatas = row_select

    dataset = IterableDistributedAnnDataCollectionDataset(
        dat,
        batch_keys={
            "x_ng": AnnDataField("X"),
            "obs_names": AnnDataField("obs_names"),
        },
    )

    if cache_size_strictly_enforced and (n_adatas > max_cache_size):
        with pytest.raises(ValueError, match="Expected the number of anndata files"):
            dataset_X = dataset[oidx]["x_ng"]
    else:
        adt_X = adt[oidx].X
        dataset_X = dataset[oidx]["x_ng"]
        np.testing.assert_array_equal(adt_X, dataset_X)

        adt_obs_names = adt[oidx].obs_names
        dataset_obs_names = dataset[oidx]["obs_names"]
        np.testing.assert_array_equal(adt_obs_names, dataset_obs_names)


def test_pickle_dataset(dat: DistributedAnnDataCollection):
    dataset = IterableDistributedAnnDataCollectionDataset(
        dat,
        batch_keys={
            "x_ng": AnnDataField("X"),
            "obs_names": AnnDataField("obs_names"),
        },
    )
    new_dataset = pickle.loads(pickle.dumps(dataset))

    assert len(new_dataset.dadc.cache) == 1

    np.testing.assert_array_equal(new_dataset[:2]["x_ng"], dataset[:2]["x_ng"])
    np.testing.assert_array_equal(new_dataset[:2]["obs_names"], dataset[:2]["obs_names"])


@pytest.fixture
def dadc(adatas_path: Path):
    # distributed anndata
    filenames = str(os.path.join(adatas_path, "adata.{000..002}.h5ad"))
    limits = [2, 5, 10]
    dadc = DistributedAnnDataCollection(
        filenames,
        limits,
        max_cache_size=2,
    )
    return dadc


@pytest.mark.parametrize(
    "attr,key,convert_fn",
    [
        ("X", None, None),
        ("obs", "A", None),
        ("obs", "C", categories_to_codes),
        ("obs_names", None, None),
        ("var_names", None, None),
        ("layers", "L", None),
    ],
)
@pytest.mark.parametrize("idx", [slice(0, 5), [0, 2, 4], 0])
def test_anndata_field(
    dadc: DistributedAnnDataCollection,
    attr: str,
    key: str | None,
    convert_fn: Callable[[Any], np.ndarray] | None,
    idx: slice | list,
):
    expected = getattr(dadc[idx], attr)
    if key is not None:
        expected = expected[key]
    if convert_fn is not None:
        expected = convert_fn(expected)

    field = AnnDataField(attr, key, convert_fn)
    actual = field(dadc, idx)

    np.testing.assert_array_equal(expected, actual)
