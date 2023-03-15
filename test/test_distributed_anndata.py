import os
import pickle

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvid.data import (
    DistributedAnnDataCollection,
    DistributedAnnDataCollectionDataset,
    read_h5ad_file,
)


@pytest.fixture
def adatas_path(tmp_path):
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
        sliced_adata.write(os.path.join(tmp_path, f"adata.00{i}.h5ad"))
    return tmp_path


@pytest.fixture
def adt(adatas_path):
    # single anndata
    adt = read_h5ad_file(str(os.path.join(adatas_path, "adata.h5ad")))
    return adt


@pytest.fixture(params=[(i, j) for i in (1, 2, 3) for j in (True, False)])
def dat(adatas_path, request):
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


def test_init_dat(dat):
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


@pytest.mark.parametrize(
    "row_select",
    [(slice(0, 2), 1), (slice(1, 4), 2), ([1, 2, 4, 4], 2), ([6, 1, 3], 3)],
    ids=["one adata", "two adatas", "sorted two adatas", "unsorted three adatas"],
)
@pytest.mark.parametrize("vidx", [slice(0, 2), [3, 1, 0]])
def test_indexing(adt, dat, row_select, vidx):
    # compare indexing single and distributed anndata
    max_cache_size = dat.max_cache_size
    cache_size_strictly_enforced = dat.cache_size_strictly_enforced
    oidx, n_adatas = row_select

    if cache_size_strictly_enforced and (n_adatas > max_cache_size):
        with pytest.raises(
            AssertionError, match="Expected the number of anndata files"
        ):
            dat_view = dat[oidx, vidx]
    else:
        adt_view = adt[oidx, vidx]
        dat_view = dat[oidx, vidx]
        np.testing.assert_array_equal(adt_view.X, dat_view.X)
        np.testing.assert_array_equal(adt_view.var_names, dat_view.var_names)
        np.testing.assert_array_equal(adt_view.layers["L"], dat_view.layers["L"])
        np.testing.assert_array_equal(adt_view.obsm["M"], dat_view.obsm["M"])
        np.testing.assert_array_equal(adt_view.obs["A"], dat_view.obs["A"])


def test_pickle(dat):
    new_dat = pickle.loads(pickle.dumps(dat))

    assert len(new_dat.cache) == 0

    new_dat_view, dat_view = new_dat[:2], dat[:2]

    np.testing.assert_array_equal(new_dat_view.X, dat_view.X)
    np.testing.assert_array_equal(new_dat_view.var_names, dat_view.var_names)
    np.testing.assert_array_equal(new_dat_view.layers["L"], dat_view.layers["L"])
    np.testing.assert_array_equal(new_dat_view.obsm["M"], dat_view.obsm["M"])
    np.testing.assert_array_equal(new_dat_view.obs["A"], dat_view.obs["A"])


@pytest.mark.parametrize(
    "row_select",
    [(slice(0, 2), 1), (slice(1, 4), 2), ([1, 2, 4, 4], 2), ([6, 1, 3], 3)],
    ids=["one adata", "two adatas", "sorted two adatas", "unsorted three adatas"],
)
def test_indexing_dataset(adt, dat, row_select):
    # compare indexing single anndata and distributed anndata dataset
    max_cache_size = dat.max_cache_size
    cache_size_strictly_enforced = dat.cache_size_strictly_enforced
    oidx, n_adatas = row_select

    dataset = DistributedAnnDataCollectionDataset(dat)

    if cache_size_strictly_enforced and (n_adatas > max_cache_size):
        with pytest.raises(
            AssertionError, match="Expected the number of anndata files"
        ):
            dataset_X = dataset[oidx]["X"]
    else:
        adt_X = adt[oidx].X
        dataset_X = dataset[oidx]["X"]
        np.testing.assert_array_equal(adt_X, dataset_X)


def test_pickle_dataset(dat):
    dataset = DistributedAnnDataCollectionDataset(dat)
    new_dataset = pickle.loads(pickle.dumps(dataset))

    assert len(new_dataset.dadc.cache) == 0

    np.testing.assert_array_equal(new_dataset[:2]["X"], dataset[:2]["X"])
