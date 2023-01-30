import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvid.data import DistributedAnnDataCollection, read_h5ad_file


@pytest.fixture
def adatas_path(tmp_path):
    n_cell, n_gene = (10, 5)
    # adata0.n_obs == 2, adata1.n_obs == 3, adata2.n_obs == 5
    limits = [2, 5, 10]

    X = np.random.randint(50, size=(n_cell, n_gene))
    L = np.random.randint(50, size=(n_cell, n_gene))
    M = np.random.normal(size=(n_cell, 2))
    V = np.random.normal(size=(n_gene, 2))
    P = np.random.normal(size=(n_gene, n_gene))
    obs = pd.DataFrame(
        {
            "A": np.random.randint(0, 2, n_cell),
            "B": np.random.randint(0, 2, n_cell),
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
    adata.write(tmp_path / "adata.h5ad")
    limits0 = [0] + limits[:-1]
    for i, limit in enumerate(zip(limits0, limits)):
        sliced_adata = adata[slice(*limit)]
        sliced_adata.write(tmp_path / f"adata.00{i}.h5ad")
    return tmp_path


@pytest.fixture
def adt(adatas_path):
    # single anndata
    adt = read_h5ad_file(str(adatas_path / "adata.h5ad"))
    return adt


@pytest.fixture
def dat(adatas_path):
    # distributed anndata
    filenames = str(adatas_path / "adata.{000..005}.h5ad")
    limits = [2, 5, 10]
    dat = DistributedAnnDataCollection(
        filenames, limits, maxsize=2, cachesize_strict=False
    )
    return dat


def test_init_dat(dat):
    # check that only one anndata was loaded during initialization
    assert len(dat.cache) == 1

    for ladata in dat.adatas:
        adata = ladata.adata
        dat.schema.validate_anndata(adata)


@pytest.mark.parametrize(
    "select",
    [slice(0, 2), slice(1, 4), [1, 2, 4, 4], [6, 1, 3]],
    ids=["one adata", "two adatas", "sorted", "unsorted"],
)
def test_indexing(adt, dat, select):
    # compare indexing single and distributed anndata
    dat_view = dat[select, :2]
    adt_view = adt[select, :2]
    np.testing.assert_array_equal(adt_view.X, dat_view.X)
    np.testing.assert_array_equal(adt_view.var_names, dat_view.var_names)
    np.testing.assert_array_equal(adt_view.layers["L"], dat_view.layers["L"])
    np.testing.assert_array_equal(adt_view.obsm["M"], dat_view.obsm["M"])
    np.testing.assert_array_equal(dat_view.obs["A"], dat_view.obs["A"])
