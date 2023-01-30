import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvid.data import AnnDataSchema, read_h5ad_file, DistributedAnnDataCollection

n_cell, n_gene = (100, 10)
limits = [(0, 5), (5, 15), (15, 30), (30, 50), (50, 75), (75, 100)]
limits = [5, 15, 30, 50, 75, 100]


@pytest.fixture
def adatas_path(tmp_path):
    X = np.random.randint(100, size=(n_cell, n_gene))
    L = np.random.randint(50, size=(n_cell, n_gene))
    M = np.random.normal(size=(n_cell, 2))
    V = np.random.normal(size=(n_gene, 2))
    P = np.random.normal(size=(n_gene, n_gene))
    obs = pd.DataFrame(
        {
            "A": np.array(["a", "b"])[np.random.randint(0, 2, n_cell)],
            "B": np.array(["c", "d"])[np.random.randint(0, 2, n_cell)],
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
    adata.write(tmp_path / "adata.h5ad")
    limits0 = [0] + limits[:-1]
    for i, limit in zip(limits):
        sliced_adata = adata[slice(*limit)]
        sliced_adata.write(tmp_path / f"adata.00{i}.h5ad")
    return tmp_path

def test_validate_adata(adatas_path):
    breakpoint()
    adata = read_h5ad_file(str(adatas_path / "adata.h5ad"))
    filenames = str(adatas_path / "adata.{000..005}.h5ad")
    dadata = DistributedAnnDataCollection(filenames)
    pass

@pytest.mark.parametrize("adatas", [np.array, csr_matrix], indirect=True)
def test_full_selection(adatas):
    dat = AnnCollection(adatas, index_unique="_")
    adt_concat = ad.concat(adatas, index_unique="_")

    # sorted selection from one adata
    dat_view = dat[:2, :2]
    for adata in (adatas[0], adt_concat):
        adt_view = adata[:2, :2]
        np.testing.assert_allclose(_dense(dat_view.X), _dense(adt_view.X))
        np.testing.assert_allclose(dat_view.obsm["o_test"], adt_view.obsm["o_test"])
        np.testing.assert_array_equal(dat_view.obs["a_test"], adt_view.obs["a_test"])

    # sorted and unsorted selection from 2 adatas
    rand_idxs = np.random.choice(dat.shape[0], 4, replace=False)
    for select in (slice(2, 5), [4, 2, 3], rand_idxs):
        dat_view = dat[select, :2]
        adt_view = adt_concat[select, :2]
        np.testing.assert_allclose(_dense(dat_view.X), _dense(adt_view.X))
        np.testing.assert_allclose(dat_view.obsm["o_test"], adt_view.obsm["o_test"])
        np.testing.assert_array_equal(dat_view.obs["a_test"], adt_view.obs["a_test"])

    # test duplicate selection
    idxs = [1, 2, 4, 4]
    dat_view = dat[idxs, :2]
    np.testing.assert_allclose(
        _dense(dat_view.X), np.array([[4, 5], [7, 8], [9, 8], [9, 8]])
    )


@pytest.mark.parametrize("adatas", [np.array, csr_matrix], indirect=True)
def test_creation(adatas):
    adatas_inner = [adatas[0], adatas[1][:, :2].copy()]

    dat = AnnCollection(adatas_inner, join_vars="inner", index_unique="_")
    adt_concat = ad.concat(adatas_inner, index_unique="_")
    np.testing.assert_array_equal(dat.var_names, adt_concat.var_names)


@pytest.mark.parametrize("adatas", [np.array], indirect=True)
def test_convert(adatas):
    dat = AnnCollection(adatas, index_unique="_")

    le = LabelEncoder()
    le.fit(dat[:].obs["a_test"])

    obs_no_convert = dat[:].obs["a_test"]
    convert = dict(obs={"a_test": lambda a: le.transform(a)})
    dat.convert = convert
    np.testing.assert_array_equal(dat[:].obs["a_test"], le.transform(obs_no_convert))
