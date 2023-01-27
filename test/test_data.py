import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvid.data import AnnDataSchema

n_cell_ref, n_cell, n_gene = (3, 5, 4)


@pytest.fixture
def ref_adata():
    X = np.ones((n_cell_ref, n_gene))
    L = np.ones((n_cell_ref, n_gene))
    M = np.ones((n_cell_ref, 2))
    V = np.zeros((n_gene, 2))
    P = np.zeros((n_gene, n_gene))
    obs = pd.DataFrame(
        {
            "A": np.array(["a", "b"])[np.random.randint(0, 2, n_cell_ref)],
            "B": np.array(["c", "d"])[np.random.randint(0, 2, n_cell_ref)],
            "C": pd.Categorical(
                np.array(["e", "f"])[np.random.randint(0, 2, n_cell_ref)],
                categories=["e", "f"],
            ),
        },
        index=[f"ref_cell{i:03d}" for i in range(n_cell_ref)],
    )
    var = pd.DataFrame(
        {
            "D": np.zeros(n_gene),
            "E": np.ones(n_gene),
            "F": np.arange(n_gene),
        },
        index=[f"gene{i:03d}" for i in range(n_gene)],
    )
    return AnnData(
        X,
        dtype=X.dtype,
        obs=obs,
        var=var,
        layers={"L": L},
        obsm={"M": M},
        varm={"V": V},
        varp={"P": P},
    )


@pytest.fixture
def adata():
    X = np.zeros((n_cell, n_gene))
    L = np.zeros((n_cell, n_gene))
    M = np.zeros((n_cell, 2))
    V = np.zeros((n_gene, 2))
    P = np.zeros((n_gene, n_gene))
    obs = pd.DataFrame(
        {
            "A": np.array(["a", "b"])[np.random.randint(0, 2, n_cell)],
            "B": np.array(["c", "d"])[np.random.randint(0, 2, n_cell)],
            "C": pd.Categorical(
                np.array(["e", "f"])[np.random.randint(0, 2, n_cell)],
                categories=["e", "f"],
            ),
        },
        index=[f"cell{i:03d}" for i in range(n_cell)],
    )
    var = pd.DataFrame(
        {
            "D": np.zeros(n_gene),
            "E": np.ones(n_gene),
            "F": np.arange(n_gene),
        },
        index=[f"gene{i:03d}" for i in range(n_gene)],
    )
    return AnnData(
        X,
        dtype=X.dtype,
        obs=obs,
        var=var,
        layers={"L": L},
        obsm={"M": M},
        varm={"V": V},
        varp={"P": P},
    )


@pytest.fixture
def schema(ref_adata):
    return AnnDataSchema(ref_adata)


@pytest.mark.parametrize("delete_ref", [False, True])
def test_validate_adata(ref_adata, adata, delete_ref):
    schema = AnnDataSchema(ref_adata)
    if delete_ref:
        del ref_adata
    schema.validate_anndata(adata)


@pytest.fixture
def permute_obs_columns(adata):
    adata.obs = adata.obs.iloc[:, [1, 0, 2]]


def test_permuted_obs_columns(schema, adata, permute_obs_columns):
    with pytest.raises(
        ValueError,
        match=".obs attribute columns for anndata passed in",
    ):
        schema.validate_anndata(adata)


@pytest.fixture
def rename_obs_columns(adata):
    cols = adata.obs.columns
    adata.obs = adata.obs.rename(columns={c: f"{c}_" for c in cols})


def test_renamed_obs_columns(schema, adata, rename_obs_columns):
    with pytest.raises(
        ValueError,
        match=".obs attribute columns for anndata passed in",
    ):
        schema.validate_anndata(adata)


@pytest.fixture
def permute_var_columns(adata):
    adata.var = adata.var.iloc[:, [1, 0, 2]]


def test_permuted_var_columns(schema, adata, permute_var_columns):
    with pytest.raises(
        ValueError,
        match=".var attribute for anndata passed in",
    ):
        schema.validate_anndata(adata)


@pytest.fixture
def change_obs_dtype(adata):
    adata.obs["A"] = np.ones(n_cell)


def test_changed_obs_dtype(schema, adata, change_obs_dtype):
    with pytest.raises(
        ValueError,
        match=".obs attribute dtypes for anndata passed in",
    ):
        schema.validate_anndata(adata)


@pytest.fixture
def change_obs_categories(adata):
    adata.obs["C"] = pd.Categorical(
        np.array(["g", "h"])[np.random.randint(0, 2, n_cell)],
        categories=["g", "h"],
    )


def test_changed_obs_categories(schema, adata, change_obs_categories):
    with pytest.raises(
        ValueError,
        match=".obs attribute dtypes for anndata passed in",
    ):
        schema.validate_anndata(adata)
