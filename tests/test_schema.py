# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellarium.ml.data import AnnDataSchema

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
def schema(ref_adata: AnnData):
    return AnnDataSchema(ref_adata)


@pytest.mark.parametrize("delete_ref", [False, True])
def test_validate_adata(ref_adata: AnnData, adata: AnnData, delete_ref: bool):
    schema = AnnDataSchema(ref_adata)
    if delete_ref:
        del ref_adata
    schema.validate_anndata(adata)


@pytest.fixture(
    params=[
        "permute_obs_columns",
        "rename_obs_columns",
        "permute_var_columns",
        "change_obs_dtype",
        "change_obs_categories",
    ]
)
def change_adata(adata: AnnData, request: pytest.FixtureRequest):
    if request.param == "permute_obs_columns":
        adata.obs = adata.obs.iloc[:, [1, 0, 2]]
        err_msg = ".obs attribute columns for anndata passed in"

    elif request.param == "rename_obs_columns":
        cols = adata.obs.columns
        adata.obs = adata.obs.rename(columns={c: f"{c}_" for c in cols})
        err_msg = ".obs attribute columns for anndata passed in"

    elif request.param == "permute_var_columns":
        adata.var = adata.var.iloc[:, [1, 0, 2]]
        err_msg = ".var attribute for anndata passed in"

    elif request.param == "change_obs_dtype":
        adata.obs["A"] = np.ones(n_cell)
        err_msg = ".obs attribute dtypes for anndata passed in"

    elif request.param == "change_obs_categories":
        adata.obs["C"] = pd.Categorical(
            np.array(["g", "h"])[np.random.randint(0, 2, n_cell)],
            categories=["g", "h"],
        )
        err_msg = ".obs attribute dtypes for anndata passed in"

    return err_msg


def test_changed_adata(schema: AnnDataSchema, adata: AnnData, change_adata: str):
    err_msg = change_adata
    with pytest.raises(ValueError, match=err_msg):
        schema.validate_anndata(adata)


@pytest.mark.parametrize("change_obs_categories", [False, True])
@pytest.mark.parametrize("subset_obs_columns", [False, True])
def test_validate_obs_columns(
    ref_adata: AnnData,
    adata: AnnData,
    change_obs_categories: bool,
    subset_obs_columns: bool,
):
    if change_obs_categories:
        adata.obs["C"] = pd.Categorical(
            np.array(["g", "h"])[np.random.randint(0, 2, n_cell)],
            categories=["g", "h"],
        )

    if subset_obs_columns:
        schema = AnnDataSchema(ref_adata, obs_columns_to_validate=["A", "B"])
    else:
        schema = AnnDataSchema(ref_adata)

    if change_obs_categories and not subset_obs_columns:
        with pytest.raises(ValueError, match=".obs attribute dtypes for anndata passed in"):
            schema.validate_anndata(adata)
    else:
        schema.validate_anndata(adata)
