import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvid.data import AnnDataSchema

n_cell_ref, n_cell, n_gene = (3, 5, 4)


@pytest.fixture
def adata():
    X = np.zeros((n_cell, n_gene))
    L = np.zeros((n_cell, n_gene))
    M = np.zeros((n_cell, 2))
    obs = pd.DataFrame(
        dict(batch=np.array(["a", "b"])[np.random.randint(0, 2, n_cell)]),
        index=[f"cell{i:03d}" for i in range(n_cell)],
    )
    var = pd.DataFrame(index=[f"gene{i:03d}" for i in range(n_gene)])
    return AnnData(X, dtype=X.dtype, obs=obs, var=var, layers={"L": L}, obsm={"M": M})


@pytest.fixture
def ref_adata():
    X = np.ones((n_cell_ref, n_gene))
    L = np.ones((n_cell_ref, n_gene))
    M = np.ones((n_cell_ref, 2))
    obs = pd.DataFrame(
        dict(batch=np.array(["a", "b"])[np.random.randint(0, 2, n_cell_ref)]),
        index=[f"ref_cell{i:03d}" for i in range(n_cell_ref)],
    )
    var = pd.DataFrame(index=[f"gene{i:03d}" for i in range(n_gene)])
    return AnnData(X, dtype=X.dtype, obs=obs, var=var, layers={"L": L}, obsm={"M": M})


def test_validate_adata(ref_adata, adata):
    schema = AnnDataSchema(ref_adata)
    schema.validate_adata(adata)
