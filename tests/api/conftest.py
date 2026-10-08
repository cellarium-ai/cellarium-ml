# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellarium.ml.api import _data_transforms
from cellarium.ml.api.cellariumdata import CellariumData


@pytest.fixture
def fits(monkeypatch) -> list[dict]:
    """The configs (the recipes) of the statistics models that `fit_or_reuse` has fit so far, one per fit."""
    fitted = []
    fit_and_register = _data_transforms.fit_and_register

    def spy(cdata, trainer, module, key, config=None, **kwargs):
        fitted.append(config)
        return fit_and_register(cdata, trainer, module, key, config=config, **kwargs)

    monkeypatch.setattr(_data_transforms, "fit_and_register", spy)
    return fitted


def _make_h5ad_files(
    tmp_path,
    n_files: int = 3,
    cells_per_file: int = 5,
    n_genes: int = 4,
    seed: int = 0,
) -> list[str]:
    rng = np.random.default_rng(seed)
    cell_types = ["T cell", "B cell", "NK cell"]
    gene_means = rng.lognormal(mean=1.0, sigma=1.5, size=n_genes)
    h5ad_paths = []
    for i in range(n_files):
        obs = pd.DataFrame(
            {
                "cell_type": pd.Categorical(rng.choice(cell_types, size=cells_per_file), categories=cell_types),
                "n_counts": rng.integers(100, 10000, size=cells_per_file).astype(np.int64),
            },
            index=[f"file{i}_cell{j}" for j in range(cells_per_file)],
        )
        obs.index.name = "barcode"
        var = pd.DataFrame(index=[f"gene{k}" for k in range(n_genes)])
        # sparse, like real single-cell count matrices -- CellariumData's default pipeline
        # expects a scipy-sparse-backed X (it converts to torch sparse CSR on load)
        X = sp.csr_matrix(rng.poisson(lam=gene_means, size=(cells_per_file, n_genes)).astype(np.float32))
        adata = ad.AnnData(X=X, obs=obs, var=var)
        path = tmp_path / f"data_{i}.h5ad"
        adata.write_h5ad(path)
        h5ad_paths.append(str(path))
    return h5ad_paths


@pytest.fixture
def make_h5ad_files(tmp_path):
    """Factory fixture: call with optional n_files/cells_per_file/n_genes/seed to write synthetic h5ad files."""

    def _make(n_files: int = 3, cells_per_file: int = 5, n_genes: int = 4, seed: int = 0) -> list[str]:
        return _make_h5ad_files(tmp_path, n_files=n_files, cells_per_file=cells_per_file, n_genes=n_genes, seed=seed)

    return _make


@pytest.fixture
def h5ad_paths(make_h5ad_files) -> list[str]:
    """A default set of small synthetic h5ad files."""
    return make_h5ad_files()


@pytest.fixture
def deltacells_uri(h5ad_paths, tmp_path) -> str:
    """The default synthetic h5ad files converted to a deltacells dataset (tiles of 4 cells, so they straddle files)."""
    pytest.importorskip("deltacells._core")
    from cellarium.ml.api import create_deltacells_dataset

    return create_deltacells_dataset(h5ad_paths, str(tmp_path / "deltacells"), tile_size=4, level=3, log=None)


@pytest.fixture
def deltacells_kwargs(tmp_path) -> dict:
    """Keeps the obs cache of the tests out of the home directory."""
    return {"cache_dir": str(tmp_path / "deltacells_cache")}


@pytest.fixture
def h5ad_cdata(make_h5ad_files) -> CellariumData:
    """Like `cdata`, for the tests of what is specific to h5ad files."""
    return CellariumData(h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30))


@pytest.fixture(params=["h5ad", "deltacells"])
def cdata(request, make_h5ad_files, tmp_path, deltacells_kwargs) -> CellariumData:
    """A CellariumData big enough to run HVG/PCA/geometric-sketch tools on, reading h5ad files and a deltacells dataset.

    n_files=2 to stay within CellariumData's hardcoded DistributedAnnDataCollection max_cache_size=2.
    """
    h5ad_paths = make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30)
    if request.param == "h5ad":
        return CellariumData(h5ad_paths=h5ad_paths)
    pytest.importorskip("deltacells._core")
    from cellarium.ml.api import create_deltacells_dataset

    uri = create_deltacells_dataset(h5ad_paths, str(tmp_path / "deltacells"), tile_size=15, level=3, log=None)
    return CellariumData.from_deltacells(uri, deltacells_kwargs=deltacells_kwargs)
