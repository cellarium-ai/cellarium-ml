# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.models import AmortizedOnlineNonNegativeMatrixFactorization
from cellarium.ml.models.nmf import NMFOutput
from cellarium.ml.models.nmf_amortized import BilinearLoadingsEncoder
from cellarium.ml.transforms import DivideByScale, Filter
from cellarium.ml.utilities.data import AnnDataField

os.environ["TORCH_COMPILE_DISABLE"] = "1"


@pytest.fixture
def small_adata():
    n, g, k = 1000, 10, 3
    rng = np.random.default_rng(0)
    z_nk = rng.standard_normal((n, k)).astype(np.float32)
    w_kg = rng.standard_normal((k, g)).astype(np.float32)
    noise = 0.3 * rng.standard_normal((n, g)).astype(np.float32)
    # NMF requires non-negative inputs
    x_ng = np.clip(z_nk @ w_kg + noise, 0, None)
    return anndata.AnnData(
        X=x_ng,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(g)]),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n)]),
    )


def _make_module(
    small_adata: anndata.AnnData,
    k_values: list[int],
    r: int,
    latent_dim: int,
    batch_size: int,
) -> CellariumModule:
    g = small_adata.shape[1]
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    amortized_nmf = AmortizedOnlineNonNegativeMatrixFactorization(
        var_names_g=var_names_g.tolist(),
        k_values=k_values,
        r=r,
        latent_dim=latent_dim,
        total_n_cells=small_adata.shape[0],
        batch_size=batch_size,
    )
    return CellariumModule(
        cpu_transforms=[
            DivideByScale(
                scale_g=torch.from_numpy(small_adata.X.std(axis=0)),
                var_names_g=var_names_g,
                eps=1e-4,
            ),
            Filter(var_names_g.tolist()),
        ],
        model=amortized_nmf,
    )


def _make_datamodule(small_adata: anndata.AnnData, batch_size: int) -> CellariumAnnDataDataModule:
    return CellariumAnnDataDataModule(
        dadc=small_adata,
        batch_size=batch_size,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=None),
            "var_names_g": AnnDataField(attr="var_names"),
            "obs_names_n": AnnDataField(attr="obs_names"),
        },
    )


def test_amortized_nmf_single_device(small_adata: anndata.AnnData) -> None:
    """Smoke test: model trains for one epoch without error on a single CPU device."""
    n = small_adata.shape[0]
    dm = _make_datamodule(small_adata, batch_size=n // 2)
    dm.setup(stage="fit")
    module = _make_module(small_adata, k_values=[3], r=2, latent_dim=16, batch_size=n // 2)
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)


def test_amortized_nmf_multiple_k_values(small_adata: anndata.AnnData) -> None:
    """Model handles multiple k values simultaneously and produces factors of the correct shape."""
    n, g = small_adata.shape
    k_values = [3, 4]
    r = 2
    dm = _make_datamodule(small_adata, batch_size=n // 2)
    dm.setup(stage="fit")
    module = _make_module(small_adata, k_values=k_values, r=r, latent_dim=16, batch_size=n // 2)
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)

    factors_dict = module.model.factors_dict
    assert set(factors_dict.keys()) == {3, 4}
    assert factors_dict[3].shape == (r, 3, g)
    assert factors_dict[4].shape == (r, 4, g)


def test_amortized_nmf_forward_returns_loss(small_adata: anndata.AnnData) -> None:
    """forward() returns a dict with a non-negative scalar 'loss' tensor."""
    g = small_adata.shape[1]
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    x_ng = torch.from_numpy(small_adata.X[:50]).float()

    amortized_nmf = AmortizedOnlineNonNegativeMatrixFactorization(
        var_names_g=var_names_g.tolist(),
        k_values=[3],
        r=2,
        latent_dim=16,
        total_n_cells=small_adata.shape[0],
        batch_size=50,
    )
    result = amortized_nmf(x_ng=x_ng, var_names_g=var_names_g)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() >= 0


def test_bilinear_loadings_encoder_output_shape() -> None:
    """BilinearLoadingsEncoder produces (R, N, K) outputs that are non-negative."""
    n_genes, latent_dim, r, n, k = 10, 16, 3, 5, 4
    encoder = BilinearLoadingsEncoder(n_genes=n_genes, latent_dim=latent_dim)
    x_ng = torch.rand(n, n_genes)
    # L1-normalize rows of w_rkg to match the convention used during training
    w_rkg = torch.rand(r, k, n_genes)
    w_rkg = w_rkg / w_rkg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    out = encoder(x_ng, w_rkg)
    assert out.shape == (r, n, k), f"Expected ({r}, {n}, {k}), got {out.shape}"
    assert (out >= 0).all(), "Encoder output should be non-negative after ReLU"


def test_amortized_nmf_infer_loadings(small_adata: anndata.AnnData) -> None:
    """After training, NMFOutput can compute consensus factors and per-cell loadings."""
    n, g = small_adata.shape
    k = 3
    # r=1 keeps the test fast and avoids the n_neighbors >= 2 requirement in compute_consensus_factors
    r = 1
    dm = _make_datamodule(small_adata, batch_size=n // 2)
    dm.setup(stage="fit")
    module = _make_module(small_adata, k_values=[k], r=r, latent_dim=16, batch_size=n // 2)
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=2)
    trainer.fit(module, dm)

    nmf_output = NMFOutput(nmf_module=module, datamodule=dm)
    # density_threshold=1 keeps all replicates regardless of density score
    nmf_output.compute_consensus_factors(k_values=k, density_threshold=1, local_neighborhood_size=0.3)

    loadings_df = nmf_output.compute_loadings(k=k, normalize=False)
    assert loadings_df.shape == (n, k)
    assert (loadings_df.values >= 0).all(), "Loadings should be non-negative"

    rec_error = nmf_output.calculate_reconstruction_error(k_values=[k])
    assert k in rec_error
    assert np.isfinite(rec_error[k]), f"Reconstruction error should be finite, got {rec_error[k]}"
