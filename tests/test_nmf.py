# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from sklearn.decomposition import NMF

from cellarium.ml import CellariumModule
from cellarium.ml.models import NonNegativeMatrixFactorization
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset

# set the number of cells, genes, cell types, and factors
n = 1000
g = 100
n_celltypes = 5
simulated_k = 3

# other constants
cell_type_coherence_factor = 20
gene_set_sparsity_factor = 0.1


@pytest.fixture(scope="module")
def d_uncorrelated_kg() -> torch.Tensor:
    """Gene programs for NMF which are uncorrelated."""
    torch.manual_seed(0)
    d_kg = torch.distributions.Dirichlet(gene_set_sparsity_factor * torch.ones(g)).sample([simulated_k])
    return d_kg


@pytest.fixture(scope="module")
def d_correlated_kg() -> torch.Tensor:
    """Gene programs for NMF which are somewhat correlated (some genes always off for example)."""
    torch.manual_seed(0)
    d_g = torch.distributions.Dirichlet(gene_set_sparsity_factor * 100 * torch.ones(g)).sample()
    d_kg = torch.distributions.Dirichlet(gene_set_sparsity_factor * 100 * d_g).sample([simulated_k])
    return d_kg


@pytest.fixture(scope="module")
def alpha_uncorrelated_nk() -> torch.Tensor:
    """Cell loadings for NMF which are uncorrelated."""
    torch.manual_seed(0)
    alpha_nk = torch.distributions.Dirichlet(torch.ones(simulated_k)).sample([n])
    return alpha_nk


@pytest.fixture(scope="module")
def alpha_correlated_nk() -> torch.Tensor:
    """Cell loadings for NMF which are correlated within celltype blocks."""
    torch.manual_seed(0)
    alpha_ck = torch.distributions.Dirichlet(torch.ones(simulated_k)).sample([n_celltypes])
    alpha_nk = (
        torch.distributions.Dirichlet(cell_type_coherence_factor * alpha_ck + 1e-5)
        .sample([n // n_celltypes])
        .reshape(n, simulated_k)
    )
    return alpha_nk


@pytest.fixture(scope="module")
def x_uncorrelated_mean_nmf_ng(alpha_uncorrelated_nk, d_uncorrelated_kg) -> torch.Tensor:
    """Data created by a sparse NMF process with no correlation between the underlying factors."""
    x_ng = alpha_uncorrelated_nk @ d_uncorrelated_kg
    x_ng = x_ng / x_ng.sum(dim=-1).mean() * g
    return x_ng


@pytest.fixture(scope="module")
def x_correlated_mean_nmf_ng(alpha_correlated_nk, d_correlated_kg) -> torch.Tensor:
    """Data created by a sparse NMF process where the underlying factors are
    drawn from the same dirichlet distribution."""
    x_ng = alpha_correlated_nk @ d_correlated_kg
    x_ng = x_ng / x_ng.sum(dim=-1).mean() * g
    return x_ng


@pytest.fixture(scope="module")
def x_nmf_ng(x_uncorrelated_mean_nmf_ng, x_correlated_mean_nmf_ng) -> dict[str, torch.Tensor]:
    """Data created by an NMF process with gaussian/poisson sampling noise
    and (no) correlation between the underlying factors and loadings."""
    torch.manual_seed(0)
    sigma = 0.6
    out = {}
    for name, mean_ng in zip(["uncorrelated", "correlated"], [x_uncorrelated_mean_nmf_ng, x_correlated_mean_nmf_ng]):
        # gaussian noise but clamped to be non-negative
        noise = sigma * torch.randn_like(mean_ng)
        out[f"gaussian_{name}"] = torch.clamp(mean_ng + noise, min=0.0)
        # poisson noise
        out[f"poisson_{name}"] = torch.distributions.Poisson(mean_ng).sample()

    return out


def run_cellarium_nmf(
    x_ng: torch.Tensor,
    var_names_g: np.ndarray,
    k: int,
    devices,
) -> NonNegativeMatrixFactorization:
    n, g = x_ng.shape

    # dataloader
    batch_size = n // 10
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            x_ng.numpy(),
            var_names_g,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # model
    cellarium_nmf = NonNegativeMatrixFactorization(
        var_names_g=var_names_g.tolist(),
        var_names_hvg=var_names_g.tolist(),
        k_values=[simulated_k],
        r=1,
        full_g=g,
        log_variational=False,
    )
    module = CellariumModule(
        model=cellarium_nmf,
    )

    # trainer
    trainer = pl.Trainer(
        barebones=False,
        accelerator="cpu",
        devices=devices,
        max_epochs=50,
    )

    # fit
    trainer.fit(module, train_dataloaders=train_loader)

    return cellarium_nmf


def get_cellarium_target(x_ng: torch.Tensor) -> torch.Tensor:
    std_g = torch.std(x_ng, dim=0) + 1e-4
    x_ng = x_ng / std_g
    x_ng = torch.clamp(x_ng, min=0.0, max=100.0)
    return x_ng


@pytest.mark.parametrize(
    "data", ["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"]
)
def test_nmf_against_sklearn_multi_device(
    x_nmf_ng: dict[str, torch.Tensor],
    data: Literal["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"],
    d_correlated_kg: torch.Tensor,
    d_uncorrelated_kg: torch.Tensor,
    alpha_correlated_nk: torch.Tensor,
    alpha_uncorrelated_nk: torch.Tensor,
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    x_ng = x_nmf_ng[data]
    x_ng = get_cellarium_target(x_ng)
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    # cellarium nmf fit
    cellarium_nmf = run_cellarium_nmf(x_ng, var_names_g=var_names_g, k=simulated_k, devices=devices)
    cellarium_nmf.get_rec_error = False  # hacking around strange code design
    cellarium_nmf.if_get_full_D = False  # hacking around strange code design
    cellarium_loadings_nk = cellarium_nmf.predict(x_ng, var_names_g=var_names_g)["alpha_nk"]
    assert isinstance(cellarium_loadings_nk, torch.Tensor)
    # cellarium_factors_kg = cellarium_nmf.factors_kg[simulated_k]
    cellarium_factors_kg = getattr(cellarium_nmf, f"D_{simulated_k}_rkg").squeeze(0)
    print(f"cellarium_loadings_nk: {cellarium_loadings_nk}")
    print(f"cellarium_factors_kg: {cellarium_factors_kg}")

    # sklearn nmf fit
    sklearn_nmf = NMF(n_components=simulated_k, init="random", solver="cd", beta_loss="frobenius", max_iter=10000)
    sklearn_loadings_nk = torch.from_numpy(sklearn_nmf.fit_transform(x_ng))
    sklearn_factors_kg = torch.from_numpy(sklearn_nmf.components_)

    # fraction of variance explained by each method
    frobenius_norm_data = torch.norm(x_ng, "fro")
    sklearn_reconstruction_ng = torch.matmul(sklearn_loadings_nk, sklearn_factors_kg)
    cellarium_reconstruction_ng = torch.matmul(cellarium_loadings_nk, cellarium_factors_kg)
    print(f"x_ng: {x_ng}")
    print(f"sklearn_reconstruction_ng: {sklearn_reconstruction_ng}")
    print(f"cellarium_reconstruction_ng: {cellarium_reconstruction_ng}")
    frobenius_norm_sklearn_residual = torch.norm(x_ng - sklearn_reconstruction_ng, "fro")
    frobenius_norm_cellarium_residual = torch.norm(x_ng - cellarium_reconstruction_ng, "fro")
    explained_variance_ratio_sklearn = 1 - (frobenius_norm_sklearn_residual**2) / (frobenius_norm_data**2)
    explained_variance_ratio_cellarium = 1 - (frobenius_norm_cellarium_residual**2) / (frobenius_norm_data**2)
    print(f"explained_variance_ratio_sklearn: {explained_variance_ratio_sklearn}")
    print(f"explained_variance_ratio_cellarium: {explained_variance_ratio_cellarium}")
    assert 0
    # np.testing.assert_allclose(expected_total_var, actual_total_var, rtol=1e-3)

    # # variance explained by each PC
    # expected_explained_var = L_g[:k]
    # actual_explained_var = ppca.L_k
    # np.testing.assert_allclose(expected_explained_var, actual_explained_var, rtol=1e-3)

    # # absolute cosine similarity between expected and actual PCs
    # abs_cos_sim = torch.abs(
    #     torch.nn.functional.cosine_similarity(
    #         ppca.U_gk,
    #         torch.as_tensor(U_gg[:, :k]),
    #         dim=0,
    #     )
    # )
    # np.testing.assert_allclose(np.ones(k), abs_cos_sim, rtol=1e-3)
