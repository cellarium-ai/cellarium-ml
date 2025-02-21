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
from cellarium.ml.models.nmf import compute_consensus_factors, get_embedding
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


def d_uncorrelated_kg(k, g, gene_set_sparsity_factor=gene_set_sparsity_factor) -> torch.Tensor:
    """Gene programs for NMF which are uncorrelated."""
    torch.manual_seed(0)
    d_kg = torch.distributions.Dirichlet(gene_set_sparsity_factor * torch.ones(g)).sample([k])
    return d_kg


@pytest.fixture(scope="module")
def fixture_d_uncorrelated_kg() -> torch.Tensor:
    return d_uncorrelated_kg(simulated_k, g)


def d_correlated_kg(k, g, gene_set_sparsity_factor=gene_set_sparsity_factor) -> torch.Tensor:
    """Gene programs for NMF which are somewhat correlated (some genes always off for example)."""
    torch.manual_seed(0)
    d_g = torch.distributions.Dirichlet(gene_set_sparsity_factor * 100 * torch.ones(g)).sample()
    d_kg = torch.distributions.Dirichlet(gene_set_sparsity_factor * 100 * d_g).sample([k])
    return d_kg


@pytest.fixture(scope="module")
def fixture_d_correlated_kg() -> torch.Tensor:
    return d_correlated_kg(simulated_k, g)


def alpha_uncorrelated_nk() -> torch.Tensor:
    """Cell loadings for NMF which are uncorrelated."""
    torch.manual_seed(0)
    alpha_nk = torch.distributions.Dirichlet(torch.ones(simulated_k)).sample([n])
    return alpha_nk


@pytest.fixture(scope="module")
def fixture_alpha_uncorrelated_nk() -> torch.Tensor:
    return alpha_uncorrelated_nk()


def alpha_correlated_nk(n, k, n_celltypes) -> torch.Tensor:
    """Cell loadings for NMF which are correlated within celltype blocks."""
    torch.manual_seed(0)
    alpha_ck = torch.distributions.Dirichlet(torch.ones(k)).sample([n_celltypes])
    alpha_nk = (
        torch.distributions.Dirichlet(cell_type_coherence_factor * alpha_ck + 1e-5)
        .sample([n // n_celltypes])
        .reshape(n, k)
    )
    return alpha_nk


@pytest.fixture(scope="module")
def fixture_alpha_correlated_nk() -> torch.Tensor:
    return alpha_correlated_nk(n, simulated_k, n_celltypes)


def x_uncorrelated_mean_nmf_ng(alpha_uncorrelated_nk, d_uncorrelated_kg) -> torch.Tensor:
    """Data created by a sparse NMF process with no correlation between the underlying factors."""
    x_ng = alpha_uncorrelated_nk @ d_uncorrelated_kg
    x_ng = x_ng / x_ng.sum(dim=-1).mean() * g * 100
    return x_ng


@pytest.fixture(scope="module")
def fixture_x_uncorrelated_mean_nmf_ng(fixture_alpha_uncorrelated_nk, fixture_d_uncorrelated_kg) -> torch.Tensor:
    return x_uncorrelated_mean_nmf_ng(fixture_alpha_uncorrelated_nk, fixture_d_uncorrelated_kg)


def x_correlated_mean_nmf_ng(alpha_correlated_nk, d_correlated_kg) -> torch.Tensor:
    """Data created by a sparse NMF process where the underlying factors are
    drawn from the same dirichlet distribution."""
    x_ng = alpha_correlated_nk @ d_correlated_kg
    x_ng = x_ng / x_ng.sum(dim=-1).mean() * g * 100
    return x_ng


@pytest.fixture(scope="module")
def fixture_x_correlated_mean_nmf_ng(fixture_alpha_uncorrelated_nk, fixture_d_uncorrelated_kg) -> torch.Tensor:
    return x_correlated_mean_nmf_ng(fixture_alpha_uncorrelated_nk, fixture_d_uncorrelated_kg)


@pytest.fixture(scope="module")
def x_nmf_ng(fixture_x_uncorrelated_mean_nmf_ng, fixture_x_correlated_mean_nmf_ng) -> dict[str, torch.Tensor]:
    """Data created by an NMF process with gaussian/poisson sampling noise
    and (no) correlation between the underlying factors and loadings."""
    torch.manual_seed(0)
    sigma = 0.1
    out = {}
    for name, mean_ng in zip(
        ["uncorrelated", "correlated"], [fixture_x_uncorrelated_mean_nmf_ng, fixture_x_correlated_mean_nmf_ng]
    ):
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
    seed: int,
    devices,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, g = x_ng.shape

    # dataloader
    batch_size = n // 10
    dataset = BoringDataset(
        data=x_ng.numpy(),
        var_names=var_names_g,
        obs_names=np.array([f"cell_{i}" for i in range(n)]),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    predict_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # model
    cellarium_nmf = NonNegativeMatrixFactorization(
        var_names_g=var_names_g.tolist(),
        var_names_hvg=var_names_g.tolist(),
        k_values=[k],
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
        max_epochs=1,
        strategy="auto" if devices == 1 else pl.strategies.DDPStrategy(broadcast_buffers=True),
    )

    # fit
    torch.manual_seed(seed)
    trainer.fit(module, train_dataloaders=train_loader)

    # get loadings and factors
    compute_consensus_factors(nmf_model=cellarium_nmf)  # sort of strange to need to do this
    cellarium_loadings_dataframe = get_embedding(
        dataset=predict_loader,
        pipeline=module.pipeline,
        k=k,
        if_get_final_gene_loading=False,
    )
    cellarium_loadings_nk = torch.tensor(cellarium_loadings_dataframe.values).float()
    assert isinstance(cellarium_loadings_nk, torch.Tensor)
    cellarium_factors_kg = getattr(cellarium_nmf, f"D_{k}_kg").squeeze(0)

    return cellarium_loadings_nk, cellarium_factors_kg


def run_sklearn_nmf(x_norm_ng: torch.Tensor, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    # sklearn nmf fit
    sklearn_nmf = NMF(
        n_components=k,
        init="random",
        solver="cd",
        beta_loss="frobenius",
        # solver="mu",
        # beta_loss="kullback-leibler",
        max_iter=1000,
        tol=1e-4,
        random_state=seed,
    )
    sklearn_loadings_nk = torch.from_numpy(sklearn_nmf.fit_transform(x_norm_ng)).float()
    sklearn_factors_kg = torch.from_numpy(sklearn_nmf.components_).float()
    return sklearn_loadings_nk, sklearn_factors_kg


def get_cellarium_normalized_data(x_ng: torch.Tensor) -> torch.Tensor:
    std_g = torch.std(x_ng, dim=0) + 1e-4
    x_ng = x_ng / std_g
    x_ng = torch.clamp(x_ng, min=0.0, max=100.0)
    return x_ng


def pairwise_cosine_similarity_cdist(tensor1_kg: torch.Tensor, tensor2_kg: torch.Tensor) -> torch.Tensor:
    # Normalize each tensor along the g dimension
    tensor1_norm_kg = tensor1_kg / tensor1_kg.norm(dim=1, keepdim=True)
    tensor2_norm_kg = tensor2_kg / tensor2_kg.norm(dim=1, keepdim=True)

    # Compute squared Euclidean distance
    squared_euclidean_dist_kk = torch.cdist(tensor1_norm_kg, tensor2_norm_kg, p=2) ** 2

    # Convert to cosine similarity
    cosine_similarity_matrix_kk = 1 - squared_euclidean_dist_kk / 2
    return cosine_similarity_matrix_kk


def pairwise_spearman_correlation(tensor1_kg: torch.Tensor, tensor2_kg: torch.Tensor) -> torch.Tensor:
    def rank_transform(tensor: torch.Tensor) -> torch.Tensor:
        """Returns the ranks of elements along each row."""
        ranks = tensor.argsort(dim=1).argsort(dim=1).to(torch.float)
        return ranks

    # Rank-transform both tensors
    ranked_tensor1_kg = rank_transform(tensor1_kg)
    ranked_tensor2_kg = rank_transform(tensor2_kg)

    # Normalize ranks (zero mean, unit variance)
    ranked_tensor1_kg = (ranked_tensor1_kg - ranked_tensor1_kg.mean(dim=1, keepdim=True)) / ranked_tensor1_kg.std(
        dim=1, unbiased=False, keepdim=True
    )
    ranked_tensor2_kg = (ranked_tensor2_kg - ranked_tensor2_kg.mean(dim=1, keepdim=True)) / ranked_tensor2_kg.std(
        dim=1, unbiased=False, keepdim=True
    )

    # Compute the Pearson correlation (dot product normalized by the number of elements)
    spearman_matrix_kk = ranked_tensor1_kg @ ranked_tensor2_kg.T / ranked_tensor1_kg.shape[1]

    return spearman_matrix_kk


def pairwise_pearson_correlation(tensor1_kg: torch.Tensor, tensor2_kg: torch.Tensor) -> torch.Tensor:
    # Normalize (zero mean, unit variance)
    norm_tensor1_kg = (tensor1_kg - tensor1_kg.mean(dim=1, keepdim=True)) / tensor1_kg.std(
        dim=1, unbiased=False, keepdim=True
    )
    norm_tensor2_kg = (tensor2_kg - tensor2_kg.mean(dim=1, keepdim=True)) / tensor2_kg.std(
        dim=1, unbiased=False, keepdim=True
    )

    # Compute the Pearson correlation (dot product normalized by the number of elements)
    pearson_matrix_kk = norm_tensor1_kg @ norm_tensor2_kg.T / norm_tensor1_kg.shape[1]

    return pearson_matrix_kk


def similarity_matrix_assign_rows_to_columns(
    similarity_kk: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    assert similarity_kk.shape[0] == similarity_kk.shape[1], "Similarity matrix must be square"
    assert similarity_kk.shape[0] > 0, "Similarity matrix must have at least one row and column"

    from scipy.optimize import linear_sum_assignment

    cost_kk = -similarity_kk

    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_kk)

    # Compute the total mean similarity
    total_similarity = similarity_kk[row_indices, col_indices].sum() / similarity_kk.shape[0]

    return total_similarity, np.asarray(row_indices), np.asarray(col_indices)


def run_nmf_and_sklearn_multi_device(
    x_ng: torch.Tensor,
    k: int = simulated_k,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    var_names_g = np.array([f"gene_{i}" for i in range(x_ng.shape[1])])

    # cellarium nmf fit
    cellarium_loadings_nk, cellarium_factors_kg = run_cellarium_nmf(
        x_ng=x_ng,
        var_names_g=var_names_g,
        k=k,
        devices=devices,
        seed=seed,
    )

    # sklearn nmf fit
    sklearn_loadings_nk, sklearn_factors_kg = run_sklearn_nmf(
        x_norm_ng=get_cellarium_normalized_data(x_ng),
        k=k,
        seed=seed,
    )

    loadings = {
        "cellarium": cellarium_loadings_nk,
        "sklearn": sklearn_loadings_nk,
    }
    factors = {
        "cellarium": cellarium_factors_kg,
        "sklearn": sklearn_factors_kg,
    }

    return loadings, factors


@pytest.mark.parametrize(
    "data", ["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"]
)
def test_nmf_against_sklearn(
    x_nmf_ng: dict[str, torch.Tensor],
    data: Literal["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"],
    fixture_d_correlated_kg: torch.Tensor,
    fixture_d_uncorrelated_kg: torch.Tensor,
    fixture_alpha_correlated_nk: torch.Tensor,
    fixture_alpha_uncorrelated_nk: torch.Tensor,
):
    # run both methods
    x_ng = x_nmf_ng[data]
    loadings, factors = run_nmf_and_sklearn_multi_device(
        x_ng,
    )
    x_norm_ng = get_cellarium_normalized_data(x_ng)
    cellarium_loadings_nk = loadings["cellarium"]
    cellarium_factors_kg = factors["cellarium"]
    sklearn_loadings_nk = loadings["sklearn"]
    sklearn_factors_kg = factors["sklearn"]

    d_correlated_kg = fixture_d_correlated_kg
    d_uncorrelated_kg = fixture_d_uncorrelated_kg
    alpha_correlated_nk = fixture_alpha_correlated_nk
    alpha_uncorrelated_nk = fixture_alpha_uncorrelated_nk

    # assert that fraction of variance explained by each method is similar
    frobenius_norm_data = torch.norm(x_norm_ng, "fro")
    sklearn_reconstruction_ng = torch.matmul(sklearn_loadings_nk, sklearn_factors_kg)
    cellarium_reconstruction_ng = torch.matmul(cellarium_loadings_nk, cellarium_factors_kg)
    print(f"x_norm_ng:\n{x_norm_ng}")
    print(f"sklearn_reconstruction_ng:\n{sklearn_reconstruction_ng}")
    print(f"cellarium_reconstruction_ng:\n{cellarium_reconstruction_ng}")
    frobenius_norm_sklearn_residual = torch.norm(x_norm_ng - sklearn_reconstruction_ng, "fro")
    frobenius_norm_cellarium_residual = torch.norm(x_norm_ng - cellarium_reconstruction_ng, "fro")
    explained_variance_ratio_sklearn = 1 - (frobenius_norm_sklearn_residual**2) / (frobenius_norm_data**2)
    explained_variance_ratio_cellarium = 1 - (frobenius_norm_cellarium_residual**2) / (frobenius_norm_data**2)
    print(f"explained_variance_ratio_sklearn: {explained_variance_ratio_sklearn}")
    print(f"explained_variance_ratio_cellarium: {explained_variance_ratio_cellarium}")

    # assert that the NMF loss is similar
    nmf_loss_sklearn = torch.nn.functional.mse_loss(x_norm_ng, sklearn_reconstruction_ng)
    nmf_loss_cellarium = torch.nn.functional.mse_loss(x_norm_ng, cellarium_reconstruction_ng)
    print(f"nmf_loss_sklearn: {nmf_loss_sklearn}")
    print(f"nmf_loss_cellarium: {nmf_loss_cellarium}")
    assert torch.abs(nmf_loss_sklearn - nmf_loss_cellarium) < 0.03, (
        f"cellarium and sklearn loss is not very similar: {torch.abs(nmf_loss_sklearn - nmf_loss_cellarium):.4f}"
    )

    # assert that the factors are similar
    pairwise_factor_similarity_kk = pairwise_cosine_similarity_cdist(cellarium_factors_kg, sklearn_factors_kg)
    total_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(pairwise_factor_similarity_kk)
    print(f"pairwise_factor_similarity_kk:\n{pairwise_factor_similarity_kk[row_indices, :][:, col_indices]}")
    print(f"total mean similarity: {total_similarity}")
    assert total_similarity > 0.95, f"factors are not very similar: {total_similarity}"

    # assert that the loadings are similar
    pairwise_loading_similarity_nn = pairwise_cosine_similarity_cdist(
        cellarium_loadings_nk,
        sklearn_loadings_nk,
    )
    total_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(
        pairwise_loading_similarity_nn
    )
    print(f"pairwise_loading_similarity_nn:\n{pairwise_loading_similarity_nn[row_indices, :][:, col_indices]}")
    print(f"total mean similarity: {total_similarity}")
    assert total_similarity > 0.95, f"loadings are not very similar: {total_similarity:.4f}"

    # truth
    if data.split("_")[-1] == "correlated":
        truth_factors_kg = d_correlated_kg
        truth_loadings_nk = alpha_correlated_nk
    else:
        truth_factors_kg = d_uncorrelated_kg
        truth_loadings_nk = alpha_uncorrelated_nk

    # assert that the cellarium factors match truth as much as the sklearn factors do
    pairwise_cellarium_factor_similarity_kk = pairwise_cosine_similarity_cdist(cellarium_factors_kg, truth_factors_kg)
    total_cellarium_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(
        pairwise_cellarium_factor_similarity_kk
    )
    print(
        f"pairwise_cellarium_factor_similarity_kk:"
        f"\n{pairwise_cellarium_factor_similarity_kk[row_indices, :][:, col_indices]}"
    )
    print(f"total mean cellarium similarity: {total_cellarium_similarity}")
    pairwise_sklearn_factor_similarity_kk = pairwise_cosine_similarity_cdist(sklearn_factors_kg, truth_factors_kg)
    total_sklearn_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(
        pairwise_sklearn_factor_similarity_kk
    )
    print(
        f"pairwise_sklearn_factor_similarity_kk:"
        f"\n{pairwise_sklearn_factor_similarity_kk[row_indices, :][:, col_indices]}"
    )
    print(f"total mean sklearn similarity: {total_sklearn_similarity}")
    assert total_sklearn_similarity - total_cellarium_similarity <= 0.042, (
        f"cellarium factors are substantially less similar to truth than sklearn factors: "
        f"{total_sklearn_similarity - total_cellarium_similarity:.4f}"
    )

    # specific threshold for each data type, intended to prevent performance regressions
    # these can be bumped up when performance is improved
    match data:
        case "gaussian_correlated":
            threshold: float = 0.25
        case "gaussian_uncorrelated":
            threshold = 0.55
        case "poisson_correlated":
            threshold = 0.19
        case "poisson_uncorrelated":
            threshold = 0.75
        case _:
            raise ValueError(f"unexpected data: {data}")

    assert total_cellarium_similarity > threshold, (
        f"cellarium factors are not very similar to truth: {total_cellarium_similarity:.4f}"
    )

    # assert that the cellarium loadings match truth as much as the sklearn loadings do
    pairwise_cellarium_loading_similarity_nn = pairwise_cosine_similarity_cdist(
        cellarium_loadings_nk,
        truth_loadings_nk,
    )
    total_cellarium_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(
        pairwise_cellarium_loading_similarity_nn
    )
    print(
        f"pairwise_cellarium_loading_similarity_nn:\n"
        f"{pairwise_cellarium_loading_similarity_nn[row_indices, :][:, col_indices]}"
    )
    print(f"total mean cellarium similarity: {total_cellarium_similarity}")
    pairwise_sklearn_loading_similarity_nn = pairwise_cosine_similarity_cdist(sklearn_loadings_nk, truth_loadings_nk)
    total_sklearn_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(
        pairwise_sklearn_loading_similarity_nn
    )
    print(
        f"pairwise_sklearn_loading_similarity_nn:"
        f"\n{pairwise_sklearn_loading_similarity_nn[row_indices, :][:, col_indices]}"
    )
    print(f"total mean sklearn similarity: {total_sklearn_similarity}")
    assert total_sklearn_similarity - total_cellarium_similarity <= 0.025, (
        "cellarium loadings are substantially less similar to truth than sklearn loadings"
        f"{total_sklearn_similarity - total_cellarium_similarity:.4f}"
    )
    assert total_similarity > 0.95, (
        f"cellarium loadings are not very similar to truth: {total_cellarium_similarity:.4f}"
    )


@pytest.mark.skip(reason="NMF does not yet work with multiple devices")
@pytest.mark.parametrize(
    "data", ["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"]
)
def test_nmf_against_sklearn_multi_device(
    x_nmf_ng: dict[str, torch.Tensor],
    data: Literal["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"],
    fixture_d_correlated_kg: torch.Tensor,
    fixture_d_uncorrelated_kg: torch.Tensor,
    fixture_alpha_correlated_nk: torch.Tensor,
    fixture_alpha_uncorrelated_nk: torch.Tensor,
):
    pass
