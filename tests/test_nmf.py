# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import NMF

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.models import OnlineNonNegativeMatrixFactorization
from cellarium.ml.models.nmf import NMFOutput
from cellarium.ml.transforms import DivideByScale, Filter
from cellarium.ml.utilities.data import AnnDataField


@pytest.fixture
def small_adata():
    n, g, k = 1000, 10, 3
    rng = np.random.default_rng(0)
    z_nk = rng.standard_normal(size=(n, k), dtype=np.float32)
    w_kg = rng.standard_normal(size=(k, g), dtype=np.float32)
    sigma = 0.6
    noise = sigma * rng.standard_normal(size=(n, g), dtype=np.float32)
    x_ng = z_nk @ w_kg + noise
    return anndata.AnnData(X=x_ng, var=pd.DataFrame(index=[f"gene_{i}" for i in range(g)]))


# set the number of cells, genes, cell types, and factors
n = 1000
g = 100
n_celltypes = 5
simulated_k = 3

# other constants
cell_type_coherence_factor = 20
gene_set_sparsity_factor = 0.1


def test_nmf_single_device(small_adata: anndata.AnnData):
    n, g = small_adata.shape
    k_values = [3, 4]
    devices = 1  # int(os.environ.get("TEST_DEVICES", "1"))

    # dataloader
    batch_size = n // 2
    dm = CellariumAnnDataDataModule(
        dadc=small_adata,
        batch_size=batch_size,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=None),
            "var_names_g": AnnDataField(attr="var_names"),
        },
    )
    dm.setup(stage="fit")
    # model
    nmf = OnlineNonNegativeMatrixFactorization(
        var_names_g=[f"gene_{i}" for i in range(g)],
        k_values=k_values,
        r=5,
    )
    module = CellariumModule(
        cpu_transforms=[
            DivideByScale(
                scale_g=torch.from_numpy(small_adata.X.std(axis=0)),
                var_names_g=np.array([f"gene_{i}" for i in range(g)]),
                eps=1e-4,
            ),
            Filter([f"gene_{i}" for i in range(g)]),
        ],
        model=nmf,
    )
    # trainer
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,
    )
    # fit
    trainer.fit(module, dm)


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
    x_ng = x_ng / x_ng.sum(dim=-1).mean() * g
    return x_ng


@pytest.fixture(scope="module")
def fixture_x_uncorrelated_mean_nmf_ng(fixture_alpha_uncorrelated_nk, fixture_d_uncorrelated_kg) -> torch.Tensor:
    return x_uncorrelated_mean_nmf_ng(fixture_alpha_uncorrelated_nk, fixture_d_uncorrelated_kg)


def x_correlated_mean_nmf_ng(alpha_correlated_nk, d_correlated_kg) -> torch.Tensor:
    """Data created by a sparse NMF process where the underlying factors are
    drawn from the same dirichlet distribution."""
    x_ng = alpha_correlated_nk @ d_correlated_kg
    x_ng = x_ng / x_ng.sum(dim=-1).mean() * g
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


def run_cellarium_online_nmf(
    x_ng: torch.Tensor,
    var_names_g: np.ndarray,
    k: int,
    seed: int,
    n_batches: int,
    devices,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, g = x_ng.shape

    # Create anndata object and datamodule
    import anndata
    import pandas as pd

    adata = anndata.AnnData(
        X=x_ng.numpy(), var=pd.DataFrame(index=var_names_g), obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    )

    # datamodule
    batch_size = n // n_batches
    from cellarium.ml.utilities.data import AnnDataField

    datamodule = CellariumAnnDataDataModule(
        dadc=adata,
        batch_size=batch_size,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=None),
            "var_names_g": AnnDataField(attr="var_names"),
            "obs_names_n": AnnDataField(attr="obs_names"),
        },
    )

    # model
    cellarium_nmf = OnlineNonNegativeMatrixFactorization(
        var_names_g=var_names_g.tolist(),
        k_values=[k],
        r=1,
    )
    module = CellariumModule(
        cpu_transforms=[
            DivideByScale(
                scale_g=x_ng.std(dim=0),
                var_names_g=var_names_g,
                eps=1e-4,
            ),
            # Filter(var_names_g.tolist()),
        ],
        model=cellarium_nmf,
    )

    # trainer
    trainer = pl.Trainer(
        barebones=False,
        accelerator="cpu",
        devices=devices,
        max_epochs=10,
        strategy="auto" if devices == 1 else pl.strategies.DDPStrategy(broadcast_buffers=True),
    )

    # fit
    torch.manual_seed(seed)
    trainer.fit(module, datamodule)

    # get loadings and factors using NMFOutput
    nmf_output = NMFOutput(nmf_module=module, datamodule=datamodule)
    nmf_output.compute_consensus_factors(k_values=k, density_threshold=1, local_neighborhood_size=0.3)
    cellarium_loadings_dataframe = nmf_output.compute_loadings(k=k, normalize=False)
    cellarium_loadings_nk = torch.tensor(cellarium_loadings_dataframe.values).float()
    assert isinstance(cellarium_loadings_nk, torch.Tensor)
    # Get consensus factors from nmf_output instead of the raw model
    consensus_factors = nmf_output.consensus[k]["consensus_D_kg"]
    assert isinstance(consensus_factors, torch.Tensor), "consensus_D_kg must be a tensor"
    cellarium_factors_kg = consensus_factors

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


def run_online_nmf_and_sklearn_multi_device(
    x_ng: torch.Tensor,
    k: int = simulated_k,
    seed: int = 0,
    n_cellarium_batches: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    var_names_g = np.array([f"gene_{i}" for i in range(x_ng.shape[1])])

    # cellarium nmf fit
    cellarium_loadings_nk, cellarium_factors_kg = run_cellarium_online_nmf(
        x_ng=x_ng,
        var_names_g=var_names_g,
        k=k,
        devices=devices,
        seed=seed,
        n_batches=n_cellarium_batches,
    )

    # sklearn nmf fit
    transform = DivideByScale(
        scale_g=x_ng.std(dim=0),
        var_names_g=var_names_g,
        eps=1e-4,
    )
    x_norm_ng = transform(x_ng=x_ng, var_names_g=var_names_g)["x_ng"]
    sklearn_loadings_nk, sklearn_factors_kg = run_sklearn_nmf(
        x_norm_ng=x_norm_ng,
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
@pytest.mark.parametrize("n_cellarium_batches", [1, 2, 10], ids=["fullbatch", "2batches", "10batches"])
def test_online_nmf_against_sklearn(
    x_nmf_ng: dict[str, torch.Tensor],
    data: Literal["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"],
    fixture_d_correlated_kg: torch.Tensor,
    fixture_d_uncorrelated_kg: torch.Tensor,
    fixture_alpha_correlated_nk: torch.Tensor,
    fixture_alpha_uncorrelated_nk: torch.Tensor,
    n_cellarium_batches: int,
):
    # run both methods
    x_ng = x_nmf_ng[data]
    var_names_g = np.array([f"gene_{i}" for i in range(x_ng.shape[1])])
    loadings, factors = run_online_nmf_and_sklearn_multi_device(
        x_ng,
        n_cellarium_batches=n_cellarium_batches,
    )
    transform = DivideByScale(
        scale_g=x_ng.std(dim=0),
        var_names_g=var_names_g,
        eps=1e-4,
    )
    x_norm_ng = transform(x_ng=x_ng, var_names_g=var_names_g)["x_ng"]
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

    # Debugging stuff
    print(f"Original data shape: {x_norm_ng.shape}")
    print(f"Cellarium factors shape: {cellarium_factors_kg.shape}")
    print(f"Cellarium loadings shape: {cellarium_loadings_nk.shape}")
    print(f"Sklearn factors shape: {sklearn_factors_kg.shape}")
    print(f"Sklearn loadings shape: {sklearn_loadings_nk.shape}")

    # Check if factors have negative values (they shouldn't for NMF)
    print(f"Cellarium factors min: {cellarium_factors_kg.min()}")
    print(f"Cellarium loadings min: {cellarium_loadings_nk.min()}")
    print(f"Sklearn factors min: {sklearn_factors_kg.min()}")
    print(f"Sklearn loadings min: {sklearn_loadings_nk.min()}")

    # Check the scales
    print(f"Original data mean: {x_norm_ng.mean()}, std: {x_norm_ng.std()}")
    print(
        f"Cellarium reconstruction mean: {cellarium_reconstruction_ng.mean()}, std: {cellarium_reconstruction_ng.std()}"
    )
    print(f"Sklearn reconstruction mean: {sklearn_reconstruction_ng.mean()}, std: {sklearn_reconstruction_ng.std()}")

    assert torch.abs(nmf_loss_sklearn - nmf_loss_cellarium) < 0.03, (
        f"cellarium and sklearn loss is not very similar: {torch.abs(nmf_loss_sklearn - nmf_loss_cellarium):.4f}"
    )

    # assert that the factors are similar
    pairwise_factor_similarity_kk = pairwise_cosine_similarity_cdist(cellarium_factors_kg, sklearn_factors_kg)
    total_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(pairwise_factor_similarity_kk)
    print(
        f"pairwise_factor_similarity_kk (cellarium and sklearn):"
        f"\n{pairwise_factor_similarity_kk[row_indices, :][:, col_indices]}"
    )
    print(f"total mean similarity: {total_similarity}")
    assert total_similarity > 0.98, f"factors are not very similar: {total_similarity}"

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
    assert total_similarity > 0.98, f"loadings are not very similar: {total_similarity:.4f}"

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
    print(f"total mean cellarium factor similarity to truth: {total_cellarium_similarity}")
    pairwise_sklearn_factor_similarity_kk = pairwise_cosine_similarity_cdist(sklearn_factors_kg, truth_factors_kg)
    total_sklearn_similarity, row_indices, col_indices = similarity_matrix_assign_rows_to_columns(
        pairwise_sklearn_factor_similarity_kk
    )
    print(
        f"pairwise_sklearn_factor_similarity_kk:"
        f"\n{pairwise_sklearn_factor_similarity_kk[row_indices, :][:, col_indices]}"
    )
    print(f"total mean sklearn factor similarity to truth: {total_sklearn_similarity}")
    assert total_sklearn_similarity - total_cellarium_similarity <= 0.01, (
        f"cellarium factors are substantially less similar to truth than sklearn factors: "
        f"{total_sklearn_similarity - total_cellarium_similarity:.4f}"
    )

    # specific threshold for each data type, intended to prevent performance regressions
    # these can be bumped up if performance is improved
    match data:
        case "gaussian_correlated":
            threshold: float = 0.2
        case "gaussian_uncorrelated":
            threshold = 0.55
        case "poisson_correlated":
            threshold = 0.15
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
    assert total_sklearn_similarity - total_cellarium_similarity <= 0.01, (
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
def test_online_nmf_against_sklearn_multi_device(
    x_nmf_ng: dict[str, torch.Tensor],
    data: Literal["gaussian_correlated", "gaussian_uncorrelated", "poisson_correlated", "poisson_uncorrelated"],
    fixture_d_correlated_kg: torch.Tensor,
    fixture_d_uncorrelated_kg: torch.Tensor,
    fixture_alpha_correlated_nk: torch.Tensor,
    fixture_alpha_uncorrelated_nk: torch.Tensor,
):
    pass


def kotliar_get_norm_counts(counts: anndata.AnnData, high_variance_genes_filter: list[str]) -> anndata.AnnData:
    """
    Slightly modified, taken from
    https://github.com/dylkot/cNMF/blob/7833a75484169cf448f8956224447cb110f4ba3d/src/cnmf/cnmf.py#L487

    Args:
        counts: Scanpy AnnData object (cells x genes) containing raw counts. Filtered such that
        no genes or cells with 0 counts

    high_variance_genes_filter: A pre-specified list of genes considered to be high-variance.
        Only these genes will be used during factorization of the counts matrix.
        Must match the .var index of counts.

    Returns:
        normcounts: anndata.AnnData, shape (cells, num_highvar_genes)
            Has a `.X` count matrix containing only the high variance genes with columns (genes)
            normalized to unit variance

    """
    ## Subset out high-variance genes
    norm_counts = counts[:, high_variance_genes_filter].copy()
    norm_counts.X = norm_counts.X.astype(np.float64)

    ## Scale genes to unit variance
    norm_counts.X /= norm_counts.X.std(axis=0, ddof=1)
    if np.isnan(norm_counts.X).sum().sum() > 0:
        print("Warning NaNs in normalized counts matrix")

    ## Check for any cells that have 0 counts of the overdispersed genes
    zerocells = np.array(norm_counts.X.sum(axis=1) == 0).reshape(-1)
    if zerocells.sum() > 0:
        examples = norm_counts.obs.index[np.ravel(zerocells)]
        raise Exception(
            f"Error: {zerocells.sum()} cells have zero counts of overdispersed genes. E.g. {', '.join(examples[:4])}. "
            "Filter those cells and re-run or adjust the number of overdispersed genes. Quitting!"
        )

    return norm_counts


def test_preprocessing_matches_kotliar():
    pass


def test_consensus_matches_kotliar():
    pass


def test_refit_all_genes_matches_kotliar():
    pass
