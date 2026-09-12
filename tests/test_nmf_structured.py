# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.models import AmortizedOnlineStructureAwareNMF
from cellarium.ml.models.nmf import solve_nnls_fista_precomputed
from cellarium.ml.models.nmf_structured import (
    MetadataAugmentedLoadingsEncoder,
    compute_metadata_nmf_init,
    group_lasso_prox,
    update_beta_group_lasso,
)
from cellarium.ml.transforms import DivideByScale, Filter
from cellarium.ml.utilities.data import AnnDataField, to_codes_column

os.environ["TORCH_COMPILE_DISABLE"] = "1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metadata_sim_adata() -> anndata.AnnData:
    """
    Simulated dataset where exactly one NMF factor is driven by a continuous metadata covariate (age).

    Ground truth:
        H_total = H_raw + H_struct,   H_struct = M * Beta
        X = H_total @ W_true + noise

    Only factor 0 is metadata-driven: Beta[0, 0] = 2.0, Beta[0, 1:] = 0.

    The fixture also stores an analytically-derived OLS coefficient matrix and the
    true gene factors in ``adata.uns["sim"]`` so tests can verify recovery.
    """
    rng = np.random.default_rng(0)
    n, g, k_true, D = 400, 20, 4, 1

    # True gene factors: random non-negative, L1-normalized rows
    W_raw = np.abs(rng.standard_normal((k_true, g))).astype(np.float32)
    W_true = W_raw / W_raw.sum(axis=1, keepdims=True).clip(1e-8)

    # True Beta: only column 0 (factor 0) is non-zero
    beta_true = np.zeros((D, k_true), dtype=np.float32)
    beta_true[0, 0] = 2.0

    # Metadata: age uniformly distributed from 20 to 80
    age_n = rng.uniform(20, 80, size=n).astype(np.float32)
    M_nd = age_n[:, None]  # (n, 1)

    # Structured component: H_struct = M @ Beta
    H_struct_nk = M_nd @ beta_true  # (n, k)

    # Idiosyncratic component: random non-negative, independent of age
    H_raw_nk = rng.exponential(scale=1.0, size=(n, k_true)).astype(np.float32)

    # Gene expression with small Gaussian noise
    H_total_nk = H_raw_nk + H_struct_nk
    noise = 0.05 * rng.standard_normal((n, g)).astype(np.float32)
    X_ng = np.clip(H_total_nk @ W_true + noise, 0, None)

    # Analytical OLS: regress M onto X (closed form since D=1)
    MtM = M_nd.T @ M_nd  # (1, 1)
    MtX = M_nd.T @ X_ng  # (1, g)
    ols_coeff_dg = np.linalg.solve(MtM + 1e-6 * np.eye(D), MtX).astype(np.float32)  # (1, g)

    metadata_mean_d = M_nd.mean(axis=0).astype(np.float32)  # (1,)
    metadata_min_d = M_nd.min(axis=0).astype(np.float32)  # (1,)
    metadata_max_d = M_nd.max(axis=0).astype(np.float32)  # (1,)

    # Binary categorical: 'healthy' (age < 50) vs 'sick' (age >= 50).
    # Alphabetical ordering → codes: healthy=0, sick=1. Both are in {0.0, 1.0}.
    disease_labels = ["sick" if a >= 50 else "healthy" for a in age_n]
    disease_cat = pd.Categorical(disease_labels, categories=["healthy", "sick"])
    metadata_mean_binary_d = np.array([disease_cat.codes.mean()], dtype=np.float32)

    adata = anndata.AnnData(
        X=X_ng,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(g)]),
        obs=pd.DataFrame(
            index=[f"cell_{i}" for i in range(n)],
            data={"age": age_n, "disease": disease_cat},
        ),
    )
    adata.obsm["metadata"] = M_nd  # (n, 1) float32
    adata.uns["sim"] = {
        "W_true": W_true,
        "beta_true": beta_true,
        "ols_coeff_dg": ols_coeff_dg,
        "metadata_mean_d": metadata_mean_d,
        "metadata_min_d": metadata_min_d,
        "metadata_max_d": metadata_max_d,
        "metadata_mean_binary_d": metadata_mean_binary_d,
        "M_nd": M_nd,
    }
    return adata


@pytest.fixture(scope="module")
def large_sim_adata() -> anndata.AnnData:
    """
    Larger simulated dataset (n=10240, g=30, k_true=6) for robust factor-recovery tests.

    n=10240 gives exactly 10 batches of 1024, matching the batch_size used in
    _build_recovery_module_and_data. Only factor 0 is metadata-driven (age),
    with beta_true[0, 0] = 2.0.
    """
    rng = np.random.default_rng(42)
    n, g, k_true, D = 10240, 30, 6, 1

    W_raw = np.abs(rng.standard_normal((k_true, g))).astype(np.float32)
    W_true = W_raw / W_raw.sum(axis=1, keepdims=True).clip(1e-8)

    beta_true = np.zeros((D, k_true), dtype=np.float32)
    beta_true[0, 0] = 2.0

    age_n = rng.uniform(20, 80, size=n).astype(np.float32)
    M_nd = age_n[:, None]

    H_struct_nk = M_nd @ beta_true
    H_raw_nk = rng.exponential(scale=1.0, size=(n, k_true)).astype(np.float32)
    H_total_nk = H_raw_nk + H_struct_nk
    noise = 0.05 * rng.standard_normal((n, g)).astype(np.float32)
    X_ng = np.clip(H_total_nk @ W_true + noise, 0, None).astype(np.float32)

    MtM = M_nd.T @ M_nd
    MtX = M_nd.T @ X_ng
    ols_coeff_dg = np.linalg.solve(MtM + 1e-6 * np.eye(D), MtX).astype(np.float32)
    metadata_mean_d = M_nd.mean(axis=0).astype(np.float32)
    metadata_min_d = M_nd.min(axis=0).astype(np.float32)
    metadata_max_d = M_nd.max(axis=0).astype(np.float32)

    adata = anndata.AnnData(
        X=X_ng,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(g)]),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n)], data={"age": age_n}),
    )
    adata.obsm["metadata"] = M_nd
    adata.uns["sim"] = {
        "W_true": W_true,
        "beta_true": beta_true,
        "ols_coeff_dg": ols_coeff_dg,
        "metadata_mean_d": metadata_mean_d,
        "metadata_min_d": metadata_min_d,
        "metadata_max_d": metadata_max_d,
        "M_nd": M_nd,
    }
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structured_module(
    adata: anndata.AnnData,
    k_values: list[int],
    r: int,
    n_metadata: int,
    metadata_mean_d: np.ndarray,
    metadata_min_d: np.ndarray | None = None,
    metadata_max_d: np.ndarray | None = None,
    ols_coeff_dg: np.ndarray | None = None,
    n_metadata_programs: int = 1,
    latent_dim: int = 16,
    lambda_align: float = 0.1,
    lambda_select: float = 0.05,
    mean_total_count: float | None = None,
    beta_lr: float = 1.0,
    batch_size: int = 64,
    exploration_epochs: int = 2,
) -> CellariumModule:
    var_names_g = np.array([f"gene_{i}" for i in range(adata.shape[1])])
    M = adata.obsm["metadata"].astype(np.float32)
    if metadata_min_d is None:
        metadata_min_d = M.min(axis=0)
    if metadata_max_d is None:
        metadata_max_d = M.max(axis=0)
    model = AmortizedOnlineStructureAwareNMF(
        var_names_g=var_names_g.tolist(),
        k_values=k_values,
        r=r,
        latent_dim=latent_dim,
        total_n_cells=adata.shape[0],
        batch_size=batch_size,
        n_metadata=n_metadata,
        metadata_mean_d=metadata_mean_d,
        metadata_min_d=metadata_min_d,
        metadata_max_d=metadata_max_d,
        n_metadata_programs=n_metadata_programs,
        ols_coeff_dg=ols_coeff_dg,
        mean_total_count=mean_total_count,
        lambda_align=lambda_align,
        lambda_select=lambda_select,
        beta_lr=beta_lr,
        exploration_epochs=exploration_epochs,
        max_solver_iter_train=30,
        max_solver_iter_cooldown=100,
    )
    return CellariumModule(
        cpu_transforms=[
            DivideByScale(
                scale_g=torch.from_numpy(adata.X.std(axis=0)).clamp(min=1e-4),
                var_names_g=var_names_g,
                eps=1e-4,
            ),
            Filter(var_names_g.tolist()),
        ],
        model=model,
    )


def _make_datamodule(adata: anndata.AnnData, batch_size: int = 64) -> CellariumAnnDataDataModule:
    return CellariumAnnDataDataModule(
        dadc=adata,
        batch_size=batch_size,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=None),
            "var_names_g": AnnDataField(attr="var_names"),
            "m_nd": AnnDataField(attr="obsm", key="metadata"),
        },
    )


# ---------------------------------------------------------------------------
# Unit tests: free functions
# ---------------------------------------------------------------------------


def test_group_lasso_prox_zeroes_weak_columns() -> None:
    """Columns whose group norm is below the threshold are exactly zeroed out."""
    torch.manual_seed(0)
    r, D, K = 2, 3, 5
    beta = torch.randn(r, D, K) * 0.01  # all columns have tiny norm
    lambda_select, lr = 1.0, 0.1
    out = group_lasso_prox(beta, lambda_select=lambda_select, lr=lr)
    # threshold = lambda_select * lr = 0.1; all column norms << 0.1 → all zeroed
    assert out.abs().max().item() == pytest.approx(0.0, abs=1e-7)


def test_group_lasso_prox_preserves_strong_columns() -> None:
    """Columns far above the threshold shrink but remain non-zero, direction preserved."""
    r, D, K = 1, 4, 3
    beta = torch.zeros(r, D, K)
    beta[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # large column 0
    lambda_select, lr = 0.01, 0.1  # threshold = 0.001; well below column-0 norm
    out = group_lasso_prox(beta, lambda_select=lambda_select, lr=lr)
    # Column 0 should shrink but remain non-zero
    assert out[0, :, 0].norm().item() > 0
    # Direction should be preserved (proportional to original)
    cos_sim = F.cosine_similarity(out[0, :, 0].unsqueeze(0), beta[0, :, 0].unsqueeze(0)).item()
    assert cos_sim == pytest.approx(1.0, abs=1e-5)
    # Other columns (zero) stay zero
    assert out[0, :, 1:].abs().max().item() == pytest.approx(0.0, abs=1e-7)


def test_compute_metadata_nmf_init_shapes() -> None:
    """Output tensors have the expected shapes."""
    torch.manual_seed(0)
    D, G, n_progs, n_reps, n_comp = 2, 30, 3, 4, 8
    ols = torch.randn(D, G)
    range_M = torch.ones(D) * 60.0  # e.g. age range 20-80
    W_out, Beta_out = compute_metadata_nmf_init(
        ols_coeff_dg=ols,
        n_metadata_programs=n_progs,
        n_replicates=n_reps,
        n_components=n_comp,
        range_M_d=range_M,
        mean_total_count=5000.0,
    )
    assert W_out.shape == (n_reps, n_progs, G)
    assert Beta_out.shape == (n_reps, D, n_progs)


def test_compute_metadata_nmf_init_l1_norms() -> None:
    """All rows of W_meta are L1-normalized and non-negative."""
    torch.manual_seed(1)
    D, G = 1, 50
    ols = torch.randn(D, G)
    range_M = torch.tensor([60.0])
    W_out, _ = compute_metadata_nmf_init(
        ols_coeff_dg=ols,
        n_metadata_programs=2,
        n_replicates=3,
        n_components=5,
        range_M_d=range_M,
        mean_total_count=8000.0,
    )
    assert (W_out >= 0).all(), "W factors must be non-negative"
    l1_norms = W_out.norm(p=1, dim=-1)  # (R, P)
    assert l1_norms.allclose(torch.ones_like(l1_norms), atol=1e-5), "W rows must be L1-normalized"


def test_compute_metadata_nmf_init_replicate_diversity() -> None:
    """Different replicates produce different W values when noise_scale > 0."""
    torch.manual_seed(2)
    D, G = 1, 40
    ols = torch.randn(D, G)
    range_M = torch.tensor([60.0])
    W_out, _ = compute_metadata_nmf_init(
        ols_coeff_dg=ols,
        n_metadata_programs=1,
        n_replicates=4,
        n_components=6,
        range_M_d=range_M,
        mean_total_count=5000.0,
        noise_scale=0.1,
    )
    # At least two replicates should differ
    assert not W_out[0].allclose(W_out[1]), "Replicates should be diverse with noise_scale > 0"


def test_metadata_encoder_output_shape() -> None:
    """MetadataAugmentedLoadingsEncoder produces (R, N, K) non-negative output."""
    torch.manual_seed(0)
    n_genes, latent_dim, n_metadata, r, n, k = 20, 16, 2, 3, 10, 5
    enc = MetadataAugmentedLoadingsEncoder(n_genes=n_genes, latent_dim=latent_dim, n_metadata=n_metadata)
    x_ng = torch.rand(n, n_genes)
    w_rkg = F.normalize(torch.rand(r, k, n_genes), p=1, dim=-1)
    m_nd = torch.rand(n, n_metadata)
    out = enc(x_ng, w_rkg, m_nd)
    assert out.shape == (r, n, k), f"Expected ({r}, {n}, {k}), got {out.shape}"
    assert (out >= 0).all(), "Encoder output should be non-negative"


# ---------------------------------------------------------------------------
# Smoke / structural tests (1 epoch, fast)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metadata_route", ["obsm_continuous", "obs_binary_cat"])
def test_structured_nmf_forward_returns_loss(metadata_route: str, metadata_sim_adata: anndata.AnnData) -> None:
    """forward() returns a dict with a non-negative scalar loss tensor.

    Exercises two metadata routes:
    - obsm_continuous: continuous age loaded from adata.obsm['metadata']
    - obs_binary_cat: binary 'disease' column loaded from adata.obs via to_codes_column
    """
    sim = metadata_sim_adata.uns["sim"]
    g = metadata_sim_adata.shape[1]
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    n_batch = 32
    x_ng = torch.from_numpy(metadata_sim_adata.X[:n_batch]).float()

    if metadata_route == "obsm_continuous":
        metadata_mean_d = sim["metadata_mean_d"]
        metadata_min_d = sim["metadata_min_d"]
        metadata_max_d = sim["metadata_max_d"]
        m_nd = torch.from_numpy(metadata_sim_adata.obsm["metadata"][:n_batch]).float()
    else:
        # Binary categorical: codes are {0.0, 1.0} — identical to min-max with min=0, max=1.
        metadata_mean_d = sim["metadata_mean_binary_d"]
        metadata_min_d = np.array([0.0], dtype=np.float32)
        metadata_max_d = np.array([1.0], dtype=np.float32)
        m_nd = torch.from_numpy(to_codes_column(metadata_sim_adata.obs["disease"][:n_batch])).float()

    model = AmortizedOnlineStructureAwareNMF(
        var_names_g=var_names_g.tolist(),
        k_values=[4],
        r=2,
        latent_dim=16,
        total_n_cells=metadata_sim_adata.shape[0],
        batch_size=64,
        n_metadata=1,
        metadata_mean_d=metadata_mean_d,
        metadata_min_d=metadata_min_d,
        metadata_max_d=metadata_max_d,
        n_metadata_programs=1,
    )
    result = model(x_ng=x_ng, var_names_g=var_names_g, m_nd=m_nd)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].ndim == 0
    assert result["loss"].item() >= 0


def test_binary_categorical_metadata_from_obs(metadata_sim_adata: anndata.AnnData) -> None:
    """Binary categorical obs column ('disease') used as metadata via to_codes_column.

    AnnDataField(attr='obs', key='disease', convert_fn=to_codes_column) produces an
    (N, 1) float32 array of codes {0.0, 1.0} — equivalent to min-max scaling with
    min=0, max=1.  Training for one epoch must complete without error and loss must be
    non-negative.
    """
    sim = metadata_sim_adata.uns["sim"]
    g = metadata_sim_adata.shape[1]
    k, r = 4, 2

    dm = CellariumAnnDataDataModule(
        dadc=metadata_sim_adata,
        batch_size=64,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=None),
            "var_names_g": AnnDataField(attr="var_names"),
            "m_nd": AnnDataField(attr="obs", key="disease", convert_fn=to_codes_column),
        },
    )

    module = CellariumModule(
        model=AmortizedOnlineStructureAwareNMF(
            var_names_g=[f"gene_{i}" for i in range(g)],
            k_values=[k],
            r=r,
            latent_dim=16,
            total_n_cells=metadata_sim_adata.shape[0],
            batch_size=64,
            n_metadata=1,
            metadata_mean_d=sim["metadata_mean_binary_d"],
            metadata_min_d=np.array([0.0], dtype=np.float32),
            metadata_max_d=np.array([1.0], dtype=np.float32),
            n_metadata_programs=1,
        ),
    )

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    trainer.fit(module, datamodule=dm)

    # After training the model must have produced a valid (non-NaN) loss.
    dm.setup(stage="fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    x_ng = batch["x_ng"].float()
    m_nd = batch["m_nd"].float()
    var_names_g = batch["var_names_g"]
    with torch.no_grad():
        result = module.model(x_ng=x_ng, var_names_g=var_names_g, m_nd=m_nd)
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() >= 0
    assert not torch.isnan(result["loss"])


def test_structured_nmf_single_device(metadata_sim_adata: anndata.AnnData) -> None:
    """Model trains for one epoch without error; factor shapes are correct."""
    sim = metadata_sim_adata.uns["sim"]
    g = metadata_sim_adata.shape[1]
    k, r = 4, 2
    dm = _make_datamodule(metadata_sim_adata, batch_size=64)
    dm.setup(stage="fit")
    module = _make_structured_module(
        adata=metadata_sim_adata,
        k_values=[k],
        r=r,
        n_metadata=1,
        metadata_mean_d=sim["metadata_mean_d"],
        n_metadata_programs=1,
    )
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)
    factors = module.model.factors_dict
    assert k in factors
    assert factors[k].shape == (r, k, g)


def test_multiple_k_values(metadata_sim_adata: anndata.AnnData) -> None:
    """Model handles multiple k values; Beta buffers have correct shapes."""
    sim = metadata_sim_adata.uns["sim"]
    D = metadata_sim_adata.obsm["metadata"].shape[1]
    k_values, r, n_progs = [5, 8], 2, 1
    dm = _make_datamodule(metadata_sim_adata, batch_size=64)
    dm.setup(stage="fit")
    module = _make_structured_module(
        adata=metadata_sim_adata,
        k_values=k_values,
        r=r,
        n_metadata=D,
        metadata_mean_d=sim["metadata_mean_d"],
        n_metadata_programs=n_progs,
    )
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)
    model = module.model
    assert isinstance(model, AmortizedOnlineStructureAwareNMF)
    for k in k_values:
        assert model.factors_dict[k].shape == (r, k, metadata_sim_adata.shape[1])
        assert getattr(model, f"beta_{k}_rdk").shape == (r, D, k)


def test_non_nominated_beta_always_zero(metadata_sim_adata: anndata.AnnData) -> None:
    """Non-nominated Beta columns stay exactly zero throughout training.

    By design, update_beta_group_lasso only touches columns 0:n_metadata_programs.
    Non-nominated columns are never updated, so they remain at exactly zero
    regardless of how many epochs are trained.
    """
    sim = metadata_sim_adata.uns["sim"]
    k, r, n_progs = 4, 2, 1
    module = _make_structured_module(
        adata=metadata_sim_adata,
        k_values=[k],
        r=r,
        n_metadata=1,
        metadata_mean_d=sim["metadata_mean_d"],
        n_metadata_programs=n_progs,
    )
    dm = _make_datamodule(metadata_sim_adata, batch_size=64)
    dm.setup(stage="fit")
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)

    model = module.model
    assert isinstance(model, AmortizedOnlineStructureAwareNMF)
    beta = getattr(model, f"beta_{k}_rdk")  # (r, D, k)
    non_nominated = beta[:, :, n_progs:]
    assert non_nominated.abs().max().item() == pytest.approx(0.0, abs=1e-7), (
        "Non-nominated Beta columns must remain exactly zero — they are never updated"
    )


# ---------------------------------------------------------------------------
# Recovery test
# ---------------------------------------------------------------------------


def test_ols_init_beta_sparsity(metadata_sim_adata: anndata.AnnData) -> None:
    """
    At initialization, nominated Beta column is non-zero and non-nominated columns
    are exactly zero. This is guaranteed by the OLS init + freeze design.
    """
    sim = metadata_sim_adata.uns["sim"]
    g = metadata_sim_adata.shape[1]
    k, r, n_progs = 4, 2, 1
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    # Instantiate model directly (no CellariumModule wrapper needed)
    model = AmortizedOnlineStructureAwareNMF(
        var_names_g=var_names_g.tolist(),
        k_values=[k],
        r=r,
        latent_dim=16,
        total_n_cells=metadata_sim_adata.shape[0],
        batch_size=64,
        n_metadata=1,
        metadata_mean_d=sim["metadata_mean_d"],
        metadata_min_d=sim["metadata_min_d"],
        metadata_max_d=sim["metadata_max_d"],
        n_metadata_programs=n_progs,
        ols_coeff_dg=sim["ols_coeff_dg"],
        mean_total_count=float(metadata_sim_adata.X.sum(axis=1).mean()),
    )
    beta_init = getattr(model, f"beta_{k}_rdk").detach()  # (r, D=1, k)
    non_nominated = beta_init[:, :, n_progs:]
    nominated = beta_init[:, :, :n_progs]
    assert non_nominated.abs().max().item() == pytest.approx(0.0, abs=1e-7), (
        "Non-nominated Beta columns must be exactly zero at initialization"
    )
    assert nominated.abs().max().item() > 0.0, "Nominated Beta column must be non-zero after OLS initialization"


def _build_recovery_module_and_data(
    large_sim_adata: anndata.AnnData,
    n_metadata_programs: int,
    k: int = 6,
    lambda_select: float = 0.1,
    lambda_align: float = 0.5,
    batch_size: int = 1024,
) -> tuple["CellariumModule", "CellariumAnnDataDataModule"]:
    """Shared setup for recovery tests: uses the large dataset with batch_size=1024."""
    sim = large_sim_adata.uns["sim"]
    r = 3
    module = _make_structured_module(
        adata=large_sim_adata,
        k_values=[k],
        r=r,
        n_metadata=1,
        metadata_mean_d=sim["metadata_mean_d"],
        ols_coeff_dg=sim["ols_coeff_dg"],
        n_metadata_programs=n_metadata_programs,
        latent_dim=32,
        lambda_align=lambda_align,
        lambda_select=lambda_select,
        mean_total_count=float(large_sim_adata.X.sum(axis=1).mean()),
        batch_size=batch_size,
        # Prevent early stopping from cutting training short before max_epochs=20.
        # The convergence trigger is designed for production runs; in tests we want
        # to run the full 20 epochs to reach the cosine-similarity threshold.
        exploration_epochs=25,
    )
    dm = _make_datamodule(large_sim_adata, batch_size=batch_size)
    return module, dm


def test_metadata_factor_recovery(large_sim_adata: anndata.AnnData) -> None:
    """
    After training, the model recovers the metadata-driven gene factor in the scaled space.

    Setup: n=10240 cells, g=30 genes, k_true=6 factors, D=1 (age).
    Only factor 0 is age-driven (beta_true[0, 0]=2.0). n_metadata_programs=1 so only
    the one nominated Beta column ever updates.

    Note on the cos-sim reference: the model trains on DivideByScale-normalized data
    (X / per_gene_std), so the optimal factor 0 in the model's W space is proportional
    to W_true[0] / per_gene_std (then L1-normalized), NOT W_true[0] itself. The test
    compares against this correctly-scaled reference.

    Checks:
    1. W[:,0,:] has cosine similarity >= 0.90 with the scaled reference W_true_scaled[0]
       in at least one replicate.
    2. Reconstruction error on scaled data beats all-zeros baseline.
    3. No NaN/Inf in encoder output.
    """
    k = 6
    module, dm = _build_recovery_module_and_data(large_sim_adata, k=k, n_metadata_programs=1)
    dm.setup(stage="fit")
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=20)
    trainer.fit(module, dm)

    sim = large_sim_adata.uns["sim"]
    W_true: np.ndarray = sim["W_true"]
    r = 3

    model = module.model
    assert isinstance(model, AmortizedOnlineStructureAwareNMF)
    beta_rdk = getattr(model, f"beta_{k}_rdk").detach()  # (r, D=1, k)
    W_rkg = getattr(model, f"D_{k}_rkg").detach()  # (r, k, g)

    # --- Assertion 1: W[:,0,:] aligns with the correctly-scaled W_true[0] ---
    # The model sees x_ng = X_ng / scale_g, so the optimal W_true in model space is
    # W_true[0] / scale_g (then L1-normalized). Compare against that reference.
    scale_g = torch.from_numpy(large_sim_adata.X.std(axis=0)).float().clamp(min=1e-4)
    W_true_unscaled = torch.from_numpy(W_true[0]).float()
    W_true_scaled = W_true_unscaled / scale_g
    # L1-normalize to match the model's W convention (rows sum to 1, non-negative)
    W_true_scaled_ref = F.normalize(W_true_scaled.unsqueeze(0), p=1, dim=-1, eps=1e-8).squeeze(0)

    cos_sims = [
        F.cosine_similarity(W_rkg[rep, 0].unsqueeze(0), W_true_scaled_ref.unsqueeze(0)).item() for rep in range(r)
    ]
    max_cos_sim = max(cos_sims)
    print(f"Max cosine similarity: {max_cos_sim:.3f} (per replicate: {[f'{c:.3f}' for c in cos_sims]})")
    assert max_cos_sim >= 0.95, (
        f"Max cosine similarity W[:,0,:] vs scaled W_true[0] should be >= 0.95; "
        f"got {max_cos_sim:.3f} (per replicate: {[f'{c:.3f}' for c in cos_sims]}). "
        f"OLS init should give a good warm start — check compute_metadata_nmf_init and "
        f"update_beta_group_lasso."
    )

    # --- Assertion 2: Reconstruction beats all-zeros on DivideByScale-normalized data ---
    # Run FISTA on the full dataset to get the optimal H_raw given the trained W and Beta.
    # This mirrors the training loop exactly: solve H_raw on X_eff = X - H_struct @ W,
    # then reconstruct with H_total = H_raw + H_struct. The encoder warm-start is only
    # used for training speed; FISTA is the ground truth for reconstruction quality.
    x_all_raw = torch.from_numpy(large_sim_adata.X).float()
    m_all = torch.from_numpy(large_sim_adata.obsm["metadata"]).float()
    x_all = x_all_raw / scale_g
    n_all = x_all.shape[0]

    model.eval()
    with torch.no_grad():
        m_scaled_all = ((m_all - model.min_M_global_d) / model.range_M_global_d).clamp(0.0, 1.0)
        H_struct_rnk = torch.einsum("nd,rdk->rnk", m_scaled_all, beta_rdk)  # linear
        X_eff_rng = x_all.unsqueeze(0) - torch.einsum("rnk,rkg->rng", H_struct_rnk, W_rkg)
        wwT_rkk = torch.einsum("rkg,rhg->rkh", W_rkg, W_rkg)
        wxT_rkn = torch.einsum("rkg,rng->rkn", W_rkg, X_eff_rng)
        H_raw_solver_kn, _ = solve_nnls_fista_precomputed(
            AtA=wwT_rkk,
            AtB=wxT_rkn,
            initial_x=torch.zeros(r, k, n_all),
            max_iter=200,
        )
        H_raw_solver_rnk = H_raw_solver_kn.transpose(-2, -1)
        H_total_rnk = H_raw_solver_rnk + H_struct_rnk
        recon_rng = torch.einsum("rnk,rkg->rng", H_total_rnk, W_rkg)
        recon_err = ((recon_rng - x_all.unsqueeze(0)) ** 2).mean().item()
        baseline_err = (x_all**2).mean().item()

    print(f"Reconstruction error: {recon_err:.4f}, Baseline error: {baseline_err:.4f}")
    assert np.isfinite(recon_err), "Reconstruction error must be finite"
    assert recon_err < baseline_err, f"Model ({recon_err:.4f}) should beat all-zeros baseline ({baseline_err:.4f})"

    # --- Assertion 3: No NaN/Inf in encoder output ---
    with torch.no_grad():
        H_raw_warm_rnk = model.encoder(x_all, W_rkg, m_scaled_all)
    assert not H_raw_warm_rnk.isnan().any(), "Encoder output must not contain NaN"
    assert not H_raw_warm_rnk.isinf().any(), "Encoder output must not contain Inf"


def test_group_lasso_prunes_excess_nominated_beta(large_sim_adata: anndata.AnnData) -> None:
    """
    Direct function-level test of group lasso pruning inside update_beta_group_lasso.

    Why a function-level test (not full training loop):
      In a stochastic training loop with k=6 factors, spurious nominated Beta columns
      (factors 1-3 with no age correlation) accumulate non-zero gradients due to finite
      overlaps between learned W factors. The exact lambda_select window where the true
      column survives but all spurious columns are pruned is narrow and depends on the
      specific W configuration at each training step. By calling update_beta_group_lasso
      directly with controlled orthogonal W factors, we guarantee the spurious gradient
      is exactly zero and the test cleanly verifies the pruning mechanism.

    Setup: block-orthogonal W factors (factor 0 covers genes 0-4, factors 1-5 cover
    disjoint gene sets) so overlap between factor 0 (age) and factors 1-3 is exactly
    zero. X is purely age-driven through factor 0. Beta starts at zero.

    Checks:
    - Beta[:,0] grows from zero (gradient from age signal >> lambda_select=5)
    - Beta[:,1:4] stay exactly at zero (zero gradient from orthogonal W)
    - Beta[:,4:6] stay at zero (never updated — not in nominated slice)
    """

    n, g, k, r, D = 1024, 30, 6, 3, 1
    n_progs = 4  # 4 nominated, but only factor 0 has age signal

    # Block-orthogonal W: factor 0 covers genes 0-4, factor 1 covers genes 5-9, etc.
    # This guarantees zero overlap between factor 0 and factors 1-3, making spurious
    # gradient exactly zero regardless of age variance or beta_true magnitude.
    W_rkg = torch.zeros(r, k, g)
    for fac in range(k):
        start = fac * 5
        W_rkg[:, fac, start : start + 5] = 1.0 / 5  # L1-normalized, 5 genes each

    # Metadata (age): take first 1024 cells from fixture; min-max scale to [0, 1].
    M_nd = torch.from_numpy(large_sim_adata.obsm["metadata"][:n]).float()  # (1024, 1)
    M_min = M_nd.min(dim=0).values
    M_range = (M_nd.max(dim=0).values - M_min).clamp(min=1e-8)
    M_scaled_nd = ((M_nd - M_min) / M_range).clamp(0.0, 1.0)  # in [0, 1]

    # X is ONLY driven by factor 0 through the age metadata (H_raw = 0 for simplicity).
    # We generate X in raw-age space; M_scaled is only used in the gradient update.
    beta_true_0 = 2.0
    H_struct_factor0 = M_nd * beta_true_0  # (1024, 1): raw ages * beta_true
    X_ng = H_struct_factor0 @ W_rkg[0, :1, :]  # (1024, g): age signal through factor 0 only

    # Beta starts at zero. With min-max scaled M and linear H_struct, at beta=0:
    #   H_struct=0, err_W = -X_res_WaT
    #   grad(col=0) ≈ -(2/n) * (M_scaled^T @ M_nd) * 2.0 * WaWaT[0,0] ≈ -24
    #   grad(col=1-3) = 0 (exactly orthogonal W factors)
    # With lambda_select=5: col 0 grows (24 >> 5), cols 1-3 stay at zero ✓
    beta_rdk = torch.zeros(r, D, k)

    beta_updated = update_beta_group_lasso(
        H_raw_rnk=torch.zeros(r, n, k),  # H_raw = 0 → X_res = X_ng exactly
        W_rkg=W_rkg,
        X_ng=X_ng,
        M_scaled_nd=M_scaled_nd,
        beta_rdk=beta_rdk,
        lambda_select=5.0,
        beta_lr=1.0,
        n_iter=200,  # enough iterations to converge from zero
        n_metadata_programs=n_progs,
    )

    # Nominated factor 0 should grow (age signal drives it; gradient >> lambda_select=5)
    beta_0_norm = beta_updated[:, :, 0].norm().item()
    assert beta_0_norm > 0.1, (
        f"Nominated Beta column 0 should be non-zero (group norm = {beta_0_norm:.4f}). "
        f"Factor 0 is the age gene direction; its gradient (≈24) >> lambda_select (5.0)."
    )

    # Nominated factors 1-3 should stay exactly at zero (zero gradient from orthogonal W)
    for col in range(1, n_progs):
        max_val = beta_updated[:, :, col].abs().max().item()
        assert max_val == pytest.approx(0.0, abs=1e-5), (
            f"Nominated Beta column {col} should remain zero (max abs = {max_val:.6f}). "
            f"Factor {col} is orthogonal to the age direction — gradient is exactly zero."
        )

    # Non-nominated factors (4:6) always stay at zero — never updated by design
    non_nominated_max = beta_updated[:, :, n_progs:].abs().max().item()
    assert non_nominated_max == pytest.approx(0.0, abs=1e-7), (
        f"Non-nominated Beta columns (index {n_progs}:{k}) must be exactly zero."
    )
