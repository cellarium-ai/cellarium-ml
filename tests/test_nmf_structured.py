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
        lambda_align=lambda_align,
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


def test_compute_metadata_nmf_init_shapes() -> None:
    """Output tensor has the expected shape."""
    torch.manual_seed(0)
    D, G, n_progs, n_reps = 2, 30, 3, 4
    ols = torch.randn(D, G)
    range_M = torch.ones(D) * 60.0  # e.g. age range 20-80
    W_out, _ = compute_metadata_nmf_init(
        ols_coeff_dg=ols,
        n_metadata_programs=n_progs,
        n_replicates=n_reps,
        n_components=n_progs,
        range_M_d=range_M,
        mean_total_count=1000.0,
    )
    assert W_out.shape == (n_reps, n_progs, G)


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
        n_components=4,
        range_M_d=range_M,
        mean_total_count=1000.0,
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
        n_components=4,
        range_M_d=range_M,
        mean_total_count=1000.0,
        noise_scale=0.1,
    )
    # At least two replicates should differ
    assert not W_out[0].allclose(W_out[1]), "Replicates should be diverse with noise_scale > 0"


def test_update_beta_group_lasso_prior_pulls_nominated_positive() -> None:
    """With lambda_prior > 0, the prior gradient pulls nominated Beta toward beta_prior_active.

    Uses X=0 and H_raw=0 so the data gradient at beta=0 is exactly zero; only the prior
    gradient acts on the first step. After 100 iterations the nominated columns must be
    positive (pulled from zero toward beta_prior_active=2.0). Non-nominated columns are
    never touched and must remain exactly zero.
    """
    torch.manual_seed(0)
    r, D, n, k, n_progs = 2, 1, 100, 4, 2
    g = k * 5

    W_rkg = torch.zeros(r, k, g)
    for i in range(k):
        W_rkg[:, i, i * 5 : (i + 1) * 5] = 0.2  # L1-normalized, disjoint blocks

    M_scaled_nd = torch.rand(n, D)
    X_ng = torch.zeros(n, g)  # zero data → data gradient is zero at beta=0
    H_raw_rnk = torch.zeros(r, n, k)
    beta_prior = torch.full((r, D, n_progs), 2.0)
    beta_rdk = torch.zeros(r, D, k)

    result = update_beta_group_lasso(
        H_raw_rnk=H_raw_rnk,
        W_rkg=W_rkg,
        X_ng=X_ng,
        M_scaled_nd=M_scaled_nd,
        beta_rdk=beta_rdk,
        lambda_select=0.0,
        beta_lr=1.0,
        n_iter=100,
        n_metadata_programs=n_progs,
        beta_prior_active=beta_prior,
        lambda_prior=1.0,
    )

    assert torch.isfinite(result).all(), "Result must be finite with lambda_prior > 0"
    assert (result >= 0).all(), "Result must remain non-negative"
    assert result[:, :, :n_progs].min().item() > 0.0, (
        "Nominated Beta should be pulled positive by the prior (beta_prior=2.0)"
    )
    assert result[:, :, n_progs:].abs().max().item() == pytest.approx(0.0, abs=1e-7), (
        "Non-nominated columns must remain exactly zero"
    )


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
    """Model handles multiple k values; factor shapes are correct."""
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


def test_nan_metadata_produces_finite_loss(metadata_sim_adata: anndata.AnnData) -> None:
    """
    forward() returns a finite loss when m_nd contains NaN for ~25% of cells.

    Exercises:
    - nan_to_num(0.0) on m_scaled_nd
    - M_c_nd.masked_fill → NaN cells excluded from the covariance penalty numerator
    - n_valid normalization in the covariance penalty
    - EMA mean computed over valid cells only (second call has _n_ema_updates > 0)
    """
    rng = np.random.default_rng(7)
    sim = metadata_sim_adata.uns["sim"]
    g = metadata_sim_adata.shape[1]
    n_batch = 64
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    x_ng = torch.from_numpy(metadata_sim_adata.X[:n_batch]).float()
    m_nd = torch.from_numpy(metadata_sim_adata.obsm["metadata"][:n_batch]).float().clone()

    # Inject NaN into ~25% of cells
    nan_indices = rng.choice(n_batch, size=n_batch // 4, replace=False)
    m_nd[nan_indices] = float("nan")
    assert torch.isnan(m_nd).any(), "Sanity: m_nd must have NaN before the model sees it"

    model = AmortizedOnlineStructureAwareNMF(
        var_names_g=var_names_g.tolist(),
        k_values=[4],
        r=2,
        latent_dim=16,
        total_n_cells=metadata_sim_adata.shape[0],
        batch_size=64,
        n_metadata=1,
        metadata_mean_d=sim["metadata_mean_d"],
        metadata_min_d=sim["metadata_min_d"],
        metadata_max_d=sim["metadata_max_d"],
        n_metadata_programs=1,
        ols_coeff_dg=sim["ols_coeff_dg"],
    )

    # First call: _n_ema_updates=0, mu_H_corrected=0
    loss1 = model(x_ng=x_ng, var_names_g=var_names_g, m_nd=m_nd)["loss"]
    assert loss1 is not None
    assert torch.isfinite(loss1), f"Loss must be finite on first call with NaN metadata; got {loss1.item()}"

    # Second call: _n_ema_updates=1, exercises the EMA-centering path in the covariance penalty
    loss2 = model(x_ng=x_ng, var_names_g=var_names_g, m_nd=m_nd)["loss"]
    assert loss2 is not None
    assert torch.isfinite(loss2), f"Loss must be finite on second call (EMA path) with NaN metadata; got {loss2.item()}"

    # W buffers must remain finite after both updates
    k = 4
    W = getattr(model, f"D_{k}_rkg")
    assert not W.isnan().any(), "W must not contain NaN after forward with NaN metadata"
    assert torch.isfinite(W).all(), "W must be finite after forward with NaN metadata"


def test_lambda_prior_finite_and_frozen(metadata_sim_adata: anndata.AnnData) -> None:
    """Model with lambda_prior > 0 produces finite loss; beta_prior buffer is frozen.

    Checks:
    - Loss is finite on both forward passes (exercises the _n_ema_updates=0 and >0 paths)
    - Beta and W remain finite and Beta remains non-negative
    - beta_prior_{k}_rdk is unchanged after training (it is a fixed prior, not a running stat)
    """
    sim = metadata_sim_adata.uns["sim"]
    g = metadata_sim_adata.shape[1]
    n_batch = 64
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    x_ng = torch.from_numpy(metadata_sim_adata.X[:n_batch]).float()
    m_nd = torch.from_numpy(metadata_sim_adata.obsm["metadata"][:n_batch]).float()

    model = AmortizedOnlineStructureAwareNMF(
        var_names_g=var_names_g.tolist(),
        k_values=[4],
        r=2,
        latent_dim=16,
        total_n_cells=metadata_sim_adata.shape[0],
        batch_size=64,
        n_metadata=1,
        metadata_mean_d=sim["metadata_mean_d"],
        metadata_min_d=sim["metadata_min_d"],
        metadata_max_d=sim["metadata_max_d"],
        n_metadata_programs=1,
        ols_coeff_dg=sim["ols_coeff_dg"],
        mean_total_count=float(metadata_sim_adata.X.sum(axis=1).mean()),
        lambda_prior=0.1,
    )
    k = 4
    beta_prior_at_init = getattr(model, f"beta_prior_{k}_rdk").clone()

    for call_idx in range(2):
        loss = model(x_ng=x_ng, var_names_g=var_names_g, m_nd=m_nd)["loss"]
        assert loss is not None
        assert torch.isfinite(loss), f"Loss must be finite (call {call_idx}); got {loss.item()}"

    beta = getattr(model, f"beta_{k}_rdk")
    beta_prior = getattr(model, f"beta_prior_{k}_rdk")
    assert torch.isfinite(beta).all(), "Beta must be finite after forward with lambda_prior > 0"
    assert (beta >= 0).all(), "Beta must remain non-negative with lambda_prior > 0"
    assert not beta.isnan().any(), "Beta must not contain NaN"
    assert beta_prior.equal(beta_prior_at_init), (
        "beta_prior buffer must be frozen — it is a fixed prior, never updated during training"
    )


# ---------------------------------------------------------------------------
# Recovery test
# ---------------------------------------------------------------------------


def _build_recovery_module_and_data(
    large_sim_adata: anndata.AnnData,
    n_metadata_programs: int,
    k: int = 6,
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
        batch_size=batch_size,
        # Prevent early stopping from cutting training short before max_epochs=20.
        exploration_epochs=25,
    )
    dm = _make_datamodule(large_sim_adata, batch_size=batch_size)
    return module, dm


def test_metadata_factor_recovery(large_sim_adata: anndata.AnnData) -> None:
    """
    After training, the model recovers the metadata-driven gene factor in the scaled space.

    Setup: n=10240 cells, g=30 genes, k_true=6 factors, D=1 (age).
    Only factor 0 is age-driven (beta_true[0, 0]=2.0). n_metadata_programs=1 so the
    first nominated program captures the age signal via the tax-free haven mechanism.

    Note on the cos-sim reference: the model trains on DivideByScale-normalized data
    (X / per_gene_std), so the optimal factor 0 in the model's W space is proportional
    to W_true[0] / per_gene_std (then L1-normalized), NOT W_true[0] itself. The test
    compares against this correctly-scaled reference.

    Checks:
    1. W[:,0,:] has cosine similarity >= 0.90 with the scaled reference W_true_scaled[0]
       in at least one replicate.
    2. Reconstruction error on scaled data beats all-zeros baseline (FISTA on full X).
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
    W_rkg = getattr(model, f"D_{k}_rkg").detach()  # (r, k, g)

    # --- Assertion 1: W[:,0,:] aligns with the correctly-scaled W_true[0] ---
    scale_g = torch.from_numpy(large_sim_adata.X.std(axis=0)).float().clamp(min=1e-4)
    W_true_unscaled = torch.from_numpy(W_true[0]).float()
    W_true_scaled = W_true_unscaled / scale_g
    W_true_scaled_ref = F.normalize(W_true_scaled.unsqueeze(0), p=1, dim=-1, eps=1e-8).squeeze(0)

    cos_sims = [
        F.cosine_similarity(W_rkg[rep, 0].unsqueeze(0), W_true_scaled_ref.unsqueeze(0)).item() for rep in range(r)
    ]
    max_cos_sim = max(cos_sims)
    print(f"Max cosine similarity: {max_cos_sim:.3f} (per replicate: {[f'{c:.3f}' for c in cos_sims]})")
    assert max_cos_sim >= 0.90, (
        f"Max cosine similarity W[:,0,:] vs scaled W_true[0] should be >= 0.90; "
        f"got {max_cos_sim:.3f} (per replicate: {[f'{c:.3f}' for c in cos_sims]}). "
        f"OLS init should give a good warm start — check compute_metadata_nmf_init."
    )

    # --- Assertion 2: Reconstruction beats all-zeros baseline ---
    # Run plain FISTA on full X given the trained W to get H_total, then reconstruct.
    x_all_raw = torch.from_numpy(large_sim_adata.X).float()
    x_all = x_all_raw / scale_g
    n_all = x_all.shape[0]

    model.eval()
    with torch.no_grad():
        wwT_rkk = torch.einsum("rkg,rhg->rkh", W_rkg, W_rkg)
        WxT_rkn = torch.einsum("rkg,ng->rkn", W_rkg, x_all)
        H_total_solver_rkn, _ = solve_nnls_fista_precomputed(
            AtA=wwT_rkk,
            AtB=WxT_rkn,
            initial_x=torch.zeros(r, k, n_all),
            max_iter=200,
        )
        H_total_rnk = H_total_solver_rkn.transpose(-2, -1)
        recon_rng = torch.einsum("rnk,rkg->rng", H_total_rnk, W_rkg)
        recon_err = ((recon_rng - x_all.unsqueeze(0)) ** 2).mean().item()
        baseline_err = (x_all**2).mean().item()

    print(f"Reconstruction error: {recon_err:.4f}, Baseline error: {baseline_err:.4f}")
    assert np.isfinite(recon_err), "Reconstruction error must be finite"
    assert recon_err < baseline_err, f"Model ({recon_err:.4f}) should beat all-zeros baseline ({baseline_err:.4f})"

    # --- Assertion 3: No NaN/Inf in encoder output ---
    m_all = torch.from_numpy(large_sim_adata.obsm["metadata"]).float()
    m_scaled_all = ((m_all - model.min_M_global_d) / model.range_M_global_d).clamp(0.0, 1.0)
    with torch.no_grad():
        H_total_warm_rnk = model.encoder(x_all, W_rkg, m_scaled_all)
    assert not H_total_warm_rnk.isnan().any(), "Encoder output must not contain NaN"
    assert not H_total_warm_rnk.isinf().any(), "Encoder output must not contain Inf"


# ---------------------------------------------------------------------------
# New mechanism tests
# ---------------------------------------------------------------------------


def test_alignment_penalty_reduces_free_program_metadata_correlation(
    metadata_sim_adata: anndata.AnnData,
) -> None:
    """
    With lambda_align > 0, the FISTA solver routes metadata variance to nominated programs,
    reducing the correlation of free programs with metadata vs. lambda_align = 0.
    """
    sim = metadata_sim_adata.uns["sim"]
    g = metadata_sim_adata.shape[1]
    n_batch = 128
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    x_ng = torch.from_numpy(metadata_sim_adata.X[:n_batch]).float()
    m_nd = torch.from_numpy(metadata_sim_adata.obsm["metadata"][:n_batch]).float()

    def make_and_run(lam: float) -> torch.Tensor:
        torch.manual_seed(0)
        model = AmortizedOnlineStructureAwareNMF(
            var_names_g=var_names_g.tolist(),
            k_values=[4],
            r=1,
            latent_dim=16,
            total_n_cells=metadata_sim_adata.shape[0],
            batch_size=n_batch,
            n_metadata=1,
            metadata_mean_d=sim["metadata_mean_d"],
            metadata_min_d=sim["metadata_min_d"],
            metadata_max_d=sim["metadata_max_d"],
            n_metadata_programs=1,
            lambda_align=lam,
        )
        out = model.online_dictionary_update(x_ng=x_ng, k=4, m_nd=m_nd, n_iterations=50)
        return out["solver_loadings_rnk"]  # (1, N, 4)

    H_no_pen = make_and_run(0.0)
    H_pen = make_and_run(10.0)

    min_M = torch.from_numpy(sim["metadata_min_d"]).float()
    max_M = torch.from_numpy(sim["metadata_max_d"]).float()
    m_scaled = ((m_nd - min_M) / (max_M - min_M)).clamp(0, 1).squeeze(1)  # (N,)
    m_c = m_scaled - m_scaled.mean()
    std_m = m_c.std().clamp(min=1e-8)

    def max_free_corr(H_rnk: torch.Tensor) -> float:
        H_free = H_rnk[0, :, 1:]  # (N, K_free): non-nominated programs
        H_c = H_free - H_free.mean(dim=0, keepdim=True)
        cov = (H_c * m_c.unsqueeze(1)).mean(dim=0)
        std_H = H_c.std(dim=0).clamp(min=1e-8)
        return (cov / (std_H * std_m)).abs().max().item()

    corr_no = max_free_corr(H_no_pen)
    corr_pen = max_free_corr(H_pen)
    assert corr_pen < corr_no, (
        f"lambda_align=10 should reduce free-program correlation with metadata; "
        f"got corr_pen={corr_pen:.4f}, corr_no={corr_no:.4f}"
    )
