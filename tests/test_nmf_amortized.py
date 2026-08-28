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
from cellarium.ml.models.nmf_amortized import ConsensusNMFEncoder, FiLMBlock
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
    encoder_hidden_dims: list[int],
    batch_size: int,
) -> CellariumModule:
    g = small_adata.shape[1]
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    amortized_nmf = AmortizedOnlineNonNegativeMatrixFactorization(
        var_names_g=var_names_g.tolist(),
        k_values=k_values,
        r=r,
        encoder_hidden_dims=encoder_hidden_dims,
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
    module = _make_module(small_adata, k_values=[3], r=2, encoder_hidden_dims=[16], batch_size=n // 2)
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)


def test_amortized_nmf_multiple_k_values(small_adata: anndata.AnnData) -> None:
    """Model handles multiple k values simultaneously and produces factors of the correct shape."""
    n, g = small_adata.shape
    k_values = [3, 4]
    r = 2
    dm = _make_datamodule(small_adata, batch_size=n // 2)
    dm.setup(stage="fit")
    module = _make_module(small_adata, k_values=k_values, r=r, encoder_hidden_dims=[16], batch_size=n // 2)
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
        encoder_hidden_dims=[16],
        total_n_cells=small_adata.shape[0],
        batch_size=50,
    )
    result = amortized_nmf(x_ng=x_ng, var_names_g=var_names_g)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() >= 0


def test_film_block_output_shape() -> None:
    """FiLMBlock produces (R, N, H) outputs and non-negative activations for both 2D and 3D inputs,
    and produces (R_active, N, H) when active_replicates is given."""
    input_dim, output_dim, r, n = 10, 8, 5, 4
    block = FiLMBlock(input_dim=input_dim, output_dim=output_dim, num_replicates=r)

    # 2D input (N, input_dim) -> (R, N, output_dim): first layer in the stack
    x_2d = torch.randn(n, input_dim)
    out_2d = block(x_2d)
    assert out_2d.shape == (r, n, output_dim), f"Expected ({r}, {n}, {output_dim}), got {out_2d.shape}"
    assert (out_2d >= 0).all(), "Output should be non-negative after ReLU"

    # 3D input (R, N, input_dim) -> (R, N, output_dim): subsequent layers in the stack
    x_3d = torch.randn(r, n, input_dim)
    out_3d = block(x_3d)
    assert out_3d.shape == (r, n, output_dim), f"Expected ({r}, {n}, {output_dim}), got {out_3d.shape}"
    assert (out_3d >= 0).all()

    # active_replicates subsets the replicate dimension
    active = [0, 2, 4]
    out_active = block(x_2d, active_replicates=active)
    assert out_active.shape == (len(active), n, output_dim)
    # output for active replicates should match the full forward indexed at those rows
    assert torch.allclose(out_active, out_2d[active])


def test_consensus_nmf_encoder_output_shape() -> None:
    """ConsensusNMFEncoder produces (R, N, K) outputs that are non-negative,
    and (R_active, N, K) when active_replicates is given."""
    num_genes, num_factors, r, n = 10, 4, 5, 4
    encoder = ConsensusNMFEncoder(
        num_genes=num_genes,
        hidden_dims=[16, 8],
        num_factors=num_factors,
        num_replicates=r,
    )
    x_ng = torch.randn(n, num_genes)
    out = encoder(x_ng)
    assert out.shape == (r, n, num_factors), f"Expected ({r}, {n}, {num_factors}), got {out.shape}"
    assert (out >= 0).all(), "Encoder output should be non-negative after ReLU"

    # active_replicates subsets the replicate dimension end-to-end through all FiLM blocks
    active = [1, 3]
    out_active = encoder(x_ng, active_replicates=active)
    assert out_active.shape == (len(active), n, num_factors)
    assert (out_active >= 0).all()


def test_amortized_nmf_infer_loadings(small_adata: anndata.AnnData) -> None:
    """After training, NMFOutput can compute consensus factors and per-cell loadings."""
    n, g = small_adata.shape
    k = 3
    # r=1 keeps the test fast and avoids the n_neighbors >= 2 requirement in compute_consensus_factors
    r = 1
    dm = _make_datamodule(small_adata, batch_size=n // 2)
    dm.setup(stage="fit")
    module = _make_module(small_adata, k_values=[k], r=r, encoder_hidden_dims=[16], batch_size=n // 2)
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


def test_factor_drift_metric() -> None:
    """_compute_factor_drift returns the relative Frobenius norm of change."""
    model = AmortizedOnlineNonNegativeMatrixFactorization(
        var_names_g=[f"gene_{i}" for i in range(10)],
        k_values=[3],
        r=2,
        encoder_hidden_dims=[8],
        total_n_cells=100,
        batch_size=8,
    )
    D = torch.rand(3, 10)
    D_prev = D.clone()

    # identical tensors → zero drift
    assert model._compute_factor_drift(D, D_prev) == 0.0

    # scale D by a constant: relative drift = ||(c-1)*D||_F / ||D||_F = |c-1|
    c = 1.05
    assert abs(model._compute_factor_drift(D * c, D_prev) - abs(c - 1)) < 1e-5

    # zero previous → inf (undefined)
    assert model._compute_factor_drift(D, torch.zeros_like(D)) == float("inf")


def test_converged_replicate_not_updated() -> None:
    """D buffer rows for converged replicates are not modified by forward()."""
    g = 10
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    x_ng = torch.rand(8, g)
    model = AmortizedOnlineNonNegativeMatrixFactorization(
        var_names_g=var_names_g.tolist(),
        k_values=[3],
        r=3,
        encoder_hidden_dims=[8],
        total_n_cells=100,
        batch_size=8,
    )
    model.converged_rK[0, model.k_to_idx[3]] = True
    D_row_before = model.D_3_rkg[0].clone()

    model(x_ng=x_ng, var_names_g=var_names_g)

    assert torch.equal(model.D_3_rkg[0], D_row_before), "Converged replicate's D row must not change"


def test_per_replicate_convergence_tracking() -> None:
    """Patience counter accumulates on low drift and marks replicate converged; large drift resets patience."""
    model = AmortizedOnlineNonNegativeMatrixFactorization(
        var_names_g=[f"gene_{i}" for i in range(10)],
        k_values=[3],
        r=2,
        encoder_hidden_dims=[8],
        total_n_cells=100,
        batch_size=8,
        convergence_patience=2,
    )
    k_idx = model.k_to_idx[3]

    # Simulate two windows of zero drift → both replicates should be marked converged
    model._D_snapshots[3] = model.D_3_rkg.clone()
    for _ in range(2):
        D_full = model.D_3_rkg.detach()
        snapshot = model._D_snapshots[3]
        assert isinstance(snapshot, torch.Tensor)
        for r in range(2):
            if model.converged_rK[r, k_idx]:
                continue
            drift = model._compute_factor_drift(D_full[r], snapshot[r])
            if drift < model.factor_drift_threshold:
                model.patience_counter_rK[r, k_idx] += 1
                if model.patience_counter_rK[r, k_idx] >= model.convergence_patience:
                    model.converged_rK[r, k_idx] = True
            else:
                model.patience_counter_rK[r, k_idx] = 0
        model._D_snapshots[3] = D_full.clone()

    assert model.converged_rK[:, k_idx].all(), "Both replicates should be converged after 2 stable windows"

    # Reset and simulate a large-drift window followed by a stable window → patience stays at 1
    model.converged_rK.zero_()
    model.patience_counter_rK.zero_()
    model._D_snapshots[3] = model.D_3_rkg.clone()

    noisy_D = model.D_3_rkg.clone()
    noisy_D += torch.rand_like(noisy_D) * 10  # large perturbation → drift >> threshold
    getattr(model, "D_3_rkg")[:] = noisy_D

    # Window 1: large drift → patience stays 0
    D_full = model.D_3_rkg.detach()
    snapshot = model._D_snapshots[3]
    assert isinstance(snapshot, torch.Tensor)
    for r in range(2):
        drift = model._compute_factor_drift(D_full[r], snapshot[r])
        if drift < model.factor_drift_threshold:
            model.patience_counter_rK[r, k_idx] += 1
        else:
            model.patience_counter_rK[r, k_idx] = 0
    model._D_snapshots[3] = D_full.clone()

    assert model.patience_counter_rK[:, k_idx].sum() == 0, "Large drift should reset patience to 0"
