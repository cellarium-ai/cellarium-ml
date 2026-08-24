# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models.ols import StreamingOrdinaryLeastSquares
from cellarium.ml.utilities.data import collate_fn


def _make_data(n: int, g: int, k: int, seed: int = 0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    x = torch.randn(n, g, generator=rng)
    y = torch.randn(n, k, generator=rng)
    var_names = np.array([f"gene_{i}" for i in range(g)])
    return x, y, var_names


def _reference_solve(x: torch.Tensor, y: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
    XtX = x.T @ x
    if ridge > 0.0:
        XtX = XtX + ridge * torch.eye(XtX.size(0), dtype=XtX.dtype)
    return torch.linalg.solve(XtX, x.T @ y)


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_streaming_matches_direct(batch_size: int):
    """Streaming accumulation over minibatches produces the same solution as a direct solve."""
    n, g, k = 100, 8, 3
    x, y, var_names = _make_data(n, g, k)

    model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, ridge_penalty=0.0)
    for start in range(0, n, batch_size):
        model.update(x[start : start + batch_size], y[start : start + batch_size])

    W_streaming = model.solve()
    W_reference = _reference_solve(x, y, ridge=0.0)

    torch.testing.assert_close(W_streaming, W_reference)


@pytest.mark.parametrize("ridge", [0.0, 1e-4, 1.0])
def test_ridge_penalty(ridge: float):
    """Ridge penalty is applied correctly and matches the reference formula (X^T X + λI)^{-1} X^T Y."""
    n, g, k = 200, 10, 4
    x, y, var_names = _make_data(n, g, k, seed=1)

    model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, ridge_penalty=ridge)
    model.update(x, y)

    W_streaming = model.solve()
    W_reference = _reference_solve(x, y, ridge=ridge)

    torch.testing.assert_close(W_streaming, W_reference)


class _OLSDataset(torch.utils.data.Dataset):
    """Dataset that yields (x_ng, var_names_g, y_nk) dicts for OLS training."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, var_names: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.var_names = var_names

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x_ng": self.x[idx, None].numpy(),
            "var_names_g": self.var_names,
            "y_nk": self.y[idx, None].numpy(),
        }


def test_lightning_integration(tmp_path):
    """
    End-to-end test: CellariumModule + Trainer runs one epoch, stops automatically,
    and the solved W_gk matches the reference direct solve.
    """
    n, g, k = 50, 6, 2
    x, y, var_names = _make_data(n, g, k, seed=42)

    dataset = _OLSDataset(x, y, var_names)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

    model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, ridge_penalty=0.0)
    module = CellariumModule(model=model)

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,  # intentionally set > 1; should_stop must cut it to 1
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    assert trainer.current_epoch == 1, "Training should have stopped after the first epoch"

    W_solved = module.model.W_gk
    W_reference = _reference_solve(x, y, ridge=0.0)
    torch.testing.assert_close(W_solved, W_reference)
