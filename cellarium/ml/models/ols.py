# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class StreamingOrdinaryLeastSquares(CellariumModel, PredictMixin):
    """
    Streaming ordinary least squares (OLS) solver.

    Accumulates X^T X and X^T Y over minibatches, then solves the normal equations
    once at the end of the first epoch. Training is stopped after one pass.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        n_targets:
            Number of target columns (k in y_nk).
        ridge_penalty:
            L2 penalty added to the diagonal of X^T X before solving.
            Recommended for numerical stability when some features have zero variance.
    """

    def __init__(self, var_names_g: np.ndarray, n_targets: int, ridge_penalty: float = 1e-6) -> None:
        super().__init__()

        self.var_names_g = var_names_g
        n_features = len(var_names_g)
        self.ridge_penalty = ridge_penalty

        self.XtX_gg: torch.Tensor
        self.XtY_gk: torch.Tensor
        self.W_gk: torch.Tensor
        self.register_buffer("XtX_gg", torch.zeros(n_features, n_features))
        self.register_buffer("XtY_gk", torch.zeros(n_features, n_targets))
        self.register_buffer("W_gk", torch.zeros(n_features, n_targets))

        self.reset_parameters()

    def forward(
        self, x_ng: torch.Tensor, var_names_g: np.ndarray, y_nk: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        """
        Accumulate sufficient statistics for a minibatch.

        Args:
            x_ng:
                Feature matrix of shape (batch_size, n_features).
            var_names_g:
                The variable names for the input data.
            y_nk:
                Target matrix of shape (batch_size, n_targets).

        Returns:
            An empty dictionary (no loss).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        self.update(x_ng, y_nk)
        return {}

    @torch.no_grad()
    def update(self, x_ng: torch.Tensor, y_nk: torch.Tensor) -> None:
        """
        Update OLS accumulators with a minibatch.

        Args:
            x_ng: Tensor of shape (batch_size, n_features).
            y_nk: Tensor of shape (batch_size, n_targets).
        """
        self.XtX_gg += x_ng.T @ x_ng
        self.XtY_gk += x_ng.T @ y_nk

    @torch.no_grad()
    def solve(self, ridge_penalty: float | None = None) -> torch.Tensor:
        """
        Solve the normal equations for the accumulated data.

        Args:
            ridge_penalty:
                L2 penalty on the diagonal of X^T X. Defaults to ``self.ridge_penalty``.

        Returns:
            Coefficient matrix of shape (n_features, n_targets).
        """
        penalty = self.ridge_penalty if ridge_penalty is None else ridge_penalty
        XtX = self.XtX_gg
        if penalty > 0.0:
            identity = torch.eye(XtX.size(0), device=XtX.device, dtype=XtX.dtype)
            XtX = XtX + penalty * identity
        return torch.linalg.solve(XtX, self.XtY_gk)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        """
        Solve the normal equations at the end of the first (and only) epoch.

        In multi-GPU training the accumulators are all-reduced before solving so
        the solution uses the full dataset rather than a single shard.
        """
        if trainer.world_size > 1:
            dist.all_reduce(self.XtX_gg, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.XtY_gk, op=dist.ReduceOp.SUM)

        self.W_gk.copy_(self.solve())
        trainer.should_stop = True

    @torch.no_grad()
    def predict(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Apply the solved coefficients to new data.

        Args:
            x_ng:
                Feature matrix of shape (batch_size, n_features).
            var_names_g:
                The variable names for the input data.

        Returns:
            A dictionary with ``y_hat_nk`` of shape (batch_size, n_targets).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        return {"y_hat_nk": x_ng @ self.W_gk}

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.XtX_gg.zero_()
        self.XtY_gk.zero_()
        self.W_gk.zero_()
