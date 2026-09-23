# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist

from cellarium.ml.models.model import CellariumModel, PredictMixin, TransformPrediction
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

# When ``fit_intercept=False``, warn if a feature's mean is at least this many standard
# deviations from zero: a through-the-origin fit on such a feature is rarely intended and
# fails silently (every coefficient can be forced to share the sign of the feature).
OFF_CENTER_WARN_RATIO = 2.0


def _solve_dtype(device: torch.device) -> torch.dtype:
    """Widest float the device supports, for the centering subtraction.

    That subtraction is where significance is lost, so it is worth doing in float64 --
    but MPS has no float64 at all, so there it stays in float32.
    """
    return torch.float32 if device.type == "mps" else torch.float64


class StreamingOrdinaryLeastSquares(CellariumModel, PredictMixin):
    """
    Streaming ordinary least squares (OLS) solver.

    Accumulates sufficient statistics over minibatches, then solves the normal equations
    once at the end of the first epoch. Training is stopped after one pass.

    Two modes are supported:

    * **Multivariate** (``univariate=False``, default): fits a single joint regression
      ``X @ W = Y`` where all features compete simultaneously. Requires accumulating
      ``X^T X`` of shape ``(n_features, n_features)`` — only tractable when ``n_features``
      is small (e.g. a gene expression matrix filtered to a relevant gene set, or a
      low-dimensional embedding from ``obsm``).

    * **Univariate** (``univariate=True``): fits ``n_features × n_targets`` independent
      simple linear regressions, one per feature–target pair. Accumulates only the sum of
      squared feature values (shape ``(n_features,)``), making it tractable for
      high-dimensional ``X`` such as a raw genotype matrix with millions of variants.
      The ridge penalty is added per-feature to the scalar denominator rather than to
      a matrix diagonal; it is otherwise equivalent in intent.

    An intercept is **not** fitted by default, which means the fit is forced through the
    origin.  That is rarely what is wanted and fails silently: with a strictly positive
    feature and non-negative targets, for instance, every coefficient is forced positive
    regardless of the data.  Pass ``fit_intercept=True`` unless a through-the-origin fit is
    genuinely intended; leaving it off emits a warning when any feature is strongly
    off-center.

    The intercept is fitted via the centered normal equations rather than by appending a
    column of ones, so ``solve()`` keeps returning slopes of shape
    ``(n_features, n_targets)``, the ``var_names_g`` schema is unchanged, and the ridge
    penalty applies to the slopes only (an intercept should never be penalized).

    .. note::
        The centering correction subtracts ``n * mean**2`` from the accumulated sum of
        squares.  When a feature's mean is very large relative to its spread (e.g. a year
        covariate, 2024 +/- 2) this is a catastrophic cancellation, and the precision lost
        while accumulating in float32 cannot be recovered at solve time.  Centering such a
        feature before it reaches this model, or accumulating in float64, avoids the issue.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        n_targets:
            Number of target columns (k in y_nk).
        univariate:
            If ``True``, run massively parallel univariate regressions instead of a single
            joint multivariate regression.
        ridge_penalty:
            L2 penalty added to the diagonal of X^T X (multivariate) or to each feature's
            sum of squares (univariate) before solving. Recommended for numerical stability.
            Never applied to the intercept.
        fit_intercept:
            If ``True``, fit an intercept per target (multivariate) or per feature-target
            pair (univariate), and expose it as ``intercept_k`` / ``intercept_gk``.
            Defaults to ``False`` for backwards compatibility.
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        n_targets: int,
        univariate: bool = False,
        ridge_penalty: float = 1e-6,
        fit_intercept: bool = False,
    ) -> None:
        super().__init__()

        self.var_names_g = var_names_g
        n_features = len(var_names_g)
        self.univariate = univariate
        self.ridge_penalty = ridge_penalty
        self.fit_intercept = fit_intercept

        if univariate:
            self.Xsq_g: torch.Tensor
            self.register_buffer("Xsq_g", torch.zeros(n_features))
        else:
            self.XtX_gg: torch.Tensor
            self.register_buffer("XtX_gg", torch.zeros(n_features, n_features))

        self.XtY_gk: torch.Tensor
        self.W_gk: torch.Tensor
        self.register_buffer("XtY_gk", torch.zeros(n_features, n_targets))
        self.register_buffer("W_gk", torch.zeros(n_features, n_targets))

        # Centering statistics.  Accumulated unconditionally: they are small next to XtY_gk
        # even in univariate mode, and the off-center warning needs them when
        # fit_intercept=False.  n_obs is int64 so the count stays exact past 2**24 rows;
        # the sums match the other accumulators' dtype because MPS has no float64.
        self.n_obs: torch.Tensor
        self.Xsum_g: torch.Tensor
        self.Ysum_k: torch.Tensor
        self.register_buffer("n_obs", torch.zeros((), dtype=torch.int64))
        self.register_buffer("Xsum_g", torch.zeros(n_features))
        self.register_buffer("Ysum_k", torch.zeros(n_targets))

        if fit_intercept:
            # Multivariate fits one joint regression per target; univariate fits
            # n_features * n_targets independent simple regressions, each with its own.
            if univariate:
                self.intercept_gk: torch.Tensor
                self.register_buffer("intercept_gk", torch.zeros(n_features, n_targets))
            else:
                self.intercept_k: torch.Tensor
                self.register_buffer("intercept_k", torch.zeros(n_targets))

        # DDP requires at least one parameter with requires_grad=True even when no
        # optimizer is used; this scalar satisfies that constraint without affecting results.
        self._dummy_param = torch.nn.Parameter(torch.empty(()))

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
        if self.univariate:
            self.Xsq_g += (x_ng**2).sum(dim=0)
        else:
            self.XtX_gg += x_ng.T @ x_ng
        self.XtY_gk += x_ng.T @ y_nk

        self.n_obs += x_ng.shape[0]
        self.Xsum_g += x_ng.sum(dim=0)
        self.Ysum_k += y_nk.sum(dim=0)

    @torch.no_grad()
    def _warn_if_off_center(self) -> None:
        """Warn when a through-the-origin fit is being run on a feature far from zero."""
        n = int(self.n_obs.item())
        if n <= 1:
            return
        dtype = _solve_dtype(self.Xsum_g.device)
        mean_g = self.Xsum_g.to(dtype) / n
        second_moment_g = (self.Xsq_g if self.univariate else torch.diagonal(self.XtX_gg)).to(dtype) / n
        sd_g = (second_moment_g - mean_g**2).clamp_min(0.0).sqrt()
        # A constant column is an intercept the caller added by hand -- not a mistake.
        informative = sd_g > 1e-12
        if not informative.any():
            return
        idx = torch.nonzero(informative, as_tuple=False).squeeze(1)
        ratio = mean_g[idx].abs() / sd_g[idx]
        if bool((ratio > OFF_CENTER_WARN_RATIO).any()):
            worst = int(idx[int(ratio.argmax())].item())
            warnings.warn(
                f"fit_intercept=False but feature {self.var_names_g[worst]!r} has mean "
                f"{mean_g[worst].item():.4g}, which is {ratio.max().item():.1f} standard deviations "
                "from zero. Regression through the origin on an off-center feature is rarely "
                "intended and fails silently -- e.g. a strictly positive feature against "
                "non-negative targets forces every coefficient positive. Pass fit_intercept=True, "
                "or center the feature, unless a through-the-origin fit is what you want.",
                UserWarning,
                stacklevel=3,
            )

    @torch.no_grad()
    def solve(
        self, ridge_penalty: float | None = None, return_intercept: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the normal equations for the accumulated data.

        When :attr:`fit_intercept` is ``True`` this solves the *centered* normal equations,
        which yields exactly the slopes of a fit with an intercept while leaving the
        returned shape unchanged.

        Args:
            ridge_penalty:
                L2 penalty. In multivariate mode it is added to the diagonal of X^T X;
                in univariate mode it is added to each feature's sum of squares.
                Defaults to ``self.ridge_penalty``. Never applied to the intercept.
            return_intercept:
                If ``True``, also return the fitted intercept: shape ``(n_targets,)`` in
                multivariate mode, ``(n_features, n_targets)`` in univariate mode. Zeros
                when :attr:`fit_intercept` is ``False``.

        Returns:
            Coefficient matrix of shape (n_features, n_targets), or a
            ``(coefficients, intercept)`` tuple when ``return_intercept`` is ``True``.
        """
        penalty = self.ridge_penalty if ridge_penalty is None else ridge_penalty
        out_dtype = self.XtY_gk.dtype

        if not self.fit_intercept:
            self._warn_if_off_center()

        n = int(self.n_obs.item())
        if self.fit_intercept and n <= 1:
            raise ValueError(f"fit_intercept=True requires at least 2 observations; accumulated {n}.")

        # The centering subtraction is widened past the accumulator dtype where the device
        # allows it: that subtraction is the step where significance is lost.
        dtype = _solve_dtype(self.XtY_gk.device)
        XtY = self.XtY_gk.to(dtype)
        x_mean_g = self.Xsum_g.to(dtype) / max(n, 1)
        y_mean_k = self.Ysum_k.to(dtype) / max(n, 1)
        if self.fit_intercept:
            XtY = XtY - n * torch.outer(x_mean_g, y_mean_k)

        if self.univariate:
            Xsq_g = self.Xsq_g.to(dtype)
            if self.fit_intercept:
                Xsq_g = Xsq_g - n * x_mean_g**2
            W_gk = XtY / (Xsq_g + penalty).unsqueeze(1)
            intercept = (
                y_mean_k.unsqueeze(0) - W_gk * x_mean_g.unsqueeze(1) if self.fit_intercept else torch.zeros_like(W_gk)
            )
        else:
            XtX = self.XtX_gg.to(dtype)
            if self.fit_intercept:
                XtX = XtX - n * torch.outer(x_mean_g, x_mean_g)
            if penalty > 0.0:
                XtX = XtX + penalty * torch.eye(XtX.size(0), device=XtX.device, dtype=XtX.dtype)
            W_gk = torch.linalg.solve(XtX, XtY)
            intercept = y_mean_k - x_mean_g @ W_gk if self.fit_intercept else torch.zeros_like(y_mean_k)

        W_gk = W_gk.to(out_dtype)
        if return_intercept:
            return W_gk, intercept.to(out_dtype)
        return W_gk

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        """
        Solve the normal equations at the end of the first (and only) epoch.

        In multi-GPU training the accumulators are all-reduced before solving so
        the solution uses the full dataset rather than a single shard.
        """
        if trainer.world_size > 1:
            if self.univariate:
                dist.all_reduce(self.Xsq_g, op=dist.ReduceOp.SUM)
            else:
                dist.all_reduce(self.XtX_gg, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.XtY_gk, op=dist.ReduceOp.SUM)
            # The centering statistics are sufficient statistics too: without these the
            # intercept (and the off-center warning) would see only this rank's shard.
            dist.all_reduce(self.n_obs, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.Xsum_g, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.Ysum_k, op=dist.ReduceOp.SUM)

        W_gk, intercept = self.solve(return_intercept=True)
        self.W_gk.copy_(W_gk)
        if self.fit_intercept:
            (self.intercept_gk if self.univariate else self.intercept_k).copy_(intercept)
        trainer.should_stop = True

    @torch.no_grad()
    def predict(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> TransformPrediction:
        """
        Apply the solved coefficients to new data.

        Args:
            x_ng:
                Feature matrix of shape (batch_size, n_features).
            var_names_g:
                The variable names for the input data.

        Returns:
            A dictionary with ``x_ng`` (misnomer) of shape (batch_size, n_targets).

        Raises:
            NotImplementedError: in univariate mode with ``fit_intercept=True``, where each
                feature-target pair has its own intercept and there is no single joint
                prediction to make.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        y_hat_nk = x_ng @ self.W_gk
        if self.fit_intercept:
            if self.univariate:
                raise NotImplementedError(
                    "predict() is not defined for univariate mode with fit_intercept=True: each "
                    "feature-target pair is a separate regression with its own intercept, so there "
                    "is no single joint prediction. Use W_gk and intercept_gk directly."
                )
            y_hat_nk = y_hat_nk + self.intercept_k

        return {"x_ng": y_hat_nk, "var_names_g": np.array([f"ols_{i}" for i in range(self.W_gk.shape[0])])}

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.univariate:
            self.Xsq_g.zero_()
        else:
            self.XtX_gg.zero_()
        self.XtY_gk.zero_()
        self.W_gk.zero_()
        self.n_obs.zero_()
        self.Xsum_g.zero_()
        self.Ysum_k.zero_()
        if self.fit_intercept:
            (self.intercept_gk if self.univariate else self.intercept_k).zero_()
        self._dummy_param.data.zero_()
