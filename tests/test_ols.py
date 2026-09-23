# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

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


def _reference_univariate_solve(x: torch.Tensor, y: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
    Xsq_g = (x**2).sum(dim=0)  # (g,)
    return (x.T @ y) / (Xsq_g + ridge).unsqueeze(1)


def _reference_intercept_solve(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least squares with an intercept, via an explicit column of ones (the naive construction)."""
    x_aug = torch.cat([torch.ones(x.shape[0], 1, dtype=x.dtype), x], dim=1)
    coef = torch.linalg.lstsq(x_aug, y).solution
    return coef[1:], coef[0]  # slopes, intercept


def _reference_univariate_intercept_solve(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """n_features x n_targets independent simple linear regressions, each with an intercept."""
    g, k = x.shape[1], y.shape[1]
    slopes = torch.empty(g, k, dtype=x.dtype)
    intercepts = torch.empty(g, k, dtype=x.dtype)
    for i in range(g):
        for j in range(k):
            s, b = _reference_intercept_solve(x[:, i : i + 1], y[:, j : j + 1])
            slopes[i, j], intercepts[i, j] = s.squeeze(), b.squeeze()
    return slopes, intercepts


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


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_univariate_streaming_matches_direct(batch_size: int):
    """Univariate streaming accumulation produces the same solution as the direct per-feature formula."""
    n, g, k = 100, 8, 3
    x, y, var_names = _make_data(n, g, k)

    model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, univariate=True, ridge_penalty=0.0)
    for start in range(0, n, batch_size):
        model.update(x[start : start + batch_size], y[start : start + batch_size])

    W_streaming = model.solve()
    W_reference = _reference_univariate_solve(x, y, ridge=0.0)

    torch.testing.assert_close(W_streaming, W_reference)


@pytest.mark.parametrize("ridge", [0.0, 1e-4, 1.0])
def test_univariate_ridge_penalty(ridge: float):
    """Univariate ridge penalty is applied correctly: W_gk = XtY_gk / (Xsq_g + ridge)."""
    n, g, k = 200, 10, 4
    x, y, var_names = _make_data(n, g, k, seed=2)

    model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, univariate=True, ridge_penalty=ridge)
    model.update(x, y)

    W_streaming = model.solve()
    W_reference = _reference_univariate_solve(x, y, ridge=ridge)

    torch.testing.assert_close(W_streaming, W_reference)


def test_univariate_differs_from_multivariate():
    """Univariate and multivariate solutions differ when features are correlated."""
    n, g, k = 200, 5, 2
    rng = torch.Generator()
    rng.manual_seed(99)
    # Introduce correlation by making features linear combinations of a smaller basis
    basis = torch.randn(n, 2, generator=rng)
    x = basis @ torch.randn(2, g, generator=rng)
    y = torch.randn(n, k, generator=rng)
    var_names = np.array([f"gene_{i}" for i in range(g)])

    mv_model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, univariate=False, ridge_penalty=1e-3)
    uv_model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, univariate=True, ridge_penalty=1e-3)
    mv_model.update(x, y)
    uv_model.update(x, y)

    W_mv = mv_model.solve()
    W_uv = uv_model.solve()

    assert not torch.allclose(W_mv, W_uv), "Multivariate and univariate solutions should differ for correlated features"


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_intercept_matches_reference(batch_size: int):
    """With fit_intercept=True the slopes and intercept match an explicit ones-column fit."""
    n, g, k = 100, 8, 3
    x, y, var_names = _make_data(n, g, k, seed=3)
    x = x + 5.0  # strongly off-center: the case a no-intercept fit gets wrong

    model = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, ridge_penalty=0.0, fit_intercept=True
    )
    for start in range(0, n, batch_size):
        model.update(x[start : start + batch_size], y[start : start + batch_size])

    W, b = model.solve(return_intercept=True)
    W_reference, b_reference = _reference_intercept_solve(x, y)

    torch.testing.assert_close(W, W_reference, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b, b_reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_univariate_intercept_matches_reference(batch_size: int):
    """Univariate mode fits one intercept per feature-target pair."""
    n, g, k = 100, 5, 3
    x, y, var_names = _make_data(n, g, k, seed=4)
    x = x + 5.0

    model = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, univariate=True, ridge_penalty=0.0, fit_intercept=True
    )
    for start in range(0, n, batch_size):
        model.update(x[start : start + batch_size], y[start : start + batch_size])

    W, b = model.solve(return_intercept=True)
    W_reference, b_reference = _reference_univariate_intercept_solve(x, y)

    assert W.shape == (g, k) and b.shape == (g, k)
    torch.testing.assert_close(W, W_reference, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b, b_reference, rtol=1e-4, atol=1e-4)


def test_intercept_equivalent_to_pre_centering():
    """Fitting an intercept gives the same slopes as centering the features by hand."""
    n, g, k = 200, 6, 2
    x, y, var_names = _make_data(n, g, k, seed=5)
    x = x * 3.0 + 40.0

    with_intercept = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, ridge_penalty=0.0, fit_intercept=True
    )
    with_intercept.update(x, y)

    pre_centered = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, ridge_penalty=0.0)
    pre_centered.update(x - x.mean(dim=0), y)

    torch.testing.assert_close(with_intercept.solve(), pre_centered.solve(), rtol=1e-4, atol=1e-4)


def test_no_intercept_on_positive_data_forces_one_sign():
    """Regression through the origin on a positive feature cannot produce negative coefficients.

    This is the silent failure fit_intercept exists to prevent: a strictly positive feature
    against non-negative targets forces every coefficient positive no matter what the data
    does, and the true (intercept-ful) slopes can even point the other way.
    """
    n = 2000
    rng = torch.Generator()
    rng.manual_seed(7)
    age = 30.0 + 60.0 * torch.rand(n, 1, generator=rng)  # strictly positive covariate
    # Two genes with genuinely negative age slopes, on a large positive baseline.
    y = torch.stack([100.0 - 0.5 * age[:, 0], 80.0 - 0.2 * age[:, 0]], dim=1)
    y = y + 0.01 * torch.randn(n, 2, generator=rng)
    var_names = np.array(["age"])

    without = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=2, ridge_penalty=0.0)
    without.update(age, y)
    with pytest.warns(UserWarning, match="fit_intercept=False"):
        W_without = without.solve()

    with_ = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=2, ridge_penalty=0.0, fit_intercept=True
    )
    with_.update(age, y)
    W_with = with_.solve()

    assert (W_without > 0).all(), "no-intercept fit on positive data should force positive coefficients"
    assert (W_with < 0).all(), "with an intercept the true negative slopes are recovered"
    torch.testing.assert_close(W_with.squeeze(), torch.tensor([-0.5, -0.2]), rtol=1e-3, atol=1e-3)


def test_off_center_warning_only_when_warranted():
    """The warning fires for off-center features, and not for centered ones or a ones column."""
    n, g, k = 200, 4, 2
    x, y, var_names = _make_data(n, g, k, seed=8)

    # Centered features: no warning.
    centered = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k)
    centered.update(x - x.mean(dim=0), y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centered.solve()

    # A constant column is a hand-rolled intercept, not a mistake: no warning.
    ones_names = np.array(["intercept"])
    ones = StreamingOrdinaryLeastSquares(var_names_g=ones_names, n_targets=k)
    ones.update(torch.ones(n, 1), y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ones.solve()

    # fit_intercept=True never warns, however off-center the data.
    fitted = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, fit_intercept=True)
    fitted.update(x + 50.0, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted.solve()


def test_intercept_does_not_change_default_behaviour():
    """fit_intercept defaults to False, preserving the original through-the-origin solution."""
    n, g, k = 100, 5, 2
    x, y, var_names = _make_data(n, g, k, seed=9)

    model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, ridge_penalty=0.0)
    assert model.fit_intercept is False
    model.update(x, y)
    torch.testing.assert_close(model.solve(), _reference_solve(x, y, ridge=0.0), rtol=1e-5, atol=1e-5)


def test_ridge_does_not_penalize_intercept():
    """The ridge penalty shrinks slopes but leaves the intercept free to absorb the mean."""
    n, k = 500, 1
    rng = torch.Generator()
    rng.manual_seed(11)
    x = torch.randn(n, 1, generator=rng)
    y = 100.0 + 0.0 * x  # large offset, zero slope
    var_names = np.array(["feature"])

    model = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, ridge_penalty=1e3, fit_intercept=True
    )
    model.update(x, y)
    W, b = model.solve(return_intercept=True)

    torch.testing.assert_close(b, torch.tensor([100.0]), rtol=1e-4, atol=1e-3)
    assert abs(W.item()) < 1e-3


def test_predict_includes_intercept():
    """predict() adds the fitted intercept in multivariate mode."""
    n, g, k = 200, 3, 2
    x, y, var_names = _make_data(n, g, k, seed=12)
    x = x + 10.0

    model = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, ridge_penalty=0.0, fit_intercept=True
    )
    model.update(x, y)
    W, b = model.solve(return_intercept=True)
    model.W_gk.copy_(W)
    model.intercept_k.copy_(b)

    y_hat = model.predict(x, var_names)["y_hat_nk"]
    torch.testing.assert_close(y_hat, x @ W + b, rtol=1e-5, atol=1e-5)


def test_predict_rejects_univariate_intercept():
    """Univariate + fit_intercept has no single joint prediction, and says so."""
    n, g, k = 50, 3, 2
    x, y, var_names = _make_data(n, g, k, seed=13)

    model = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, univariate=True, fit_intercept=True
    )
    model.update(x, y)
    with pytest.raises(NotImplementedError, match="univariate"):
        model.predict(x, var_names)


def test_reset_parameters_clears_centering_statistics():
    """reset_parameters zeroes the new accumulators, so a reused model does not double count."""
    n, g, k = 60, 4, 2
    x, y, var_names = _make_data(n, g, k, seed=14)

    model = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, ridge_penalty=0.0, fit_intercept=True
    )
    model.update(x + 7.0, y)
    model.reset_parameters()
    assert model.n_obs.item() == 0
    torch.testing.assert_close(model.Xsum_g, torch.zeros(g))
    torch.testing.assert_close(model.Ysum_k, torch.zeros(k))

    model.update(x + 7.0, y)
    W_reference, _ = _reference_intercept_solve(x + 7.0, y)
    torch.testing.assert_close(model.solve(), W_reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("univariate", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_buffers_survive_accelerator_move(univariate: bool, fit_intercept: bool):
    """Every buffer must be movable to the training device.

    MPS has no float64, so a float64 accumulator raises on .to(device) the moment the
    model is used with accelerator="auto" on Apple silicon -- a failure CPU-only tests
    never see.
    """
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    n, g, k = 40, 4, 2
    x, y, var_names = _make_data(n, g, k, seed=16)
    x = x + 6.0

    for device in devices:
        model = StreamingOrdinaryLeastSquares(
            var_names_g=var_names,
            n_targets=k,
            univariate=univariate,
            ridge_penalty=0.0,
            fit_intercept=fit_intercept,
        ).to(device)
        model.update(x.to(device), y.to(device))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = model.solve()
        assert W.shape == (g, k)
        assert W.device.type == torch.device(device).type
        assert torch.isfinite(W).all(), f"non-finite solution on {device}"


def test_fit_intercept_requires_two_observations():
    n, g, k = 1, 3, 2
    x, y, var_names = _make_data(n, g, k, seed=15)
    model = StreamingOrdinaryLeastSquares(var_names_g=var_names, n_targets=k, fit_intercept=True)
    model.update(x, y)
    with pytest.raises(ValueError, match="at least 2 observations"):
        model.solve()


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


def test_lightning_integration_with_intercept(tmp_path):
    """End-to-end with fit_intercept=True: both W_gk and intercept_k are populated."""
    n, g, k = 50, 6, 2
    x, y, var_names = _make_data(n, g, k, seed=43)
    x = x + 20.0

    dataset = _OLSDataset(x, y, var_names)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

    model = StreamingOrdinaryLeastSquares(
        var_names_g=var_names, n_targets=k, ridge_penalty=0.0, fit_intercept=True
    )
    module = CellariumModule(model=model)

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    assert trainer.current_epoch == 1
    assert module.model.n_obs.item() == n, "centering statistics must see every row"

    W_reference, b_reference = _reference_intercept_solve(x, y)
    torch.testing.assert_close(module.model.W_gk, W_reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(module.model.intercept_k, b_reference, rtol=1e-3, atol=1e-3)
