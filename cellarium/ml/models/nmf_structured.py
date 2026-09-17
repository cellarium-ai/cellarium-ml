# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

from cellarium.ml.models.nmf import (
    compute_reconstruction_error_compiled,
    frobenius_loss_trace_compiled,
    nmf_compute_factors_fista,
)
from cellarium.ml.models.nmf_amortized import (
    AmortizedOnlineNonNegativeMatrixFactorization,
    BilinearLoadingsEncoder,
)
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


@torch.compile()
@torch.no_grad()
def solve_structure_aware_nnls_fista(
    AtA: torch.Tensor,
    AtB: torch.Tensor,
    initial_x: torch.Tensor,
    M_c_nd: torch.Tensor,
    mu_H_rk: torch.Tensor,
    lambda_align: float,
    n_metadata_programs: int = 0,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    FISTA solver for structure-aware NMF with a covariance decorrelation penalty on
    non-nominated programs only.

    Minimizes::

        (1/2) ||X - H W||_F^2
        + lambda_align * ||M_c^T (H_free - mu_H_free)||_F^2

    subject to H >= 0.

    ``H_free = H[:, n_metadata_programs:]`` receives the alignment penalty, making it
    expensive for those programs to absorb metadata variance.  The first
    ``n_metadata_programs`` columns are exempt — they are the "tax-free haven" where FISTA
    naturally routes metadata-correlated variance.

    Args:
        AtA: (R, K, K) — W W^T, precomputed.
        AtB: (R, K, N) — W X^T, precomputed (full X, not a residual).
        initial_x: (R, K, N) — warm-start H^T.
        M_c_nd: (N, D) — batch-centered scaled metadata, NaN rows zeroed.
        mu_H_rk: (R, K) — bias-corrected EMA mean of H_total, for centering.
        lambda_align: Covariance penalty strength on non-nominated programs.
        n_metadata_programs: Number of nominated programs exempt from the penalty.
        max_iter: Number of FISTA iterations.

    Returns:
        x: (R, K, N) — solved H_total^T, non-negative.
    """
    # --- Lipschitz constant: reconstruction term via power iteration ---
    v = torch.ones(*AtA.shape[:-1], 1, device=AtA.device, dtype=AtA.dtype)
    for _ in range(10):
        v = AtA @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-8)
    L_A = (v.transpose(-2, -1) @ AtA @ v).clamp(min=1e-12)  # (R, 1, 1)

    # Lipschitz constant: penalty term (only applies to free programs)
    L_M = (2.0 * lambda_align) * (M_c_nd**2).sum()  # scalar

    # Per-program step sizes: nominated programs use L_A, free programs use L_A + L_M.
    K = AtA.shape[1]
    L = L_A.expand(-1, K, 1).clone()  # (R, K, 1)
    if lambda_align > 0 and n_metadata_programs < K:
        L[:, n_metadata_programs:, :] += L_M

    mu_H_rk1 = mu_H_rk.unsqueeze(-1)  # (R, K, 1)

    x = initial_x.clone()
    y = initial_x.clone()
    t = 1.0

    for _ in range(max_iter):
        grad = AtA @ y - AtB

        if lambda_align > 0 and n_metadata_programs < grad.shape[1]:
            # Penalty applied only to free (non-nominated) programs.
            # Right-to-left order avoids forming the N×N matrix M_c M_c^T.
            y_free = y[:, n_metadata_programs:, :]  # (R, K_free, N)
            mu_free = mu_H_rk1[:, n_metadata_programs:, :]  # (R, K_free, 1)
            tmp_rkd = (y_free - mu_free) @ M_c_nd  # (R, K_free, D)
            grad[:, n_metadata_programs:, :] += (2.0 * lambda_align) * (tmp_rkd @ M_c_nd.mT)

        x_new = torch.clamp(y - grad / L, min=0.0)

        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t**2)) / 2.0
        momentum = (t - 1.0) / t_new
        y = x_new + momentum * (x_new - x)
        x = x_new
        t = t_new

    return x


@torch.no_grad()
def compute_metadata_nmf_init(
    ols_coeff_dg: torch.Tensor,
    n_metadata_programs: int,
    n_replicates: int,
    range_M_d: torch.Tensor,
    noise_scale: float = 0.05,
) -> torch.Tensor:
    """
    Derives NMF W initialization for nominated metadata-capturing factors from OLS coefficients.

    Runs truncated SVD on the range-scaled OLS coefficient matrix to extract dominant modes
    of metadata-gene co-variation in min-max scaled space. Applies a sign convention so the
    primary gene direction is positive, then ReLU-projects to non-negative and L1-normalizes.

    Args:
        ols_coeff_dg: OLS coefficient matrix of shape (D, G), in raw metadata units.
        n_metadata_programs: Number of metadata factors to initialize.
        n_replicates: Number of NMF replicates.
        range_M_d: Global metadata range (max - min) of shape (D,).
        noise_scale: Relative noise std for replicate diversity (default 0.05).

    Returns:
        W_meta_rkg: shape (R, n_metadata_programs, G) — nominated gene factor vectors,
            L1-normalized by row.
    """
    D, G = ols_coeff_dg.shape
    n_singular = min(n_metadata_programs, D)

    # Scale OLS rows by range_M_d to get directions in min-max scaled space
    ols_coeff_dg_scaled = range_M_d.unsqueeze(1) * ols_coeff_dg  # (D, G)
    U, S, Vh = torch.linalg.svd(ols_coeff_dg_scaled, full_matrices=False)
    U = U[:, :n_singular]
    S = S[:n_singular]
    Vh = Vh[:n_singular]

    # Sign convention: largest absolute value in each V row should be positive
    for j in range(n_singular):
        max_abs_idx = torch.argmax(torch.abs(Vh[j]))
        if Vh[j, max_abs_idx] < 0:
            Vh[j] = -Vh[j]
            U[:, j] = -U[:, j]

    W_base: list[torch.Tensor] = []
    for j in range(n_metadata_programs):
        if j < n_singular:
            w_j = F.relu(Vh[j]) + 1e-8  # (G,)
            w_j = F.normalize(w_j, p=1, dim=0)
        else:
            # More programs than SVD modes: random init for extras
            w_j = F.normalize(torch.rand(G, device=ols_coeff_dg.device) + 1e-8, p=1, dim=0)
        W_base.append(w_j)

    W_meta_rkg = torch.zeros(n_replicates, n_metadata_programs, G, device=ols_coeff_dg.device)
    for r in range(n_replicates):
        for j in range(n_metadata_programs):
            w_j = W_base[j]
            w_mean = w_j.mean().item()
            noise_w = torch.randn(G, device=ols_coeff_dg.device) * noise_scale * max(w_mean, 1e-8)
            w_r = F.relu(w_j + noise_w) + 1e-8
            W_meta_rkg[r, j] = F.normalize(w_r, p=1, dim=0)

    return W_meta_rkg


class MetadataAugmentedLoadingsEncoder(BilinearLoadingsEncoder):
    """
    Encoder that augments the cell embedding with a metadata side-channel before computing
    bilinear affinities with gene factors.

    The encoder is trained to predict H_total (the FISTA solver output). Seeing the metadata
    M directly allows the encoder to learn warm-starts that account for both idiosyncratic
    and metadata-driven variance.
    """

    def __init__(self, n_genes: int, latent_dim: int, n_metadata: int):
        super().__init__(n_genes=n_genes, latent_dim=latent_dim)
        self.metadata_encoder = torch.nn.Linear(n_metadata, latent_dim, bias=False)

    def forward(self, x_ng: torch.Tensor, w_rkg: torch.Tensor, m_nd: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts of shape (N, G).
            w_rkg: Current NMF factors of shape (R, K, G), L1-normalized by row.
            m_nd: Metadata matrix of shape (N, D), raw (uncentered). Required.

        Returns:
            h_rnk: Warm-start loadings of shape (R, N, K), non-negative,
                scaled so each cell's loadings sum approximately to its total count.
        """
        assert m_nd is not None, "m_nd is required for MetadataAugmentedLoadingsEncoder"
        c_ne = self.cell_encoder(torch.log1p(x_ng)) + self.metadata_encoder(m_nd)
        f_rke = self.factor_encoder(w_rkg)
        logits_rnk = torch.einsum("ne,rke->rnk", c_ne, f_rke) * self.scale
        h_sparse_rnk = F.relu(logits_rnk)
        h_norm_rnk = h_sparse_rnk / h_sparse_rnk.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return h_norm_rnk * x_ng.sum(dim=-1, keepdim=True).unsqueeze(0)


class AmortizedOnlineStructureAwareNMF(AmortizedOnlineNonNegativeMatrixFactorization):
    """
    Structure-aware extension of AmortizedOnlineNonNegativeMatrixFactorization.

    Uses a **solve-then-harvest** paradigm:

    - **Training**: FISTA solves directly for ``H_total >= 0`` on the full X.
      Nominated programs (first ``n_metadata_programs``) are exempt from the alignment
      penalty — they are the "tax-free haven" for metadata-correlated variance.
      Non-nominated programs pay a covariance penalty, so FISTA routes metadata variance
      to the cheaper nominated columns. Beta is never part of the training loop.

    - **Inference** (``predict``): Beta is harvested post-hoc via OLS from the nominated
      H_total columns: ``Beta = (M^T M + eps I)^{-1} M^T H_nominated``, clamped >= 0.

    **Metadata convention:**
    ``m_nd`` should contain **raw non-negative** metadata values. The model internally
    applies min-max scaling to ``[0, 1]``. Provide ``metadata_min_d`` and
    ``metadata_max_d`` computed from the training set.

    Args:
        var_names_g: Gene names.
        k_values: List of NMF component counts to train in parallel.
        r: Number of replicates.
        latent_dim: Encoder latent dimension.
        total_n_cells: Total dataset size (for EMA period computation).
        batch_size: Training batch size.
        n_metadata: Number of metadata columns (D).
        metadata_mean_d: Precomputed global metadata mean, shape (D,).
        metadata_min_d: Precomputed global metadata minimum, shape (D,).
        metadata_max_d: Precomputed global metadata maximum, shape (D,).
        n_metadata_programs: Number of nominated metadata factors (exempt from alignment
            penalty). Defaults to ``2 * n_metadata``. Must be < min(k_values).
        ols_coeff_dg: OLS coefficient matrix (D, G) for initializing W factors.
            If None, nominated factors use random initialization.
        metadata_noise_scale: Relative noise std for replicate diversity (default 0.05).
        lambda_align: Strength of the covariance decorrelation penalty on non-nominated
            programs in the FISTA solver. Any value > 0 creates a preference for metadata
            variance to route to nominated columns. Penalty is normalized by sqrt(n_valid)
            to be batch-size independent.
        solver: Inner solver for H_total. Only ``"fista"`` is supported.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        k_values: list[int],
        r: int,
        latent_dim: int,
        total_n_cells: int,
        batch_size: int,
        n_metadata: int,
        metadata_mean_d: list[float] | np.ndarray,
        metadata_min_d: list[float] | np.ndarray,
        metadata_max_d: list[float] | np.ndarray,
        n_metadata_programs: int | None = None,
        ols_coeff_dg: np.ndarray | None = None,
        metadata_noise_scale: float = 0.05,
        lambda_align: float = 0.1,
        solver: Literal["fista"] = "fista",
        encoder_improvement_threshold: float = 2e-3,
        reconstruction_improvement_threshold: float = 2e-3,
        trigger_patience: int = 3,
        cooldown_patience: int = 5,
        max_cooldown_epochs: float = 1.0,
        trailing_window_fraction: float = 0.25,
        exploration_epochs: int = 2,
        max_solver_iter_train: int = 50,
        max_solver_iter_cooldown: int = 200,
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
    ) -> None:
        if n_metadata_programs is None:
            n_metadata_programs = 2 * n_metadata
        for k in k_values:
            if n_metadata_programs >= k:
                raise ValueError(
                    f"n_metadata_programs ({n_metadata_programs}) must be strictly less than "
                    f"every k in k_values, but k={k} violates this. "
                    f"Reduce n_metadata_programs or increase k values."
                )

        super().__init__(
            var_names_g=var_names_g,
            k_values=k_values,
            r=r,
            latent_dim=latent_dim,
            total_n_cells=total_n_cells,
            batch_size=batch_size,
            solver=solver,
            encoder_improvement_threshold=encoder_improvement_threshold,
            reconstruction_improvement_threshold=reconstruction_improvement_threshold,
            trigger_patience=trigger_patience,
            cooldown_patience=cooldown_patience,
            max_cooldown_epochs=max_cooldown_epochs,
            trailing_window_fraction=trailing_window_fraction,
            exploration_epochs=exploration_epochs,
            max_solver_iter_train=max_solver_iter_train,
            max_solver_iter_cooldown=max_solver_iter_cooldown,
            init=init,
            transformed_data_mean=transformed_data_mean,
        )
        # super().__init__() sets self.encoder = BilinearLoadingsEncoder, registers
        # A/B/D buffers, and calls reset_parameters() once (early return because
        # subclass attrs not yet set).

        g = len(self.var_names_g)
        self.n_metadata = n_metadata
        self.n_metadata_programs = n_metadata_programs
        self.lambda_align = lambda_align

        self._ols_coeff_dg: torch.Tensor | None = (
            torch.from_numpy(np.array(ols_coeff_dg)).float() if ols_coeff_dg is not None else None
        )
        self._metadata_mean_d_init = torch.from_numpy(np.array(metadata_mean_d)).float()
        self._metadata_min_d_init = torch.from_numpy(np.array(metadata_min_d)).float()
        self._metadata_max_d_init = torch.from_numpy(np.array(metadata_max_d)).float()
        self._metadata_noise_scale = metadata_noise_scale

        # Replace encoder with metadata-augmented version
        self.encoder = MetadataAugmentedLoadingsEncoder(n_genes=g, latent_dim=latent_dim, n_metadata=n_metadata)

        # Fixed global metadata min-max scaling stats (never updated during training)
        range_M = (self._metadata_max_d_init - self._metadata_min_d_init).clamp(min=1e-8)
        mu_M_scaled = (self._metadata_mean_d_init - self._metadata_min_d_init) / range_M
        self.register_buffer("min_M_global_d", self._metadata_min_d_init.clone())
        self.register_buffer("range_M_global_d", range_M)
        self.register_buffer("mu_M_scaled_d", mu_M_scaled)

        # Per-k mu_H EMA buffers (track H_total mean)
        for k in k_values:
            self.register_buffer(f"mu_H_ema_{k}_rk", torch.zeros(r, k))

        self._n_ema_updates: int = 0

        # Full reset with OLS W init and new encoder
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        self._n_ema_updates = 0

        # Zero mu_H_ema buffers
        for k in self.k_values:
            buf_name = f"mu_H_ema_{k}_rk"
            if hasattr(self, buf_name):
                getattr(self, buf_name).zero_()

        # Guard: called from super().__init__() before subclass attrs are set
        if not hasattr(self, "n_metadata_programs"):
            return

        if self._ols_coeff_dg is None:
            return

        # OLS-based W initialization for nominated programs
        range_M = getattr(
            self,
            "range_M_global_d",
            (self._metadata_max_d_init - self._metadata_min_d_init).clamp(min=1e-8),
        )

        for k in self.k_values:
            D_rkg = getattr(self, f"D_{k}_rkg")
            device = D_rkg.device

            W_meta_rkg = compute_metadata_nmf_init(
                ols_coeff_dg=self._ols_coeff_dg.to(device),
                n_metadata_programs=self.n_metadata_programs,
                n_replicates=self.r,
                range_M_d=range_M.to(device),
                noise_scale=self._metadata_noise_scale,
            )

            D_rkg[:, : self.n_metadata_programs, :].copy_(W_meta_rkg)

    def online_dictionary_update(
        self,
        x_ng: torch.Tensor,
        k: int,
        n_iterations: int = 100,
        m_nd: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Structure-aware online dictionary update for one k value.

        FISTA solves for H_total directly on the full X. Nominated programs (index
        0:n_metadata_programs) are exempt from the alignment penalty, making them the
        natural route for metadata variance. No Beta update occurs during training.

        Args:
            x_ng: Gene counts (N, G).
            k: The k value to run.
            n_iterations: Solver iterations.
            m_nd: Metadata matrix (N, D), raw uncentered non-negative values.

        Returns:
            dict with keys:
                ``loss``: encoder loss (SmoothL1 warm-start loss), has gradients.
                ``solver_loadings_rnk``: H_total detached (R, N, K).
                ``encoder_loadings_rnk``: H_total_warm detached (R, N, K).
        """
        assert m_nd is not None, "m_nd must be provided for AmortizedOnlineStructureAwareNMF"
        n = x_ng.shape[0]

        W_rkg = getattr(self, f"D_{k}_rkg")
        A_rkk = getattr(self, f"A_{k}_rkk")
        B_rkg = getattr(self, f"B_{k}_rkg")
        mu_H_ema_rk = getattr(self, f"mu_H_ema_{k}_rk")

        # --- Step 0: min-max scale metadata to [0, 1] ---
        nan_mask_n = m_nd.isnan().any(dim=1)  # (N,) True where any metadata dim is NaN
        n_valid = int((~nan_mask_n).sum().item())
        m_scaled_nd = ((m_nd - self.min_M_global_d) / self.range_M_global_d).clamp(0.0, 1.0)
        m_scaled_nd = m_scaled_nd.nan_to_num(0.0)  # NaN cells → 0

        # --- Step 1: encoder warm-start for H_total (has gradients) ---
        H_total_warm_rnk = self.encoder(x_ng, W_rkg.detach(), m_scaled_nd)

        with torch.no_grad():
            # --- Precompute FISTA inputs on full X (no H_struct subtraction) ---
            wwT_rkk = torch.einsum("rkg,rhg->rkh", W_rkg, W_rkg)  # (R, K, K)
            WxT_rkn = torch.einsum("rkg,ng->rkn", W_rkg, x_ng)  # (R, K, N)

            # --- Precompute M_c and mu_H for the covariance penalty ---
            M_c_nd = m_scaled_nd - self.mu_M_scaled_d  # (N, D), centered scaled metadata
            M_c_nd = M_c_nd.masked_fill(nan_mask_n.unsqueeze(1), 0.0)  # zero NaN-cell rows
            # Normalize by sqrt(n_valid): penalty scales as average over valid cells,
            # making lambda_align batch-size independent.
            M_c_nd = M_c_nd / math.sqrt(max(n_valid, 1))

            ema_rho = float(np.exp(-1.0 / self.n_batches_for_forgetting_momentum))
            if self._n_ema_updates > 0:
                bias_correction = max(1.0 - ema_rho**self._n_ema_updates, 1e-8)
                mu_H_corrected_rk = mu_H_ema_rk / bias_correction
            else:
                mu_H_corrected_rk = torch.zeros_like(mu_H_ema_rk)

        H_total_solver_rkn = solve_structure_aware_nnls_fista(
            AtA=wwT_rkk,
            AtB=WxT_rkn,
            initial_x=H_total_warm_rnk.detach().transpose(-2, -1),  # (R, K, N)
            M_c_nd=M_c_nd,
            mu_H_rk=mu_H_corrected_rk,
            lambda_align=self.lambda_align,
            n_metadata_programs=self.n_metadata_programs,
            max_iter=n_iterations,
        )
        H_total_solver_rnk = H_total_solver_rkn.transpose(-2, -1)  # (R, N, K)

        with torch.no_grad():
            # --- Accumulate A, B using H_total ---
            A_rkk_new = (
                self.exponential_decay_rho * A_rkk
                + torch.bmm(H_total_solver_rnk.transpose(1, 2), H_total_solver_rnk) / n
            )
            B_rkg_new = self.exponential_decay_rho * B_rkg + torch.einsum("rnk,ng->rkg", H_total_solver_rnk, x_ng) / n

        # --- Update W via FISTA ---
        W_rkg_new, _ = nmf_compute_factors_fista(
            w_rkg=W_rkg,
            A_rkk=A_rkk_new,
            B_rkg=B_rkg_new,
            max_iter=n_iterations,
        )
        W_rkg_new = F.normalize(W_rkg_new.reshape(-1, W_rkg_new.shape[-1]), p=1, dim=-1, eps=1e-8).reshape_as(W_rkg_new)

        setattr(self, f"A_{k}_rkk", A_rkk_new)
        setattr(self, f"B_{k}_rkg", B_rkg_new)
        setattr(self, f"D_{k}_rkg", W_rkg_new)

        # --- Update mu_H_ema with H_total over valid cells ---
        with torch.no_grad():
            if n_valid > 0:
                batch_mean_rk = H_total_solver_rnk[:, ~nan_mask_n, :].mean(dim=1)
            else:
                batch_mean_rk = H_total_solver_rnk.mean(dim=1)
            mu_H_ema_new = ema_rho * mu_H_ema_rk + (1.0 - ema_rho) * batch_mean_rk
            setattr(self, f"mu_H_ema_{k}_rk", mu_H_ema_new)

        # --- Encoder loss: chase H_total_solver ---
        encoder_loss = self.encoder_loss_fn(H_total_warm_rnk.contiguous(), H_total_solver_rnk.detach().contiguous())

        return {
            "loss": encoder_loss,
            "solver_loadings_rnk": H_total_solver_rnk.detach(),
            "encoder_loadings_rnk": H_total_warm_rnk.detach(),
        }

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        m_nd: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng: Gene counts matrix (N, G).
            var_names_g: Variable names matching self.var_names_g.
            m_nd: Metadata matrix (N, D), raw uncentered non-negative values.

        Returns:
            dict with key ``loss`` (encoder loss averaged over active k values).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        encoder_losses: list[torch.Tensor] = []
        nmf_reconstruction_errors: list[torch.Tensor] = []

        for k in self.k_values:
            if self._k_done[k]:
                continue
            n_iter = self.max_solver_iter_cooldown if self._k_in_final_epoch[k] else self.max_solver_iter_train
            out = self.online_dictionary_update(x_ng=x_ng, k=k, n_iterations=n_iter, m_nd=m_nd)
            encoder_loss = out["loss"]
            solver_loadings_rnk = out["solver_loadings_rnk"]
            encoder_losses.append(encoder_loss)

            with torch.no_grad():
                self._last_encoder_loss[k] = encoder_loss.detach().item()
                W_rkg = getattr(self, f"D_{k}_rkg")
                squared_error_r = frobenius_loss_trace_compiled(x_ng=x_ng, h_rnk=solver_loadings_rnk, w_rkg=W_rkg)
                nmf_reconstruction_error = squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1])
                nmf_reconstruction_errors.append(nmf_reconstruction_error)
                self._last_recon_loss[k] = nmf_reconstruction_error.item()

        self._n_ema_updates += 1

        with torch.no_grad():
            minibatch_nmf_loss = (
                sum(nmf_reconstruction_errors) / len(nmf_reconstruction_errors) if nmf_reconstruction_errors else None
            )
            beta_ema = np.exp(-1.0 / self.n_batches_for_forgetting_momentum)
            val = (
                beta_ema * self._train_nmf_loss_ema + (1.0 - beta_ema) * minibatch_nmf_loss
                if self._train_nmf_loss_ema is not None
                else minibatch_nmf_loss
            )
            assert isinstance(val, torch.Tensor)
            self._train_nmf_loss_ema = val

        loss = sum(encoder_losses) / len(encoder_losses) if encoder_losses else None
        assert isinstance(loss, torch.Tensor) or loss is None
        return {"loss": loss}

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        m_nd: torch.Tensor | None = None,
    ) -> None:
        """
        Validation step computing reconstruction error using the encoder's H_total prediction.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        if m_nd is None:
            raise ValueError("m_nd must be provided for AmortizedOnlineStructureAwareNMF.validate")

        m_scaled_nd = ((m_nd - self.min_M_global_d) / self.range_M_global_d).clamp(0.0, 1.0)
        m_scaled_nd = m_scaled_nd.nan_to_num(0.0)
        nmf_reconstruction_errors = []
        for k in self.k_values:
            W_rkg = getattr(self, f"D_{k}_rkg")
            H_total_rnk = self.encoder(x_ng, W_rkg.detach(), m_scaled_nd)
            squared_error_r = compute_reconstruction_error_compiled(
                x_ng=x_ng, loadings_rnk=H_total_rnk, factors_rkg=W_rkg
            )
            nmf_reconstruction_error = squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1])
            nmf_reconstruction_errors.append(nmf_reconstruction_error)

        minibatch_nmf_loss = (
            sum(nmf_reconstruction_errors) / len(nmf_reconstruction_errors) if nmf_reconstruction_errors else None
        )
        beta_ema = np.exp(-1.0 / self.n_batches_for_forgetting_momentum)
        val = (
            beta_ema * self._val_nmf_loss_ema + (1.0 - beta_ema) * minibatch_nmf_loss
            if self._val_nmf_loss_ema is not None
            else minibatch_nmf_loss
        )
        assert isinstance(val, torch.Tensor) or val is None
        self._val_nmf_loss_ema = val

        if self._val_nmf_loss_ema is not None:
            pl_module.log("val_nmf_loss", self._val_nmf_loss_ema, sync_dist=True, on_epoch=True)

    @torch.no_grad()
    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        m_nd: torch.Tensor,
        n_iterations: int = 200,
    ) -> dict[str, dict[int, torch.Tensor]]:
        """
        Post-hoc inference: FISTA on full X to get H_total, then OLS to harvest Beta.

        Beta is computed as the non-negative OLS solution mapping scaled metadata to the
        nominated H_total columns:
        ``Beta = (M^T M + eps I)^{-1} M^T H_nominated``, clamped >= 0.

        Args:
            x_ng: Gene counts (N, G).
            var_names_g: Variable names matching self.var_names_g.
            m_nd: Metadata matrix (N, D), raw uncentered non-negative values.
            n_iterations: FISTA iterations for the final solve.

        Returns:
            dict with keys:
                ``H_total``: {k: tensor (R, N, K)} — full non-negative loadings.
                ``Beta``: {k: tensor (R, D, n_metadata_programs)} — harvested Beta >= 0.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        nan_mask_n = m_nd.isnan().any(dim=1)
        n_valid = int((~nan_mask_n).sum().item())
        m_scaled_nd = ((m_nd - self.min_M_global_d) / self.range_M_global_d).clamp(0.0, 1.0)
        m_scaled_nd = m_scaled_nd.nan_to_num(0.0)

        M_c_nd = m_scaled_nd - self.mu_M_scaled_d
        M_c_nd = M_c_nd.masked_fill(nan_mask_n.unsqueeze(1), 0.0)
        M_c_nd = M_c_nd / math.sqrt(max(n_valid, 1))

        H_total_out: dict[int, torch.Tensor] = {}
        Beta_out: dict[int, torch.Tensor] = {}

        for k in self.k_values:
            W_rkg = getattr(self, f"D_{k}_rkg")
            mu_H_ema_rk = getattr(self, f"mu_H_ema_{k}_rk")
            if self._n_ema_updates > 0:
                ema_rho = float(np.exp(-1.0 / self.n_batches_for_forgetting_momentum))
                bias_correction = max(1.0 - ema_rho**self._n_ema_updates, 1e-8)
                mu_H_corrected_rk = mu_H_ema_rk / bias_correction
            else:
                mu_H_corrected_rk = torch.zeros_like(mu_H_ema_rk)

            n = x_ng.shape[0]
            wwT_rkk = torch.einsum("rkg,rhg->rkh", W_rkg, W_rkg)
            WxT_rkn = torch.einsum("rkg,ng->rkn", W_rkg, x_ng)

            H_total_solver_rkn = solve_structure_aware_nnls_fista(
                AtA=wwT_rkk,
                AtB=WxT_rkn,
                initial_x=torch.zeros(self.r, k, n, device=x_ng.device, dtype=x_ng.dtype),
                M_c_nd=M_c_nd,
                mu_H_rk=mu_H_corrected_rk,
                lambda_align=self.lambda_align,
                n_metadata_programs=self.n_metadata_programs,
                max_iter=n_iterations,
            )
            H_total_solver_rnk = H_total_solver_rkn.transpose(-2, -1)  # (R, N, K)
            H_total_out[k] = H_total_solver_rnk

            # OLS harvest: solve (M^T M + eps I) Beta = M^T H_nominated, clamp >= 0
            p = self.n_metadata_programs
            H_nominated_rnp = H_total_solver_rnk[:, :, :p]  # (R, N, p)
            MtM = m_scaled_nd.T @ m_scaled_nd  # (D, D)
            eps = 1e-6 * torch.eye(MtM.shape[0], device=MtM.device, dtype=MtM.dtype)
            MtH_rdp = torch.einsum("nd,rnp->rdp", m_scaled_nd, H_nominated_rnp)  # (R, D, p)
            beta_rdp = torch.linalg.solve(
                (MtM + eps).unsqueeze(0).expand(self.r, -1, -1),  # (R, D, D)
                MtH_rdp,  # (R, D, p)
            ).clamp(min=0.0)
            Beta_out[k] = beta_rdp

        return {"H_total": H_total_out, "Beta": Beta_out}
