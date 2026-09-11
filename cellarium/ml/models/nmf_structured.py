# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

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
    solve_nnls_fista_precomputed,
)
from cellarium.ml.models.nmf_amortized import (
    AmortizedOnlineNonNegativeMatrixFactorization,
    BilinearLoadingsEncoder,
)
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


@torch.no_grad()
def group_lasso_prox(beta_rdk: torch.Tensor, lambda_select: float, lr: float | torch.Tensor) -> torch.Tensor:
    """
    Block soft-thresholding proximal operator for the Group Lasso penalty.

    For each factor column k, computes the L2 norm across the metadata dimension d
    and shrinks the entire column toward zero. Columns whose group norm falls below
    ``lambda_select * lr`` are zeroed out exactly, achieving factor-level sparsity:
    only a few factors are metadata-driven.

    Args:
        beta_rdk: Beta tensor of shape (R, D, K).
        lambda_select: Group Lasso regularization strength.
        lr: Proximal gradient step size.

    Returns:
        Updated beta_rdk with group-sparse columns.
    """
    group_norms_r1k = beta_rdk.norm(dim=1, keepdim=True)  # (R, 1, K)
    threshold = lambda_select * lr
    scale = (1.0 - threshold / group_norms_r1k.clamp(min=1e-10)).clamp(min=0.0)
    scale = torch.where(group_norms_r1k > threshold, scale, torch.zeros_like(scale))
    return beta_rdk * scale


@torch.no_grad()
def update_beta_group_lasso(
    H_raw_rnk: torch.Tensor,
    W_rkg: torch.Tensor,
    X_ng: torch.Tensor,
    M_nd: torch.Tensor,
    beta_rdk: torch.Tensor,
    lambda_select: float,
    beta_lr: float,
    n_iter: int,
    n_metadata_programs: int,
) -> torch.Tensor:
    """
    Proximal gradient update for Beta under Group Lasso regularization.

    Updates only the nominated columns ``0:n_metadata_programs`` of Beta.
    Non-nominated columns are never touched — they stay at exactly zero by design.
    This means the Group Lasso prunes *among* nominated columns: if
    ``n_metadata_programs`` was over-specified, unnecessary nominated columns are
    driven to zero; only those with genuine metadata-correlation survive.

    The gradient is computed with z-scored metadata so the gradient magnitude is
    unit-free (not sensitive to metadata units or scale). The step size is set by
    the Lipschitz constant of the gradient, which accounts for the mixed raw-M
    forward / z-scored-M gradient computation.

    Args:
        H_raw_rnk: Idiosyncratic loadings of shape (R, N, K).
        W_rkg: Gene factors of shape (R, K, G).
        X_ng: Gene counts of shape (N, G).
        M_nd: Metadata matrix of shape (N, D), raw (uncentered).
        beta_rdk: Current Beta of shape (R, D, K).
        lambda_select: Group Lasso strength (in z-scored gradient units).
        beta_lr: Upper bound on the proximal gradient step size. The actual step is
            ``min(beta_lr, 1/L)`` where L is the per-batch Lipschitz constant.
            Setting ``beta_lr=1.0`` uses the Lipschitz-optimal step when L≥1.
        n_iter: Number of proximal gradient iterations per forward call.
        n_metadata_programs: Number of nominated metadata columns to update.
            Non-nominated columns (``n_metadata_programs:K``) are never updated.

    Returns:
        Updated beta_rdk of shape (R, D, K).
    """
    n = X_ng.shape[0]
    W_active = W_rkg[:, :n_metadata_programs, :]  # (R, P, G), P = n_metadata_programs

    # Z-score metadata to remove mean-level bias from the gradient.
    # Without centering, large mean(M) (e.g. age ≈ 50) creates a constant gradient term
    # proportional to mean(M) * mean(X_res), which would drive all nominated columns positive
    # regardless of whether they correlate with M. Centering removes this bias.
    # Dividing by std(M) additionally makes the gradient magnitude unit-free, so
    # lambda_select is interpretable across different metadata scales.
    # Note: the Lipschitz constant still scales with std(M) because the forward uses raw M.
    M_z_nd = M_nd - M_nd.mean(dim=0, keepdim=True)  # center
    M_z_nd = M_z_nd / M_z_nd.std(dim=0, keepdim=True).clamp(min=1e-8)  # scale to unit variance

    # Lipschitz constant of the (mixed) beta gradient.
    # The gradient uses M_z but the forward uses raw M, so the gradient function is:
    #   grad(beta) = (2/n) * M_z^T @ (M @ beta @ W - X) @ W^T
    # Its Hessian is (2/n) * (M_z^T @ M) ⊗ (W W^T), with Lipschitz constant:
    #   L = (2/n) * ||M_z^T @ M||_op * lambda_max(WaWaT)
    # Using M_z^T @ M_z ≈ n (the self-product) instead of the cross-product M_z^T @ M
    # underestimates L by a factor of std(M), causing gradient steps that overshoot
    # by the same factor and diverge from the correct solution.
    MztM = M_z_nd.T @ M_nd  # (D, D): cross-product of z-scored and raw metadata
    lambda_max_MztM = torch.linalg.matrix_norm(MztM, ord=2)  # scalar tensor, no CPU sync
    # WaWaT is precomputed here and reused for both the Lipschitz bound and the loop gradient.
    WaWaT_rpp = torch.einsum("rpg,rqg->rpq", W_active, W_active)  # (R, P, P)
    # Trace upper-bounds lambda_max for a PSD matrix; avoids eigvalsh and its CPU sync.
    lambda_max_WaWaT = WaWaT_rpp.diagonal(dim1=-2, dim2=-1).sum(dim=-1).max()  # scalar tensor
    L = (2.0 / n) * lambda_max_MztM * lambda_max_WaWaT
    effective_lr = torch.minimum(
        torch.tensor(beta_lr, device=L.device, dtype=L.dtype),
        1.0 / L.clamp(min=1e-10),
    )  # 0-d tensor, no CPU sync

    # Precompute G-space projections into the latent space so the hot loop never touches G.
    # X_res @ Wa.T = (X - H_raw @ W_all) @ Wa.T = X @ Wa.T - H_raw @ (W_all @ Wa.T)
    X_WaT_rnp = torch.einsum("ng,rpg->rnp", X_ng, W_active)  # (R, N, P)
    WWaT_rkp = torch.einsum("rkg,rpg->rkp", W_rkg, W_active)  # (R, K, P)
    X_res_WaT_rnp = X_WaT_rnp - torch.einsum("rnk,rkp->rnp", H_raw_rnk, WWaT_rkp)  # (R, N, P)

    beta_active = beta_rdk[:, :, :n_metadata_programs].clone()

    for _ in range(n_iter):
        # All operations in (R, N, P) space — G dimension eliminated from the loop.
        M_beta_rnp = torch.einsum("nd,rdp->rnp", M_nd, beta_active)  # (R, N, P)
        pred_W_rnp = torch.einsum("rnp,rpq->rnq", M_beta_rnp, WaWaT_rpp)  # (R, N, P)
        err_W_rnp = pred_W_rnp - X_res_WaT_rnp  # (R, N, P)
        grad_active = (2.0 / n) * torch.einsum("nd,rnp->rdp", M_z_nd, err_W_rnp)  # (R, D, P)

        beta_active = beta_active - effective_lr * grad_active
        beta_active = group_lasso_prox(beta_active, lambda_select, effective_lr)
        beta_active = beta_active.clamp(min=0.0)

    result = beta_rdk.clone()
    result[:, :, :n_metadata_programs] = beta_active
    return result


@torch.no_grad()
def compute_metadata_nmf_init(
    ols_coeff_dg: torch.Tensor,
    n_metadata_programs: int,
    n_replicates: int,
    n_components: int,
    mean_M_d: torch.Tensor,
    mean_total_count: float,
    noise_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derives NMF initialization for nominated metadata-capturing factors from OLS coefficients.

    Runs truncated SVD on the OLS coefficient matrix to extract the dominant modes of
    metadata-gene co-variation. Applies a sign convention so the primary gene direction
    is positive, then ReLU-projects to non-negative (upregulation only) and L1-normalizes
    to match the NMF factor convention. Beta columns are calibrated so that
    ``mean(M) @ Beta_j ≈ mean_total_count / n_components``.

    Recommended workflow:
        1. Run ``StreamingOrdinaryLeastSquares(var_names_g=gene_names, n_targets=n_genes)``
           with ``x_ng=M_metadata`` and ``y_nk=X_genes`` for one epoch.
        2. Call ``.solve()`` to get ``ols_coeff_dg`` of shape (D, G).
        3. Pass ``ols_coeff_dg`` to this function.

    Args:
        ols_coeff_dg: OLS coefficient matrix of shape (D, G).
        n_metadata_programs: Number of metadata factors to initialize.
        n_replicates: Number of NMF replicates.
        n_components: Total NMF components k (for loading scale calibration).
        mean_M_d: Global metadata mean of shape (D,).
        mean_total_count: Expected total UMI count per cell.
        noise_scale: Relative noise std for replicate diversity (default 0.05).

    Returns:
        W_meta_rkg: shape (R, n_metadata_programs, G) — nominated gene factor vectors.
        Beta_meta_rdk: shape (R, D, n_metadata_programs) — nominated Beta columns.
    """
    D, G = ols_coeff_dg.shape
    n_singular = min(n_metadata_programs, D)

    U, S, Vh = torch.linalg.svd(ols_coeff_dg, full_matrices=False)
    # U: (D, min(D,G)), S: (min(D,G),), Vh: (min(D,G), G)
    U = U[:, :n_singular]  # (D, n_singular)
    S = S[:n_singular]  # (n_singular,)
    Vh = Vh[:n_singular]  # (n_singular, G)

    # Sign convention: largest absolute value in each V row should be positive
    for j in range(n_singular):
        max_abs_idx = torch.argmax(torch.abs(Vh[j]))
        if Vh[j, max_abs_idx] < 0:
            Vh[j] = -Vh[j]
            U[:, j] = -U[:, j]

    # Build base W and Beta for each nominated program
    W_base: list[torch.Tensor] = []
    Beta_base: list[torch.Tensor] = []
    target_loading = mean_total_count / max(n_components, 1)
    mean_M_sum = mean_M_d.abs().sum().clamp(min=1e-8).item()

    for j in range(n_metadata_programs):
        if j < n_singular:
            w_j = F.relu(Vh[j]) + 1e-8  # (G,)
            w_j = F.normalize(w_j, p=1, dim=0)

            beta_j = F.relu(U[:, j])  # (D,) non-negative
            mean_loading = (mean_M_d * beta_j).sum().item()
            if mean_loading > 1e-8:
                beta_j = beta_j * (target_loading / mean_loading)
            else:
                beta_j = torch.ones(D, device=ols_coeff_dg.device) * (target_loading / mean_M_sum)
        else:
            # More programs than SVD modes: random init for extras
            w_j = F.normalize(torch.rand(G, device=ols_coeff_dg.device) + 1e-8, p=1, dim=0)
            beta_j = torch.ones(D, device=ols_coeff_dg.device) * (target_loading / mean_M_sum)

        W_base.append(w_j)
        Beta_base.append(beta_j)

    W_meta_rkg = torch.zeros(n_replicates, n_metadata_programs, G, device=ols_coeff_dg.device)
    Beta_meta_rdk = torch.zeros(n_replicates, D, n_metadata_programs, device=ols_coeff_dg.device)

    for r in range(n_replicates):
        for j in range(n_metadata_programs):
            w_j = W_base[j]
            beta_j = Beta_base[j]

            w_mean = w_j.mean().item()
            noise_w = torch.randn(G, device=ols_coeff_dg.device) * noise_scale * max(w_mean, 1e-8)
            w_r = F.relu(w_j + noise_w) + 1e-8
            W_meta_rkg[r, j] = F.normalize(w_r, p=1, dim=0)

            beta_mean = beta_j.mean().item()
            if beta_mean > 1e-10:
                noise_b = torch.randn(D, device=ols_coeff_dg.device) * noise_scale * beta_mean
                Beta_meta_rdk[r, :, j] = F.relu(beta_j + noise_b)
            else:
                Beta_meta_rdk[r, :, j] = beta_j

    return W_meta_rkg, Beta_meta_rdk


class MetadataAugmentedLoadingsEncoder(BilinearLoadingsEncoder):
    """
    Encoder that augments the cell embedding with a metadata side-channel before computing
    bilinear affinities with gene factors.

    The metadata projection is added to the cell embedding so the encoder can learn to
    predict the idiosyncratic component H_raw — which should be uncorrelated with metadata M —
    by seeing M and learning to subtract the structured variance.

    The encoder is trained with two signals:
    1. SmoothL1 matching loss: match the FISTA solver's H_raw output.
    2. Covariance penalty: ``lambda_align * ||anchored_cov(H_raw_encoder, M)||_F^2``,
       which directly teaches the encoder to produce decorrelated warm-starts.
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
            h_rnk: Warm-start idiosyncratic loadings of shape (R, N, K), non-negative,
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

    Decomposes cell loadings into:

        H_total = H_raw + H_struct = H_raw + M * Beta

    where H_raw is idiosyncratic (penalized to be uncorrelated with metadata M) and
    H_struct = M * Beta is the metadata-driven structured component. Gene factors W are
    shared across both components.

    Objective (implicit, enforced by training signals):

        L_recon = ||X - H_total * W||_F^2
        L_align = lambda_align * ||anchored_cov(H_raw_encoder, M)||_F^2
        L_select = lambda_select * sum_k ||Beta[:, k]||_2   (Group Lasso)

    Group Lasso on Beta achieves **factor-level sparsity**: only a few nominated factors
    become metadata-driven; the rest stay free. Lambda_align teaches the encoder to
    produce H_raw warm-starts that are already decorrelated from M.

    **Nominated factors and initialization:**
    The first ``n_metadata_programs`` factors are "nominated" as metadata factors.
    They are initialized from OLS coefficients (see ``compute_metadata_nmf_init``), and
    their Beta columns are the only ones ever updated during training. Non-nominated Beta
    columns (index ``n_metadata_programs:K``) stay exactly at zero throughout. Group Lasso
    prunes *among* the nominated columns: if ``n_metadata_programs`` is over-specified,
    unnecessary nominated Beta columns are driven to zero; only genuinely metadata-correlated
    factors survive. Different replicates receive independent noise, ensuring diversity for
    the consensus step.

    **Metadata convention:**
    ``m_nd`` should contain **raw (uncentered) non-negative** metadata values (e.g., donor
    age in years). Non-negativity ensures H_struct = M * Beta >= 0 (since Beta >= 0),
    keeping H_total >= 0.

    **Recommended workflow:**

    .. code-block:: python

        # Step 1: pre-training OLS pass (one epoch)
        ols = StreamingOrdinaryLeastSquares(var_names_g=gene_names, n_targets=n_genes)
        # train ols with x_ng=M_metadata, y_nk=X_genes for one epoch
        ols_coeff_dg = ols.solve().numpy()   # (D, G)

        # Step 2: metadata mean
        metadata_mean_d = M.mean(axis=0)    # (D,)

        # Step 3: train structured NMF
        model = AmortizedOnlineStructureAwareNMF(
            ...,
            n_metadata=D,
            metadata_mean_d=metadata_mean_d,
            ols_coeff_dg=ols_coeff_dg,
            n_metadata_programs=2,          # or 2 * D as a starting point
        )

        # Step 4: consensus
        consensus_factors = compute_consensus_factors(model)
        # Stability of the first 1-2 factor clusters reveals how many metadata
        # programs were genuinely needed.

    Args:
        var_names_g: Gene names.
        k_values: List of NMF component counts to train in parallel.
        r: Number of replicates.
        latent_dim: Encoder latent dimension.
        total_n_cells: Total dataset size (for EMA period computation).
        batch_size: Training batch size.
        n_metadata: Number of metadata columns (D).
        metadata_mean_d: Precomputed global metadata mean, shape (D,). Required.
        n_metadata_programs: Number of nominated metadata factors. Defaults to
            ``2 * n_metadata``. Must be < min(k_values).
        ols_coeff_dg: OLS coefficient matrix (D, G) from StreamingOrdinaryLeastSquares.
            If None, nominated factors use random initialization.
        mean_total_count: Expected total UMI count per cell, used for Beta scale
            calibration. If None, a rough estimate of ``10 * min(k_values)`` is used.
        metadata_noise_scale: Relative noise std for replicate diversity (default 0.05).
        lambda_align: Strength of the anchored covariance decorrelation penalty on
            the encoder output.
        lambda_select: Group Lasso strength on Beta columns.
        beta_lr: Upper bound on the Beta proximal gradient step size. The actual step is
            ``min(beta_lr, 1/L)`` where L is the per-batch Lipschitz constant of the
            Beta gradient. Setting ``beta_lr=1.0`` (the default) is equivalent to always
            using the theoretically optimal step size; smaller values slow Beta down.
        beta_n_iter: Number of proximal gradient iterations per forward call.
        solver: Inner solver for H_raw. Only ``"fista"`` is supported (HALS requires
            replicate-varying residuals which are not yet implemented).
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
        metadata_mean_d: np.ndarray,
        n_metadata_programs: int | None = None,
        ols_coeff_dg: np.ndarray | None = None,
        mean_total_count: float | None = None,
        metadata_noise_scale: float = 0.05,
        lambda_align: float = 0.1,
        lambda_select: float = 0.05,
        beta_lr: float = 1.0,
        beta_n_iter: int = 20,
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
        # super().__init__() has: set self.encoder = BilinearLoadingsEncoder, registered
        # A/B/D buffers, and called reset_parameters() once (early return because
        # subclass attrs not yet set).

        g = len(self.var_names_g)
        self.n_metadata = n_metadata
        self.n_metadata_programs = n_metadata_programs
        self.lambda_align = lambda_align
        self.lambda_select = lambda_select
        self.beta_lr = beta_lr
        self.beta_n_iter = beta_n_iter

        # Store init data for reset_parameters
        self._ols_coeff_dg: torch.Tensor | None = (
            torch.from_numpy(np.array(ols_coeff_dg)).float() if ols_coeff_dg is not None else None
        )
        self._metadata_mean_d_init = torch.from_numpy(np.array(metadata_mean_d)).float()
        self._mean_total_count = mean_total_count
        self._metadata_noise_scale = metadata_noise_scale

        # Replace encoder with metadata-augmented version
        self.encoder = MetadataAugmentedLoadingsEncoder(n_genes=g, latent_dim=latent_dim, n_metadata=n_metadata)

        # Fixed global metadata mean (never updated during training)
        self.register_buffer("mu_M_global_d", self._metadata_mean_d_init.clone())

        # Per-k Beta and H_raw EMA buffers
        for k in k_values:
            self.register_buffer(f"beta_{k}_rdk", torch.zeros(r, n_metadata, k))
            self.register_buffer(f"mu_H_ema_{k}_rk", torch.zeros(r, k))

        self._n_ema_updates: int = 0

        # Full reset with OLS init and new encoder
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

        # Zero all Beta buffers (non-nominated columns must start exactly at zero)
        for k in self.k_values:
            buf_name = f"beta_{k}_rdk"
            if hasattr(self, buf_name):
                getattr(self, buf_name).zero_()

        if self._ols_coeff_dg is None:
            # No OLS: nominated Beta columns get small uniform random values
            for k in self.k_values:
                buf_name = f"beta_{k}_rdk"
                if not hasattr(self, buf_name):
                    continue
                getattr(self, buf_name)[:, :, : self.n_metadata_programs].uniform_(0.01, 0.1)
            return

        # OLS-based initialization
        mu_M = getattr(self, "mu_M_global_d", self._metadata_mean_d_init)

        for k in self.k_values:
            D_rkg = getattr(self, f"D_{k}_rkg")
            device = D_rkg.device
            mean_count = float(self._mean_total_count) if self._mean_total_count is not None else 10.0 * k

            W_meta_rkg, Beta_meta_rdk = compute_metadata_nmf_init(
                ols_coeff_dg=self._ols_coeff_dg.to(device),
                n_metadata_programs=self.n_metadata_programs,
                n_replicates=self.r,
                n_components=k,
                mean_M_d=mu_M.to(device),
                mean_total_count=mean_count,
                noise_scale=self._metadata_noise_scale,
            )

            D_rkg[:, : self.n_metadata_programs, :].copy_(W_meta_rkg)
            getattr(self, f"beta_{k}_rdk")[:, :, : self.n_metadata_programs].copy_(Beta_meta_rdk)

    def online_dictionary_update(
        self,
        x_ng: torch.Tensor,
        k: int,
        n_iterations: int = 100,
        m_nd: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Structure-aware online dictionary update for one k value.

        Solves for H_raw on the effective residual X - H_struct * W, updates Beta
        via Group Lasso proximal gradient, accumulates A/B using H_total = H_raw + H_struct,
        and updates W via FISTA. Computes the encoder loss (SmoothL1 + anchored covariance
        penalty) but does NOT call encoder.backward() — that is handled by Lightning.

        Args:
            x_ng: Gene counts (N, G).
            k: The k value to run.
            n_iterations: Solver iterations for both H_raw and W.
            m_nd: Metadata matrix (N, D), raw uncentered non-negative values.

        Returns:
            dict with keys:
                ``loss``: encoder loss (SmoothL1 + covariance penalty), has gradients.
                ``solver_loadings_rnk``: H_total detached (R, N, K), for recon error tracking.
                ``encoder_loadings_rnk``: H_raw_warm detached (R, N, K).
        """
        assert m_nd is not None, "m_nd must be provided for AmortizedOnlineStructureAwareNMF"
        n = x_ng.shape[0]

        W_rkg = getattr(self, f"D_{k}_rkg")
        beta_rdk = getattr(self, f"beta_{k}_rdk")
        A_rkk = getattr(self, f"A_{k}_rkk")
        B_rkg = getattr(self, f"B_{k}_rkg")
        mu_H_ema_rk = getattr(self, f"mu_H_ema_{k}_rk")

        # --- Step 1: encoder warm-start for H_raw (has gradients) ---
        H_raw_warm_rnk = self.encoder(x_ng, W_rkg.detach(), m_nd)

        # --- Step 2: structural component H_struct = M * Beta ---
        with torch.no_grad():
            H_struct_rnk = torch.einsum("nd,rdk->rnk", m_nd, beta_rdk)

            # --- Step 3 & 4: FISTA solver setup (no (R, N, G) materialization) ---
            # W @ X_eff.T = W @ X.T - (W @ W.T) @ H_struct.T, avoiding explicit X_eff.
            wwT_rkk = torch.einsum("rkg,rhg->rkh", W_rkg, W_rkg)  # (R, K, K)
            WxT_rkn = torch.einsum("rkg,ng->rkn", W_rkg, x_ng)  # (R, K, N)
            wwT_Hstruct_rkn = torch.einsum("rkh,rnh->rkn", wwT_rkk, H_struct_rnk)  # (R, K, N)
            wxT_eff_rkn = WxT_rkn - wwT_Hstruct_rkn  # (R, K, N)

        H_raw_solver_kn, _ = solve_nnls_fista_precomputed(
            AtA=wwT_rkk,
            AtB=wxT_eff_rkn,
            initial_x=H_raw_warm_rnk.detach().transpose(-2, -1),  # (R, K, N)
            max_iter=n_iterations,
        )
        H_raw_solver_rnk = H_raw_solver_kn.transpose(-2, -1)  # (R, N, K)

        with torch.no_grad():
            # --- Step 5: Beta update via Group Lasso proximal gradient ---
            beta_rdk_updated = update_beta_group_lasso(
                H_raw_rnk=H_raw_solver_rnk,
                W_rkg=W_rkg,
                X_ng=x_ng,
                M_nd=m_nd,
                beta_rdk=beta_rdk,
                lambda_select=self.lambda_select,
                beta_lr=self.beta_lr,
                n_iter=self.beta_n_iter,
                n_metadata_programs=self.n_metadata_programs,
            )
            setattr(self, f"beta_{k}_rdk", beta_rdk_updated)

            # --- Step 6: recompute H_struct and H_total ---
            H_struct_updated_rnk = torch.einsum("nd,rdk->rnk", m_nd, beta_rdk_updated)
            H_total_rnk = H_raw_solver_rnk + H_struct_updated_rnk

            # --- Step 7: accumulate A, B using H_total (Mairal update with rho decay) ---
            A_rkk_new = self.exponential_decay_rho * A_rkk + torch.bmm(H_total_rnk.transpose(1, 2), H_total_rnk) / n
            B_rkg_new = self.exponential_decay_rho * B_rkg + torch.einsum("rnk,ng->rkg", H_total_rnk, x_ng) / n

        # --- Step 8: update W via FISTA factors ---
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

        # --- Step 9: anchored covariance penalty on encoder output (pre-EMA-update) ---
        # Uses historical mu_H_ema (before this batch) to avoid circularity.
        M_c_nd = m_nd - self.mu_M_global_d  # (N, D), exact centering
        ema_rho = float(np.exp(-1.0 / self.n_batches_for_forgetting_momentum))
        if self._n_ema_updates > 0:
            bias_correction = max(1.0 - ema_rho**self._n_ema_updates, 1e-8)
            mu_H_corrected_rk = mu_H_ema_rk / bias_correction
        else:
            mu_H_corrected_rk = torch.zeros_like(mu_H_ema_rk)

        # H_raw_warm_c is centered by the historical mean; gradient flows through H_raw_warm_rnk
        H_raw_warm_c_rnk = H_raw_warm_rnk - mu_H_corrected_rk.detach().unsqueeze(1)  # (R, N, K)
        cov_rdk = torch.einsum("nd,rnk->rdk", M_c_nd, H_raw_warm_c_rnk) / n  # (R, D, K)
        cov_penalty = self.lambda_align * (cov_rdk**2).sum()

        # --- Step 10: update mu_H_ema with solver H_raw (after penalty computation) ---
        with torch.no_grad():
            batch_mean_rk = H_raw_solver_rnk.mean(dim=1)  # (R, K)
            mu_H_ema_new = ema_rho * mu_H_ema_rk + (1.0 - ema_rho) * batch_mean_rk
            setattr(self, f"mu_H_ema_{k}_rk", mu_H_ema_new)

        # --- Step 11: encoder loss ---
        encoder_loss = (
            self.encoder_loss_fn(H_raw_warm_rnk.contiguous(), H_raw_solver_rnk.detach().contiguous()) + cov_penalty
        )

        return {
            "loss": encoder_loss,
            "solver_loadings_rnk": H_total_rnk.detach(),  # H_total for recon error tracking
            "encoder_loadings_rnk": H_raw_warm_rnk.detach(),
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
        Validation step computing reconstruction error using H_total = H_raw_encoder + H_struct.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        if m_nd is None:
            raise ValueError("m_nd must be provided for AmortizedOnlineStructureAwareNMF.validate")

        nmf_reconstruction_errors = []
        for k in self.k_values:
            W_rkg = getattr(self, f"D_{k}_rkg")
            beta_rdk = getattr(self, f"beta_{k}_rdk")
            H_raw_warm_rnk = self.encoder(x_ng, W_rkg.detach(), m_nd)
            H_struct_rnk = torch.einsum("nd,rdk->rnk", m_nd, beta_rdk.detach())
            H_total_rnk = H_raw_warm_rnk + H_struct_rnk
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
