# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings
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
    max_iter: int = 100,
) -> torch.Tensor:
    """
    FISTA solver for structure-aware NMF that includes the covariance decorrelation penalty
    directly in the gradient, so ``H_solver`` genuinely balances reconstruction with
    metadata-decorrelation.

    Minimizes::

        (1/2) ||X_eff - H W||_F^2  +  lambda_align * ||M_c^T (H - mu_H)||_F^2

    subject to H >= 0.

    Args:
        AtA: (R, K, K) — W W^T, precomputed.
        AtB: (R, K, N) — W X_eff^T, precomputed.
        initial_x: (R, K, N) — warm-start (H^T in FISTA convention).
        M_c_nd: (N, D) — batch-centered scaled metadata, NaN rows zeroed.
        mu_H_rk: (R, K) — bias-corrected EMA mean of H_raw, used for H centering.
        lambda_align: Covariance penalty strength.
        max_iter: Number of FISTA iterations.

    Returns:
        x: (R, K, N) — solved H^T.
    """
    # --- Lipschitz constant: reconstruction term via power iteration ---
    # Avoids eigvalsh / SVD (which can fail on ill-conditioned matrices and trigger
    # CPU-GPU sync).  Same strategy as solve_nnls_fista_precomputed.
    v = torch.ones(*AtA.shape[:-1], 1, device=AtA.device, dtype=AtA.dtype)
    for _ in range(10):
        v = AtA @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-8)
    L_A = (v.transpose(-2, -1) @ AtA @ v).clamp(min=1e-12)  # (R, 1, 1)

    # --- Lipschitz constant: penalty term ---
    # Gradient of lambda_align * ||M_c^T (H - mu_H)||_F^2 w.r.t. H has Lipschitz constant
    # 2 * lambda_align * ||M_c M_c^T||_op.  For a rank-D matrix M_c the Frobenius norm
    # squared is an upper bound on the spectral norm (exact when D=1), and avoids SVD.
    L_M = (2.0 * lambda_align) * (M_c_nd**2).sum()  # scalar

    L = L_A + L_M  # (R, 1, 1), broadcasts correctly

    # mu_H broadcast shape: (R, K, 1) so it subtracts from (R, K, N) without forming NxN
    mu_H_rk1 = mu_H_rk.unsqueeze(-1)  # (R, K, 1)

    x = initial_x.clone()
    y = initial_x.clone()
    t = 1.0

    for _ in range(max_iter):
        grad = AtA @ y - AtB

        if lambda_align > 0:
            # Right-to-left: avoids forming the N×N matrix M_c M_c^T.
            # Cost: O(NKD) per iteration — negligible for small D.
            tmp = (y - mu_H_rk1) @ M_c_nd  # (R, K, D)
            grad = grad + (2.0 * lambda_align) * (tmp @ M_c_nd.mT)  # (R, K, N)

        x_new = torch.clamp(y - grad / L, min=0.0)

        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t**2)) / 2.0
        momentum = (t - 1.0) / t_new
        y = x_new + momentum * (x_new - x)

        x = x_new
        t = t_new

    return x


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
    M_scaled_nd: torch.Tensor,
    beta_rdk: torch.Tensor,
    lambda_select: float,
    beta_lr: float,
    n_iter: int,
    n_metadata_programs: int,
    n_valid: int | None = None,
    beta_prior_active: torch.Tensor | None = None,
    lambda_prior: float = 0.0,
) -> torch.Tensor:
    """
    Proximal gradient update for Beta under Group Lasso regularization.

    Updates only the nominated columns ``0:n_metadata_programs`` of Beta.
    Non-nominated columns are never touched — they stay at exactly zero by design.
    This means the Group Lasso prunes *among* nominated columns: if
    ``n_metadata_programs`` was over-specified, unnecessary nominated columns are
    driven to zero; only those with genuine metadata-correlation survive.

    The forward is linear: ``H_struct = M_scaled @ Beta``. Since ``M_scaled`` is
    min-max scaled to ``[0, 1]`` and ``Beta >= 0`` (enforced by clamping), H_struct
    is guaranteed non-negative. The gradient is the standard linear least-squares
    gradient with no chain-rule correction needed.

    Args:
        H_raw_rnk: Idiosyncratic loadings of shape (R, N, K).
        W_rkg: Gene factors of shape (R, K, G).
        X_ng: Gene counts of shape (N, G).
        M_scaled_nd: Min-max scaled metadata of shape (N, D), values in [0, 1].
            Cells with missing metadata should have their rows pre-filled with 0
            before this call; their gradient contribution is then exactly zero.
        beta_rdk: Current Beta of shape (R, D, K).
        lambda_select: Group Lasso strength.
        beta_lr: Upper bound on the proximal gradient step size. The actual step is
            ``min(beta_lr, 1/L)`` where L is the per-batch Lipschitz constant.
            Setting ``beta_lr=1.0`` uses the Lipschitz-optimal step when L≥1.
        n_iter: Number of proximal gradient iterations per forward call.
        n_metadata_programs: Number of nominated metadata columns to update.
            Non-nominated columns (``n_metadata_programs:K``) are never updated.
        n_valid: Number of cells with non-missing metadata. When provided, the
            Lipschitz constant and gradient are normalized by ``n_valid`` instead
            of the total batch size, so missing-metadata cells do not dilute the
            effective step size. Defaults to the total batch size.
        beta_prior_active: OLS-derived prior for nominated Beta columns, shape
            (R, D, n_metadata_programs). When provided with ``lambda_prior > 0``,
            adds ``lambda_prior * ||Beta_active - beta_prior_active||_F^2`` to the
            Beta objective, pulling Beta toward the OLS initialization.
        lambda_prior: Strength of the L2 prior toward ``beta_prior_active``.
            Default 0 disables the prior. The Lipschitz constant is updated to
            account for the prior term so step sizes remain valid.

    Returns:
        Updated beta_rdk of shape (R, D, K).
    """
    n = X_ng.shape[0]
    n_eff = max(n_valid, 1) if n_valid is not None else n
    W_active = W_rkg[:, :n_metadata_programs, :]  # (R, P, G), P = n_metadata_programs

    # Lipschitz constant of the linear beta gradient:
    #   ||grad(beta)||_2 <= (2/n_eff) * ||M_sc^T M_sc||_op * lambda_max(WaWaT) * ||delta_beta||_2
    MscTMsc = M_scaled_nd.T @ M_scaled_nd  # (D, D): self-product of scaled metadata
    lambda_max_MscTMsc = torch.linalg.matrix_norm(MscTMsc, ord=2)  # scalar tensor, no CPU sync
    # WaWaT is precomputed here and reused for both the Lipschitz bound and the loop gradient.
    WaWaT_rpp = torch.einsum("rpg,rqg->rpq", W_active, W_active)  # (R, P, P)
    # Trace upper-bounds lambda_max for a PSD matrix; avoids eigvalsh and its CPU sync.
    lambda_max_WaWaT = WaWaT_rpp.diagonal(dim1=-2, dim2=-1).sum(dim=-1).max()  # scalar tensor
    L = (2.0 / n_eff) * lambda_max_MscTMsc * lambda_max_WaWaT + 2.0 * lambda_prior
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
        H_struct_rnp = torch.einsum("nd,rdp->rnp", M_scaled_nd, beta_active)  # (R, N, P): linear
        pred_W_rnp = torch.einsum("rnp,rpq->rnq", H_struct_rnp, WaWaT_rpp)  # (R, N, P)
        err_W_rnp = pred_W_rnp - X_res_WaT_rnp  # (R, N, P)
        grad_active = (2.0 / n_eff) * torch.einsum("nd,rnp->rdp", M_scaled_nd, err_W_rnp)  # linear
        if lambda_prior > 0.0 and beta_prior_active is not None:
            grad_active = grad_active + 2.0 * lambda_prior * (beta_active - beta_prior_active)

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
    range_M_d: torch.Tensor,
    mean_total_count: float,
    noise_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derives NMF initialization for nominated metadata-capturing factors from OLS coefficients.

    Runs truncated SVD on the range-scaled OLS coefficient matrix (rows scaled by range_M_d)
    to extract dominant modes of metadata-gene co-variation in min-max scaled space. Applies
    a sign convention so the primary gene direction is positive, then ReLU-projects to
    non-negative and L1-normalizes. Beta columns are calibrated so that a cell at the maximum
    metadata value contributes approximately ``mean_total_count / n_components`` to its
    H_struct loading: ``sum(Beta_j) ≈ mean_total_count / n_components`` (since M_scaled=1
    at maximum for each covariate).

    Recommended workflow:
        1. Run ``StreamingOrdinaryLeastSquares(var_names_g=gene_names, n_targets=n_genes)``
           with ``x_ng=M_metadata`` and ``y_nk=X_genes`` for one epoch.
        2. Call ``.solve()`` to get ``ols_coeff_dg`` of shape (D, G).
        3. Pass ``ols_coeff_dg`` to this function.

    Args:
        ols_coeff_dg: OLS coefficient matrix of shape (D, G), in raw metadata units.
        n_metadata_programs: Number of metadata factors to initialize.
        n_replicates: Number of NMF replicates.
        n_components: Total NMF components k (for loading scale calibration).
        range_M_d: Global metadata range (max - min) of shape (D,). Used to convert OLS
            directions to min-max scaled space and to calibrate Beta scale.
        mean_total_count: Expected total UMI count per cell.
        noise_scale: Relative noise std for replicate diversity (default 0.05).

    Returns:
        W_meta_rkg: shape (R, n_metadata_programs, G) — nominated gene factor vectors.
        Beta_meta_rdk: shape (R, D, n_metadata_programs) — nominated Beta columns.
    """
    D, G = ols_coeff_dg.shape
    n_singular = min(n_metadata_programs, D)

    # Scale OLS rows by range_M_d to get scaled-space directions: dX/dM_scaled = range_M * dX/dM
    ols_coeff_dg_scaled = range_M_d.unsqueeze(1) * ols_coeff_dg  # (D, G)
    U, S, Vh = torch.linalg.svd(ols_coeff_dg_scaled, full_matrices=False)
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

    # Build base W and Beta for each nominated program.
    # Beta calibration: at full range (M_scaled=1 for all d), contribution = beta_j.sum().
    # Target: beta_j.sum() ≈ target_loading. Fallback: uniform target_loading / D per entry.
    W_base: list[torch.Tensor] = []
    Beta_base: list[torch.Tensor] = []
    target_loading = mean_total_count / max(n_components, 1)

    for j in range(n_metadata_programs):
        if j < n_singular:
            w_j = F.relu(Vh[j]) + 1e-8  # (G,)
            w_j = F.normalize(w_j, p=1, dim=0)

            beta_j = F.relu(U[:, j])  # (D,) non-negative, in scaled space
            full_range_loading = beta_j.sum().item()
            if full_range_loading > 1e-8:
                beta_j = beta_j * (target_loading / full_range_loading)
            else:
                beta_j = torch.ones(D, device=ols_coeff_dg.device) * (target_loading / max(D, 1))
        else:
            # More programs than SVD modes: random init for extras
            w_j = F.normalize(torch.rand(G, device=ols_coeff_dg.device) + 1e-8, p=1, dim=0)
            beta_j = torch.ones(D, device=ols_coeff_dg.device) * (target_loading / max(D, 1))

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

    The encoder is trained with a single signal: SmoothL1 matching loss against the FISTA
    solver's H_raw output. The FISTA solver is itself penalized (via ``lambda_align``) to
    produce H_raw solutions that are decorrelated from M, so the encoder indirectly learns
    decorrelated warm-starts by chasing H_solver.
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

        L_recon   = ||X - H_total * W||_F^2
        L_select  = lambda_select * sum_k ||Beta[:, k]||_2   (Group Lasso on Beta)

    The FISTA solver for H_raw minimizes an augmented objective that includes a covariance
    decorrelation penalty directly in the gradient steps::

        min_{H>=0} (1/2)||X_eff - H W||_F^2
                   + lambda_align * ||M_c^T (H - mu_H) / sqrt(n_valid)||_F^2

    where ``X_eff = X - H_struct @ W``, ``M_c`` is batch-centered scaled metadata with NaN
    rows zeroed, and ``mu_H`` is a bias-corrected EMA of the mean H_raw loading.  Dividing
    by ``sqrt(n_valid)`` makes ``lambda_align`` batch-size independent (comparable to
    ``E[m_c^2]`` rather than ``N * E[m_c^2]``).  The encoder then learns decorrelated
    warm-starts indirectly by chasing the FISTA H_solver via the SmoothL1 matching loss.

    Group Lasso on Beta achieves **factor-level sparsity**: only a few nominated factors
    become metadata-driven; the rest stay free.

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
    ``m_nd`` should contain **raw non-negative** metadata values (e.g., donor age in years).
    The model internally applies min-max scaling: ``M_scaled = (M - min) / (max - min)``,
    clipped to ``[0, 1]``. This guarantees ``H_struct = M_scaled * Beta >= 0`` (since
    ``Beta >= 0``), keeping ``H_total >= 0``. Provide ``metadata_min_d`` and
    ``metadata_max_d`` computed from the training set (or representative percentiles for
    robustness to outliers).

    **Recommended workflow:**

    .. code-block:: python

        # Step 1: pre-training OLS pass (one epoch)
        ols = StreamingOrdinaryLeastSquares(var_names_g=gene_names, n_targets=n_genes)
        # train ols with x_ng=M_metadata, y_nk=X_genes for one epoch
        ols_coeff_dg = ols.solve().numpy()   # (D, G)

        # Step 2: metadata statistics
        metadata_mean_d = M.mean(axis=0)    # (D,)
        metadata_min_d  = M.min(axis=0)     # (D,)
        metadata_max_d  = M.max(axis=0)     # (D,)

        # Step 3: train structured NMF
        model = AmortizedOnlineStructureAwareNMF(
            ...,
            n_metadata=D,
            metadata_mean_d=metadata_mean_d,
            metadata_min_d=metadata_min_d,
            metadata_max_d=metadata_max_d,
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
        metadata_mean_d: Precomputed global metadata mean, shape (D,). Used for
            covariance penalty centering in scaled space.
        metadata_min_d: Precomputed global metadata minimum, shape (D,). Required.
        metadata_max_d: Precomputed global metadata maximum, shape (D,). Required.
        n_metadata_programs: Number of nominated metadata factors. Defaults to
            ``2 * n_metadata``. Must be < min(k_values).
        ols_coeff_dg: OLS coefficient matrix (D, G) from StreamingOrdinaryLeastSquares.
            If None, nominated factors use random initialization.
        mean_total_count: Expected total UMI count per cell, used for Beta scale
            calibration. If None, a rough estimate of ``10 * min(k_values)`` is used.
        metadata_noise_scale: Relative noise std for replicate diversity (default 0.05).
        lambda_align: Strength of the covariance decorrelation penalty applied to H_raw
            inside the FISTA solver. The penalty is normalized by ``sqrt(n_valid)`` so its
            effective strength is batch-size independent. A value of 0.1–1.0 is a gentle
            nudge relative to the reconstruction term.
        lambda_select: Group Lasso strength on Beta columns.
        beta_lr: Upper bound on the Beta proximal gradient step size. The actual step is
            ``min(beta_lr, 1/L)`` where L is the per-batch Lipschitz constant of the
            Beta gradient. Setting ``beta_lr=1.0`` (the default) is equivalent to always
            using the theoretically optimal step size; smaller values slow Beta down.
        beta_n_iter: Number of proximal gradient iterations per forward call.
        lambda_prior: Strength of the L2 prior pulling Beta toward its OLS initialization.
            Adds ``lambda_prior * ||Beta_active - Beta_prior||_F^2`` to the Beta objective.
            Requires ``ols_coeff_dg``; without it, beta_prior is zero and the prior acts
            as an extra L2 regularizer toward zero. Default 0 disables the prior.
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
        metadata_mean_d: list[float] | np.ndarray,
        metadata_min_d: list[float] | np.ndarray,
        metadata_max_d: list[float] | np.ndarray,
        n_metadata_programs: int | None = None,
        ols_coeff_dg: np.ndarray | None = None,
        mean_total_count: float | None = None,
        metadata_noise_scale: float = 0.05,
        lambda_align: float = 0.1,
        lambda_select: float = 0.05,
        beta_lr: float = 1.0,
        beta_n_iter: int = 20,
        lambda_prior: float = 0.0,
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
        self.lambda_prior = lambda_prior

        if lambda_prior > 0.0 and ols_coeff_dg is None:
            warnings.warn(
                "lambda_prior > 0 but ols_coeff_dg is None: beta_prior will be all zeros, "
                "so the prior acts as an extra L2 regularizer toward zero rather than toward "
                "the OLS estimate. Consider providing ols_coeff_dg or setting lambda_prior=0.",
                UserWarning,
                stacklevel=2,
            )

        # Store init data for reset_parameters
        self._ols_coeff_dg: torch.Tensor | None = (
            torch.from_numpy(np.array(ols_coeff_dg)).float() if ols_coeff_dg is not None else None
        )
        self._metadata_mean_d_init = torch.from_numpy(np.array(metadata_mean_d)).float()
        self._metadata_min_d_init = torch.from_numpy(np.array(metadata_min_d)).float()
        self._metadata_max_d_init = torch.from_numpy(np.array(metadata_max_d)).float()
        self._mean_total_count = mean_total_count
        self._metadata_noise_scale = metadata_noise_scale

        # Replace encoder with metadata-augmented version
        self.encoder = MetadataAugmentedLoadingsEncoder(n_genes=g, latent_dim=latent_dim, n_metadata=n_metadata)

        # Fixed global metadata min-max scaling stats (never updated during training)
        range_M = (self._metadata_max_d_init - self._metadata_min_d_init).clamp(min=1e-8)
        mu_M_scaled = (self._metadata_mean_d_init - self._metadata_min_d_init) / range_M
        self.register_buffer("min_M_global_d", self._metadata_min_d_init.clone())
        self.register_buffer("range_M_global_d", range_M)
        self.register_buffer("mu_M_scaled_d", mu_M_scaled)

        # Per-k Beta and H_raw EMA buffers
        for k in k_values:
            self.register_buffer(f"beta_{k}_rdk", torch.zeros(r, n_metadata, k))
            self.register_buffer(f"beta_prior_{k}_rdk", torch.zeros(r, n_metadata, n_metadata_programs))
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
            prior_buf_name = f"beta_prior_{k}_rdk"
            if hasattr(self, prior_buf_name):
                getattr(self, prior_buf_name).zero_()

        if self._ols_coeff_dg is None:
            # No OLS: nominated Beta columns get small uniform random values
            for k in self.k_values:
                buf_name = f"beta_{k}_rdk"
                if not hasattr(self, buf_name):
                    continue
                getattr(self, buf_name)[:, :, : self.n_metadata_programs].uniform_(0.01, 0.1)
            return

        # OLS-based initialization
        range_M = getattr(
            self,
            "range_M_global_d",
            (self._metadata_max_d_init - self._metadata_min_d_init).clamp(min=1e-8),
        )

        for k in self.k_values:
            D_rkg = getattr(self, f"D_{k}_rkg")
            device = D_rkg.device
            mean_count = float(self._mean_total_count) if self._mean_total_count is not None else 10.0 * k

            W_meta_rkg, Beta_meta_rdk = compute_metadata_nmf_init(
                ols_coeff_dg=self._ols_coeff_dg.to(device),
                n_metadata_programs=self.n_metadata_programs,
                n_replicates=self.r,
                n_components=k,
                range_M_d=range_M.to(device),
                mean_total_count=mean_count,
                noise_scale=self._metadata_noise_scale,
            )

            D_rkg[:, : self.n_metadata_programs, :].copy_(W_meta_rkg)
            getattr(self, f"beta_{k}_rdk")[:, :, : self.n_metadata_programs].copy_(Beta_meta_rdk)
            getattr(self, f"beta_prior_{k}_rdk").copy_(Beta_meta_rdk)

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
        and updates W via FISTA. Computes the encoder loss (SmoothL1 warm-start loss) but
        does NOT call encoder.backward() — that is handled by Lightning.

        Args:
            x_ng: Gene counts (N, G).
            k: The k value to run.
            n_iterations: Solver iterations for both H_raw and W.
            m_nd: Metadata matrix (N, D), raw uncentered non-negative values.

        Returns:
            dict with keys:
                ``loss``: encoder loss (SmoothL1 warm-start loss), has gradients.
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

        # --- Step 0: min-max scale metadata to [0, 1] ---
        nan_mask_n = m_nd.isnan().any(dim=1)  # (N,) True where any metadata dim is NaN
        n_valid = int((~nan_mask_n).sum().item())
        m_scaled_nd = ((m_nd - self.min_M_global_d) / self.range_M_global_d).clamp(0.0, 1.0)  # (N, D)
        m_scaled_nd = m_scaled_nd.nan_to_num(0.0)  # NaN cells → 0; H_struct=0, Beta grad=0

        # --- Step 1: encoder warm-start for H_raw (has gradients) ---
        H_raw_warm_rnk = self.encoder(x_ng, W_rkg.detach(), m_scaled_nd)

        # --- Step 2: structural component H_struct = M_scaled @ Beta (linear, non-negative) ---
        with torch.no_grad():
            H_struct_rnk = torch.einsum("nd,rdk->rnk", m_scaled_nd, beta_rdk)

            # --- Step 3 & 4: FISTA solver setup (no (R, N, G) materialization) ---
            # W @ X_eff.T = W @ X.T - (W @ W.T) @ H_struct.T, avoiding explicit X_eff.
            wwT_rkk = torch.einsum("rkg,rhg->rkh", W_rkg, W_rkg)  # (R, K, K)
            WxT_rkn = torch.einsum("rkg,ng->rkn", W_rkg, x_ng)  # (R, K, N)
            wwT_Hstruct_rkn = torch.einsum("rkh,rnh->rkn", wwT_rkk, H_struct_rnk)  # (R, K, N)
            wxT_eff_rkn = WxT_rkn - wwT_Hstruct_rkn  # (R, K, N)

            # --- Precompute M_c and mu_H for the structure-aware FISTA solver ---
            M_c_nd = m_scaled_nd - self.mu_M_scaled_d  # (N, D), centered scaled metadata
            M_c_nd = M_c_nd.masked_fill(nan_mask_n.unsqueeze(1), 0.0)  # zero out NaN-cell rows
            # Normalize by sqrt(n_valid) so the penalty scales as an average over valid cells,
            # making lambda_align batch-size independent (comparable to E[m_c^2] * 2*lambda).
            M_c_nd = M_c_nd / math.sqrt(max(n_valid, 1))
            ema_rho = float(np.exp(-1.0 / self.n_batches_for_forgetting_momentum))
            if self._n_ema_updates > 0:
                bias_correction = max(1.0 - ema_rho**self._n_ema_updates, 1e-8)
                mu_H_corrected_rk = mu_H_ema_rk / bias_correction
            else:
                mu_H_corrected_rk = torch.zeros_like(mu_H_ema_rk)

        H_raw_solver_kn = solve_structure_aware_nnls_fista(
            AtA=wwT_rkk,
            AtB=wxT_eff_rkn,
            initial_x=H_raw_warm_rnk.detach().transpose(-2, -1),  # (R, K, N)
            M_c_nd=M_c_nd,
            mu_H_rk=mu_H_corrected_rk,
            lambda_align=self.lambda_align,
            max_iter=n_iterations,
        )
        H_raw_solver_rnk = H_raw_solver_kn.transpose(-2, -1)  # (R, N, K)

        with torch.no_grad():
            # --- Step 5: Beta update via Group Lasso proximal gradient ---
            beta_prior_rdk = getattr(self, f"beta_prior_{k}_rdk")
            beta_rdk_updated = update_beta_group_lasso(
                H_raw_rnk=H_raw_solver_rnk,
                W_rkg=W_rkg,
                X_ng=x_ng,
                M_scaled_nd=m_scaled_nd,
                beta_rdk=beta_rdk,
                lambda_select=self.lambda_select,
                beta_lr=self.beta_lr,
                n_iter=self.beta_n_iter,
                n_metadata_programs=self.n_metadata_programs,
                n_valid=n_valid,
                beta_prior_active=beta_prior_rdk,
                lambda_prior=self.lambda_prior,
            )
            setattr(self, f"beta_{k}_rdk", beta_rdk_updated)

            # --- Step 6: recompute H_struct and H_total ---
            H_struct_updated_rnk = torch.einsum("nd,rdk->rnk", m_scaled_nd, beta_rdk_updated)
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

        # --- Step 9: update mu_H_ema with solver H_raw over valid cells only ---
        with torch.no_grad():
            if n_valid > 0:
                batch_mean_rk = H_raw_solver_rnk[:, ~nan_mask_n, :].mean(dim=1)  # (R, K)
            else:
                batch_mean_rk = H_raw_solver_rnk.mean(dim=1)  # fallback: all cells
            mu_H_ema_new = ema_rho * mu_H_ema_rk + (1.0 - ema_rho) * batch_mean_rk
            setattr(self, f"mu_H_ema_{k}_rk", mu_H_ema_new)

        # --- Step 10: encoder loss ---
        # The encoder is trained purely as a warm-start predictor; structure is enforced
        # directly inside the FISTA solver via the covariance penalty on H_solver.
        encoder_loss = self.encoder_loss_fn(H_raw_warm_rnk.contiguous(), H_raw_solver_rnk.detach().contiguous())

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

        m_scaled_nd = ((m_nd - self.min_M_global_d) / self.range_M_global_d).clamp(0.0, 1.0)  # (N, D)
        m_scaled_nd = m_scaled_nd.nan_to_num(0.0)  # NaN cells → 0; H_struct=0 for missing metadata
        nmf_reconstruction_errors = []
        for k in self.k_values:
            W_rkg = getattr(self, f"D_{k}_rkg")
            beta_rdk = getattr(self, f"beta_{k}_rdk")
            H_raw_warm_rnk = self.encoder(x_ng, W_rkg.detach(), m_scaled_nd)
            H_struct_rnk = torch.einsum("nd,rdk->rnk", m_scaled_nd, beta_rdk.detach())
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
