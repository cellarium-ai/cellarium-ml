# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import cellarium.ml

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.strategies import DDPStrategy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.transforms import Filter, NormalizeTotal
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

warnings.filterwarnings("ignore")


def _get_logger():
    """Get a logger that works well in both regular Python and Jupyter notebooks."""
    logger = logging.getLogger(__name__)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(name)s - %(levelname)s - %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent propagation to avoid duplicate messages
        logger.propagate = False

    return logger


logger = _get_logger()


def nmf_frobenius_loss(x_ng: torch.Tensor, loadings_nk: torch.Tensor, factors_kg: torch.Tensor):
    # compute prediction error as the frobenius norm
    return F.mse_loss(torch.matmul(loadings_nk, factors_kg), x_ng, reduction="sum")


def solve_nnls_fista(A, B, max_iter=1000, tol=1e-6):
    """
    FISTA algorithm for NNLS solving Ax = B for x >= 0

    Args:
        A: Coefficient matrix of shape (..., m, n)
        B: Right-hand side of shape (..., m, k)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        x: Solution of shape (..., n, k) with x >= 0
    """
    # Handle batch dimensions
    *batch_dims, m, n = A.shape
    *batch_dims_B, m_B, k = B.shape

    assert m == m_B, f"Incompatible dimensions: A has {m} rows, B has {m_B} rows"

    # Precompute AtA and AtB for efficiency
    AtA = A.transpose(-2, -1) @ A  # (..., n, n)
    AtB = A.transpose(-2, -1) @ B  # (..., n, k)

    # Compute Lipschitz constant (largest eigenvalue of AtA)
    eigenvals = torch.linalg.eigvals(AtA).real  # (..., n)
    L = eigenvals.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (..., 1, 1)

    # Initialize variables
    x = torch.zeros(*batch_dims, n, k, device=A.device, dtype=A.dtype)
    y = x.clone()
    t = torch.ones(*batch_dims, 1, 1, device=A.device, dtype=A.dtype)

    for i in range(max_iter):
        x_old = x.clone()

        # Gradient step: grad = AtA @ y - AtB
        grad = AtA @ y - AtB
        x_new = torch.clamp(y - grad / L, min=0)

        # Momentum update
        t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

        # Check convergence
        if torch.norm(x - x_old) < tol:
            break

    return x


@torch.no_grad()
def nmf_torch_update_loadings_hals(
    x_ng: torch.Tensor,
    w_rkg: torch.Tensor,
    h_rnk: torch.Tensor,
    max_iter: int = 200,
    h_tol: float = 0.05,
    # l1_reg_H: float = 0.0,
    # l2_reg_H: float = 0.0,
) -> None:
    # https://github.com/lilab-bcb/nmf-torch/blob/188747777c30e72626a14fe9b9d57c0ffda3efbb/
    # nmf/nmf_models/_nmf_online_hals.py#L64C13-L88C34
    # Online update H.
    assert x_ng.shape[0] == h_rnk.shape[-2]
    assert x_ng.shape[1] == w_rkg.shape[-1]
    assert w_rkg.shape[-2] == h_rnk.shape[-1]

    # wwT_rkk = torch.bmm(w_rkg, w_rkg.transpose(1, 2))
    wwT_rkk = torch.einsum("rkg,rhg->rkh", w_rkg, w_rkg)
    # xwT_rnk = torch.bmm(x_ng.unsqueeze(0), w_rkg.transpose(1, 2))
    xwT_rnk = torch.einsum("ng,rkg->rnk", x_ng, w_rkg)

    nonconverged_logic_r = torch.ones(h_rnk.shape[0], device=h_rnk.device).bool()

    for i in range(max_iter):
        cur_max_r = torch.zeros(h_rnk.shape[0], device=h_rnk.device)

        # TODO this does unnecessary compute on reps that have already converged
        # but would it be faster this way if compiled with torch.compile()?
        # alternatively could do the compute only on non-converged reps by indexing
        for k in range(h_rnk.shape[-1]):
            # numer_rn1 = xwT_rnk[..., k] - torch.bmm(h_rnk, wwT_rkk[..., k])
            numer_rn = xwT_rnk[..., k] - torch.einsum("rnk,rk->rn", h_rnk, wwT_rkk[..., k])
            # if l1_reg_H > 0.0:
            #     numer -= l1_reg_H
            # if l2_reg_H > 0.0:
            #     denom = WWT[k, k] + l2_reg_H
            #     hvec = h[:, k] * (WWT[k, k] / denom) + numer / denom
            # else:
            h_rn = h_rnk[..., k]
            hvec_rn = h_rn + numer_rn / wwT_rkk[:, k, k].unsqueeze(-1)
            if torch.isnan(hvec_rn).sum() > 0:
                # hvec_rn1[:] = 0.0  # divide zero error: set hvec to 0
                hvec_rn.fill_(0.0)
            else:
                hvec_rn = torch.clamp(hvec_rn, min=0.0)
            cur_max_r = torch.max(cur_max_r, (h_rn - hvec_rn).abs().max())
            h_rnk[nonconverged_logic_r, :, k] = hvec_rn[nonconverged_logic_r, ...]

        # remove replicates meeting stop criteria from further updates
        nonconverged_logic_r = (
            nonconverged_logic_r  # once converged, always converged
            & ((cur_max_r / h_rnk.mean(dim=(-2, -1))) >= h_tol)
        )

        # if j + 1 < max_iter and cur_max / h_rnk.mean() < h_tol:
        if nonconverged_logic_r.sum() == 0:
            print(f"NMF HALS loadings update converged in {i + 1} iterations.")
            break

    if i == max_iter - 1:
        print("NMF HALS loadings update reached max iterations without convergence.")


@torch.compile
@torch.no_grad()
def nmf_torch_update_loadings_hals_compiled(
    x_ng: torch.Tensor,
    w_rkg: torch.Tensor,
    h_rnk: torch.Tensor,
    max_iter: int = 200,
    h_tol: float = 0.05,
) -> tuple[torch.Tensor, int]:
    """
    Compiled version of HALS loadings update.
    Returns updated h_rnk and number of iterations to convergence.
    """
    assert x_ng.shape[0] == h_rnk.shape[-2]
    assert x_ng.shape[1] == w_rkg.shape[-1]
    assert w_rkg.shape[-2] == h_rnk.shape[-1]

    wwT_rkk = torch.einsum("rkg,rhg->rkh", w_rkg, w_rkg)
    xwT_rnk = torch.einsum("ng,rkg->rnk", x_ng, w_rkg)

    # Use a fixed number of iterations instead of dynamic convergence
    # This makes the function more amenable to compilation
    for i in range(max_iter):
        for k in range(h_rnk.shape[-1]):
            numer_rn = xwT_rnk[..., k] - torch.einsum("rnk,rk->rn", h_rnk, wwT_rkk[..., k])
            h_rn = h_rnk[..., k]
            
            # Avoid division by zero using torch.where instead of manual checks
            denom = wwT_rkk[:, k, k].unsqueeze(-1)
            hvec_rn = torch.where(
                denom > 1e-12,
                torch.clamp(h_rn + numer_rn / denom, min=0.0),
                torch.zeros_like(h_rn)
            )
            h_rnk[..., k] = hvec_rn

    return h_rnk, max_iter


@torch.no_grad()
def nmf_torch_update_loadings_hals_with_compile(
    x_ng: torch.Tensor,
    w_rkg: torch.Tensor,
    h_rnk: torch.Tensor,
    max_iter: int = 200,
    h_tol: float = 0.05,
) -> None:
    """
    Wrapper that uses compiled function with convergence checking.
    """
    # Use smaller chunks of iterations with convergence checks
    chunk_size = 10
    total_iterations = 0
    
    while total_iterations < max_iter:
        h_old = h_rnk.clone()
        current_chunk = min(chunk_size, max_iter - total_iterations)
        
        # Run compiled function for a chunk of iterations
        h_rnk_new, _ = nmf_torch_update_loadings_hals_compiled(
            x_ng, w_rkg, h_rnk, current_chunk, h_tol
        )
        h_rnk.copy_(h_rnk_new)
        
        total_iterations += current_chunk
        
        # Check convergence
        max_change = (h_rnk - h_old).abs().max()
        mean_h = h_rnk.mean(dim=(-2, -1))
        relative_change = (max_change / torch.clamp(mean_h.max(), min=1e-12))
        
        if relative_change < h_tol:
            # print(f"NMF HALS loadings update converged in {total_iterations} iterations.")
            break
    # else:
    #     print("NMF HALS loadings update reached max iterations without convergence.")


@torch.no_grad()
def nmf_torch_update_factors_hals(
    w_rkg: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    max_iter: int = 200,
    w_tol: float = 0.05,
) -> None:
    # https://github.com/lilab-bcb/nmf-torch/blob/188747777c30e72626a14fe9b9d57c0ffda3efbb/
    # nmf/nmf_models/_nmf_online_hals.py#L106C13-L127C26

    nonconverged_logic_r = torch.ones(A_rkk.shape[0], device=A_rkk.device).bool()

    # Online update W.
    for j in range(max_iter):
        cur_max_r = torch.zeros(A_rkk.shape[0], device=A_rkk.device)

        # TODO this does unnecessary compute on reps that have already converged
        # but would it be faster this way if compiled with torch.compile()?
        # alternatively could do the compute only on non-converged reps by indexing
        for k in range(A_rkk.shape[-1]):
            # numer_r1g = B_rkg[:, k, :] - torch.bmm(A_rkk[:, k, :], w_rkg)
            numer_rg = B_rkg[:, k, :] - torch.einsum("rk,rkg->rg", A_rkk[:, k, :], w_rkg)
            # if l1_reg_W > 0.0:
            #     numer -= l1_reg_W
            # if l2_reg_W > 0.0:
            #     denom = A[k, k] + l2_reg_W
            #     w_new = self.W[k, :] * (A[k, k] / denom) + numer / denom
            # else:
            w_new_rg = w_rkg[:, k, :] + numer_rg / A_rkk[:, k, k].unsqueeze(-1)
            if torch.isnan(w_new_rg).sum() > 0:
                # w_new_r1g[:] = 0.0 # divide zero error: set w_new to 0
                w_new_rg.fill_(0.0)
            else:
                w_new_rg = torch.clamp(w_new_rg, min=0.0)
            cur_max_r = torch.max(cur_max_r, (w_rkg[:, k, :] - w_new_rg).abs().max())
            w_rkg[nonconverged_logic_r, k, :] = w_new_rg[nonconverged_logic_r, ...]

        # remove replicates meeting stop criteria from further updates
        nonconverged_logic_r = (
            nonconverged_logic_r  # once converged, always converged
            & ((cur_max_r / w_rkg.mean(dim=(-2, -1))) >= w_tol)
        )

        # if j + 1 < max_iter and cur_max_r / w_rkg.mean() < w_tol:
        if nonconverged_logic_r.sum() == 0:
            print(f"NMF HALS factors update converged in {j + 1} iterations.")
            break

    if j == max_iter - 1:
        print("NMF HALS factors update reached max iterations without convergence.")


@torch.compile
@torch.no_grad()
def nmf_torch_update_factors_hals_compiled(
    w_rkg: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    max_iter: int = 200,
    w_tol: float = 0.05,
) -> tuple[torch.Tensor, int]:
    """
    Compiled version of HALS factors update.
    Returns updated w_rkg and number of iterations to convergence.
    """
    # Use a fixed number of iterations instead of dynamic convergence
    # This makes the function more amenable to compilation
    for j in range(max_iter):
        for k in range(A_rkk.shape[-1]):
            numer_rg = B_rkg[:, k, :] - torch.einsum("rk,rkg->rg", A_rkk[:, k, :], w_rkg)
            w_new_rg = w_rkg[:, k, :] + numer_rg / A_rkk[:, k, k].unsqueeze(-1)
            
            # Avoid division by zero using torch.where instead of manual checks
            w_new_rg = torch.where(
                torch.isnan(w_new_rg),
                torch.zeros_like(w_new_rg),
                torch.clamp(w_new_rg, min=0.0)
            )
            w_rkg[:, k, :] = w_new_rg

    return w_rkg, max_iter


@torch.no_grad()
def nmf_torch_update_factors_hals_with_compile(
    w_rkg: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    max_iter: int = 200,
    w_tol: float = 0.05,
) -> None:
    """
    Wrapper that uses compiled function with convergence checking.
    """
    # Use smaller chunks of iterations with convergence checks
    chunk_size = 10
    total_iterations = 0
    
    while total_iterations < max_iter:
        w_old = w_rkg.clone()
        current_chunk = min(chunk_size, max_iter - total_iterations)
        
        # Run compiled function for a chunk of iterations
        w_rkg_new, _ = nmf_torch_update_factors_hals_compiled(
            w_rkg, A_rkk, B_rkg, current_chunk, w_tol
        )
        w_rkg.copy_(w_rkg_new)
        
        total_iterations += current_chunk
        
        # Check convergence
        max_change = (w_rkg - w_old).abs().max()
        mean_w = w_rkg.mean(dim=(-2, -1))
        relative_change = (max_change / torch.clamp(mean_w.max(), min=1e-12))
        
        if relative_change < w_tol:
            # print(f"NMF HALS factors update converged in {total_iterations} iterations.")
            break
    # else:
    #     print("NMF HALS factors update reached max iterations without convergence.")


# def efficient_ols_all_cols(X, Y, XtX, XtY, normalize_y=True):
#     """
#     https://github.com/dylkot/cNMF/blob/7833a75484169cf448f8956224447cb110f4ba3d/src/cnmf/cnmf.py#L55
#     Solve OLS: Beta = (X^T X)^{-1} X^T Y,
#     accumulating X^T X and X^T Y in row-batches.

#     Optionally mean/variance-normalize each column of Y *globally*
#     (using the entire dataset's mean/var), while still only converting
#     each row-batch to dense on-the-fly.

#     Parameters
#     ----------
#     X : np.ndarray, shape (n_samples, n_predictors)
#         Predictor matrix.
#     Y : np.ndarray or scipy.sparse.spmatrix, shape (n_samples, n_targets)
#         Outcomes. Each column is one target variable.
#     batch_size : int
#         Number of rows to process per chunk.
#     normalize_y : bool
#         If True, compute global mean & var of Y columns, then subtract mean
#         and divide by std for each batch.

#     Returns
#     -------
#     Beta : np.ndarray, shape (n_predictors, n_targets)
#         The OLS coefficients for each target.
#     """

#     def _get_mean_var(X):
#         scaler = StandardScaler(with_mean=False)
#         scaler.fit(X)
#         return (scaler.mean_, scaler.var_)

#     # -- Optionally compute global mean & variance of Y columns
#     if normalize_y:
#         meanY, varY = _get_mean_var(Y)

#         # Avoid zero or near-zero std
#         eps = 1e-12
#         varY[varY < eps] = eps
#         stdY = np.sqrt(varY)

#     # -- Optionally apply normalization
#     if normalize_y:
#         Y = (Y - meanY) / stdY

#     # -- Accumulate partial sums
#     XtX += X.T @ X
#     XtY += X.T @ Y

#     # -- Solve the normal equations
#     #    Beta = (X^T X)^(-1) X^T Y
#     #    Using lstsq for stability.
#     Beta, residuals, rank, s = np.linalg.lstsq(XtX, XtY, rcond=None)
#     return Beta, XtX, XtY


def find_local_minima(mean_neighbor_distance_m: torch.Tensor, n_bins: int = 100) -> list[float]:
    """
    Helper function to find local minima in a histogram of distance values.

    Args:
        mean_neighbor_distance_m: 1D tensor of distance values
        n_bins: Number of bins for the histogram (default: 100)

    Returns:
        List of distance values corresponding to local minima (bin centers)
    """
    # Convert to numpy for histogram computation
    distances = mean_neighbor_distance_m.cpu().numpy()

    # Create histogram
    counts, bin_edges = np.histogram(distances, bins=n_bins, range=(0, 1))

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth the histogram slightly to reduce noise
    if len(counts) >= 3:
        # Simple smoothing: each bin becomes average of itself and neighbors
        smoothed_counts = counts.copy()
        for i in range(1, len(counts) - 1):
            smoothed_counts[i] = (counts[i - 1] + counts[i] + counts[i + 1]) / 3
        counts = smoothed_counts

    # Find local minima in the histogram counts
    # A local minimum is a point that is smaller than or equal to both neighbors
    # but at least one neighbor must be strictly larger
    local_minima_indices = []

    for i in range(1, len(counts) - 1):
        left_count = counts[i - 1]
        center_count = counts[i]
        right_count = counts[i + 1]

        # Check if this is a local minimum
        if (
            center_count <= left_count
            and center_count <= right_count
            and (center_count < left_count or center_count < right_count)
        ):
            local_minima_indices.append(i)

    # Convert indices to actual distance values (bin centers)
    candidate_values = [bin_centers[i] for i in local_minima_indices]

    # Also consider the first and last bins as potential candidates
    # if they are at the boundaries and could be minima
    if len(counts) > 1:
        # Add first bin if it's a potential minimum
        if counts[0] <= counts[1]:
            candidate_values.insert(0, bin_centers[0])

        # Add last bin if it's a potential minimum
        if counts[-1] <= counts[-2]:
            candidate_values.append(bin_centers[-1])

    # Remove duplicates and sort
    candidate_values = sorted(list(set(candidate_values)))

    # Filter out values that are too close to each other (within 5% of range)
    min_distance = 0.05
    filtered_candidates: list[float] = []
    for val in candidate_values:
        if not any(abs(val - existing) < min_distance for existing in filtered_candidates):
            filtered_candidates.append(val)

    candidate_values = filtered_candidates

    return candidate_values


def compute_loadings(
    x_ng: torch.Tensor,
    factors_rkg: torch.Tensor,
    n_iterations: int,
    alpha_tol: float = 1e-5,
) -> torch.Tensor:
    """
    Algorithm 1 step 4 from Mairal et al. [1] for computing the loadings.

    Args:
        x_ng: The data.
        factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The max number of iterations to perform.
        alpha_tol: The tolerance for the change in alpha for stopping.

    Returns:
        The computed loadings.
    """
    n, _ = x_ng.shape
    r, n_factors, g = factors_rkg.shape

    # initialization
    alpha_rnk = torch.rand((r, n, n_factors), device=factors_rkg.device).abs()
    # alpha_rnk = torch.zeros((r, n, n_factors), device=factors_rkg.device)

    with torch.no_grad():
        for rep in range(r):
            D_kg = factors_rkg[rep]  # (k, g)

            # scaled random based on data statistics
            data_scale = torch.mean(x_ng)
            factor_scale = torch.mean(torch.abs(D_kg))
            if factor_scale > 1e-10:
                init_scale = data_scale / (factor_scale * n_factors)
                alpha_rnk[rep] = init_scale * torch.rand((n, n_factors), device=factors_rkg.device)
            else:
                # last resort - simple uniform random
                alpha_rnk[rep] = 0.1 * torch.rand((n, n_factors), device=factors_rkg.device)

    alpha_buffer_rnk = alpha_rnk.clone()

    DDT = torch.bmm(factors_rkg, factors_rkg.transpose(1, 2))
    xDT = torch.bmm(x_ng.expand(r, n, g), factors_rkg.transpose(1, 2))

    for i in range(n_iterations):
        for k in range(n_factors):
            scalar = DDT[:, k, k].view(r, 1, 1)
            a_rk1 = DDT[:, :, k].unsqueeze(2)
            b_rn1 = xDT[:, :, k].unsqueeze(2)

            # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
            u_r1g = torch.clamp(
                alpha_rnk[:, :, k].unsqueeze(2) + (b_rn1 - torch.bmm(alpha_rnk, a_rk1)) / scalar,
                min=0.0,
            )
            alpha_rnk[:, :, k] = u_r1g.squeeze(2)

        alpha_max_diff = (
            F.mse_loss(alpha_rnk, alpha_buffer_rnk, reduction="none").sum(dim=[-2, -1])
            / alpha_rnk.square().sum(dim=[-2, -1])
        ).max()
        if alpha_max_diff <= alpha_tol:
            break
        alpha_buffer_rnk = alpha_rnk.clone()

    return alpha_rnk


def compute_factors(
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    factors_rkg: torch.Tensor,
    n_iterations: int,
    D_tol: float = 2e-5,
) -> torch.Tensor:
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        A_rkk: Mairal's matrix A.
        B_rkg: Mairal's matrix B.
        factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
        D_tol: The tolerance for the change in D for stopping.

    Returns:
        The updated dictionary.
    """
    factors_buffer_rkg = factors_rkg.clone()
    updated_factors_rkg = factors_rkg.clone()
    r, n_factors, g = factors_rkg.shape

    for i in range(n_iterations):
        for k in range(n_factors):
            scalar = A_rkk[:, k, k].view(r, 1, 1)
            a_r1k = A_rkk[:, k, :].unsqueeze(1)
            b_r1g = B_rkg[:, k, :].unsqueeze(1)

            # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
            u_r1g = torch.clamp(
                updated_factors_rkg[:, k, :].unsqueeze(1) + (b_r1g - torch.bmm(a_r1k, updated_factors_rkg)) / scalar,
                min=0.0,
            )

            updated_factors_r1g = u_r1g / torch.clamp(
                torch.linalg.vector_norm(u_r1g, ord=2, dim=-1, keepdim=True), min=1.0
            )
            updated_factors_rkg[:, k, :] = updated_factors_r1g.squeeze(1)

        D_max_diff = (
            F.mse_loss(updated_factors_rkg, factors_buffer_rkg, reduction="none").sum(dim=[-2, -1])
            / updated_factors_rkg.square().sum(dim=[-2, -1])
        ).max()
        if D_max_diff <= D_tol:
            break
        factors_buffer_rkg = updated_factors_rkg.clone()

    return updated_factors_rkg


# def compute_factors_nmf_torch_online_mu(
#     A_rkk: torch.Tensor,
#     B_rkg: torch.Tensor,
#     factors_rkg: torch.Tensor,
#     n_iterations: int,
#     D_tol: float = 2e-5,
# ) -> torch.Tensor:
#     updated_factors_rkg = factors_rkg

#     for _ in range(n_iterations):
#         denom_rkg = torch.bmm(A_rkk, factors_rkg)
#         rates_rkg = B_rkg / torch.clamp(denom_rkg, min=1e-10)
#         rates_rkg[denom_rkg < 1e-10] = 0.0
#         cur_max = (torch.abs(1.0 - rates_rkg) * updated_factors_rkg).max()
#         updated_factors_rkg = factors_rkg * rates_rkg
#         if cur_max <= D_tol:
#             break

#     return updated_factors_rkg


def online_dictionary_update_nmf_torch_hals(
    x_ng: torch.Tensor,
    factors_rkg: torch.Tensor,
    loadings_rnk: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    n_iterations: int = 200,
    alpha_tol: float = 0.05,
    D_tol: float = 0.05,
) -> dict[str, torch.Tensor]:
    """
    Algorithm adapted from the nmf-torch github library.

    Args:
        x_ng: The data.
        factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
        loadings_rnk: The matrix of cell loadings (Mairal's coefficients alpha).
        A_rkk: Mairal's matrix A.
        B_rkg: Mairal's matrix B.
        n_iterations: The number of iterations to perform.
        alpha_tol: The tolerance for the change in alpha for stopping.
        D_tol: The tolerance for the change in D for stopping.

    Returns:
        dict with keys:
            "factors_rkg": The updated dictionary factors_rkg.
            "A_rkk": The updated matrix A.
            "B_rkg": The updated matrix B.
    """

    n, g = x_ng.shape
    r, _, _ = factors_rkg.shape

    # TODO: need access to local latent loadings_rnk for nmf-torch hals update

    # inplace update loadings_rnk
    nmf_torch_update_loadings_hals_with_compile(
        x_ng=x_ng,
        w_rkg=factors_rkg,
        h_rnk=loadings_rnk,
        max_iter=n_iterations,
        h_tol=alpha_tol,
    )

    with torch.no_grad():
        # update A and B, Mairal Algorithm 1 step 5 and 6
        A_rkk = A_rkk + torch.bmm(loadings_rnk.transpose(1, 2), loadings_rnk) / n
        B_rkg = B_rkg + torch.bmm(loadings_rnk.transpose(1, 2), x_ng.expand(r, n, g)) / n

    # inplace update factors_rkg
    nmf_torch_update_factors_hals_with_compile(
        w_rkg=factors_rkg,
        A_rkk=A_rkk,
        B_rkg=B_rkg,
        max_iter=n_iterations,
        w_tol=D_tol,
    )

    return {"factors_rkg": factors_rkg, "A_rkk": A_rkk, "B_rkg": B_rkg}


def online_dictionary_update_mairal(
    x_ng: torch.Tensor,
    factors_rkg: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    n_iterations: int = 100,
    alpha_tol: float = 1e-5,
    D_tol: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """
    Algorithm 1 from Mairal et al. [1] for online dictionary learning.

    Args:
        x_ng: The data.
        factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
        A_rkk: Mairal's matrix A.
        B_rkg: Mairal's matrix B.
        n_iterations: The number of iterations to perform.
        alpha_tol: The tolerance for the change in alpha for stopping.
        D_tol: The tolerance for the change in D for stopping.

    Returns:
        dict with keys:
            "factors_rkg": The updated dictionary factors_rkg.
            "A_rkk": The updated matrix A.
            "B_rkg": The updated matrix B.
    """

    n, g = x_ng.shape
    r, _, _ = factors_rkg.shape

    # TODO benchmark compute_loadings vs solve_nnls_fista
    # update loadings, Mairal Algorithm 1 step 4
    # loadings_rnk = compute_loadings(
    #     x_ng=x_ng,
    #     factors_rkg=factors_rkg,
    #     n_iterations=n_iterations,
    #     alpha_tol=alpha_tol,
    # )
    loadings_rnk = solve_nnls_fista(
        factors_rkg.transpose(1, 2), x_ng.t(), tol=alpha_tol, max_iter=n_iterations
    ).transpose(1, 2)

    with torch.no_grad():
        # update A and B, Mairal Algorithm 1 step 5 and 6
        A_rkk = A_rkk + torch.bmm(loadings_rnk.transpose(1, 2), loadings_rnk) / n
        B_rkg = B_rkg + torch.bmm(loadings_rnk.transpose(1, 2), x_ng.expand(r, n, g)) / n

        # TODO benchmark compute_factors vs compute_factors_nmf_torch_online_mu
        # update D, Mairal Algorithm 1 step 7
        updated_factors_rkg = compute_factors(
            factors_rkg=factors_rkg,
            A_rkk=A_rkk,
            B_rkg=B_rkg,
            n_iterations=n_iterations,
            D_tol=D_tol,
        )
        # updated_factors_rkg = compute_factors_nmf_torch_online_mu(
        #     factors_rkg=factors_rkg,
        #     A_rkk=A_rkk,
        #     B_rkg=B_rkg,
        #     n_iterations=n_iterations,
        #     D_tol=D_tol,
        # )

    return {"factors_rkg": updated_factors_rkg, "A_rkk": A_rkk, "B_rkg": B_rkg}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class NMFInit:
    @staticmethod
    def __call__(x: torch.Tensor, k: int, transformed_data_mean: float | None = None) -> None:
        """Modify the values of x in place as a way to initialize a dictionary factor matrix for NMF."""
        pass


class NMFInitSklearnRandom(NMFInit):
    @staticmethod
    def __call__(x: torch.Tensor, k: int, transformed_data_mean: float | None = None) -> None:
        """Modify the values of x in place according to the sklearn NMF init random recipe."""
        # https://github.com/scikit-learn/scikit-learn/blob/
        # 99bf3d8e4eed5ba5db19a1869482a238b6223ffd/sklearn/decomposition/_nmf.py#L304-L315
        assert transformed_data_mean is not None
        factor = np.sqrt(transformed_data_mean / k)
        x.normal_(0.0, factor).abs_()


class NMFInitUniformRandom(NMFInit):
    @staticmethod
    def __call__(x: torch.Tensor, k: int, transformed_data_mean: float | None = None) -> None:
        """Modify the values of x in place according to a Joshua Welch NMF init random recipe."""
        # https://www.nature.com/articles/s41587-021-00867-x#Sec10
        # algorithm 1 step 2
        x.uniform_(0.0, 2.0)


class NonNegativeMatrixFactorization(ABC, CellariumModel):
    """
    Abstract base class for non-negative matrix factorization implementations.

    This class defines the interface that all NMF implementations must provide
    to work with NMFOutput, which can run consensus and downstream analyses.
    """

    def __init__(self, var_names_g: Sequence[str], k_values: list[int], **kwargs):
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.k_values = k_values
        # Create the HVG filter transform that all implementations will need
        self.transform__filter_to_hvgs = Filter([str(s) for s in self.var_names_g])

    @property
    @abstractmethod
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """
        Return the learned factors for each k value.

        Returns:
            Dictionary mapping k -> factor tensor of shape (r, k, g) where:
            - r is number of replicates (could be 1 for some implementations)
            - k is number of factors
            - g is number of genes
        """
        pass

    @abstractmethod
    def infer_loadings(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
        k: int,
        normalize: bool = True,
        obs_names_n: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Infer the loadings of each program for the input count matrix.

        Args:
            x_ng: Gene counts matrix
            var_names_g: Variable names
            consensus_factors: Consensus factors from consensus computation
            k: Number of factors
            normalize: Whether to normalize loadings
            obs_names_n: Cell names

        Returns:
            Loadings tensor of shape (n, k)
        """
        pass

    @abstractmethod
    def reconstruction_error(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
    ) -> dict[int, float]:
        """
        Compute reconstruction error for each k value.

        Args:
            x_ng: Gene counts matrix
            var_names_g: Variable names
            consensus_factors: Consensus factors from consensus computation

        Returns:
            Dictionary mapping k -> reconstruction error
        """
        pass


class OnlineNonNegativeMatrixFactorization(NonNegativeMatrixFactorization):
    """
    Use the online NMF algorithm of Mairal et al. [1] to factorize the count matrix
    into a dictionary of gene expression programs and a matrix of cell program loadings.

    **References:**

    1. `Online learning for matrix factorization and sparse coding. Mairal, Bach, Ponce, Sapiro. JMLR 2009.

    Args:
        var_names_g: The variable names schema for the input data: should be highly variable genes.
        k_values: A list of the number of gene expression programs to infer.
        r: Number of NMF replicates (for consensus).
        algorithm: The algorithm to use for the online NMF. Currently only "mairal" is supported.
        init: The initialization method to use for the NMF factors, in ["sklearn_random", "uniform_random"].
        transformed_data_mean: The mean of the transformed data, used for initialization if an only if
            `init` is "sklearn_random".
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        k_values: list[int],
        r: int,
        algorithm: Literal["mairal", "nmf_torch_hals"] = "mairal",
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
        n_cells_total: int | None = None,
    ) -> None:
        super().__init__(var_names_g=var_names_g, k_values=k_values)
        g = len(self.var_names_g)
        self.algorithm = algorithm
        self.obs_names_to_index_map: dict[str, int] = {}  # used for local latents
        if algorithm == "nmf_torch_hals":
            if n_cells_total is None:
                raise ValueError("n_cells_total must be provided for nmf_torch_hals algorithm")
        self.r = r
        self.transformed_data_mean = transformed_data_mean
        self.init = init
        if init == "sklearn_random":
            if transformed_data_mean is None:
                raise ValueError("transformed_data_mean must be provided when using the sklearn_random initialization")

        for i in self.k_values:
            self.register_buffer(f"A_{i}_rkk", torch.empty(r, i, i))
            self.register_buffer(f"B_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_rkg", torch.empty(r, i, g))
            if self.algorithm == "nmf_torch_hals":
                assert isinstance(n_cells_total, int)
                self.register_buffer(f"loadings_{i}_rnk", torch.empty(r, n_cells_total, i))  # initialized later

        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self._D_tol = 1e-5
        self._alpha_tol = 1e-5
        self._hals_tol = 1e-4
        self.reset_parameters()

    def reset_parameters(self) -> None:
        match self.init:
            case "sklearn_random":
                init_fn: NMFInit = NMFInitSklearnRandom()
            case "uniform_random":
                init_fn = NMFInitUniformRandom()
            case _:
                raise ValueError(f"Unknown initialization method: {self.init}")

        for i in self.k_values:
            getattr(self, f"A_{i}_rkk").zero_()
            getattr(self, f"B_{i}_rkg").zero_()
            init_fn(getattr(self, f"D_{i}_rkg"), k=i, transformed_data_mean=self.transformed_data_mean)
            if self.algorithm == "nmf_torch_hals":
                init_fn(getattr(self, f"loadings_{i}_rnk"), k=i, transformed_data_mean=self.transformed_data_mean)

        if self.algorithm == "nmf_torch_hals":
            self._prev_err_rk: torch.Tensor | None = None
            self._init_err_rk: torch.Tensor | None = None
            self._err_running_sum_rk = torch.zeros((self.r, len(self.k_values)), device=self._dummy_param.device)
            self._cells_seen_in_epoch = 0  # Track cells seen in current epoch

    @property
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """Return the learned factors for each k value."""
        return {k: getattr(self, f"D_{k}_rkg") for k in self.k_values}

    def online_dictionary_update(self, x_ng: torch.Tensor, k: int, minibatch_indices_n: torch.Tensor | None) -> None:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            x_ng: The data.
            k: The value of k to run.
            minibatch_indices_n: The indices of the cells in the current minibatch.
        """
        # get running values
        A_rkk = getattr(self, f"A_{k}_rkk")
        B_rkg = getattr(self, f"B_{k}_rkg")
        factors_rkg = getattr(self, f"D_{k}_rkg")

        if self.algorithm == "mairal":
            # run algorithm 1
            updated_values = online_dictionary_update_mairal(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                A_rkk=A_rkk,
                B_rkg=B_rkg,
                n_iterations=100,
                alpha_tol=self._alpha_tol,
                D_tol=self._D_tol,
            )
        elif self.algorithm == "nmf_torch_hals":
            assert isinstance(minibatch_indices_n, torch.Tensor), (
                "minibatch_indices_n must be provided for nmf_torch_hals algorithm"
            )
            loadings_rnk = getattr(self, f"loadings_{k}_rnk")[:, minibatch_indices_n, :]
            # run nmf-torch hals online update
            updated_values = online_dictionary_update_nmf_torch_hals(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                loadings_rnk=loadings_rnk,
                A_rkk=A_rkk,
                B_rkg=B_rkg,
                # n_iterations=100,
                # alpha_tol=self._alpha_tol,
                # D_tol=self._D_tol,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # update running values
        setattr(self, f"A_{k}_rkk", updated_values["A_rkk"])
        setattr(self, f"B_{k}_rkg", updated_values["B_rkg"])
        setattr(self, f"D_{k}_rkg", updated_values["factors_rkg"])

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        obs_names_n: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.
            obs_names_n: The names of the cells in the current minibatch (used when there are local latents).

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        # handle the mapping of obs_names to indices for local latents
        minibatch_indices_n: torch.Tensor | None = None
        if self.algorithm == "nmf_torch_hals":
            assert obs_names_n is not None, "obs_names_n must be provided for nmf_torch_hals algorithm"
            new_obs_names = set(obs_names_n.tolist()) - set(self.obs_names_to_index_map.keys())
            if len(new_obs_names) > 0:
                self.obs_names_to_index_map |= {
                    name: i for i, name in enumerate(new_obs_names, start=len(self.obs_names_to_index_map))
                }
            vectorized_lookup = np.vectorize(lambda x: self.obs_names_to_index_map[x])
            minibatch_indices_n = torch.from_numpy(vectorized_lookup(obs_names_n)).to(x_ng.device)

        if self.algorithm == "nmf_torch_hals":
            # for error computation to assess convergence
            if self._init_err_rk is None:
                assert isinstance(minibatch_indices_n, torch.Tensor)
                self._loss(x_ng=x_ng, minibatch_indices_n=minibatch_indices_n)
                # Take sqrt and normalize by number of cells in this batch
                self._init_err_rk = self._err_running_sum_rk.clone() / x_ng.shape[0]
                assert isinstance(self._init_err_rk, torch.Tensor)
                self._prev_err_rk = self._init_err_rk.clone()
                self._err_running_sum_rk.zero_()  # Reset after initialization
                self._cells_seen_in_epoch = 0  # Reset counter

        for k in self.k_values:
            self.online_dictionary_update(x_ng=x_ng, k=k, minibatch_indices_n=minibatch_indices_n)

        if self.algorithm == "nmf_torch_hals":
            # for error computation to assess convergence
            assert isinstance(minibatch_indices_n, torch.Tensor)
            self._loss(x_ng=x_ng, minibatch_indices_n=minibatch_indices_n)

        return {}

    def _loss(self, x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor) -> None:
        """
        Simple and efficient NMF reconstruction loss computation.
        Computes ||X - WH||_F^2 for the current batch using einsum.
        """
        with torch.no_grad():
            for i, k in enumerate(self.k_values):
                factors_rkg = getattr(self, f"D_{k}_rkg")  # (r, k, g)
                loadings_rnk = getattr(self, f"loadings_{k}_rnk")[:, minibatch_indices_n, :]  # (r, batch_size, k)
                
                # Compute reconstruction: WH for current batch using einsum
                # loadings_rnk: (r, batch_size, k), factors_rkg: (r, k, g)
                # -> reconstruction_rng: (r, batch_size, g)
                reconstruction_rng = torch.einsum("rnk,rkg->rng", loadings_rnk, factors_rkg)
                
                # Compute squared reconstruction error using einsum
                # x_ng: (batch_size, g) -> expand to (r, batch_size, g) and compute ||X - WH||_F^2
                x_expanded_rng = x_ng.unsqueeze(0).expand(self.r, -1, -1)  # (r, batch_size, g)
                
                # Compute squared Frobenius norm for each replicate using einsum
                squared_error_r = F.mse_loss(x_expanded_rng, reconstruction_rng, reduction='none').sum(dim=(1, 2))
                
                # Accumulate the squared error
                self._err_running_sum_rk[:, i] += squared_error_r
            
            # Track cells seen in this epoch
            if self.algorithm == "nmf_torch_hals":
                self._cells_seen_in_epoch += x_ng.shape[0]

    # def _loss(self, x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor) -> None:
    #     with torch.no_grad():
    #         for i, k in enumerate(self.k_values):
    #             factors_rkg = getattr(self, f"D_{k}_rkg")
    #             loadings_rnk = getattr(self, f"loadings_{k}_rnk")[:, minibatch_indices_n, :]
    #             xWT_rnk = torch.einsum("ng,rkg->rnk", x_ng, factors_rkg)
    #             h_rnk = loadings_rnk
    #             hth_rkk = torch.einsum("rnk,rnj->rkj", h_rnk, h_rnk)
    #             WWT_rkk = torch.einsum("rkg,rjg->rkj", factors_rkg, factors_rkg)
    #             X_SS_half = (x_ng.norm(p=2) ** 2 / 2).double()
    #             sum_h_err_r = self._trace(WWT_rkk, hth_rkk) / 2.0 - self._trace(h_rnk, xWT_rnk)
    #             # Accumulate squared error, don't take sqrt yet
    #             self._err_running_sum_rk[:, i] += 2.0 * (sum_h_err_r + X_SS_half)
    #     self._cells_seen_in_epoch += x_ng.shape[0]

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "OnlineNonNegativeMatrixFactorization requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is True, (
                "OnlineNonNegativeMatrixFactorization requires that the `broadcast_buffers` parameter of "
                "lightning.pytorch.strategies.DDPStrategy is set to True"
            )

    def _trace(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # batched frobenius inner product
        return torch.einsum("rij,rij->r", a, b)

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        # convergence criterion check
        if self.algorithm == "nmf_torch_hals":
            # Take sqrt of accumulated squared errors and normalize by number of cells seen
            # cur_err_rk = torch.sqrt(self._err_running_sum_rk / self._cells_seen_in_epoch)
            cur_err_rk = self._err_running_sum_rk / self._cells_seen_in_epoch
            assert isinstance(self._prev_err_rk, torch.Tensor)
            assert isinstance(self._init_err_rk, torch.Tensor)

            current_overall_err_rk = torch.abs((self._prev_err_rk - cur_err_rk) / self._init_err_rk)
            if (
                current_overall_err_rk.max() < self._hals_tol
            ):
                trainer.should_stop = True
                print(f"Stopping early: converged, loss={cur_err_rk}")

            print(f"Epoch {trainer.current_epoch} reconstruction error: {current_overall_err_rk.max()}")
            print(f"Per-cell loss - Current: {cur_err_rk.max():.6f}, Previous: {self._prev_err_rk.max():.6f}")
            print(f"Cells seen this epoch: {self._cells_seen_in_epoch}")
            self._prev_err_rk = cur_err_rk.clone()
            self._err_running_sum_rk.zero_()
            self._cells_seen_in_epoch = 0  # Reset for next epoch

        # this hard reset to zero is equivalent to forgetting momentum each epoch
        for i in self.k_values:
            getattr(self, f"A_{i}_rkk").zero_()
            getattr(self, f"B_{i}_rkg").zero_()

    def on_end(self, trainer: pl.Trainer) -> None:
        trainer.save_checkpoint(trainer.default_root_dir + "/NMF.ckpt")

    @torch.no_grad()
    def infer_loadings(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
        k: int,
        normalize: bool = False,
        obs_names_n: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Infer the loadings of each program for the input count matrix.
        To be run after the model has been trained.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        x_filtered_ng = self.transform__filter_to_hvgs(x_ng, var_names_g)["x_ng"]
        D_kg = consensus_factors[k]["consensus_D_kg"]
        assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"

        # compute loadings, Mairal Algorithm 1 step 4
        # alpha_nk = compute_loadings(
        #     x_ng=x_filtered_ng,
        #     factors_rkg=D_kg.to(x_filtered_ng.device).unsqueeze(0),
        #     n_iterations=1000,
        #     alpha_tol=self._alpha_tol,
        # ).squeeze(0)
        alpha_nk = (
            solve_nnls_fista(
                D_kg.to(x_filtered_ng.device).unsqueeze(0).transpose(1, 2),
                x_ng.t(),
                tol=self._alpha_tol * 0.1,
                max_iter=1000,
            )
            .transpose(1, 2)
            .squeeze(0)
        )

        if normalize:
            alpha_nk = F.normalize(alpha_nk, p=1, dim=-1)

        return alpha_nk

    @torch.no_grad()
    def reconstruction_error(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
    ) -> dict[int, float]:
        """
        Compute the reconstruction error for each k_value using trained consensus factors D_kg.

        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.
            consensus_factors: The consensus factors for each k_value are in consensus_factors[k]["consensus_D_kg"].

        Returns:
            A dictionary mapping each k_value to its reconstruction error.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        x_filtered_ng = self.transform__filter_to_hvgs(x_ng, var_names_g)["x_ng"]

        rec_error = {}
        for k in consensus_factors.keys():
            D_kg = consensus_factors[k]["consensus_D_kg"]
            assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"
            if (D_kg == 0).all():
                raise ValueError("D_kg is all zeros, please train the model and run compute_consensus_factors() first")

            alpha_nk = self.infer_loadings(
                x_ng=x_filtered_ng,
                var_names_g=var_names_g,
                consensus_factors=consensus_factors,
                k=k,
                normalize=False,
            ).squeeze(0)

            rec_error[k] = (
                nmf_frobenius_loss(
                    x_ng=x_filtered_ng,
                    loadings_nk=alpha_nk.to(x_filtered_ng.device),
                    factors_kg=D_kg.to(x_filtered_ng.device),
                )
                .sum()
                .item()
            )

        return rec_error


def consensus(D_rkg: torch.Tensor, density_threshold: float, local_neighborhood_size: float, plot_only=False):
    assert local_neighborhood_size > 0 and local_neighborhood_size < 1, (
        "local_neighborhood_size must be between 0 and 1"
    )
    assert density_threshold > 0 and density_threshold <= 1, "density_threshold must be > 0 and <= 1"
    r, num_component, g = D_rkg.shape
    d_norm_rkg = F.normalize(D_rkg, dim=-1, p=2)
    d_norm_mg = d_norm_rkg.reshape(r * num_component, g)

    if r > 1:
        n_neighbors = int(r * local_neighborhood_size)
        if n_neighbors < 2:
            raise UserWarning(
                f"local_neighborhood_size {local_neighborhood_size} is too small for k={num_component}. "
                f"n_neighbors = int(replicates * local_neighborhood_size) = {n_neighbors}. "
                "We want n_neighbors >= 2. Increase local_neighborhood_size."
            )

        # euclidean distance to every other run
        euclidean_dist_mm = torch.cdist(d_norm_mg, d_norm_mg, p=2)
        euclidean_dist_mm.fill_diagonal_(0)  # correct for roundoff errors that may be present

        # top n_neighbors plus self (distance 0)
        n_nearest_dist_including_self_mL, _ = torch.topk(euclidean_dist_mm, n_neighbors + 1, largest=False)

        # distances to top n_neighbors
        n_nearest_dist_ml = n_nearest_dist_including_self_mL[:, 1:]

        # mean distance to top n_neighbors
        mean_neighbor_distance_m = n_nearest_dist_ml.mean(dim=1)

        if plot_only:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(5, 2))
            plt.hist(mean_neighbor_distance_m.cpu().numpy(), bins=75)
            plt.title(f"Local Neighborhood Distances: k = {num_component}")
            plt.ylabel("Number of NMF factors\n(total is replicates times k)")
            plt.xlabel(f"Average distance to nearest {n_neighbors} neighbors")
            plt.xlim([-0.05, 1.05])
            plt.show()
            return

        # filter out runs considered outliers based on threshold
        logic = mean_neighbor_distance_m < density_threshold
        euclidean_dist_ff = euclidean_dist_mm[logic, :][:, logic]
        n_nearest_dist_fl = n_nearest_dist_ml[logic, :]
        d_norm_fg = d_norm_mg[logic, :]

        if len(d_norm_fg) == 0:
            raise UserWarning(
                f"No samples found for k={num_component} after applying density threshold {density_threshold}. "
                "Please run with plot=True and consult histogram"
            )

        # run k-means clustering on the remaining f programs
        df_d_norm_fg = pd.DataFrame(d_norm_fg.cpu().numpy())
        kmeans = KMeans(n_clusters=num_component, n_init=10, random_state=1)
        kmeans.fit(df_d_norm_fg)
        kmeans_cluster_labels_f = pd.Series(kmeans.labels_ + 1, index=df_d_norm_fg.index)

        # compute the consensus k programs as the median of each cluster
        median_D_kg = df_d_norm_fg.groupby(kmeans_cluster_labels_f).median()
        median_D_kg = torch.tensor(median_D_kg.values)

        silhouette = silhouette_score(df_d_norm_fg.values, kmeans_cluster_labels_f, metric="euclidean")

    else:
        euclidean_dist_ff = None
        n_nearest_dist_fl = None
        n_nearest_dist_ml = None
        n_neighbors = 0
        median_D_kg = D_rkg.reshape(r * num_component, g)  # just grab the factor without normalization if r=1
        silhouette = 1.0

    factors_kg = F.normalize(median_D_kg, dim=-1, p=1)

    return {
        "filtered_euclidean_distance_matrix": euclidean_dist_ff,
        "filtered_neighbor_distances": n_nearest_dist_fl,
        "all_neighbor_distances": n_nearest_dist_ml,
        "n_neighbors": n_neighbors,
        "consensus_D_kg": factors_kg,
        "stability": silhouette,
        "local_neighborhood_size": local_neighborhood_size,
        "density_threshold": density_threshold,
    }


def plot_density_histograms(
    nmf_model: NonNegativeMatrixFactorization,
    local_neighborhood_size: float = 0.3,
    k_values: list[int] | None = None,
):
    """
    Plot a histogram of local neighborhood distances for each value of k.

    Args:
        local_neighborhood_size: Determines number of neighbors to use for calculating kNN distance as
            int(local_neighborhood_size * replicates)
        k_values: A list of k values to plot histograms for (default: None, which loops over all k values)
    """
    if k_values is None:
        k_values = nmf_model.k_values
    for k in k_values:
        consensus(
            D_rkg=getattr(nmf_model, f"D_{k}_rkg"),
            density_threshold=0.5,  # ignored when plot_only=True
            local_neighborhood_size=local_neighborhood_size,
            plot_only=True,
        )


def compute_consensus_factors(
    nmf_model: NonNegativeMatrixFactorization,
    k_values: list[int] | None = None,
    density_threshold: float = 0.5,
    local_neighborhood_size: float = 0.3,
) -> dict[int, dict[str, torch.Tensor | float]]:
    """
    Run the consensus step of consensus NMF, and store the consensus factors as attributes of `nmf_model`.

    Args:
        nmf_model: The trained NMF model.
        k_values: k-values to compute consensus for.
        density_threshold: The threshold for the density of the local neighborhood.
        local_neighborhood_size: The size of the local neighborhood.

    Returns:
        A dictionary of the consensus outputs for each value in k_values, keyed by k_values.
    """
    torch.manual_seed(0)
    if k_values is None:
        k_values = nmf_model.k_values

    consensus_stat = {}
    for k in k_values:
        D_rkg = nmf_model.factors_dict[k]
        consensus_output = consensus(
            D_rkg=D_rkg,
            density_threshold=density_threshold,
            local_neighborhood_size=local_neighborhood_size,
            plot_only=False,
        )
        consensus_stat[k] = consensus_output

    return consensus_stat


def k_selection_plot(
    consensus_output: dict[int, dict[str, float | torch.Tensor]],
    reconstruction_error: dict[int, float],
):
    from matplotlib import pyplot as plt

    k_values = list(consensus_output.keys())
    assert set(consensus_output.keys()) == set(reconstruction_error.keys()), (
        "consensus_output and reconstruction_error keys k do not match"
    )

    silhouette_scores = {}
    for k in k_values:
        silhouette_scores[k] = consensus_output[k]["stability"]
    eval_metrics = pd.DataFrame.from_dict(silhouette_scores, orient="index")
    eval_metrics.columns = ["stability"]
    eval_metrics["rec_error"] = reconstruction_error

    plt.figure(figsize=(10, 5))
    plt.plot(eval_metrics.index, eval_metrics["stability"], "o-", color="b")
    plt.grid(True)
    plt.ylabel("Stability", color="b")
    plt.xlabel("Number of components: k")
    plt.xticks(k_values)
    plt.gca().tick_params(axis="y", colors="b")
    plt.twinx()
    plt.plot(eval_metrics.index, eval_metrics["rec_error"], "o-", color="r")
    plt.ylabel("Reconstruction error", color="r")
    plt.gca().tick_params(axis="y", colors="r")
    plt.grid(False)
    plt.show()


def plot_clustermap(
    consensus_output: dict[int, dict[str, float | torch.Tensor]],
    k: int,
):
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn is not installed. Please install with `pip install seaborn`")
        return

    filtered_dist_matrix = consensus_output[k]["filtered_euclidean_distance_matrix"]
    assert isinstance(filtered_dist_matrix, torch.Tensor)
    cg = sns.clustermap(
        filtered_dist_matrix.cpu().numpy(),
        row_cluster=True,
        col_cluster=True,
        cbar_pos=(0.05, 0.25, 0.03, 0.15),
        figsize=(5, 5),
        xticklabels=False,
        yticklabels=False,
        rasterized=True,
    )
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    # cg.cax.set_visible(False)
    cg.cax.set_ylabel("Euclidean distance")
    cg.ax_heatmap.set_title(f"k = {k}")
    plt.show()

    plt.figure(figsize=(5, 2))
    all_neighbor_distances = consensus_output[k]["all_neighbor_distances"]
    assert isinstance(all_neighbor_distances, torch.Tensor), "all_neighbor_distances must be a tensor"
    dists = all_neighbor_distances.mean(dim=1).cpu().numpy()
    plt.hist(dists, bins=75)
    plt.vlines(consensus_output[k]["density_threshold"], 0, plt.gca().get_ylim()[1], colors="r")
    plt.title(
        f"Local Neighborhood Distances: k = {k}\ndensity_threshold filters "
        f"{100 * (dists > consensus_output[k]['density_threshold']).mean():.1f}% of runs"
    )
    plt.ylabel("Number of NMF factors\n(total is replicates times k)")
    plt.xlabel(f"Average distance to nearest {consensus_output[k]['n_neighbors']} neighbors")
    plt.xlim([-0.05, 1.05])
    plt.show()


class NMFOutput:
    """
    A class to facilitate interaction with a trained NMF model and computation of downstream outputs.
    """

    def __init__(
        self,
        nmf_module: "cellarium.ml.CellariumModule",
        datamodule: "cellarium.ml.CellariumAnnDataDataModule",
    ):
        """
        Initialize the NMFOutput class.

        Args:
            nmf_module: The trained NMF module to use.
            datamodule: The data module to use.
        """
        self.datamodule = datamodule
        self.datamodule.setup(stage="predict")  # can remove cpu transforms from datamodule
        self.nmf_module = nmf_module
        self.nmf_module.setup(stage="predict")  # send cpu transforms to datamodule, but only if these share a Trainer
        self._consensus: dict = {}
        self._rec_error: dict | None = None
        self._tpm_D_kg: torch.Tensor | None = None
        self._tpm_A_kk: torch.Tensor | None = None
        self._tpm_B_kg: torch.Tensor | None = None
        if not isinstance(self.nmf_module.model, NonNegativeMatrixFactorization):
            raise ValueError("NMFOutput requires nmf_module with a NonNegativeMatrixFactorization in nmf_module.model")

    def __repr__(self) -> str:
        indent = "    "
        return (
            f"NMFOutput(\n{indent}nmf_module="
            + str(self.nmf_module).replace("\n", f"\n{indent}{indent}")
            + f",\n{indent}datamodule="
            + str(self.datamodule).replace("\n", f"\n{indent}{indent}")
            + "\n)\n"
            + f"with consensus {list(self._consensus.keys())}"
        )

    def plot_density_histograms(self, k_values: int | list[int] | None = None, local_neighborhood_size: float = 0.3):
        """
        Plot density histograms for the given choice of local_neighborhood_size.

        Args:
            local_neighborhood_size: The fraction of replicate runs that are considered neighbors.
            k_values: The list of k values to plot. If None, use all available k values.
        """
        if isinstance(k_values, int):
            k_values = [k_values]
        assert isinstance(self.nmf_module.model, NonNegativeMatrixFactorization)
        plot_density_histograms(
            nmf_model=self.nmf_module.model,
            local_neighborhood_size=local_neighborhood_size,
            k_values=k_values,
        )

    def compute_consensus_factors(
        self, k_values: int | list[int], density_threshold: float, local_neighborhood_size: float = 0.3
    ) -> None:
        """
        Run the "consensus" step of consensus NMF by filtering outliers and clustering replicates,
        taking the median program in each cluster.

        Args:
            density_threshold: The density threshold to use for filtering. Kotliar default is 0.5
            local_neighborhood_size: The local neighborhood size to use for clustering. Kotliar default is 0.3

        Returns:
            None, but updates :attr:`_consensus`, accessible via the property :property:`consensus`.
        """
        assert isinstance(self.nmf_module.model, NonNegativeMatrixFactorization)

        if isinstance(k_values, int):
            k_values = [k_values]

        consensus_output = compute_consensus_factors(
            nmf_model=self.nmf_module.model,
            k_values=k_values,
            density_threshold=density_threshold,
            local_neighborhood_size=local_neighborhood_size,
        )
        self._consensus = self._consensus | consensus_output
        self._rec_error = None  # remove this value, as it wil need updating with new consensus factors

    @property
    def consensus(self) -> dict[int, dict[str, torch.Tensor | float]]:
        if len(self._consensus) > 0:
            return self._consensus
        raise UserWarning("Compute a consensus using compute_consensus_factors() -- entails hyperparameter choices")

    def calculate_reconstruction_error(self, k_values: list[int] | None = None) -> dict[int, float]:
        """
        Calculate the reconstruction error for each k value.
        Stores the output in :property:`reconstruction_error`.

        Returns:
            dict[int, float]: The reconstruction error for each k value.
        """
        assert isinstance(self.nmf_module.model, NonNegativeMatrixFactorization)
        rec_error = {k: 0.0 for k in self.nmf_module.model.k_values}
        for batch in tqdm(self.datamodule.train_dataloader()):
            errors_keyed_by_k = self.nmf_module.model.reconstruction_error(
                x_ng=batch["x_ng"],
                var_names_g=batch["var_names_g"],
                consensus_factors=self.consensus,
            )
            for k, error in errors_keyed_by_k.items():
                rec_error[k] += error

        self._rec_error = rec_error
        return rec_error

    @property
    def reconstruction_error(self) -> dict[int, float]:
        if self._rec_error is not None:
            return self._rec_error
        raise UserWarning("Compute reconstruction error using calculate_reconstruction_error()")

    @torch.no_grad()
    def compute_loadings(
        self,
        k: int,
        datamodule: Optional["cellarium.ml.CellariumAnnDataDataModule"] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the per-cell factor loadings for a given k.

        Args:
            k: The number of consensus NMF factors
            datamodule: The datamodule to use for prediction. If None, uses the current datamodule
                and all its data. If provided, it can be a smaller subset of data.
            normalize: Whether to normalize the loadings to sum to 1.
        """
        assert isinstance(self.nmf_module.model, NonNegativeMatrixFactorization)
        if datamodule is None:
            datamodule = self.datamodule
        else:
            datamodule.setup(stage="predict")  # as this may not have been called... cpu_transforms are tricky here

        # TODO fix this hacky manual stuff
        # grab the transforms
        transforms = []
        # for transform in self.nmf_module.cpu_transforms:
        #     transforms.append(transform)
        for transform in self.nmf_module.transforms:
            transforms.append(transform)

        embedding = []
        index = []
        for batch in tqdm(datamodule.predict_dataloader()):
            # apply transforms to the data before inferring loadings
            for transform in transforms:
                batch |= transform(x_ng=batch["x_ng"], var_names_g=batch["var_names_g"])

            alpha_nk = self.nmf_module.model.infer_loadings(
                x_ng=batch["x_ng"],
                var_names_g=batch["var_names_g"],
                obs_names_n=batch.get("obs_names_n", None),
                consensus_factors=self.consensus,
                k=k,
                normalize=normalize,
            )
            embedding += [alpha_nk.cpu()]
            index.extend(batch["obs_names_n"])

        return pd.DataFrame(torch.cat(embedding).numpy(), index=index)

    @torch.no_grad()
    def refit_consensus_factor_for_all_genes(
        self,
        k: int,
        normalize_tpm_spectra: bool,
    ) -> dict[str, torch.Tensor]:
        print("WARNING: at this point, the cellarium implmentation may differ from Kotliar cNMF")
        if k not in self.consensus:
            raise KeyError(f"Missing consensus_factors key k={k}. Choose from {list(self.consensus.keys())}")
        # self.datamodule.setup(stage="predict")

        # Initialize tensors if needed
        if self._tpm_D_kg is None:
            consensus_D_kg = self.consensus[k]["consensus_D_kg"]
            assert isinstance(consensus_D_kg, torch.Tensor)
            self._tpm_D_kg = consensus_D_kg.clone()
            self._tpm_A_kk = torch.zeros(k, k, device=consensus_D_kg.device)
            self._tpm_B_kg = torch.zeros(k, consensus_D_kg.shape[1], device=consensus_D_kg.device)

        for batch in tqdm(self.datamodule.predict_dataloader()):
            # Get std_g for normalization
            x_ng = batch["x_ng"]
            std_g = torch.std(x_ng, dim=0) + 1e-4

            consensus_D_kg = self.consensus[k]["consensus_D_kg"]
            assert isinstance(consensus_D_kg, torch.Tensor)
            assert self._tpm_D_kg is not None
            assert self._tpm_A_kk is not None
            assert self._tpm_B_kg is not None

            refit = self._refit(
                x_ng=x_ng,
                var_names_g=batch["var_names_g"],
                std_g=std_g.numpy(),
                consensus_D_kg=consensus_D_kg,
                refit_D_kg=self._tpm_D_kg,
                A_kk=self._tpm_A_kk,
                B_kg=self._tpm_B_kg,
                normalize_tpm_spectra=normalize_tpm_spectra,
            )
            self._tpm_D_kg = refit["D_kg"]
            self._tpm_A_kk = refit["A_kk"]
            self._tpm_B_kg = refit["B_kg"]

        # Ensure all return values are tensors
        assert self._tpm_D_kg is not None
        assert self._tpm_A_kk is not None
        assert self._tpm_B_kg is not None
        return {"D_kg": self._tpm_D_kg, "A_kk": self._tpm_A_kk, "B_kg": self._tpm_B_kg}

    def _refit(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        std_g: np.ndarray,
        consensus_D_kg: torch.Tensor,
        refit_D_kg: torch.Tensor,
        A_kk: torch.Tensor,
        B_kg: torch.Tensor,
        normalize_tpm_spectra: bool,
    ) -> dict[str, torch.Tensor]:
        # filter to HVGs to compute the loadings according to the model
        assert isinstance(self.nmf_module.model, NonNegativeMatrixFactorization)
        x_filtered_ng = self.nmf_module.model.transform__filter_to_hvgs(x_ng, var_names_g)["x_ng"]

        # get the final alpha_nk - no log_variational attribute, use std normalization
        x_ = x_filtered_ng / torch.from_numpy(std_g).to(x_filtered_ng.device)

        # compute loadings, called "norm_usages" in Kotliar, based on consensus factors
        k = consensus_D_kg.shape[0]
        alpha_rnk = self.nmf_module.model.infer_loadings(
            x_ng=x_,
            var_names_g=var_names_g,
            consensus_factors={k: {"consensus_D_kg": consensus_D_kg}},
            k=k,
            normalize=False,
        )

        # normalize counts to TPM
        if normalize_tpm_spectra:
            tpm_transform = NormalizeTotal(target_count=1_000_000)
            x_ng = tpm_transform(x_ng)
        n, g = x_ng.shape
        r = 1

        with torch.no_grad():
            # update A and B, Mairal Algorithm 1 step 5 and 6
            A_rkk = A_kk.unsqueeze(0) + torch.bmm(alpha_rnk.transpose(1, 2), alpha_rnk) / n
            B_rkg = B_kg.unsqueeze(0) + torch.bmm(alpha_rnk.transpose(1, 2), x_ng.expand(r, n, g)) / n

            # update D, Mairal Algorithm 1 step 7
            updated_factors_rkg = compute_factors(
                factors_rkg=refit_D_kg.unsqueeze(0),
                A_rkk=A_rkk,
                B_rkg=B_rkg,
                n_iterations=1000,
                D_tol=self.nmf_module.model._D_tol,
            )

        # # update A and B
        # D_kg, A_kk, B_kg = efficient_ols_all_cols(
        #     alpha_nk.cpu().numpy(), x_ng.cpu().numpy(), A_kk.cpu().numpy(), B_kg.cpu().numpy()
        # )
        return {"D_kg": updated_factors_rkg.squeeze(0), "A_kk": A_rkk.squeeze(0), "B_kg": B_rkg.squeeze(0)}

    def default_k_selection_plot(self):
        """
        Make a k-selection plot with stability and reconstruction error curves as a function of k.
        This uses the Kotliar default hyperparameters to run consensus.
        """
        density_threshold_default = 0.5
        local_neighborhood_size_default = 0.3

        logger.info("Computing consensus factors with default hyperparameters...")
        self.compute_consensus_factors(
            k_values=self.nmf_module.model.k_values,
            density_threshold=density_threshold_default,
            local_neighborhood_size=local_neighborhood_size_default,
        )

        logger.info("Calculating reconstruction error (requires an entire pass through the data)...")
        self.calculate_reconstruction_error()

        k_selection_plot(
            consensus_output=self.consensus,
            reconstruction_error=self.reconstruction_error,
        )

    def maximal_stability_k_selection_plot(
        self,
        fast_or_exhaustive: str = "fast",
        density_threshold_limits: list = [0.05, 0.3],
        local_neighborhood_size: float = 0.3,
        max_tolerable_fraction_eliminated: float = 0.3,
    ):
        """
        Make a k-selection plot with stability and reconstruction error curves as a function of k.
        This uses the Kotliar default hyperparameter for local_neighborhood_size, but then makes
        the determination of density_threshold into an optimization problem, which it solves separately
        for each k. Arguably, this is an even better way to ultimately choose k.

        Args:
            fast_or_exhaustive: Whether to use fast or exhaustive search for density thresholds.
            density_threshold_limits: The lower and upper bounds for the density threshold search.
            local_neighborhood_size: Determines the number of neighboring runs to average distances over.
            max_tolerable_fraction_eliminated: Do not consider a density_threshold which eliminates more than
                this fraction of NMF runs.
        """
        logger.info("Computing consensus factors, searching for best density thresholds...")
        for k in tqdm(self.nmf_module.model.k_values):
            if fast_or_exhaustive == "fast":
                # try to look for local minima in the density histogram
                # first compute preliminary consensus to get neighbor distances
                if k not in self._consensus:
                    self.compute_consensus_factors(
                        k_values=[k],
                        density_threshold=0.5,  # high threshold to include most data points
                        local_neighborhood_size=local_neighborhood_size,
                    )
                n_nearest_dist_ml = self.consensus[k]["all_neighbor_distances"]
                assert isinstance(n_nearest_dist_ml, torch.Tensor)
                mean_neighbor_distance_m = n_nearest_dist_ml.mean(dim=1)
                candidate_values = [
                    dt
                    for dt in find_local_minima(mean_neighbor_distance_m)
                    if (
                        (dt >= density_threshold_limits[0])
                        and (dt <= density_threshold_limits[1])
                        and (mean_neighbor_distance_m > dt).float().mean() <= max_tolerable_fraction_eliminated
                    )
                ]
                if not candidate_values:
                    logger.warning(
                        f"Unable to find local minima in k={k} density histogram... 'fast' mode "
                        "will fall back to 'exhaustive'"
                    )
                    candidate_values = np.arange(
                        density_threshold_limits[0], density_threshold_limits[1], 0.01
                    ).tolist()
            elif fast_or_exhaustive == "exhaustive":
                candidate_values = np.arange(density_threshold_limits[0], density_threshold_limits[1], 0.01).tolist()
            else:
                raise ValueError("fast_or_exhaustive must be 'fast' or 'exhaustive'")

            best_density_threshold = 0.05
            max_stability = 0.0
            for density_threshold in candidate_values:
                self.compute_consensus_factors(
                    k_values=[k],
                    density_threshold=density_threshold,
                    local_neighborhood_size=local_neighborhood_size,
                )
                # check for improvement
                stability = self.consensus[k]["stability"]
                assert isinstance(stability, (int, float)), "stability must be a number"
                if stability > max_stability:
                    max_stability = stability
                    best_density_threshold = density_threshold

            # recompute best density threshold, storing results
            self.compute_consensus_factors(
                k_values=[k],
                density_threshold=best_density_threshold,
                local_neighborhood_size=local_neighborhood_size,
            )

        logger.info("Computing reconstruction errors...")
        self.calculate_reconstruction_error()

        k_selection_plot(
            consensus_output=self.consensus,
            reconstruction_error=self.reconstruction_error,
        )

    def k_selection_plot(self):
        """
        Make the k-selection plot with stability and reconstruction error curves as a function of k.
        """
        k_selection_plot(
            consensus_output=self.consensus,
            reconstruction_error=self.reconstruction_error,
        )

    def plot_clustermap(self, k: int | list[int] | None, density_threshold: float | None = None):
        """
        Make a clustermap plot of replicate factors to see how they cluster.

        Args:
            k: The number of components for NMF.
            density_threshold: The density threshold for filtering factors. If None, it will use
                the cached computation of consensus for k. If a value is provided, it will
                recompute consensus with that density_threshold before plotting, and the new
                consensus results will be cached.
        """
        if k is None:
            k_values = self.nmf_module.model.k_values
        else:
            if isinstance(k, int):
                k_values = [k]
            elif isinstance(k, list):
                k_values = k
            else:
                raise ValueError("k must be int or None")

        if density_threshold is not None:
            # Check if we need to recompute consensus
            first_k = k_values[0] if k_values else list(self.consensus.keys())[0]
            if first_k in self.consensus:
                current_threshold = self.consensus[first_k]["density_threshold"]
                if density_threshold != current_threshold:
                    self.compute_consensus_factors(
                        k_values=k_values,
                        density_threshold=density_threshold,
                    )

        for k_val in k_values:
            if k_val not in self.nmf_module.model.k_values:
                raise ValueError(f"Invalid k value for trained model. Choose from {self.nmf_module.model.k_values}")

            plot_clustermap(
                consensus_output=self.consensus,
                k=k_val,
            )
