# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

warnings.filterwarnings("ignore")


@torch.compile()
@torch.no_grad()
def solve_nnls_fista_precomputed(
    AtA: torch.Tensor,
    AtB: torch.Tensor,
    initial_x: torch.Tensor,
    max_iter: int = 100,
    return_history: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Highly optimized FISTA core for torch.compile.
    Assumes AtA and AtB are already precomputed.
    """
    # Compute the Lipschitz constant L (largest eigenvalue of AtA).
    # eigvalsh is faster and exact, but not implemented on MPS; power iteration
    # uses only matmul and works on all devices.
    if AtA.device.type == "mps":
        v = torch.ones(*AtA.shape[:-1], 1, device=AtA.device, dtype=AtA.dtype)
        for _ in range(10):
            v = AtA @ v
            v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-8)
        L = (v.transpose(-2, -1) @ AtA @ v).clamp(min=1e-12)  # [r, 1, 1]
    else:
        eigenvals = torch.linalg.eigvalsh(AtA)
        L = torch.clamp(eigenvals[..., -1:], min=1e-12).unsqueeze(-1)  # [r, 1, 1]

    x = initial_x.clone()
    y = initial_x.clone()

    # Initialize momentum scalar
    t = 1.0

    # torch.compile specializes on Python booleans, so the False branch generates
    # a much smaller kernel (no per-iteration tensors kept alive), which is required
    # to stay within Metal's 31 constant-buffer limit on MPS.
    history: torch.Tensor | None = torch.zeros(max_iter, device=AtA.device, dtype=AtA.dtype) if return_history else None

    # Fixed iteration loop for compilation compatibility
    for i in range(max_iter):
        # Gradient step
        grad = AtA @ y - AtB
        x_new = torch.clamp(y - grad / L, min=0.0)

        if return_history:
            assert history is not None
            history[i] = (x_new - x).abs().max()

        # Beck & Teboulle Momentum update
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t**2)) / 2.0
        momentum = (t - 1.0) / t_new
        y = x_new + momentum * (x_new - x)

        x = x_new
        t = t_new

    return x, history


@torch.compile()
@torch.no_grad()
def nmf_compute_factors_fista(
    w_rkg: torch.Tensor,
    A_rkk: torch.Tensor,  # This is H^T H
    B_rkg: torch.Tensor,  # This is H^T X
    max_iter: int = 100,
    return_history: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Wrapper to update Factors using FISTA.
    """
    # w_rkg has shape [r, k, g].
    # A_rkk @ w_rkg works directly natively via batched matrix multiplication.
    w_rkg_new, history = solve_nnls_fista_precomputed(
        AtA=A_rkk, AtB=B_rkg, initial_x=w_rkg, max_iter=max_iter, return_history=return_history
    )

    return w_rkg_new, history


@torch.compile()
@torch.no_grad()
def nmf_compute_loadings_fista(
    x_ng: torch.Tensor,
    w_rkg: torch.Tensor,
    h_rnk: torch.Tensor,
    max_iter: int = 100,
    return_history: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Wrapper to update Loadings using FISTA.
    """
    # Precompute the transposed equivalents: W W^T and W X^T
    # W W^T shape: [r, k, k]
    wwT_rkk = torch.einsum("rkg,rhg->rkh", w_rkg, w_rkg)

    # W X^T shape: [r, k, n]
    wxT_rkn = torch.einsum("rkg,ng->rkn", w_rkg, x_ng)

    # h_rnk is [r, n, k]. We transpose to [r, k, n] to match FISTA's Ax=B expectation
    h_transposed_rkn = h_rnk.transpose(-2, -1)

    # Solve for transposed H
    h_new_transposed, history = solve_nnls_fista_precomputed(
        AtA=wwT_rkk, AtB=wxT_rkn, initial_x=h_transposed_rkn, max_iter=max_iter, return_history=return_history
    )

    return h_new_transposed.transpose(-2, -1), history


def nmf_frobenius_loss(x_ng: torch.Tensor, loadings_nk: torch.Tensor, factors_kg: torch.Tensor):
    # compute prediction error as the frobenius norm
    return F.mse_loss(torch.matmul(loadings_nk, factors_kg), x_ng, reduction="sum")


def _reduction_dtype(device: torch.device) -> torch.dtype:
    """Widest float the device supports, for the scalar reductions in the Frobenius loss.

    MPS has no float64 at all, so there the reductions stay in float32.  That is accurate to
    roughly 1e-6 relative to the loss unless the fit is nearly exact.
    """
    return torch.float32 if device.type == "mps" else torch.float64


def compute_x_squared_sum(x_ng: torch.Tensor) -> torch.Tensor:
    """
    ``||X||_F^2`` as a scalar, for sharing across calls to :func:`frobenius_loss_trace_compiled`
    (e.g. when looping over ``k``) rather than recomputing it each time.
    """
    return x_ng.pow(2).sum(dtype=_reduction_dtype(x_ng.device))


@torch.compile()
@torch.no_grad()
def frobenius_loss_trace_compiled(
    x_ng: torch.Tensor,
    h_rnk: torch.Tensor,
    w_rkg: torch.Tensor,
    x_squared_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-replicate ``||X - H W||_F^2`` computed *without* materializing the reconstruction.

    Uses :math:`\\|X - HW\\|_F^2 = \\|X\\|^2 - 2\\langle HX, W\\rangle + \\langle H^\\top H,
    WW^\\top\\rangle`, so the intermediates are ``(r, k, g)`` and ``(r, k, k)`` rather than
    ``(r, n, g)``.  At ``r=100, n=2048, g=2000`` that is 80 MB instead of 1.6 GB.

    The three scalar terms are reduced in float64 where the device supports it (float32 on MPS),
    which removes cancellation in the final combination.  The ``(r, k, g)`` and ``(r, k, k)``
    contractions themselves run in the input dtype.

    Args:
        x_ng: Data, shape ``(n, g)``.
        h_rnk: Loadings, shape ``(r, n, k)``.
        w_rkg: Factors, shape ``(r, k, g)``.
        x_squared_sum: Optional precomputed :func:`compute_x_squared_sum` of ``x_ng``, shared across
            calls that use the same ``x_ng``.  Computed here if not given.

    Returns:
        Sum of squared errors per replicate, shape ``(r,)``.
    """
    reduction_dtype = _reduction_dtype(x_ng.device)
    if x_squared_sum is None:
        x_squared_sum = compute_x_squared_sum(x_ng)
    hx_rkg = torch.einsum("rnk,ng->rkg", h_rnk, x_ng)
    hth_rkk = torch.einsum("rnk,rnj->rkj", h_rnk, h_rnk)
    wwt_rkk = torch.einsum("rkg,rjg->rkj", w_rkg, w_rkg)
    cross_r = (hx_rkg * w_rkg).sum(dim=(-2, -1), dtype=reduction_dtype)
    quad_r = (hth_rkk * wwt_rkk).sum(dim=(-2, -1), dtype=reduction_dtype)
    sse_r = x_squared_sum.to(reduction_dtype) - 2.0 * cross_r + quad_r
    return sse_r.clamp(min=0.0).to(w_rkg.dtype)


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

    # Compute Lipschitz constant (largest eigenvalue of AtA).
    # eigvals is not implemented on MPS; power iteration uses only matmul.
    if AtA.device.type == "mps":
        v = torch.ones(*AtA.shape[:-1], 1, device=AtA.device, dtype=AtA.dtype)
        for _ in range(20):
            v = AtA @ v
            v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-8)
        L = (v.transpose(-2, -1) @ AtA @ v).clamp(min=1e-12)  # (..., 1, 1)
    else:
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


@torch.compile()
@torch.no_grad()
def nmf_torch_update_loadings_hals_compiled(
    x_ng: torch.Tensor,
    w_rkg: torch.Tensor,
    h_rnk: torch.Tensor,
    max_iter: int = 200,
) -> None:
    """
    Compiled version of the HALS loadings update, run for exactly ``max_iter`` iterations.
    Updates ``h_rnk`` in place; convergence is checked by the caller between calls.
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
            hvec_rn = torch.where(denom > 1e-12, torch.clamp(h_rn + numer_rn / denom, min=0.0), torch.zeros_like(h_rn))
            h_rnk[..., k] = hvec_rn


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
        nmf_torch_update_loadings_hals_compiled(x_ng, w_rkg, h_rnk, current_chunk)

        total_iterations += current_chunk

        # Check convergence
        max_change = (h_rnk - h_old).abs().max()
        mean_h = h_rnk.mean(dim=(-2, -1))
        relative_change = max_change / torch.clamp(mean_h.max(), min=1e-12)

        if relative_change < h_tol:
            # print(f"NMF HALS loadings update converged in {total_iterations} iterations.")
            break
    # else:
    #     print("NMF HALS loadings update reached max iterations without convergence.")


@torch.compile()
@torch.no_grad()
def nmf_torch_update_factors_hals_compiled(
    w_rkg: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    max_iter: int = 200,
) -> None:
    """
    Compiled version of the HALS factors update, run for exactly ``max_iter`` iterations.
    Updates ``w_rkg`` in place; convergence is checked by the caller between calls.
    """
    # Use a fixed number of iterations instead of dynamic convergence
    # This makes the function more amenable to compilation
    for j in range(max_iter):
        for k in range(A_rkk.shape[-1]):
            numer_rg = B_rkg[:, k, :] - torch.einsum("rk,rkg->rg", A_rkk[:, k, :], w_rkg)
            w_new_rg = w_rkg[:, k, :] + numer_rg / A_rkk[:, k, k].unsqueeze(-1)

            # Avoid division by zero using torch.where instead of manual checks
            w_new_rg = torch.where(torch.isnan(w_new_rg), torch.zeros_like(w_new_rg), torch.clamp(w_new_rg, min=0.0))
            w_rkg[:, k, :] = w_new_rg


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
        nmf_torch_update_factors_hals_compiled(w_rkg, A_rkk, B_rkg, current_chunk)

        total_iterations += current_chunk

        # Check convergence
        max_change = (w_rkg - w_old).abs().max()
        mean_w = w_rkg.mean(dim=(-2, -1))
        relative_change = max_change / torch.clamp(mean_w.max(), min=1e-12)

        if relative_change < w_tol:
            break


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


def online_dictionary_update_nmf_torch_hals(
    x_ng: torch.Tensor,
    factors_rkg: torch.Tensor,
    loadings_rnk: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    n_iterations: int = 200,
    alpha_tol: float = 0.05,
    D_tol: float = 0.05,
    exponential_decay_rho: float = 1.0,
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
        exponential_decay_rho: The exponential decay factor for A and B updates (default: 1, no decay).

    Returns:
        dict with keys:
            "factors_rkg": The updated dictionary factors_rkg.
            "A_rkk": The updated matrix A.
            "B_rkg": The updated matrix B.
    """

    # inplace update loadings_rnk
    nmf_torch_update_loadings_hals_with_compile(
        x_ng=x_ng,
        w_rkg=factors_rkg,
        h_rnk=loadings_rnk,
        max_iter=n_iterations,
        h_tol=alpha_tol,
    )

    with torch.no_grad():
        # update A and B, Mairal Algorithm 1 step 5 and 6 (sums are taken over the cells of all ranks if distributed)
        AtA_rkk, AtX_rkg, n_global = all_reduce_mairal_sufficient_statistics(
            *mairal_sufficient_statistics(loadings_rnk, x_ng)
        )
        A_rkk = exponential_decay_rho * A_rkk + AtA_rkk / n_global
        B_rkg = exponential_decay_rho * B_rkg + AtX_rkg / n_global

    # inplace update factors_rkg
    nmf_torch_update_factors_hals_with_compile(
        w_rkg=factors_rkg,
        A_rkk=A_rkk,
        B_rkg=B_rkg,
        max_iter=n_iterations,
        w_tol=D_tol,
    )

    return {"factors_rkg": factors_rkg, "A_rkk": A_rkk, "B_rkg": B_rkg}


def online_dictionary_update_fista(
    x_ng: torch.Tensor,
    factors_rkg: torch.Tensor,
    loadings_rnk: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    n_iterations: int = 100,
    exponential_decay_rho: float = 1.0,
    return_history: bool = False,
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
        exponential_decay_rho: The exponential decay factor for A and B updates (default: 1, no decay).

    Returns:
        dict with keys:
            "factors_rkg": The updated dictionary factors_rkg.
            "A_rkk": The updated matrix A.
            "B_rkg": The updated matrix B.
    """

    # update loadings_rnk
    loadings_rnk, loadings_history = nmf_compute_loadings_fista(
        x_ng=x_ng,
        w_rkg=factors_rkg,
        h_rnk=loadings_rnk,
        max_iter=n_iterations,
        return_history=return_history,
    )

    with torch.no_grad():
        # update A and B, Mairal Algorithm 1 step 5 and 6 (sums are taken over the cells of all ranks if distributed)
        AtA_rkk, AtX_rkg, n_global = all_reduce_mairal_sufficient_statistics(
            *mairal_sufficient_statistics(loadings_rnk, x_ng)
        )
        A_rkk = exponential_decay_rho * A_rkk + AtA_rkk / n_global
        B_rkg = exponential_decay_rho * B_rkg + AtX_rkg / n_global

    # update factors_rkg
    factors_rkg, factors_history = nmf_compute_factors_fista(
        w_rkg=factors_rkg,
        A_rkk=A_rkk,
        B_rkg=B_rkg,
        max_iter=n_iterations,
        return_history=return_history,
    )

    result = {
        "factors_rkg": factors_rkg,
        "A_rkk": A_rkk,
        "B_rkg": B_rkg,
        "loadings_rnk": loadings_rnk,
    }
    if loadings_history is not None:
        result["loadings_history"] = loadings_history
    if factors_history is not None:
        result["factors_history"] = factors_history
    return result


def l1_normalize_factors(factors_rkg: torch.Tensor) -> torch.Tensor:
    """
    L1-normalize each gene expression program (each row of ``factors_rkg``).

    The unconstrained NNLS dictionary update of :func:`online_dictionary_update_fista` leaves the scale of
    ``factors_rkg`` free (the loadings absorb it), so the caller normalizes after each update to keep it fixed.
    """
    return F.normalize(factors_rkg, p=1, dim=-1, eps=1e-8)


def _distributed_world_size() -> int:
    """Number of processes in the default process group, or 1 if not running distributed."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def mairal_sufficient_statistics(
    loadings_rnk: torch.Tensor,
    x_ng: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Per-minibatch sums that Mairal's matrices A and B accumulate. These are linear in the cells, so sums computed
    on disjoint sets of cells can simply be added together.

    Args:
        loadings_rnk: The loadings (Mairal's alpha) of the cells in the minibatch.
        x_ng: The data for the cells in the minibatch.

    Returns:
        Tuple of the sum over cells of alpha alpha^T (shape r, k, k), the sum over cells of alpha x^T
        (shape r, k, g), and the number of cells.
    """
    r, n, _ = loadings_rnk.shape
    g = x_ng.shape[1]
    alpha_t_rkn = loadings_rnk.transpose(1, 2)
    return torch.bmm(alpha_t_rkn, loadings_rnk), torch.bmm(alpha_t_rkn, x_ng.expand(r, n, g)), n


def all_reduce_mairal_sufficient_statistics(
    AtA_rkk: torch.Tensor,
    AtX_rkg: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Sum the output of :func:`mairal_sufficient_statistics` across all processes, if running distributed.
    This makes the result equivalent to a single minibatch made of the union of every rank's cells.
    """
    if _distributed_world_size() == 1:
        return AtA_rkk, AtX_rkg, n

    n_t = torch.tensor([n], dtype=AtA_rkk.dtype, device=AtA_rkk.device)
    packed = torch.cat([AtA_rkk.flatten(), AtX_rkg.flatten(), n_t])
    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    AtA_size = AtA_rkk.numel()
    AtA_rkk = packed[:AtA_size].view_as(AtA_rkk)
    AtX_rkg = packed[AtA_size : AtA_size + AtX_rkg.numel()].view_as(AtX_rkg)
    return AtA_rkk, AtX_rkg, int(round(packed[-1].item()))


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

    An implementation only has to provide the learned factors of each replicate (:attr:`factors_dict`). Given
    consensus factors computed from them (see :class:`~cellarium.ml.api.NMFOutput`), the loadings of new cells and the
    reconstruction error are the same for every implementation, so they are implemented here.
    """

    _alpha_tol = 1e-5

    def __init__(self, var_names_g: Sequence[str], k_values: list[int], **kwargs):
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.k_values = k_values

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
        Infer the loadings of each program for the input count matrix, by non-negative least squares against the
        consensus factors. To be run after the model has been trained and consensus factors have been computed.

        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.
            consensus_factors: The consensus factors for each k value are in ``consensus_factors[k]["consensus_D_kg"]``.
            k: Number of factors.
            normalize: Whether to normalize the loadings of each cell to sum to 1.
            obs_names_n: Cell names (not used).

        Returns:
            Loadings tensor of shape (n, k).
        """
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        D_kg = consensus_factors[k]["consensus_D_kg"]
        assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"

        alpha_nk = (
            solve_nnls_fista(
                D_kg.to(x_ng.device).unsqueeze(0).transpose(1, 2),
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
        Compute the reconstruction error for each k value in ``consensus_factors``, using the consensus factors
        ``consensus_factors[k]["consensus_D_kg"]``.

        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.
            consensus_factors: The consensus factors for each k value.

        Returns:
            A dictionary mapping each k value to its reconstruction error.
        """
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        rec_error = {}
        for k in consensus_factors.keys():
            D_kg = consensus_factors[k]["consensus_D_kg"]
            assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"
            if (D_kg == 0).all():
                raise ValueError("D_kg is all zeros, please train the model and compute consensus factors first")

            alpha_nk = self.infer_loadings(
                x_ng=x_ng,
                var_names_g=var_names_g,
                consensus_factors=consensus_factors,
                k=k,
                normalize=False,
            ).squeeze(0)

            rec_error[k] = (
                nmf_frobenius_loss(
                    x_ng=x_ng,
                    loadings_nk=alpha_nk.to(x_ng.device),
                    factors_kg=D_kg.to(x_ng.device),
                )
                .sum()
                .item()
            )

        return rec_error


class OnlineNonNegativeMatrixFactorization(NonNegativeMatrixFactorization):
    """
    Use the online NMF algorithm of Mairal et al. [1] to factorize the count matrix
    into a dictionary of gene expression programs and a matrix of cell program loadings.

    **References:**

    1. `Online learning for matrix factorization and sparse coding. Mairal, Bach, Ponce, Sapiro. JMLR 2009.`

    Args:
        var_names_g: The variable names schema for the input data: should be highly variable genes.
        k_values: A list of the number of gene expression programs to infer.
        r: Number of NMF replicates (for consensus).
        solver: The solver for the loadings and dictionary updates, in ["fista", "hals"].
            ``"fista"`` (FISTA for non-negative least squares) recomputes each minibatch's loadings from scratch, so
            memory does not grow with the number of cells, and it supports multi-process training.
            ``"hals"`` (hierarchical alternating least squares, following nmf-torch) stores the loadings of every
            cell (see :attr:`stores_loadings`), requires ``n_cells_total`` and ``obs_names_n``, and does not
            support multi-process training.
        init: The initialization method to use for the NMF factors, in ["sklearn_random", "uniform_random"].
        transformed_data_mean: The mean of the transformed data, used for initialization if an only if
            `init` is "sklearn_random".
        n_cells_total: The total number of cells in the dataset, used if and only if `solver` is "hals".
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        k_values: list[int],
        r: int,
        solver: Literal["fista", "hals"] = "fista",
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
        n_cells_total: int | None = None,
    ) -> None:
        super().__init__(var_names_g=var_names_g, k_values=k_values)
        g = len(self.var_names_g)
        if solver not in ("fista", "hals"):
            raise ValueError(f"Unknown solver: {solver}")
        self.solver = solver
        self.obs_names_to_index_map: dict[str, int] = {}  # used for local latents
        if self.stores_loadings and n_cells_total is None:
            raise ValueError("n_cells_total must be provided when solver='hals'")
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
            if self.stores_loadings:
                assert isinstance(n_cells_total, int)
                self.register_buffer(f"loadings_{i}_rnk", torch.empty(r, n_cells_total, i))  # initialized later

        self._dummy_param = torch.nn.Parameter(torch.empty(()))
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
            if self.stores_loadings:
                init_fn(getattr(self, f"loadings_{i}_rnk"), k=i, transformed_data_mean=self.transformed_data_mean)

        if self.stores_loadings:
            self._prev_err_rk: torch.Tensor | None = None
            self._init_err_rk: torch.Tensor | None = None
            self._err_running_sum_rk = torch.zeros((self.r, len(self.k_values)), device=self._dummy_param.device)
            self._cells_seen_in_epoch = 0  # Track cells seen in current epoch

    @property
    def stores_loadings(self) -> bool:
        """
        Whether the loadings of every cell are stored across minibatches, as buffers of shape
        ``(r, n_cells_total, k)``. True for the ``"hals"`` solver, which warm-starts from them; False for ``"fista"``,
        which recomputes each minibatch's loadings from scratch.
        """
        return self.solver == "hals"

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

        if self.solver == "fista":
            # the loadings of the minibatch are solved from scratch, so nothing per-cell is stored
            updated_values = online_dictionary_update_fista(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                loadings_rnk=torch.zeros(self.r, x_ng.shape[0], k, dtype=factors_rkg.dtype, device=factors_rkg.device),
                A_rkk=A_rkk,
                B_rkg=B_rkg,
            )
            updated_factors_rkg = l1_normalize_factors(updated_values["factors_rkg"])
        else:
            assert isinstance(minibatch_indices_n, torch.Tensor), (
                "minibatch_indices_n must be provided for solver='hals'"
            )
            stored_loadings_rnk = getattr(self, f"loadings_{k}_rnk")
            loadings_rnk = stored_loadings_rnk[:, minibatch_indices_n, :]  # a copy, which hals updates in place
            # run nmf-torch hals online update
            updated_values = online_dictionary_update_nmf_torch_hals(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                loadings_rnk=loadings_rnk,
                A_rkk=A_rkk,
                B_rkg=B_rkg,
            )
            updated_factors_rkg = updated_values["factors_rkg"]
            # store the updated loadings of the minibatch cells, to warm-start the next time these cells are seen
            stored_loadings_rnk[:, minibatch_indices_n, :] = loadings_rnk

        # update running values
        setattr(self, f"A_{k}_rkk", updated_values["A_rkk"])
        setattr(self, f"B_{k}_rkg", updated_values["B_rkg"])
        setattr(self, f"D_{k}_rkg", updated_factors_rkg)

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

        if self.stores_loadings and _distributed_world_size() > 1:
            # the per-cell loadings are rank-local buffers, which DDP would overwrite with rank 0's values
            raise NotImplementedError("solver='hals' does not support multi-process training.")

        # handle the mapping of obs_names to indices for local latents
        minibatch_indices_n: torch.Tensor | None = None
        if self.stores_loadings:
            assert obs_names_n is not None, "obs_names_n must be provided for solver='hals'"
            new_obs_names = set(obs_names_n.tolist()) - set(self.obs_names_to_index_map.keys())
            if len(new_obs_names) > 0:
                self.obs_names_to_index_map |= {
                    name: i for i, name in enumerate(new_obs_names, start=len(self.obs_names_to_index_map))
                }
            vectorized_lookup = np.vectorize(lambda x: self.obs_names_to_index_map[x])
            minibatch_indices_n = torch.from_numpy(vectorized_lookup(obs_names_n)).to(x_ng.device)

        if self.stores_loadings:
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

        if self.stores_loadings:
            # for error computation to assess convergence
            assert isinstance(minibatch_indices_n, torch.Tensor)
            self._loss(x_ng=x_ng, minibatch_indices_n=minibatch_indices_n)

        return {}

    # @torch.compile()
    @torch.no_grad()
    def _loss(self, x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor) -> None:
        """
        Simple and efficient NMF reconstruction loss computation.
        Computes ||X - WH||_F^2 for the current batch.
        """
        x_squared_sum = compute_x_squared_sum(x_ng)
        for i, k in enumerate(self.k_values):
            factors_rkg = getattr(self, f"D_{k}_rkg")  # (r, k, g)
            loadings_rnk = getattr(self, f"loadings_{k}_rnk")[:, minibatch_indices_n, :]  # (r, batch_size, k)

            squared_error_r = frobenius_loss_trace_compiled(
                x_ng=x_ng,
                h_rnk=loadings_rnk,
                w_rkg=factors_rkg,
                x_squared_sum=x_squared_sum,
            )  # (r,)

            # Accumulate the squared error
            self._err_running_sum_rk[:, i] += squared_error_r

        # Track cells seen in this epoch
        self._cells_seen_in_epoch += x_ng.shape[0]

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "OnlineNonNegativeMatrixFactorization requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is True, (
                "OnlineNonNegativeMatrixFactorization requires that the `broadcast_buffers` parameter of "
                "lightning.pytorch.strategies.DDPStrategy is set to True"
            )

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        # convergence criterion check
        if self.stores_loadings:
            cur_err_rk = self._err_running_sum_rk / self._cells_seen_in_epoch
            assert isinstance(self._prev_err_rk, torch.Tensor)
            assert isinstance(self._init_err_rk, torch.Tensor)

            current_overall_err_rk = torch.abs((self._prev_err_rk - cur_err_rk) / self._init_err_rk)
            if current_overall_err_rk.max() < self._hals_tol:
                trainer.should_stop = True
                print(f"Stopping early: converged, loss={cur_err_rk}")

            self._prev_err_rk = cur_err_rk.clone()
            self._err_running_sum_rk.zero_()
            self._cells_seen_in_epoch = 0  # Reset for next epoch

        # this hard reset to zero is equivalent to forgetting momentum each epoch
        for i in self.k_values:
            getattr(self, f"A_{i}_rkk").zero_()
            getattr(self, f"B_{i}_rkg").zero_()

    def on_end(self, trainer: pl.Trainer) -> None:
        trainer.save_checkpoint(trainer.default_root_dir + "/NMF.ckpt")
