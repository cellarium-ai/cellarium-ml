# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Amortized consensus NMF: a slot-attention transformer that learns to *solve* NMF for any ``k``.

Standard consensus NMF (cNMF) requires thousands of independent NMF optimizations to sweep
:math:`k` (number of programs) and :math:`R` (random restarts).  This module trains a single
lightweight transformer to act as an amortized solver for one dataset: given a minibatch of cells,
a value of :math:`k`, and :math:`R` independent noise seeds, it emits :math:`R` replicate
factorizations plus a differentiable Sinkhorn consensus.  Once trained, the stability / error
trade-off curve for every :math:`k` can be measured in minutes rather than GPU-days.

**References:**

1. `Identifying gene expression programs of cell-type identity and cellular activity with
   single-cell RNA-Seq. Kotliar et al. eLife 2019.`
2. `Object-Centric Learning with Slot Attention. Locatello et al. NeurIPS 2020.`
"""

import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from cellarium.ml.models.geometric_sketch import StreamingGeometricSketch
from cellarium.ml.models.model import PredictMixin, ValidateMixin
from cellarium.ml.models.nmf import (
    NonNegativeMatrixFactorization,
    nmf_compute_factors_fista,
    nmf_compute_loadings_fista,
)
from cellarium.ml.models.nmf import consensus as nmf_consensus
from cellarium.ml.utilities.core import call_func_with_batch
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

_EPS = 1e-8
# Size of the residual noise added to a data-dependent slot seed, relative to the spread of the
# cell embeddings.  Only there to separate two slots that seeded on the same cell.
_SEED_JITTER = 0.01


# -----------------------------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------------------------


def l1_normalize_rows(w: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    L1-normalize the last dimension.

    This fixes the NMF scale gauge: rows of ``W`` sum to one and all magnitude lives in ``H``,
    which removes a flat direction from the loss landscape.
    """
    return w / w.sum(dim=-1, keepdim=True).clamp(min=eps)


def l2_normalize_rows(w: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """L2-normalize the last dimension (for cosine distances)."""
    return w / w.norm(dim=-1, keepdim=True).clamp(min=eps)


def sinusoidal_k_encoding(k: int, k_max: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Sinusoidal encoding of the integer ``k`` into a ``dim``-dimensional vector.

    Uses ``k`` directly as the positional input with ``base = k_max``, giving frequencies that
    span ``[1, 1/k_max]``:

    * ``i = 0``: ``sin(k)`` — high frequency, distinguishes adjacent ``k`` values.
    * ``i = dim/2 - 1``: ``sin(k / k_max)`` — low frequency, one full sweep over ``[1, k_max]``.

    Every component carries meaningful signal for ``k ∈ [1, k_max]``.  Using ``k / k_max`` as
    the input with the standard base of 10000 would collapse most components to near zero, because
    the input range ``(0, 1]`` is too narrow for those low frequencies to vary.

    Args:
        k: The integer number of factors for this forward pass.
        k_max: The largest ``k`` in the model's ``k_values``; sets the base for the frequency
            schedule so all components are active for ``k ∈ [1, k_max]``.
        dim: Output dimensionality (should equal ``latent_dim``).
        device: Target device.
        dtype: Target dtype.

    Returns:
        1D tensor of shape ``(dim,)``.
    """
    half = dim // 2
    # Frequencies: k_max^(-2i/dim) for i in [0, half), ranging from 1 (i=0) to 1/k_max (i→half)
    freq = torch.pow(
        torch.tensor(float(k_max), device=device, dtype=torch.float32),
        -torch.arange(half, device=device, dtype=torch.float32) * 2.0 / dim,
    )
    t = torch.tensor(float(k), device=device, dtype=torch.float32)
    args = t * freq
    enc = torch.cat([args.sin(), args.cos()], dim=0)  # (dim,) if dim is even
    if dim % 2 == 1:
        enc = torch.cat([enc, torch.zeros(1, device=device, dtype=torch.float32)])
    return enc.to(dtype=dtype)


def log_sinkhorn(cost_kj: torch.Tensor, epsilon: float = 0.05, n_iterations: int = 50) -> torch.Tensor:
    """
    Entropic optimal transport plan between two equal-size sets, computed in log space.

    The returned plan has unit row sums *and* unit column sums, so contracting it against either
    axis produces a convex combination.  The log-domain formulation is not optional: a naive
    ``exp(-C / epsilon)`` underflows for ``epsilon = 0.05`` and cosine costs approaching 2.

    Args:
        cost_kj: Cost matrix of shape ``(..., k, k)``.
        epsilon: Entropic regularization.  Smaller is closer to a hard permutation but less stable.
        n_iterations: Number of Sinkhorn-Knopp iterations.

    Returns:
        Transport plan of shape ``(..., k, k)``.
    """
    if cost_kj.shape[-1] != cost_kj.shape[-2]:
        raise ValueError(f"log_sinkhorn expects a square cost matrix, got {tuple(cost_kj.shape)}")
    # Sinkhorn is numerically delicate; always run it in fp32 even under autocast.
    log_plan_kj = (-cost_kj / epsilon).float()
    f_k = torch.zeros_like(log_plan_kj[..., :, 0])
    g_j = torch.zeros_like(log_plan_kj[..., 0, :])
    for _ in range(n_iterations):
        f_k = -torch.logsumexp(log_plan_kj + g_j.unsqueeze(-2), dim=-1)
        g_j = -torch.logsumexp(log_plan_kj + f_k.unsqueeze(-1), dim=-2)
    return torch.exp(log_plan_kj + f_k.unsqueeze(-1) + g_j.unsqueeze(-2)).to(cost_kj.dtype)


def align_factors(
    w_rkg: torch.Tensor,
    reference_kg: torch.Tensor,
    epsilon: float = 0.05,
    n_iterations: int = 50,
    detach_plan: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sinkhorn-match every replicate's factors onto a reference set of factors.

    Matching uses cosine distance in **gene space** rather than latent space: it is closer to
    Kotliar's procedure, and it needs no assumption that averaging in the latent space corresponds
    to averaging in gene space (it does not -- the encoder ``LayerNorm`` and the decoder
    ``Softplus`` are both nonlinear).

    Args:
        w_rkg: Replicate factors, shape ``(r, k, g)``, non-negative.
        reference_kg: Reference factors, shape ``(k, g)``.
        epsilon: Sinkhorn entropic regularization.
        n_iterations: Sinkhorn iterations.
        detach_plan: If ``True``, no gradient flows through the transport plan.  The matching is a
            discrete *decision*; differentiating through many Sinkhorn iterations is expensive and
            is the most likely source of instability.  Gradient still flows through the aligned
            factor values themselves.

    Returns:
        ``(aligned_rkg, cosine_rk, plan_rkj, similarity_rkj)``, where ``aligned_rkg[r, j]`` is
        replicate ``r``'s factor matched to reference factor ``j``, ``cosine_rk[r, j]`` is its
        cosine similarity to that reference factor, and ``similarity_rkj[r, i, j]`` is the full
        cosine similarity matrix between replicate factors and reference factors (before matching).
    """
    reference_normalized_kg = l2_normalize_rows(reference_kg)
    similarity_rkj = torch.einsum("rkg,jg->rkj", l2_normalize_rows(w_rkg), reference_normalized_kg)
    cost_rkj = 1.0 - similarity_rkj
    plan_rkj = log_sinkhorn(cost_rkj.detach() if detach_plan else cost_rkj, epsilon, n_iterations)
    if detach_plan:
        plan_rkj = plan_rkj.detach()
    aligned_rkg = torch.einsum("rkj,rkg->rjg", plan_rkj, w_rkg)
    cosine_rk = torch.einsum("rjg,jg->rj", l2_normalize_rows(aligned_rkg), reference_normalized_kg)
    return aligned_rkg, cosine_rk, plan_rkj, similarity_rkj


def matched_distance(similarity_rkj: torch.Tensor, plan_rkj: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance from each factor to the reference factor it was matched to, using a **hard**
    assignment (the argmax of the transport plan).

    This matters for the drift criterion.  The soft-aligned factor returned by
    :func:`align_factors` is a *blend* of factors weighted by the transport plan, so its cosine to
    the reference is strictly less than 1 even for two **identical** factor sets -- at
    ``epsilon = 0.05`` the residual is on the order of ``1e-3``.  That would give the drift metric a
    nonzero floor far above any sensible tolerance, and early stopping would never fire.  With a
    hard assignment, identical factor sets give exactly zero.

    Args:
        similarity_rkj: Cosine similarities between factors and reference factors.
        plan_rkj: Transport plan from :func:`align_factors`.

    Returns:
        Distance per factor, shape ``(r, k)``.
    """
    matched_rk1 = plan_rkj.argmax(dim=-1, keepdim=True)
    return 1.0 - similarity_rkj.gather(-1, matched_rk1).squeeze(-1)


def matched_silhouette(similarity_rkj: torch.Tensor, plan_rkj: torch.Tensor) -> torch.Tensor:
    """
    Silhouette-like stability score for each matched factor, in ``[-1, 1]``.

    Mean cosine similarity is *not* usable as a stability metric for non-negative factors: two
    independent random non-negative vectors already score around 0.75, and after matching and
    averaging they score around 0.94, so the entire interesting range is compressed into the last
    few percent.  Kotliar's silhouette avoids this because it is a *contrast* -- own-cluster
    tightness against nearest-other-cluster separation -- and the baseline cancels.

    This is the direct analogue: for each replicate factor, ``a`` is its distance to the reference
    factor it was matched to and ``b`` is its distance to the nearest *other* reference factor, and
    the score is ``(b - a) / max(a, b)``.  It is near 1 when a factor is unambiguously reproduced,
    and near 0 when it sits equally close to two consensus programs.

    Args:
        similarity_rkj: Cosine similarities between replicate factors and reference factors.
        plan_rkj: Transport plan from :func:`align_factors`.

    Returns:
        Score per replicate factor, shape ``(r, k)``.
    """
    k = similarity_rkj.shape[-1]
    if k == 1:
        # A single factor is trivially unambiguous; there is no "nearest other" to contrast with.
        return torch.ones_like(similarity_rkj[..., 0])
    matched_rk1 = plan_rkj.argmax(dim=-1, keepdim=True)
    a_rk = matched_distance(similarity_rkj, plan_rkj)
    other_rkj = similarity_rkj.scatter(-1, matched_rk1, float("-inf"))
    b_rk = 1.0 - other_rkj.max(dim=-1).values
    return (b_rk - a_rk) / torch.maximum(a_rk, b_rk).clamp(min=_EPS)


def sinkhorn_consensus(
    w_rkg: torch.Tensor,
    epsilon: float = 0.05,
    n_iterations: int = 50,
    outlier_gamma: float = 5.0,
    n_refine: int = 2,
    detach_plan: bool = True,
    anchor: int = 0,
) -> dict[str, torch.Tensor | None]:
    """
    Differentiable consensus over ``r`` replicate factorizations, replacing cNMF's k-means step.

    K-means cannot be used here: it lacks a one-to-one matching constraint (so it mode-collapses)
    and its hard assignments sever the computation graph.  Sinkhorn-Knopp gives a GPU-native,
    differentiable, doubly-stochastic matching instead.

    Outliers are handled the way cNMF handles them, but natively inside the forward pass: a
    replicate factor that matches poorly (large cosine distance to the running consensus) receives
    weight ``exp(-outlier_gamma * distance)``, decaying toward zero.

    The consensus is refined toward a barycenter rather than anchored on one replicate.  Anchoring
    permanently on replicate 0 would make the consensus asymmetric and give that replicate
    privileged, unpenalized status, unlike k-means which is symmetric.  Refinement rounds are
    detached: they are a fixed-point *search*, and only the final aggregation carries gradient.

    Args:
        w_rkg: Replicate factors of shape ``(r, k, g)``, non-negative.
        epsilon: Sinkhorn entropic regularization.
        n_iterations: Sinkhorn iterations.
        outlier_gamma: Decay rate of the outlier-masking weights.
        n_refine: Number of detached barycenter refinement rounds.
        detach_plan: See :func:`align_factors`.
        anchor: Replicate index used to seed the refinement.

    Returns:
        Dict with keys ``consensus_kg`` (L1-normalized), ``stability`` (mean
        :func:`matched_silhouette` score over replicate factors), ``agreement`` (mean cosine
        similarity to the consensus, which is what the outlier weights are built from),
        ``cosine_rk``, ``weights_rk`` and ``plan_rkj`` (``None`` when ``r == 1``).
    """
    r = w_rkg.shape[0]
    if r == 1:
        ones_rk = w_rkg.new_ones(w_rkg.shape[:2])
        return {
            "consensus_kg": l1_normalize_rows(w_rkg[0]),
            "stability": w_rkg.new_ones(()),
            "agreement": w_rkg.new_ones(()),
            "cosine_rk": ones_rk,
            "weights_rk": ones_rk,
            "plan_rkj": None,
        }

    # Detached barycenter refinement: settle on a reference that no single replicate owns.
    reference_kg = w_rkg[anchor].detach()
    w_detached_rkg = w_rkg.detach()
    for _ in range(max(0, n_refine)):
        aligned_rkg, cosine_rk, _, _ = align_factors(w_detached_rkg, reference_kg, epsilon, n_iterations, True)
        weights_rk1 = torch.exp(-outlier_gamma * (1.0 - cosine_rk)).unsqueeze(-1)
        reference_kg = ((weights_rk1 * aligned_rkg).sum(dim=0) / weights_rk1.sum(dim=0).clamp(min=_EPS)).detach()

    # Final, differentiable pass against the settled reference.
    aligned_rkg, cosine_rk, plan_rkj, similarity_rkj = align_factors(
        w_rkg, reference_kg, epsilon, n_iterations, detach_plan
    )
    weights_rk = torch.exp(-outlier_gamma * (1.0 - cosine_rk))
    weights_rk1 = weights_rk.unsqueeze(-1)
    consensus_kg = (weights_rk1 * aligned_rkg).sum(dim=0) / weights_rk1.sum(dim=0).clamp(min=_EPS)
    return {
        "consensus_kg": l1_normalize_rows(consensus_kg),
        "stability": matched_silhouette(similarity_rkj.detach(), plan_rkj.detach()).mean(),
        "agreement": cosine_rk.detach().mean(),
        "cosine_rk": cosine_rk,
        "weights_rk": weights_rk,
        "plan_rkj": plan_rkj,
    }


def match_stability(
    w1_kg: torch.Tensor,
    w2_kg: torch.Tensor,
    epsilon: float = 0.05,
    n_iterations: int = 50,
) -> torch.Tensor:
    """
    Mean cosine similarity between two independently derived factor sets after optimal matching.

    This is the *cross-batch* stability statistic.  Within-batch stability (see
    :func:`sinkhorn_consensus`) folds in initialization variance only, exactly like Kotliar's
    silhouette.  This statistic additionally folds in *sampling* variance, and therefore answers
    "is this program a property of the data, or of this particular sample of cells?".  The gap
    between the two is the diagnostic for how much stability is a sampling artifact.

    Uses the same :func:`matched_silhouette` contrast as within-batch stability, so the two are on
    a comparable scale and free of the non-negative cosine baseline.
    """
    _, _, plan_rkj, similarity_rkj = align_factors(w1_kg.unsqueeze(0), w2_kg, epsilon, n_iterations, detach_plan=True)
    return matched_silhouette(similarity_rkj, plan_rkj).mean()


def frobenius_loss_trace(
    x_ng: torch.Tensor,
    h_rnk: torch.Tensor,
    w_rkg: torch.Tensor,
    x_squared_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-replicate ``||X - H W||_F^2`` computed *without* materializing the reconstruction.

    Uses :math:`\\|X - HW\\|_F^2 = \\|X\\|^2 - 2\\langle HX, W\\rangle + \\langle H^\\top H,
    WW^\\top\\rangle`, so the intermediates are ``(r, k, g)`` and ``(r, k, k)`` rather than
    ``(r, n, g)``.  At ``r=100, n=2048, g=2000`` that is 80 MB instead of 1.6 GB.  And because
    ``H`` is detached and ``X`` is data, both ``HX`` and ``H^T H`` are autograd constants: the
    entire backward reduces to :math:`\\nabla_W = -2 HX + 2 (H^\\top H) W`.

    This is why the loss is Frobenius rather than Poisson KL.  KL does not factorize this way, and
    its Poisson justification does not survive per-gene rescaling of the input anyway.

    Args:
        x_ng: Data, shape ``(n, g)``.
        h_rnk: Loadings, shape ``(r, n, k)``.
        w_rkg: Factors, shape ``(r, k, g)``.
        x_squared_sum: Optional precomputed ``(x_ng ** 2).sum()``, shared across replicates.

    Returns:
        Sum of squared errors per replicate, shape ``(r,)``.
    """
    if x_squared_sum is None:
        x_squared_sum = x_ng.pow(2).sum()
    hx_rkg = torch.einsum("rnk,ng->rkg", h_rnk, x_ng)
    hth_rkk = torch.einsum("rnk,rnj->rkj", h_rnk, h_rnk)
    wwt_rkk = torch.einsum("rkg,rjg->rkj", w_rkg, w_rkg)
    cross_r = (hx_rkg * w_rkg).sum(dim=(-2, -1))
    quad_r = (hth_rkk * wwt_rkk).sum(dim=(-2, -1))
    # Upcasting only the three reduced scalars is free and removes any cancellation concern.
    sse_r = x_squared_sum.double() - 2.0 * cross_r.double() + quad_r.double()
    return sse_r.clamp(min=0.0).to(w_rkg.dtype)


def _all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across DDP ranks in place (no-op when not distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= torch.distributed.get_world_size()
    return tensor


def _broadcast_from_rank_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast rank 0's copy of a tensor to all ranks in place (no-op when not distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=0)
    return tensor


def _weights_init(m: torch.nn.Module) -> None:
    """Re-initialize a module's parameters.  Required because models are built on the meta device."""
    if isinstance(m, torch.nn.MultiheadAttention):
        m._reset_parameters()
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        if m.elementwise_affine:
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, std=0.02)


# -----------------------------------------------------------------------------------------------
# Sub-modules
# -----------------------------------------------------------------------------------------------


class LinearCellEncoder(torch.nn.Module):
    """
    Deliberately low-capacity cell encoder: one linear layer plus ``LayerNorm``, nothing else.

    Keeping the encoder shallow forces the transformer's attention layers to do all of the
    optimization work rather than letting a deep MLP absorb it.  ``LayerNorm`` discards per-cell
    magnitude, which is fine here: the embedding only feeds attention (where "which program"
    matters, not "how much") and the loadings hot start, whose scale is set analytically and then
    polished by FISTA.
    """

    def __init__(self, n_genes: int, latent_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_genes, latent_dim)
        self.norm = torch.nn.LayerNorm(latent_dim)

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x_ng))


class LinearFactorDecoder(torch.nn.Module):
    """
    Deliberately low-capacity factor decoder: one linear layer, ``Softplus``, L1-normalized rows.

    The L1 normalization matches the convention used by
    :class:`~cellarium.ml.models.AmortizedOnlineNonNegativeMatrixFactorization`, so factors can be
    handed off between the two models.  Do not change this to L2 without changing that too.
    """

    def __init__(self, latent_dim: int, n_genes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(latent_dim, n_genes)

    def forward(self, z_rke: torch.Tensor) -> torch.Tensor:
        return l1_normalize_rows(F.softplus(self.linear(z_rke)))


class SlotAttentionBlock(torch.nn.Module):
    """
    One layer of the "physics engine": slot self-attention, then slot *cross*-attention to cells.

    Two departures from :class:`torch.nn.TransformerDecoderLayer`, both load-bearing:

    1. **The cross-attention softmax is over the slot (``k``) axis, not the cell axis.**  Ordinary
       cross-attention gives the ``k`` factors no reason not to be duplicates -- self-attention is
       permutation-equivariant and carries no repulsive term, so the usual outcome is mode
       collapse.  Slot Attention's anti-collapse mechanism is competition *between* slots for each
       input, which is exactly what a softmax over ``k`` provides.
    2. **Aggregation over cells is a normalized weighted mean, not a weighted sum.**  This makes
       the update depend on the empirical *distribution* of cells rather than on their count, so
       the block behaves the same whether it sees 2,048 cells or 500,000.

    Keys and values are projected once by the parent model and shared across layers (as in Slot
    Attention, which applies one recurrent module repeatedly) and across replicates -- expanding
    the cell memory to ``(r, n, e)`` per layer would duplicate ~1.3 GB at ``r=100`` for no gain.

    Each block receives a per-forward-pass conditioning vector ``cond_e`` derived from the
    sinusoidal encoding of ``k``.  It uses Adaptive LayerNorm (AdaLN / FiLM) to scale and shift
    all three normalization points, so ``k`` conditions the block's computation at every step rather
    than only at slot initialization.  The AdaLN projection is zero-initialized so the block starts
    as identity with respect to ``k`` conditioning and learns to use it gradually.
    """

    def __init__(self, latent_dim: int, n_self_attention_heads: int = 8, ffn_mult: int = 4) -> None:
        super().__init__()
        # Non-affine LayerNorms: scale and shift are supplied by AdaLN instead.
        self.norm_self_attention = torch.nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.self_attention = torch.nn.MultiheadAttention(latent_dim, n_self_attention_heads, batch_first=True)
        self.norm_cross_attention = torch.nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.to_query = torch.nn.Linear(latent_dim, latent_dim, bias=False)
        self.norm_ffn = torch.nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, ffn_mult * latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ffn_mult * latent_dim, latent_dim),
        )
        self.scale = latent_dim**-0.5
        # AdaLN: one linear layer projects the shared conditioning vector to (scale, shift) for
        # each of the three norm points (self-attn, cross-attn, ffn) → 6 * latent_dim outputs.
        # Zero-initialized so the block starts as identity w.r.t. k conditioning.
        # skip_init avoids the default kaiming init (which would consume random numbers and then be
        # discarded), keeping the global random state stable so other parameter initializations
        # (especially slot_mu) are unaffected by whether AdaLN is present.
        adaLN = torch.nn.utils.skip_init(torch.nn.Linear, latent_dim, 6 * latent_dim)
        torch.nn.init.zeros_(adaLN.weight)
        torch.nn.init.zeros_(adaLN.bias)
        self.adaLN = adaLN

    def forward(
        self,
        slots_rke: torch.Tensor,
        key_ne: torch.Tensor,
        value_ne: torch.Tensor,
        cond_e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slots_rke: Slot (factor) tokens, shape ``(r, k, e)``.
            key_ne: Projected cell keys, shape ``(n, e)``, shared across replicates.
            value_ne: Projected cell values, shape ``(n, e)``, shared across replicates.
            cond_e: AdaLN conditioning vector, shape ``(e,)``, derived from ``k``.

        Returns:
            Updated slots, shape ``(r, k, e)``.
        """
        # Unpack the 6 AdaLN parameters (scale1, shift1, scale2, shift2, scale3, shift3).
        params = self.adaLN(cond_e)  # (6e,)
        s1, b1, s2, b2, s3, b3 = params.chunk(6, dim=-1)  # each (e,)

        # Slots attend to each other within a replicate.
        normed_rke = (1 + s1) * self.norm_self_attention(slots_rke) + b1
        attended_rke, _ = self.self_attention(normed_rke, normed_rke, normed_rke, need_weights=False)
        slots_rke = slots_rke + attended_rke

        # Slots compete for cells (softmax over k), then aggregate by normalized weighted mean.
        query_rke = self.to_query((1 + s2) * self.norm_cross_attention(slots_rke) + b2) * self.scale
        logits_rkn = torch.einsum("rke,ne->rkn", query_rke, key_ne)
        attention_rkn = logits_rkn.softmax(dim=-2)
        attention_rkn = attention_rkn / attention_rkn.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        slots_rke = slots_rke + torch.einsum("rkn,ne->rke", attention_rkn, value_ne)

        return slots_rke + self.ffn((1 + s3) * self.norm_ffn(slots_rke) + b3)


# -----------------------------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------------------------


class CNMFTransformer(NonNegativeMatrixFactorization, ValidateMixin, PredictMixin):
    """
    Universal amortized consensus-NMF solver: one transformer that factorizes for any ``k``.

    Every training step samples a ``k`` from a growing curriculum window, embeds the cells with a
    strictly linear encoder, runs ``R`` independent noise seeds through a slot-attention stack
    (cells as keys/values), and decodes each layer's slots to replicate factor matrices ``W``.
    Loadings ``H`` are solved with a detached FISTA polish of the final-layer ``W``.  Training
    minimizes a **self-consistency loss**: the mean squared error between each layer's decoded ``W``
    and a stop-gradient target ``W'`` produced by running :func:`nmf_compute_factors_fista` on the
    final-layer ``W`` given the stop-gradient ``H``.  Per-layer decoders (the AlphaFold trick)
    give early layers a direct gradient path without running FISTA per layer.

    Three design constraints are deliberate and should not be "simplified" away:

    * **Stop-gradient on H.**  The FISTA polish is fully detached.  If gradients flowed through the
      solver, the transformer would spend capacity learning to invert FISTA instead of shaping the
      geometric basins of ``W``.
    * **Low-capacity encoder and per-layer decoders.**  Single linear layers only.  All of the
      optimization work belongs to the attention stack.
    * **Self-consistency loss.**  The training signal is ``||W_l - stop_grad(W')||²`` summed over
      layers, where ``W'`` is produced by running ``fista_w_iterations_train`` FISTA steps on the
      final-layer ``W`` given the stop-gradient ``H``.  This is zero if and only if the transformer
      outputs a true NMF fixed point, which avoids the gradient symmetry that causes mode collapse
      under a pure Frobenius reconstruction loss.

    Convergence is judged by *factor drift* -- how far the consensus programs rotate per optimizer
    step, measured with fixed noise on a fixed set of cells -- and never by the training loss.  The
    loss is not usable as a convergence signal here: ``k`` is resampled every step, so consecutive
    losses come from different problems, and the curriculum systematically shifts the ``k``
    distribution over time, so the loss series carries a trend driven by the schedule rather than
    by learning.

    .. note::
        If ``cross_batch_stability`` is ``True`` the incoming minibatch is split in half, so set the
        dataloader ``batch_size`` to **twice** the number of cells each factorization should
        condition on.

    .. note::
        ``broadcast_buffers=False`` is recommended for
        :class:`~lightning.pytorch.strategies.DDPStrategy` with this model.  Everything that must
        agree across ranks is either derived deterministically or synchronized explicitly, and the
        persisted consensus factors are large enough that broadcasting all buffers every step is
        pure waste.

    Args:
        var_names_g: The variable names schema for the input data: should be highly variable genes.
        k_values: The universe of ``k`` values the model may be asked to solve.  Sorted ascending
            and revealed progressively by the curriculum.  This is also the grid on which the
            stability and error curves are measured, so pass every integer you want on the plot --
            stability curves are frequently discontinuous and a coarse grid will miss structure.
        latent_dim: Width of the latent/slot space.
        n_iterations: Number of slot-attention passes.  In recurrent mode this is the number of
            times the *same* block and decoder are applied; in non-recurrent mode it is the number
            of independent block/decoder pairs (each pass gets its own weights).
        recurrent: When ``True`` (default) a single :class:`SlotAttentionBlock` and
            :class:`LinearFactorDecoder` are shared across all ``n_iterations`` passes, mirroring an
            unrolled iterative solver.  When ``False`` each pass gets its own independent block and
            decoder; parameter count scales with ``n_iterations`` but each stage can specialize.
        n_self_attention_heads: Heads for slot-to-slot self-attention.
        ffn_mult: Feed-forward expansion factor inside each block.
        n_replicates: ``R`` used during training.  Fresh replicates are drawn every step and
            information accumulates over thousands of steps, so this can be far smaller than
            Kotliar's 100 without hurting the gradient.
        cross_batch_stability: Split the minibatch in half and assign half the replicates to each,
            enabling the cross-batch stability statistic.
        min_cells_per_split: Minimum cells per half; below this the split is skipped.
        shuffle_split: Randomly permute rows before splitting, guarding against a dataloader that
            returns cells in a systematic order.
        fista_iterations_train: FISTA iterations for the loadings polish during training.  This is
            a *polish* of a good hot start, not a solve from scratch, so it can be small.
        fista_w_iterations_train: FISTA iterations for polishing ``W`` into the self-consistency
            target ``W'`` during training.  More steps → cleaner training signal; the default 100
            is usually well-converged for the hot-started ``W``.
        sinkhorn_epsilon: Entropic regularization for consensus matching.
        sinkhorn_iterations: Sinkhorn iterations.
        sinkhorn_refine_rounds: Detached barycenter refinement rounds.
        outlier_gamma: Decay rate of the consensus outlier-masking weights.
        detach_sinkhorn_plan: Do not backpropagate through the transport plan.
        curriculum_warmup_steps: Steps over which the ``k`` window grows to cover all ``k_values``.
        curriculum_initial_k_count: How many of the smallest ``k_values`` are available at step 0.
        stability_burn_in_steps: Do not accumulate monitoring EMAs before this step.  The default
            is 0 (accumulate from the first step); set it if you want to discard the noisiest
            early-training data from the per-k EMA curves.
        stability_ema_beta: Per-``k`` EMA decay for the monitoring curves.
        drift_k_values: ``k`` values at which factor drift is measured.  Defaults to the smallest,
            median and largest of ``k_values``.  Reduced with a max, so the worst ``k`` governs.
        drift_eval_n_cells: Size of the fixed cell set used for the drift measurement.
        drift_check_every_n_steps: Steps between drift checks.  Zero disables early stopping.
        drift_tol: Convergence threshold on ``1 - cosine similarity`` **per optimizer step** after
            the LR reduction fires.  Calibrate for the post-reduction learning rate
            (``initial_lr / lr_reduction_factor``), not the training LR.
        drift_patience_checks: Consecutive checks below ``drift_tol`` required to stop.
        drift_settle_steps: Extra steps after the curriculum finishes before the LR is reduced and
            stopping is allowed.  Checking earlier is meaningless: while the ``k`` window is still
            expanding, new ``k`` values are still being introduced and drift *should* be nonzero.
        lr_reduction_factor: Factor by which the optimizer learning rate is divided once the
            curriculum and settle window have elapsed.  The reduced LR lowers the gradient-update
            floor so that ``drift_tol`` becomes reachable.  Set to ``1.0`` to disable (keeps the
            original behavior where the LR is never touched and stopping is allowed from the end of
            the settle window).
        measure_at_end: Run the measurement phase automatically in ``on_train_end``.
        measurement_n_batches: Minibatches used by the measurement phase.
        k_sampling_seed: Seed for ``k`` sampling.  ``k`` is drawn from a step-seeded generator so
            that every DDP rank draws the *same* ``k``; independent per-rank sampling would produce
            mismatched tensor shapes and hang the gradient all-reduce.
        noise_seed: Seed for the fixed drift-evaluation noise.
        log_every_n_steps: Interval for logging monitoring scalars.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        k_values: list[int],
        latent_dim: int = 256,
        n_iterations: int = 3,
        recurrent: bool = True,
        n_self_attention_heads: int = 8,
        ffn_mult: int = 4,
        loss_discount_ratio: float = 2.0,
        n_replicates: int = 32,
        cross_batch_stability: bool = True,
        min_cells_per_split: int = 64,
        shuffle_split: bool = True,
        fista_iterations_train: int = 25,
        fista_w_iterations_train: int = 100,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 50,
        sinkhorn_refine_rounds: int = 2,
        outlier_gamma: float = 0.0,
        detach_sinkhorn_plan: bool = True,
        curriculum_warmup_steps: int = 100,
        curriculum_initial_k_count: int = 1,
        stability_burn_in_steps: int = 0,
        stability_ema_beta: float = 0.9,
        drift_k_values: list[int] | None = None,
        drift_eval_n_cells: int = 2048,
        drift_check_every_n_steps: int = 100,
        drift_tol: float = 1e-6,
        drift_patience_checks: int = 5,
        drift_settle_steps: int = 1000,
        lr_reduction_factor: float = 10.0,
        measure_at_end: bool = True,
        measurement_n_batches: int = 50,
        k_sampling_seed: int = 0,
        noise_seed: int = 1,
        log_every_n_steps: int = 50,
        use_reservoir: bool = True,
        reservoir_n_bits: int = 12,
        reservoir_max_cells_per_bucket: int = 2,
        reservoir_seed: int = 0,
    ) -> None:
        if len(k_values) == 0:
            raise ValueError("k_values must not be empty")
        if min(k_values) < 1:
            raise ValueError(f"k_values must all be >= 1, got minimum {min(k_values)}")
        if len(set(k_values)) != len(k_values):
            raise ValueError("k_values must not contain duplicates")
        sorted_k_values = sorted(k_values)
        if list(k_values) != sorted_k_values:
            warnings.warn("k_values was not sorted ascending; sorting it so the curriculum is monotone.", UserWarning)

        super().__init__(var_names_g=var_names_g, k_values=sorted_k_values)
        g = len(self.var_names_g)
        self.n_genes = g
        self.latent_dim = latent_dim
        self.n_iterations = n_iterations
        self.loss_discount_ratio = loss_discount_ratio
        # Geometric discount weights: later iterations get more weight.  Normalized to sum to 1
        # so the loss scale is stable regardless of n_iterations or loss_discount_ratio.
        raw = [loss_discount_ratio**i for i in range(n_iterations)]
        total = sum(raw)
        self._iter_weights: list[float] = [w / total for w in raw]
        self.n_replicates = n_replicates
        self.cross_batch_stability = cross_batch_stability
        self.min_cells_per_split = min_cells_per_split
        self.shuffle_split = shuffle_split
        self.fista_iterations_train = fista_iterations_train
        self.fista_w_iterations_train = fista_w_iterations_train
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_refine_rounds = sinkhorn_refine_rounds
        self.outlier_gamma = outlier_gamma
        self.detach_sinkhorn_plan = detach_sinkhorn_plan
        self.curriculum_warmup_steps = curriculum_warmup_steps
        self.curriculum_initial_k_count = max(1, min(len(self.k_values), curriculum_initial_k_count))
        self.stability_burn_in_steps = stability_burn_in_steps
        self.stability_ema_beta = stability_ema_beta
        self.drift_eval_n_cells = drift_eval_n_cells
        self.drift_check_every_n_steps = drift_check_every_n_steps
        self.drift_tol = drift_tol
        self.drift_patience_checks = drift_patience_checks
        self.drift_settle_steps = drift_settle_steps
        self.lr_reduction_factor = lr_reduction_factor
        self.measure_at_end = measure_at_end
        self.measurement_n_batches = measurement_n_batches
        self.k_sampling_seed = k_sampling_seed
        self.noise_seed = noise_seed
        self.log_every_n_steps = log_every_n_steps

        self.k_max = max(self.k_values)
        self._k_to_index = {k: i for i, k in enumerate(self.k_values)}
        self.metric_names: tuple[str, ...] = ("stability", "stability_cross", "error", "error_cross")

        self.drift_k_values = self._default_drift_k_values() if drift_k_values is None else sorted(drift_k_values)
        unknown = [k for k in self.drift_k_values if k not in self._k_to_index]
        if unknown:
            raise ValueError(f"drift_k_values {unknown} are not in k_values")

        # --- modules ---
        self.encoder = LinearCellEncoder(g, latent_dim)
        # Keys and values are projected once and shared across iterations and replicates.
        self.to_key = torch.nn.Linear(latent_dim, latent_dim, bias=False)
        self.to_value = torch.nn.Linear(latent_dim, latent_dim, bias=False)
        self.recurrent = recurrent
        n_modules = 1 if recurrent else n_iterations
        self.blocks = torch.nn.ModuleList(
            [SlotAttentionBlock(latent_dim, n_self_attention_heads, ffn_mult) for _ in range(n_modules)]
        )
        self.decoders = torch.nn.ModuleList([LinearFactorDecoder(latent_dim, g) for _ in range(n_modules)])
        # Non-affine on purpose: the loadings hot start is inside a no_grad region (stop-gradient on
        # H), so learnable parameters here could never receive a gradient and would sit dead.
        self.norm_hot_start = torch.nn.LayerNorm(latent_dim, elementwise_affine=False)
        # Slot initialization, as in Slot Attention but seeded on the data manifold: the noise
        # tokens pick which cell embedding seeds each slot (see _initial_slots), and slot_mu is a
        # learned offset applied to every seed.  slot_log_sigma is intentionally frozen at 0
        # (sigma = 1) and now only scales the residual tie-breaking jitter; the exploration that
        # k-selection needs comes from the noise choosing different seed cells per replicate, which
        # no parameter can suppress (posterior collapse is structurally ruled out rather than
        # merely discouraged).
        self.slot_mu = torch.nn.Parameter(torch.empty(latent_dim))
        self.slot_log_sigma = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
        # k-conditioning via sinusoidal encoding of k/k_max → shared MLP → conditioning vector.
        # The MLP output is passed into each block's AdaLN and also used to shift slot initialization.
        # Sinusoidal encoding (rather than a learned discrete embedding) enables smooth
        # generalization to all k values, including those underrepresented during training.
        self.k_to_cond = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(latent_dim, latent_dim),
        )

        # --- buffers: declared here, valued in reset_parameters (models are built on meta) ---
        self._step: torch.Tensor
        self.drift_slot_noise_rke: torch.Tensor
        self._drift_x_ng: torch.Tensor
        self._drift_n_captured: torch.Tensor
        self._drift_has_previous: torch.Tensor
        self._drift_below_tol_count: torch.Tensor
        self._drift_rate: torch.Tensor
        self._lr_reduced: torch.Tensor
        self._measured_n_batches: torch.Tensor
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("drift_slot_noise_rke", torch.empty(n_replicates, self.k_max, latent_dim))
        self.register_buffer("_drift_x_ng", torch.empty(drift_eval_n_cells, g))
        self.register_buffer("_drift_n_captured", torch.zeros((), dtype=torch.long))
        self.register_buffer("_drift_has_previous", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_drift_below_tol_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_drift_rate", torch.zeros(()))
        self.register_buffer("_lr_reduced", torch.zeros((), dtype=torch.bool))
        for k in self.drift_k_values:
            self.register_buffer(f"_drift_previous_w_{k}_kg", torch.empty(k, g))

        n_k = len(self.k_values)
        for name in self.metric_names:
            self.register_buffer(f"_ema_{name}_k", torch.empty(n_k))
            self.register_buffer(f"_ema_count_{name}_k", torch.empty(n_k))
        # _measured_{name}_mean_k, _measured_{name}_sem_k, and _measured_k_values are registered
        # dynamically by run_measurement_phase() because their size depends on measurement-time
        # k_values, which may differ from the training k_values.
        self.register_buffer("_measured_n_batches", torch.zeros((), dtype=torch.long))

        for k in self.k_values:
            self.register_buffer(f"consensus_D_{k}_kg", torch.empty(k, g))
        # D_{k}_rkg buffers are registered dynamically by run_measurement_phase() when
        # store_replicates_k_values is specified there.

        self.predict_k: int | None = None
        self._last_k = self.k_values[0]
        self._last_sc_loss: float = float("nan")

        self.use_reservoir = use_reservoir
        self._reservoir_cell_counter: int = 0
        if use_reservoir:
            self.reservoir: StreamingGeometricSketch | None = StreamingGeometricSketch(
                var_names_g=np.array(self.var_names_g),
                n_bits=reservoir_n_bits,
                max_cells_per_bucket=reservoir_max_cells_per_bucket,
                store_cell_data=True,
                seed=reservoir_seed,
            )
            # Disable the dummy DDP-compatibility param; CNMFTransformer's own params handle that.
            self.reservoir._dummy_param.requires_grad_(False)
        else:
            self.reservoir = None

        self.reset_parameters()

    # -------------------------------------------------------------------------------------------
    # construction / initialization
    # -------------------------------------------------------------------------------------------

    def _default_drift_k_values(self) -> list[int]:
        return sorted({self.k_values[0], self.k_values[len(self.k_values) // 2], self.k_values[-1]})

    def reset_parameters(self) -> None:
        self.apply(_weights_init)
        torch.nn.init.normal_(self.slot_mu, std=0.02)
        # slot_log_sigma is frozen at 0 (sigma=1); do not reinitialize it here.

        self._step.zero_()
        self._step_cache = 0
        self._last_k = self.k_values[0]
        self._last_sc_loss = float("nan")

        # Fixed, reproducible drift noise.  Generated here rather than in __init__ because __init__
        # runs on the meta device; a CPU generator keeps it identical across devices and runs.
        generator = torch.Generator().manual_seed(self.noise_seed)
        self.drift_slot_noise_rke.copy_(torch.randn(tuple(self.drift_slot_noise_rke.shape), generator=generator))
        self._drift_x_ng.zero_()
        self._drift_n_captured.zero_()
        self._drift_cells_full = False
        self._drift_has_previous.fill_(False)
        self._drift_below_tol_count.zero_()
        self._drift_rate.zero_()
        self._lr_reduced.fill_(False)
        for k in self.drift_k_values:
            getattr(self, f"_drift_previous_w_{k}_kg").zero_()

        for name in self.metric_names:
            getattr(self, f"_ema_{name}_k").zero_()
            getattr(self, f"_ema_count_{name}_k").zero_()
        self._measured_n_batches.zero_()
        # Drop all dynamically-registered inference buffers so post-reset state is clean.
        for name in self.metric_names:
            self._buffers.pop(f"_measured_{name}_mean_k", None)
            self._buffers.pop(f"_measured_{name}_sem_k", None)
        self._buffers.pop("_measured_k_values", None)
        for key in [k for k in self._buffers if k.startswith("D_") and k.endswith("_rkg")]:
            self._buffers.pop(key, None)

        for k in self.k_values:
            getattr(self, f"consensus_D_{k}_kg").zero_()

        if self.use_reservoir and self.reservoir is not None:
            self.reservoir.reset_parameters()
            self._reservoir_cell_counter = 0

    @property
    def device(self) -> torch.device:
        return self.slot_mu.device

    # -------------------------------------------------------------------------------------------
    # curriculum
    # -------------------------------------------------------------------------------------------

    def curriculum_k_count(self, step: int | None = None) -> int:
        """Number of ``k_values`` (from the smallest up) currently available to the sampler."""
        step = self._step_cache if step is None else step
        n = len(self.k_values)
        if self.curriculum_warmup_steps <= 0:
            return n
        progress = min(1.0, step / self.curriculum_warmup_steps)
        count = self.curriculum_initial_k_count + progress * (n - self.curriculum_initial_k_count)
        return int(min(n, max(1, math.ceil(count))))

    def sample_k(self, step: int | None = None) -> int:
        """
        Draw ``k`` from the curriculum window using a step-seeded generator.

        Determinism here is not cosmetic: under DDP every rank must draw the *same* ``k``, or the
        replicate tensors have different shapes on different ranks and the gradient all-reduce
        deadlocks.  The noise seeds are deliberately left rank-dependent -- extra diversity there
        is useful.
        """
        step = self._step_cache if step is None else step
        generator = torch.Generator().manual_seed(self.k_sampling_seed * 1_000_003 + step)
        index = int(torch.randint(0, self.curriculum_k_count(step), (1,), generator=generator).item())
        return self.k_values[index]

    # -------------------------------------------------------------------------------------------
    # core solver
    # -------------------------------------------------------------------------------------------

    def _k_cond(self, k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute the shared AdaLN conditioning vector for ``k``."""
        enc = sinusoidal_k_encoding(k, self.k_max, self.latent_dim, device, dtype)
        return self.k_to_cond(enc)

    # def _initial_slots(
    #     self,
    #     k: int,
    #     slot_noise_rke: torch.Tensor,
    #     cond_e: torch.Tensor,
    #     x_emb_ne: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Data-dependent slot initialization: every noise token is a random *direction*, and the slot
    #     it seeds is the cell embedding whose direction best matches it.

    #     Isotropic ``N(0, I)`` slots live off the manifold the encoder maps cells onto, so the
    #     cross-attention logits ``slot @ key^T`` in :class:`SlotAttentionBlock` carry no information
    #     about which cells a slot ought to claim.  The competitive softmax over ``k`` then sees
    #     near-equal logits for every slot, splits each cell evenly, and every slot aggregates the
    #     same global mean -- the intra-replicate factor collapse this seeding exists to prevent.
    #     Seeding on the manifold makes the very first round of competition meaningful, and the
    #     softmax sharpens that separation over subsequent blocks rather than having to create it.

    #     Directions are compared after L2 normalization so that the argmax spreads over the angular
    #     extent of the manifold; an unnormalized dot product would keep picking whichever few cells
    #     have the largest embedding norm, no matter the direction asked for.

    #     The noise alone still determines the seed, which keeps the two properties the rest of the
    #     model leans on: replicates remain independent random restarts (so consensus across them is
    #     meaningful), and a fixed noise tensor still gives a bit-for-bit reproducible solve, which
    #     :meth:`_check_drift` relies on to stay rank-consistent without a reduction.
    #     """
    #     similarity_rkn = torch.einsum("rke,ne->rkn",
    # F.normalize(slot_noise_rke, dim=-1), F.normalize(x_emb_ne, dim=-1))
    #     seed_rke = x_emb_ne.detach()[similarity_rkn.argmax(dim=-1)]
    #     # Jitter, scaled to the embeddings, so two slots seeded on the same cell still diverge.
    #     jitter = _SEED_JITTER * x_emb_ne.detach().std().clamp(min=_EPS)
    #     return seed_rke + self.slot_mu + jitter * torch.exp(self.slot_log_sigma) * slot_noise_rke + cond_e

    def _initial_slots(
        self,
        k: int,
        slot_noise_rke: torch.Tensor,
        cond_e: torch.Tensor,
        x_emb_ne: torch.Tensor,
    ) -> torch.Tensor:
        n_cells = x_emb_ne.shape[0]
        n_replicates = slot_noise_rke.shape[0]
        seed_rke = torch.empty(n_replicates, k, self.latent_dim, device=x_emb_ne.device, dtype=x_emb_ne.dtype)

        # Use a local generator seeded by the noise tensor.
        # This guarantees DDP sync (deterministic) but allows us to use randperm
        # to guarantee K UNIQUE cells, completely avoiding the Hypersphere Cone trap!
        for r in range(n_replicates):
            # Deterministic seed derived from the noise for this replicate
            seed_val = int(abs(slot_noise_rke[r, 0, 0].item() * 1e6))
            gen = torch.Generator(device=x_emb_ne.device).manual_seed(seed_val)

            if n_cells >= k:
                rand_idx = torch.randperm(n_cells, generator=gen, device=x_emb_ne.device)[:k]
            else:
                rand_idx = torch.randint(0, n_cells, (k,), generator=gen, device=x_emb_ne.device)

            seed_rke[r] = x_emb_ne[rand_idx].detach()

        jitter = _SEED_JITTER * x_emb_ne.detach().std().clamp(min=_EPS)
        return seed_rke + self.slot_mu + jitter * torch.exp(self.slot_log_sigma) * slot_noise_rke + cond_e

    # @torch.no_grad()
    def hot_start_loadings(self, x_ng: torch.Tensor, x_emb_ne: torch.Tensor, slots_rke: torch.Tensor) -> torch.Tensor:
        """
        Analytic hot start for ``H`` from a slot-competition map over the target cells.

        Because the decoder L1-normalizes each factor, ``sum_g x[n, g] == sum_k H[n, k]`` exactly.
        A softmax over ``k`` already sums to one per cell, so scaling it by each cell's total puts
        the hot start on precisely the right scale, leaving FISTA only the direction to fix.
        """
        # # original
        # logits_rkn = torch.einsum("rke,ne->rkn", self.norm_hot_start(slots_rke), x_emb_ne) * self.latent_dim**-0.5
        # attention_rkn = logits_rkn.softmax(dim=-2)
        # return (attention_rkn * x_ng.sum(dim=-1)).transpose(-2, -1).contiguous()

        # no temp
        logits_rkn = torch.einsum("rke,ne->rkn", self.norm_hot_start(slots_rke), x_emb_ne)
        attention_rkn = logits_rkn.softmax(dim=-2)
        return (attention_rkn * x_ng.sum(dim=-1)).transpose(-2, -1).contiguous()

        # # not great
        # logits_rkn = torch.einsum("rke,ne->rkn", self.norm_hot_start(slots_rke), x_emb_ne)
        # max_idx_r1n = logits_rkn.argmax(dim=-2, keepdim=True)
        # hard_attention_rkn = torch.zeros_like(logits_rkn).scatter_(-2, max_idx_r1n, 1.0)
        # return (hard_attention_rkn * x_ng.sum(dim=-1)).transpose(-2, -1).contiguous()

    @torch.no_grad()
    def _solve_loadings(
        self,
        x_ng: torch.Tensor,
        x_emb_ne: torch.Tensor,
        slots_rke: torch.Tensor,
        w_rkg: torch.Tensor,
        n_iterations: int,
    ) -> torch.Tensor:
        """Hot start, then FISTA polish.  Fully detached: see the class docstring on stop-gradient."""
        h_rnk = self.hot_start_loadings(x_ng, x_emb_ne, slots_rke.detach())
        h_rnk, _ = nmf_compute_loadings_fista(
            x_ng=x_ng, w_rkg=w_rkg.detach().contiguous(), h_rnk=h_rnk, max_iter=n_iterations
        )
        return h_rnk.clamp(min=0.0)

    def solve(
        self,
        x_ng: torch.Tensor,
        k: int,
        slot_noise_rke: torch.Tensor,
        n_iterations: int,
        x_target_ng: torch.Tensor | None = None,
        compute_loadings: bool = True,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        One amortized factorization: ``R`` replicate ``W`` matrices per layer, and optionally ``H``.

        Each transformer layer decodes its own slot state into a factor matrix, enabling the
        AlphaFold-style per-layer auxiliary loss.  ``w_rkg`` is always the final-layer output.

        Args:
            x_ng: Cells to condition on (the cross-attention memory).
            k: Number of factors.
            slot_noise_rke: Standard normal noise, shape ``(r, k, latent_dim)``.
            n_iterations: FISTA iterations for the loadings.
            x_target_ng: Cells on which to solve loadings.  Defaults to ``x_ng``.
            compute_loadings: Skip the FISTA solve when only ``W`` is needed (e.g. drift checks).

        Returns:
            Dict with ``w_rkg`` (final layer), ``w_layers_rkg`` (list of W per layer),
            ``slots_rke``, and if requested ``h_rnk`` and ``x_target_ng``.
        """
        x_emb_ne = self.encoder(torch.log1p(x_ng))
        key_ne = self.to_key(x_emb_ne)
        value_ne = self.to_value(x_emb_ne)

        cond_e = self._k_cond(k, x_ng.device, x_ng.dtype)
        slots_initial_rke = self._initial_slots(k, slot_noise_rke, cond_e, x_emb_ne)
        slots_rke = slots_initial_rke
        w_layers_rkg: list[torch.Tensor] = []
        for i in range(self.n_iterations):
            idx = 0 if self.recurrent else i
            slots_rke = self.blocks[idx](slots_rke, key_ne, value_ne, cond_e)
            w_layers_rkg.append(self.decoders[idx](slots_rke))

        w_rkg = w_layers_rkg[-1]
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "w_rkg": w_rkg,
            "w_layers_rkg": w_layers_rkg,
            "slots_rke": slots_rke,
            "x_emb_ne": x_emb_ne,
        }
        if compute_loadings:
            target_ng = x_ng if x_target_ng is None else x_target_ng
            target_emb_ne = x_emb_ne if x_target_ng is None else self.encoder(target_ng)

            # # attempt to resolve collapse
            # # h_hot uses pre-attention initial slots: diversity guaranteed by slot_noise regardless
            # # of student collapse, making it a robust seed for the decoupled teacher ALS.
            # out["h_hot_rnk"] = self.hot_start_loadings(target_ng, target_emb_ne, slots_initial_rke.detach())

            # h_rnk uses final-layer slots: best quality for inference and measurement phase.
            out["h_rnk"] = self._solve_loadings(target_ng, target_emb_ne, slots_rke, w_rkg, n_iterations)
            out["x_target_ng"] = target_ng
        return out

    def _consensus(self, w_rkg: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return sinkhorn_consensus(
            w_rkg,
            epsilon=self.sinkhorn_epsilon,
            n_iterations=self.sinkhorn_iterations,
            outlier_gamma=self.outlier_gamma,
            n_refine=self.sinkhorn_refine_rounds,
            detach_plan=self.detach_sinkhorn_plan,
        )

    @torch.no_grad()
    def _consensus_loadings(
        self,
        x_ng: torch.Tensor,
        h_rnk: torch.Tensor,
        consensus_kg: torch.Tensor,
        plan_rkj: torch.Tensor | None,
        weights_rk: torch.Tensor,
        n_iterations: int,
    ) -> torch.Tensor:
        """
        Loadings for the consensus factors, hot started by permuting and averaging the replicates'
        loadings with the same transport plans and outlier weights used to build the consensus.
        """
        if plan_rkj is None:
            h_nk = h_rnk[0]
        else:
            aligned_rnj = torch.einsum("rkj,rnk->rnj", plan_rkj, h_rnk)
            weights_r1k = weights_rk.unsqueeze(1)
            h_nk = (weights_r1k * aligned_rnj).sum(dim=0) / weights_r1k.sum(dim=0).clamp(min=_EPS)
        h_1nk, _ = nmf_compute_loadings_fista(
            x_ng=x_ng,
            w_rkg=consensus_kg.detach().unsqueeze(0).contiguous(),
            h_rnk=h_nk.unsqueeze(0).contiguous(),
            max_iter=n_iterations,
        )
        return h_1nk.squeeze(0).clamp(min=0.0)

    @torch.no_grad()
    def _solve_loadings_cold(self, x_ng: torch.Tensor, w_kg: torch.Tensor, n_iterations: int) -> torch.Tensor:
        """NNLS loadings for a factor set that did not come from this batch's forward pass."""
        h_1nk, _ = nmf_compute_loadings_fista(
            x_ng=x_ng,
            w_rkg=w_kg.detach().unsqueeze(0).contiguous(),
            h_rnk=x_ng.new_zeros((1, x_ng.shape[0], w_kg.shape[0])),
            max_iter=n_iterations,
        )
        return h_1nk.clamp(min=0.0)

    @torch.no_grad()
    def _compute_w_target(
        self,
        x_ng: torch.Tensor,
        h_rnk: torch.Tensor,
        w_rkg: torch.Tensor,
    ) -> torch.Tensor:
        """
        FISTA-improved W as a stop-gradient training target.

        Hot-starts from the transformer's final-layer W, runs ``fista_w_iterations_train`` steps
        given the stop-gradient H, then L1-normalizes to match the decoder output convention.
        The result is zero-gradient: gradients flow only through the transformer output W_l on the
        other side of the MSE loss.
        """
        # Scale H down to a maximum of 1.0, and W up proportionally.
        # This keeps the Lipschitz constant small and step sizes large.
        c = h_rnk.max().clamp(min=1e-8)
        h_scaled_rnk = h_rnk / c
        w_scaled_rkg = w_rkg.detach().contiguous() * c

        hth_rkk = torch.einsum("rnk,rnj->rkj", h_scaled_rnk, h_scaled_rnk)
        htx_rkg = torch.einsum("rnk,ng->rkg", h_scaled_rnk, x_ng)
        w_prime_rkg, _ = nmf_compute_factors_fista(
            w_rkg=w_scaled_rkg,
            A_rkk=hth_rkk,
            B_rkg=htx_rkg,
            max_iter=self.fista_w_iterations_train,
        )
        return l1_normalize_rows(w_prime_rkg.clamp(min=0.0))

    # -------------------------------------------------------------------------------------------
    # forward / training
    # -------------------------------------------------------------------------------------------

    def _split_batch(self, x_ng: torch.Tensor) -> list[tuple[torch.Tensor, int]]:
        """Split the minibatch into conditioning sets, one per group of replicates."""
        n = x_ng.shape[0]
        if not self.cross_batch_stability or self.n_replicates < 2 or n < 2 * self.min_cells_per_split:
            return [(x_ng, self.n_replicates)]
        if self.shuffle_split:
            x_ng = x_ng[torch.randperm(n, device=x_ng.device)]
        half = n // 2
        r_first = self.n_replicates // 2
        return [(x_ng[:half], r_first), (x_ng[half : 2 * half], self.n_replicates - r_first)]

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng: Gene counts matrix (already transformed).
            var_names_g: The list of the variable names in the input data.

        Returns:
            A dictionary with the ``loss`` and per-step monitoring scalars.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        assert x_ng.min() >= 0.0, "x_ng must be nonnegative"

        # print(x_ng.min(), x_ng.max(), x_ng.sum(-1).min(), x_ng.sum(-1).max(), x_ng.sum(-1).std())

        self._maybe_capture_drift_cells(x_ng)

        k = self.sample_k()
        self._last_k = k

        # Retrieve strictly historical reservoir (empty on step 0; updated after training below).
        x_reservoir_ng: torch.Tensor | None = None
        if self.use_reservoir and self.reservoir is not None and self.reservoir.total_cells > 0:
            reservoir_ng = self.reservoir.get_reservoir()["x_ng"]
            assert isinstance(reservoir_ng, torch.Tensor)
            x_reservoir_ng = reservoir_ng.to_dense().to(device=x_ng.device, dtype=x_ng.dtype)

        recon_losses: list[torch.Tensor] = []
        sc_losses: list[torch.Tensor] = []

        # A hyperparameter to balance the two losses.
        # Start small, e.g., 0.1, to let the FISTA teacher do the heavy lifting while
        # the recon loss just acts as a guardrail against collapse.
        lambda_recon = 0.1

        for x_half_ng, r in self._split_batch(x_ng):
            # Both halves of the cross-batch split condition on the same historical reservoir.
            x_aug_ng = torch.cat([x_half_ng, x_reservoir_ng], dim=0) if x_reservoir_ng is not None else x_half_ng
            slot_noise_rke = torch.randn(r, k, self.latent_dim, device=x_ng.device, dtype=x_ng.dtype)

            solved = self.solve(x_aug_ng, k, slot_noise_rke, self.fista_iterations_train)
            assert isinstance(solved["h_rnk"], torch.Tensor)
            assert isinstance(solved["w_rkg"], torch.Tensor)
            w_target_rkg = self._compute_w_target(x_aug_ng, solved["h_rnk"], solved["w_rkg"])

            # # wild attempts to solve collapse, did not work
            # solved = self.solve(x_aug_ng, k, slot_noise_rke, self.fista_iterations_train)
            # # Decoupled teacher ALS.  All three steps are no_grad (enforced by their decorators /
            # # the nmf_compute_loadings_fista decorator).
            # h_hot_rnk = solved["h_hot_rnk"]
            # # Step A: W from guaranteed-diverse h_hot; student W warm-starts the FISTA.
            # # w_1_rkg = self._compute_w_target(x_aug_ng, h_hot_rnk, solved["w_rkg"])
            # # Create a dummy uniform matrix so the student's collapse CANNOT leak.
            # dummy_w_rkg = torch.ones_like(solved["w_rkg"]) / self.n_genes
            # w_1_rkg = self._compute_w_target(x_aug_ng, h_hot_rnk, dummy_w_rkg)
            # # Step B: relax hard clusters into soft NMF mixtures using the diverse W_1.
            # h_1_rnk, _ = nmf_compute_loadings_fista(
            #     x_ng=x_aug_ng,
            #     w_rkg=w_1_rkg,
            #     h_rnk=h_hot_rnk,
            #     max_iter=self.fista_iterations_train,
            # )
            # # Step C: final W target conditioned on soft H; warm-started from W_1.
            # w_target_rkg = self._compute_w_target(x_aug_ng, h_1_rnk.clamp(min=0.0), w_1_rkg)
            sc_losses.append(
                sum(
                    self._iter_weights[i] * (w_l - w_target_rkg).pow(2).sum(dim=-1).mean()
                    for i, w_l in enumerate(solved["w_layers_rkg"])
                )
            )

            # Use the differentiable routing so gradients flow all the way back to the encoder
            assert isinstance(solved["x_emb_ne"], torch.Tensor)
            assert isinstance(solved["slots_rke"], torch.Tensor)
            h_diff_rnk = self.hot_start_loadings(x_aug_ng, solved["x_emb_ne"], solved["slots_rke"])

            # Use O(r*k*g) trace function. Normalize by numel to keep scales sane.
            recon_error_r = frobenius_loss_trace(x_aug_ng, h_diff_rnk, solved["w_rkg"])
            recon_losses.append(recon_error_r.mean() / x_aug_ng.numel())

        total_sc_loss = torch.stack(sc_losses).mean()
        total_recon_loss = torch.stack(recon_losses).mean()
        loss = total_sc_loss + (lambda_recon * total_recon_loss)
        self._last_sc_loss = float(total_sc_loss.detach())
        self._last_recon_loss = float(total_recon_loss.detach())

        # Update reservoir after training so it only ever contains historical cells.
        if self.use_reservoir and self.reservoir is not None:
            n = x_ng.shape[0]
            obs_names = np.array([str(self._reservoir_cell_counter + i) for i in range(n)])
            self.reservoir.update(x_ng.detach(), obs_names)
            self._reservoir_cell_counter += n

        return {"loss": loss}

    @torch.no_grad()
    def _step_metrics(
        self,
        stabilities: list[torch.Tensor],
        consensus_losses: list[torch.Tensor],
        consensus_list: list[torch.Tensor],
        cells_list: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        metrics = {
            "stability": torch.stack([s.detach() for s in stabilities]).mean(),
            "error": torch.stack([e.detach() for e in consensus_losses]).mean(),
        }
        if len(consensus_list) == 2:
            metrics["stability_cross"] = match_stability(
                consensus_list[0], consensus_list[1], self.sinkhorn_epsilon, self.sinkhorn_iterations
            )
            # Cross error: the consensus from one half scored on the other half's cells.  Cheap, and
            # it is the direct read on whether W is solution-grade or only hot-start-grade.
            cross_errors = []
            for source, target_ng in ((0, cells_list[1]), (1, cells_list[0])):
                consensus_kg = consensus_list[source]
                h_1nk = self._solve_loadings_cold(target_ng, consensus_kg, max(50, self.fista_iterations_train))
                sse_1 = frobenius_loss_trace(target_ng, h_1nk, consensus_kg.unsqueeze(0))
                cross_errors.append(sse_1.squeeze(0) / target_ng.numel())
            metrics["error_cross"] = torch.stack(cross_errors).mean()
        return metrics

    @torch.no_grad()
    def _update_ema(self, k: int, metrics: dict[str, torch.Tensor]) -> None:
        """
        Per-``k`` EMA of the monitoring metrics.

        These are per-bin rather than global-step EMAs: each ``k`` is drawn only a fraction of the
        time, so a global decay would give different effective time constants to different ``k``.
        They are for *monitoring* -- the curriculum makes large-``k`` bins both sparse and
        contaminated by a still-learning network, which is why the measurement phase exists.
        """
        if self._step_cache < self.stability_burn_in_steps:
            return
        available = [name for name in self.metric_names if name in metrics]
        if not available:
            return
        # One collective for all metrics.  Without this, per-rank accumulations diverge (and
        # broadcast_buffers, if enabled, would silently discard every non-zero rank's).
        values = _all_reduce_mean(torch.stack([metrics[name].detach().float() for name in available]))
        index = self._k_to_index[k]
        beta = self.stability_ema_beta
        for name, value in zip(available, values):
            ema_k = getattr(self, f"_ema_{name}_k")
            count_k = getattr(self, f"_ema_count_{name}_k")
            ema_k[index] = beta * ema_k[index] + (1.0 - beta) * value.to(ema_k.dtype)
            count_k[index] += 1

    def ema_curve(self, name: str) -> torch.Tensor:
        """Bias-corrected per-``k`` EMA of a monitoring metric, ``NaN`` where nothing accumulated."""
        if name not in self.metric_names:
            raise ValueError(f"unknown metric {name!r}; choose from {self.metric_names}")
        ema_k = getattr(self, f"_ema_{name}_k")
        count_k = getattr(self, f"_ema_count_{name}_k")
        correction = 1.0 - self.stability_ema_beta ** count_k.clamp(min=1)
        curve = ema_k / correction
        return torch.where(count_k > 0, curve, torch.full_like(curve, float("nan")))

    # -------------------------------------------------------------------------------------------
    # drift-based convergence
    # -------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _maybe_capture_drift_cells(self, x_ng: torch.Tensor) -> None:
        """
        Fill the fixed drift cell buffer from the first training batches.

        Taken from training data on purpose: the drift metric measures whether the *weights* have
        stopped moving, not generalization, so there is nothing to be gained from held-out cells --
        and this way the criterion works with no validation dataloader configured.
        """
        # Short-circuit on a Python flag rather than reading the buffer: this runs on every training
        # step, and `.item()` on a device tensor forces a host synchronization.
        if self._drift_cells_full:
            return
        n_captured = int(self._drift_n_captured.item())
        if n_captured >= self.drift_eval_n_cells:
            self._drift_cells_full = True
            return
        take = min(x_ng.shape[0], self.drift_eval_n_cells - n_captured)
        self._drift_x_ng[n_captured : n_captured + take] = x_ng[:take].detach().to(self._drift_x_ng.dtype)
        self._drift_n_captured.fill_(n_captured + take)
        if n_captured + take >= self.drift_eval_n_cells:
            self._drift_cells_full = True
            # All ranks must measure drift on identical cells, or they disagree about stopping.
            _broadcast_from_rank_zero(self._drift_x_ng)

    @torch.no_grad()
    def drift_check(self) -> float | None:
        """
        Measure factor drift and update the convergence counter.

        Deterministic by construction -- fixed noise, fixed cells, fixed ``k`` -- which is the whole
        point: there is no sampling noise, so no smoothing is needed, and every rank computes the
        same number and reaches the same decision without any reduction.

        Returns:
            The per-step drift rate, or ``None`` if there is nothing to compare against yet.
        """
        n_captured = int(self._drift_n_captured.item())
        if n_captured == 0:
            return None
        x_ng = self._drift_x_ng[:n_captured]

        rates: list[torch.Tensor] = []
        for k in self.drift_k_values:
            slot_noise_rke = self.drift_slot_noise_rke[:, :k, :].to(x_ng.dtype)
            solved = self.solve(x_ng, k, slot_noise_rke, 0, compute_loadings=False)
            assert isinstance(solved["w_rkg"], torch.Tensor)
            consensus_kg = self._consensus(solved["w_rkg"])["consensus_kg"]
            assert isinstance(consensus_kg, torch.Tensor)
            previous_kg = getattr(self, f"_drift_previous_w_{k}_kg")
            if bool(self._drift_has_previous):
                # Re-align before comparing: without matching, one permutation flip in the slot
                # ordering reads as catastrophic drift.  The comparison uses the hard assignment
                # (see matched_distance) so that unchanged weights give exactly zero drift.
                _, _, plan_1kj, similarity_1kj = align_factors(
                    consensus_kg.unsqueeze(0), previous_kg, self.sinkhorn_epsilon, self.sinkhorn_iterations
                )
                distance = matched_distance(similarity_1kj, plan_1kj).mean()
                rates.append(distance / max(1, self.drift_check_every_n_steps))
            previous_kg.copy_(consensus_kg)

        self._drift_has_previous.fill_(True)
        if not rates:
            return None

        rate = torch.stack(rates).max()  # the worst k governs
        self._drift_rate.copy_(rate.to(self._drift_rate.dtype))
        if float(rate) < self.drift_tol:
            self._drift_below_tol_count += 1
        else:
            self._drift_below_tol_count.zero_()
        return float(rate)

    @property
    def stopping_allowed(self) -> bool:
        """Stopping is only meaningful once the LR has been reduced (or the settle window has
        elapsed when ``lr_reduction_factor == 1.0`` and there is nothing to reduce)."""
        if self.lr_reduction_factor != 1.0:
            return bool(self._lr_reduced.item())
        return self._step_cache >= self.curriculum_warmup_steps + self.drift_settle_steps

    @property
    def converged(self) -> bool:
        """Drift has stayed below tolerance for ``drift_patience_checks`` consecutive checks."""
        return self.stopping_allowed and int(self._drift_below_tol_count.item()) >= self.drift_patience_checks

    # -------------------------------------------------------------------------------------------
    # lightning hooks
    # -------------------------------------------------------------------------------------------

    def on_train_start(self, trainer: pl.Trainer) -> None:
        # Restore the Python-side caches from their buffers (one sync, at startup only, so the hot
        # path stays free).  Doing this here rather than in reset_parameters is what makes a resumed
        # run continue the curriculum and the convergence criterion instead of restarting them.
        self._step_cache = int(self._step.item())
        self._drift_cells_full = int(self._drift_n_captured.item()) >= self.drift_eval_n_cells
        if trainer.world_size > 1 and getattr(trainer.strategy, "_ddp_kwargs", {}).get("broadcast_buffers", False):
            warnings.warn(
                "CNMFTransformer recommends DDPStrategy(broadcast_buffers=False). Everything that must "
                "agree across ranks is derived deterministically or synchronized explicitly, and this "
                "model's persisted consensus factors make a per-step buffer broadcast expensive.",
                UserWarning,
            )

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        self._step_cache += 1
        self._step.fill_(self._step_cache)

        if self.log_every_n_steps > 0 and self._step_cache % self.log_every_n_steps == 0:
            self._log_monitoring(trainer)

        settle_boundary = self.curriculum_warmup_steps + self.drift_settle_steps
        if (
            self.lr_reduction_factor != 1.0
            and not bool(self._lr_reduced.item())
            and self._step_cache >= settle_boundary
        ):
            self._apply_lr_reduction(trainer)

        if self.drift_check_every_n_steps > 0 and self._step_cache % self.drift_check_every_n_steps == 0:
            rate = self.drift_check()
            pl_module = self._lightning_module(trainer)
            if rate is not None and pl_module is not None:
                pl_module.log("factor_drift", self._drift_rate, prog_bar=True)
            if self.converged:
                trainer.should_stop = True
                print(
                    f"Stopping early: factor drift below {self.drift_tol} for "
                    f"{self.drift_patience_checks} consecutive checks"
                )

    @staticmethod
    def _lightning_module(trainer: pl.Trainer) -> pl.LightningModule | None:
        module = trainer.model
        return module if isinstance(module, pl.LightningModule) else None

    def _apply_lr_reduction(self, trainer: pl.Trainer) -> None:
        """Divide the optimizer LR by ``lr_reduction_factor`` once, then reset drift state."""
        pl_module = self._lightning_module(trainer)
        if pl_module is None:
            return

        if pl_module.lr_schedulers() is not None:
            warnings.warn(
                "lr_reduction_factor is set but a LR scheduler is active; the scheduler will "
                "overwrite the reduced LR on the next step.  Either set lr_reduction_factor=1.0 "
                "or remove the scheduler.",
                UserWarning,
            )

        optimizers = pl_module.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt in optimizers:
            # unwrap Lightning's optimizer wrapper if present
            raw_opt = opt.optimizer if hasattr(opt, "optimizer") else opt
            for pg in raw_opt.param_groups:
                pg["lr"] = pg["lr"] / self.lr_reduction_factor
                if "initial_lr" in pg:
                    pg["initial_lr"] = pg["initial_lr"] / self.lr_reduction_factor

        self._lr_reduced.fill_(True)

        # Reset drift state so convergence is measured relative to the post-reduction baseline.
        self._drift_has_previous.fill_(False)
        self._drift_below_tol_count.zero_()
        for k in self.drift_k_values:
            getattr(self, f"_drift_previous_w_{k}_kg").zero_()

        print(
            f"Step {self._step_cache}: LR reduced by {self.lr_reduction_factor}×. "
            f"Now checking drift for early stopping (tol={self.drift_tol})."
        )

    def _log_monitoring(self, trainer: pl.Trainer) -> None:
        pl_module = self._lightning_module(trainer)
        if pl_module is None:
            return
        pl_module.log("curriculum_k_max", float(self.k_values[self.curriculum_k_count() - 1]))
        pl_module.log("sampled_k", float(self._last_k))
        if not math.isnan(self._last_sc_loss):
            pl_module.log("sc_loss", self._last_sc_loss, prog_bar=True)
        if not math.isnan(self._last_recon_loss):
            pl_module.log("recon_loss", self._last_recon_loss, prog_bar=True)
        if self.use_reservoir and self.reservoir is not None:
            pl_module.log("res_cells", int(self.reservoir.total_cells), prog_bar=True)
            pl_module.log("res_fill", self.reservoir.bucket_fill_fraction, prog_bar=True)

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        output = self(*args, **kwargs)
        loss = output.get("loss")
        if loss is not None:
            pl_module.log("val_loss", loss, sync_dist=True, on_epoch=True)

    def on_train_end(self, trainer: pl.Trainer) -> None:
        if not self.measure_at_end:
            return
        if trainer.global_rank == 0:
            dataloader = self._measurement_dataloader(trainer)
            if dataloader is None:
                warnings.warn("Could not obtain a dataloader for the measurement phase; skipping.", UserWarning)
            else:
                # The model normally sits at the end of a CellariumPipeline, so the (GPU) transforms
                # have to be applied by hand when pulling batches from the datamodule directly.
                run_measurement_phase(
                    self,
                    dataloader=dataloader,
                    transforms=list(getattr(trainer.model, "transforms", []) or []),
                    n_batches=self.measurement_n_batches,
                )
        trainer.strategy.barrier()

    @staticmethod
    def _measurement_dataloader(trainer: pl.Trainer) -> Iterable | None:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None
        for name in ("predict_dataloader", "train_dataloader"):
            factory = getattr(datamodule, name, None)
            if callable(factory):
                try:
                    return factory()
                except Exception:  # noqa: BLE001 -- try the next candidate
                    continue
        return None

    def on_end(self, trainer: pl.Trainer) -> None:
        trainer.save_checkpoint(trainer.default_root_dir + "/CNMFTransformer.ckpt")

    # -------------------------------------------------------------------------------------------
    # outputs
    # -------------------------------------------------------------------------------------------

    @property
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """
        Replicate factors per ``k``, shape ``(r, k, g)``.

        For ``k`` values whose replicate buffer ``D_{k}_rkg`` was stored by
        :func:`run_measurement_phase` (via ``store_replicates_k_values``), returns genuine
        replicates suitable for :func:`~cellarium.ml.models.nmf.plot_density_histograms`.
        Otherwise falls back to the batch-averaged consensus as a single replicate — stability
        computed from a consensus is trivially 1.0.  Use :attr:`consensus_factors` instead for
        those entries.
        """
        out: dict[int, torch.Tensor] = {}
        for k in self.k_values:
            buf_name = f"D_{k}_rkg"
            if buf_name in self._buffers:
                val = self._buffers[buf_name]
                assert isinstance(val, torch.Tensor)
                out[k] = val
            else:
                out[k] = getattr(self, f"consensus_D_{k}_kg").unsqueeze(0)
        return out

    @property
    def consensus_factors(self) -> dict[int, dict[str, torch.Tensor | float]]:
        """
        Measured consensus factors in the format expected by :meth:`infer_loadings`,
        :meth:`reconstruction_error` and :class:`~cellarium.ml.models.nmf.NMFOutput`.

        Covers all ``k`` values for which a ``consensus_D_{k}_kg`` buffer exists — both training
        k values (always present) and any untrained k values measured via
        :func:`run_measurement_phase`.
        """
        if "_measured_k_values" in self._buffers:
            measured_k_list = self._measured_k_values.tolist()
            measured_stability = getattr(self, "_measured_stability_mean_k")
            stability_by_k = {k: float(measured_stability[i]) for i, k in enumerate(measured_k_list)}
        else:
            stability_by_k = {}
        all_k = sorted(
            int(name[len("consensus_D_") : -len("_kg")])
            for name in self._buffers
            if name.startswith("consensus_D_") and name.endswith("_kg")
        )
        return {
            k: {
                "consensus_D_kg": self._buffers[f"consensus_D_{k}_kg"],
                "stability": stability_by_k.get(k, float("nan")),
            }
            for k in all_k
        }

    def selection_curves(self, k_values: list[int] | None = None) -> dict[str, np.ndarray]:
        """
        The measured k-selection curves as numpy arrays, keyed by metric name plus ``"k"``.

        ``stability`` is a Kotliar-comparable silhouette-like contrast (see
        :func:`matched_silhouette`) reflecting initialization variance only.  ``stability_cross``
        additionally folds in sampling variance; the *gap* between the two says how much of the
        stability is a sampling artifact.  ``error`` and ``error_cross`` are per-entry mean squared
        errors.  The ``*_sem`` entries are standard errors across measurement batches -- on the full
        integer ``k`` grid these are what let you tell a real discontinuity from measurement noise.
        """
        if "_measured_k_values" not in self._buffers:
            raise RuntimeError("selection_curves() called before run_measurement_phase()")
        measured_k = self._measured_k_values.tolist()
        if k_values is None:
            selected = measured_k
            indices = list(range(len(measured_k)))
        else:
            k_set = set(k_values)
            indices = [i for i, k in enumerate(measured_k) if k in k_set]
            selected = [measured_k[i] for i in indices]
        curves: dict[str, np.ndarray] = {"k": np.asarray(selected)}
        for name in self.metric_names:
            full_mean = getattr(self, f"_measured_{name}_mean_k").detach().cpu().numpy()
            full_sem = getattr(self, f"_measured_{name}_sem_k").detach().cpu().numpy()
            curves[name] = full_mean[indices]
            curves[f"{name}_sem"] = full_sem[indices]
        return curves

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
        """Infer per-cell loadings for a given ``k`` by solving NNLS against the consensus factors."""
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        consensus_D_kg = consensus_factors[k]["consensus_D_kg"]
        assert isinstance(consensus_D_kg, torch.Tensor), "consensus_D_kg must be a tensor"
        consensus_D_kg = consensus_D_kg.to(device=x_ng.device, dtype=x_ng.dtype)
        if bool((consensus_D_kg == 0).all()):
            raise ValueError(
                f"consensus factors for k={k} are all zeros; run run_measurement_phase() before inferring loadings"
            )
        alpha_nk = self._solve_loadings_cold(x_ng, consensus_D_kg, 1000).squeeze(0)
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
        """Sum of squared reconstruction error for each ``k`` in ``consensus_factors``."""
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        errors: dict[int, float] = {}
        for k in consensus_factors:
            consensus_D_kg = consensus_factors[k]["consensus_D_kg"]
            assert isinstance(consensus_D_kg, torch.Tensor)
            consensus_D_kg = consensus_D_kg.to(device=x_ng.device, dtype=x_ng.dtype)
            alpha_nk = self.infer_loadings(x_ng, var_names_g, consensus_factors, k)
            errors[k] = float(frobenius_loss_trace(x_ng, alpha_nk.unsqueeze(0), consensus_D_kg.unsqueeze(0)).squeeze(0))
        return errors

    @torch.no_grad()
    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        k: int | None = None,
        normalize: bool = True,
        obs_names_n: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Per-cell loadings against the measured consensus factors for ``k``."""
        k = self.predict_k if k is None else k
        if k is None:
            raise ValueError("set `predict_k` on the model or pass `k` to predict()")
        alpha_nk = self.infer_loadings(x_ng, var_names_g, self.consensus_factors, k, normalize=normalize)
        out: dict[str, np.ndarray | torch.Tensor] = {"alpha_nk": alpha_nk}
        if obs_names_n is not None:
            out["obs_names_n"] = obs_names_n
        return out


# -----------------------------------------------------------------------------------------------
# Measurement phase
# -----------------------------------------------------------------------------------------------


@torch.no_grad()
def run_measurement_phase(
    model: CNMFTransformer,
    dataloader: Iterable,
    transforms: Iterable[torch.nn.Module] = (),
    n_batches: int | None = None,
    k_values: list[int] | None = None,
    n_replicates: int = 100,
    store_replicates_k_values: list[int] | None = None,
    r_store: int = 20,
    fista_iterations: int = 150,
    density_threshold: float = 1.0,
    local_neighborhood_size: float = 0.30,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Measure the stability / error curves and the batch-averaged consensus factors.

    Run after training, when the network has settled.  This exists because the per-``k`` EMAs
    collected *during* training are non-stationary in a ``k``-dependent way: the curriculum reveals
    large ``k`` late, so exactly the bins near the interesting part of the curve have both the
    fewest samples and the most contamination from a still-learning network.  Here the weights are
    frozen, so the estimate is clean, and with no activations retained ``R`` can be raised to a
    Kotliar-comparable value.

    The loop is **batches outer, k inner**: each minibatch is fetched once and reused for every
    ``k``.  For out-of-core data the dataloader is usually the bottleneck, so this matters more than
    the arithmetic.

    Cross-batch statistics are computed against the *previous* batch's consensus rather than by
    splitting each batch, which keeps the full ``R`` for both statistics at one solve per
    ``(batch, k)``.  (Training uses a within-batch split instead, since it has no cross-step state.)

    Consensus follows Kotliar et al. 2019: the ``r * k`` L2-normalized factor rows are density-
    filtered and then clustered with k-means into ``k`` groups; the per-cluster median row is taken
    as the consensus program and L1-renormalized.  This is structurally different from the Sinkhorn
    approach used during training for drift measurement — k-means assigns each row to whichever
    cluster it belongs to, so replicates that missed a program never contaminate that program's
    consensus centroid.

    Args:
        model: A trained :class:`CNMFTransformer`.
        dataloader: Yields batch dicts with ``x_ng`` and ``var_names_g``.
        transforms: GPU transforms to apply to each batch (the model normally sits at the end of a
            :class:`~cellarium.ml.core.CellariumPipeline` and receives transformed data).
        n_batches: Number of minibatches.  Defaults to ``model.measurement_n_batches``.
        k_values: Defaults to every ``k`` in ``model.k_values``.
        n_replicates: Number of independent noise seeds per ``(batch, k)`` solve.  A
            Kotliar-comparable value (50–100) is usually appropriate.
        store_replicates_k_values: ``k`` values for which raw replicate factors from the first
            batch are persisted as ``D_{k}_rkg`` buffers on the model, for later use by
            :func:`plot_density_histograms` without a dataloader.  ``None`` (default) skips
            replicate storage; any k being measured is a valid choice.
        r_store: Maximum number of replicates to keep per entry in ``store_replicates_k_values``.
            Clamped to ``n_replicates`` if larger.
        fista_iterations: FISTA iterations per solve.  More steps give a more converged error axis;
            150 is usually sufficient.
        density_threshold: Mean neighbor distance above which a factor row is excluded before
            k-means.  Range ``(0, 2]``; ``1.0`` (no filtering) is a safe starting point — run
            ``nmf_consensus(..., plot_only=True)`` to pick a tighter value for your data.
        local_neighborhood_size: Fraction of replicates used to define the local neighborhood for
            density filtering.  Range ``(0, 1)``.
        device: Device on which to run.  Defaults to the model's device.
        verbose: Show a progress bar.

    Returns:
        The same dict as :meth:`CNMFTransformer.selection_curves`.
    """
    was_training = model.training
    model.eval()
    try:
        k_values = list(model.k_values) if k_values is None else sorted(k_values)
        n_batches = model.measurement_n_batches if n_batches is None else n_batches
        store_k_set: set[int] = set(store_replicates_k_values) if store_replicates_k_values is not None else set()
        r_store_actual = min(r_store, n_replicates)
        device = model.device if device is None else torch.device(device)
        transform_list = list(transforms)
        dtype = model.slot_mu.dtype

        names = model.metric_names
        n_k = len(k_values)
        sums = {name: torch.zeros(n_k, dtype=torch.float64, device=device) for name in names}
        sums_of_squares = {name: torch.zeros(n_k, dtype=torch.float64, device=device) for name in names}
        counts = {name: torch.zeros(n_k, dtype=torch.float64, device=device) for name in names}

        consensus_sums: dict[int, torch.Tensor] = {}
        consensus_reference: dict[int, torch.Tensor] = {}
        previous_consensus: dict[int, torch.Tensor] = {}
        n_consensus: dict[int, int] = {k: 0 for k in k_values}

        # Retrieve the reservoir once so all batches condition on the same historical context.
        # If the reservoir is empty (first run, or post-checkpoint before re-warming) this is a
        # no-op and measurement degrades gracefully to the non-reservoir path.
        x_reservoir_ng: torch.Tensor | None = None
        if model.use_reservoir and model.reservoir is not None and model.reservoir.total_cells > 0:
            reservoir_ng = model.reservoir.get_reservoir()["x_ng"]
            assert isinstance(reservoir_ng, torch.Tensor)
            x_reservoir_ng = reservoir_ng.to_dense().to(device=device, dtype=dtype)

        iterator: Iterable = enumerate(dataloader)
        if verbose:
            iterator = tqdm(iterator, total=n_batches, desc="measuring k-selection curves")

        batches_used = 0
        for batch_index, batch in iterator:
            if batch_index >= n_batches:
                break
            for transform in transform_list:
                batch |= call_func_with_batch(transform.forward, batch)
            x_ng = batch["x_ng"].to(device=device, dtype=dtype)
            if x_ng.shape[0] < 2:
                continue
            # x_squared_sum and denominator are over the current batch only; error metrics stay
            # anchored to fresh cells rather than the (biased) geometric sketch.
            x_squared_sum = x_ng.pow(2).sum()
            denominator = x_ng.numel()
            batches_used += 1
            x_aug_ng = torch.cat([x_ng, x_reservoir_ng], dim=0) if x_reservoir_ng is not None else x_ng

            for k_index, k in enumerate(k_values):
                slot_noise_rke = torch.randn(n_replicates, k, model.latent_dim, device=device, dtype=dtype)
                solved = model.solve(x_aug_ng, k, slot_noise_rke, fista_iterations)
                assert isinstance(solved["w_rkg"], torch.Tensor)
                try:
                    kmeans_result = nmf_consensus(
                        solved["w_rkg"].detach().float().cpu(), density_threshold, local_neighborhood_size
                    )
                except (UserWarning, ValueError) as exc:
                    warnings.warn(f"k={k}, batch {batch_index}: skipping — {exc}")
                    continue
                consensus_kg = kmeans_result["consensus_D_kg"].to(device=device, dtype=dtype)

                # Stability via Sinkhorn matched-silhouette against the k-means consensus.
                # The k-means sklearn silhouette measures cluster tightness, which stays high
                # for any k an amortized model solves consistently.  The Sinkhorn matched-
                # silhouette is a contrast metric — it drops when replicates disagree about
                # which consensus factor a given program maps to, which is the correct signal
                # for k > k_true.
                _, _, plan_rkj, similarity_rkj = align_factors(
                    solved["w_rkg"].detach(),
                    consensus_kg,
                    model.sinkhorn_epsilon,
                    model.sinkhorn_iterations,
                )
                stability = matched_silhouette(similarity_rkj.detach(), plan_rkj.detach()).mean()

                h_1nk = model._solve_loadings_cold(x_ng, consensus_kg, fista_iterations)
                error = (
                    frobenius_loss_trace(x_ng, h_1nk, consensus_kg.unsqueeze(0), x_squared_sum).squeeze(0) / denominator
                )
                observations: dict[str, torch.Tensor] = {"stability": stability, "error": error}

                if k in previous_consensus:
                    observations["stability_cross"] = match_stability(
                        previous_consensus[k], consensus_kg, model.sinkhorn_epsilon, model.sinkhorn_iterations
                    )
                    h_1nk = model._solve_loadings_cold(x_ng, previous_consensus[k], fista_iterations)
                    observations["error_cross"] = (
                        frobenius_loss_trace(x_ng, h_1nk, previous_consensus[k].unsqueeze(0), x_squared_sum).squeeze(0)
                        / denominator
                    )

                for name, value in observations.items():
                    scalar = value.detach().double()
                    sums[name][k_index] += scalar
                    sums_of_squares[name][k_index] += scalar * scalar
                    counts[name][k_index] += 1.0

                # Running consensus mean, Sinkhorn-aligned to the first batch's reference.
                detached_kg = consensus_kg.detach()
                if k not in consensus_reference:
                    consensus_reference[k] = detached_kg.clone()
                    consensus_sums[k] = detached_kg.double().clone()
                else:
                    aligned_1kg, _, _, _ = align_factors(
                        detached_kg.unsqueeze(0),
                        consensus_reference[k],
                        model.sinkhorn_epsilon,
                        model.sinkhorn_iterations,
                    )
                    consensus_sums[k] += aligned_1kg.squeeze(0).double()
                n_consensus[k] += 1
                previous_consensus[k] = detached_kg

                # Replicate factors for the diagnostic plots, from the first batch only.
                if batches_used == 1 and k in store_k_set:
                    rkg = solved["w_rkg"].detach()[:r_store_actual].cpu()
                    model.register_buffer(f"D_{k}_rkg", rkg)

        if batches_used == 0:
            raise RuntimeError("the measurement dataloader yielded no usable batches")

        # Register measurement-time k_values so selection_curves() knows the layout.
        model.register_buffer("_measured_k_values", torch.tensor(k_values, dtype=torch.long, device="cpu"))
        for name in names:
            mean_k = torch.full((len(k_values),), float("nan"), dtype=dtype, device="cpu")
            sem_k = torch.full((len(k_values),), float("nan"), dtype=dtype, device="cpu")
            for k_index, k in enumerate(k_values):
                count = float(counts[name][k_index])
                if count < 1.0:
                    continue
                mean = sums[name][k_index] / count
                mean_k[k_index] = mean.to(mean_k.dtype)
                if count >= 2.0:
                    variance = (sums_of_squares[name][k_index] / count - mean * mean).clamp(min=0.0)
                    variance = variance * count / (count - 1.0)  # sample variance
                    sem_k[k_index] = (variance / count).sqrt().to(sem_k.dtype)
                else:
                    sem_k[k_index] = 0.0
            model.register_buffer(f"_measured_{name}_mean_k", mean_k)
            model.register_buffer(f"_measured_{name}_sem_k", sem_k)

        for k in k_values:
            if n_consensus[k] == 0:
                continue
            averaged_kg = (consensus_sums[k] / n_consensus[k]).to(dtype)
            normalized_kg = l1_normalize_rows(averaged_kg).cpu()
            buf_name = f"consensus_D_{k}_kg"
            if buf_name in model._buffers:
                t = model._buffers[buf_name]
                assert isinstance(t, torch.Tensor)
                t.copy_(normalized_kg.to(t.dtype))
            else:
                model.register_buffer(buf_name, normalized_kg)

        model._measured_n_batches.fill_(batches_used)
        if verbose:
            print(f"measurement phase complete: {batches_used} batches x {len(k_values)} k values")
        return model.selection_curves(k_values=k_values)
    finally:
        model.train(was_training)


def export_hot_start(
    model: CNMFTransformer,
    k_values: list[int] | None = None,
    r: int = 1,
) -> dict[int, torch.Tensor]:
    """
    Export measured consensus factors as ``{k: (r, k, g)}`` for seeding another NMF model.

    Intended for :class:`~cellarium.ml.models.AmortizedOnlineNonNegativeMatrixFactorization`, whose
    ``D_{k}_rkg`` buffers use the same L1-normalized-row convention.

    .. warning::
        Hot starting a *consensus* run partly defeats the consensus.  Kotliar's ``r`` replicates are
        meant to be independent random initializations, and that independence is what makes the
        downstream silhouette meaningful; seeding them all from one consensus makes them converge
        together and inflates stability regardless of whether the programs are real.  Use ``r=1``
        (or a small ``r``) to *polish* factors at a ``k`` you have already chosen.  To obtain a
        Kotliar-comparable stability number, run the other model cold instead.

    .. note::
        The receiving model accumulates Mairal's ``A_rkk`` / ``B_rkg`` from zero, so its first
        factor update optimizes against a single minibatch and will pull the injected factors a long
        way.  To preserve the hot start, accumulate ``A`` / ``B`` over a few batches with the
        injected factors frozen before letting them update.

    Args:
        model: A :class:`CNMFTransformer` whose measurement phase has been run.
        k_values: Which ``k`` to export.  Defaults to all of ``model.k_values``.
        r: Number of (identical) replicates to stack.

    Returns:
        Dict mapping ``k`` to a factor tensor of shape ``(r, k, g)``.
    """
    k_values = list(model.k_values) if k_values is None else sorted(k_values)
    missing = [k for k in k_values if f"consensus_D_{k}_kg" not in model._buffers]
    if missing:
        raise ValueError(
            f"k values {missing} have no consensus factors; run run_measurement_phase() first"
        )

    out: dict[int, torch.Tensor] = {}
    for k in k_values:
        consensus_kg = model._buffers[f"consensus_D_{k}_kg"].detach()
        if bool((consensus_kg == 0).all()):
            raise ValueError(f"consensus factors for k={k} are all zeros; run run_measurement_phase() first")
        row_sums = consensus_kg.sum(dim=-1)
        if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)):
            raise ValueError(
                f"consensus factors for k={k} are not L1-normalized by row, which is required for "
                "compatibility with AmortizedOnlineNonNegativeMatrixFactorization"
            )
        out[k] = consensus_kg.unsqueeze(0).expand(r, -1, -1).clone()
    return out


def plot_density_histograms(
    model: CNMFTransformer,
    dataloader: Iterable | None = None,
    transforms: Iterable[torch.nn.Module] = (),
    n_batches: int = 1,
    n_replicates: int = 50,
    fista_iterations: int = 150,
    device: torch.device | str | None = None,
    density_threshold: float = 1.0,
    local_neighborhood_size: float = 0.30,
    k_values: list[int] | None = None,
    n_bins: int = 75,
):
    """
    Histogram of mean neighbor distances for each ``k``, to guide the choice of
    ``density_threshold`` and ``local_neighborhood_size`` in :func:`run_measurement_phase`.

    Can be called **before** :func:`run_measurement_phase` by passing a ``dataloader`` — this
    is the intended workflow, since ``density_threshold`` must be chosen before running the full
    measurement.  When a dataloader is provided, a lightweight solve (no consensus, no stability
    metrics) is run on ``n_batches`` minibatches and the replicate tensors are held locally without
    mutating any model buffers.

    Each subplot shows the distribution of mean L2 distances from each L2-normalized factor row to
    its ``n_neighbors`` nearest neighbors (where
    ``n_neighbors = int(r_total * local_neighborhood_size)`` and
    ``r_total = n_replicates * n_batches``).
    A red vertical line marks ``density_threshold``; rows to the right of that line would be treated
    as outliers and excluded before k-means.  The title reports what fraction would be filtered.

    Args:
        model: A trained :class:`CNMFTransformer`.
        dataloader: Yields batch dicts with ``x_ng`` and ``var_names_g``.  If provided, a fresh
            solve is run and ``k_values`` defaults to all of ``model.k_values``.  If ``None``,
            reads from the replicate buffers populated by :func:`run_measurement_phase`.
        transforms: GPU transforms to apply to each batch (same as :func:`run_measurement_phase`).
        n_batches: Number of minibatches to solve.  More batches give a larger replicate pool and
            a better-sampled density distribution.  Only used when ``dataloader`` is provided.
        n_replicates: Independent replicate solves per ``(batch, k)``.  Only used when
            ``dataloader`` is provided.
        fista_iterations: FISTA iterations for the replicate solve.  Only used when ``dataloader``
            is provided.
        device: Device on which to run.  Defaults to the model's device.  Only used when
            ``dataloader`` is provided.
        density_threshold: Shown as a red vertical line.  Rows with mean neighbor distance above
            this value would be filtered.  Range ``(0, 2]``; ``1.0`` is a safe no-filter starting
            point for L2-normalized non-negative vectors.
        local_neighborhood_size: Fraction of replicates used to define the local neighborhood.
            ``n_neighbors = int(r_total * local_neighborhood_size)`` — matches the formula in
            :func:`~cellarium.ml.models.nmf.consensus`, measuring within-cluster density rather
            than cross-program distances.
            Range ``(0, 1)``.
        k_values: ``k`` values to plot.  When a dataloader is provided, defaults to all of
            ``model.k_values``; otherwise defaults to whatever ``k`` values have stored replicate
            buffers (set via ``store_replicates_k_values`` in :func:`run_measurement_phase`).
        n_bins: Number of histogram bins.

    Returns:
        The :class:`matplotlib.figure.Figure` — call ``plt.show()`` or ``fig.savefig(...)``
        on it yourself.
    """
    from matplotlib import pyplot as plt

    if dataloader is not None:
        # --- fresh-solve mode: run lightweight solves, accumulate replicates locally ---
        if k_values is None:
            k_values = list(model.k_values)
        else:
            k_values = sorted(k_values)

        device_ = model.device if device is None else torch.device(device)
        dtype = model.slot_mu.dtype
        transform_list = list(transforms)

        # w_batches[k] accumulates (r, k, g) tensors from each batch
        w_batches: dict[int, list[torch.Tensor]] = {k: [] for k in k_values}

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= n_batches:
                        break
                    for transform in transform_list:
                        batch |= transform(x_ng=batch["x_ng"], var_names_g=batch["var_names_g"])
                    x_ng = batch["x_ng"].to(device=device_, dtype=dtype)
                    if x_ng.shape[0] < 2:
                        continue
                    for k in k_values:
                        slot_noise_rke = torch.randn(n_replicates, k, model.latent_dim, device=device_, dtype=dtype)
                        solved = model.solve(x_ng, k, slot_noise_rke, fista_iterations)
                        assert isinstance(solved["w_rkg"], torch.Tensor)
                        w_batches[k].append(solved["w_rkg"].detach().float().cpu())
        finally:
            model.train(was_training)

        # replicate_factors[k]: (r_total, k, g) where r_total = n_replicates * n_batches_used
        replicate_factors: dict[int, torch.Tensor] = {
            k: torch.cat(w_batches[k], dim=0) for k in k_values if w_batches[k]
        }
        k_values = [k for k in k_values if k in replicate_factors]
        if not k_values:
            raise RuntimeError("dataloader yielded no usable batches")

    else:
        # --- buffer mode: read from replicate buffers stored by run_measurement_phase ---
        stored_k = [int(name[2:-4]) for name in model._buffers if name.startswith("D_") and name.endswith("_rkg")]
        if not stored_k:
            raise RuntimeError(
                "no replicate buffers found; pass a dataloader, or call run_measurement_phase() "
                "with store_replicates_k_values to pre-store replicates"
            )
        if k_values is None:
            k_values = sorted(stored_k)
        else:
            k_values = sorted(k_values)
            missing = [k for k in k_values if k not in stored_k]
            if missing:
                raise ValueError(
                    f"k values {missing} have no stored replicate buffer; "
                    "pass store_replicates_k_values to run_measurement_phase() to cover them"
                )

        replicate_factors = {}
        for k in k_values:
            t = model._buffers[f"D_{k}_rkg"]
            assert isinstance(t, torch.Tensor)
            replicate_factors[k] = t.detach().float()

    n_panels = len(k_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3), squeeze=False)
    axes = axes[0]

    for ax, k in zip(axes, k_values):
        w_rkg = replicate_factors[k]  # (r_total, k, g)
        r_total = w_rkg.shape[0]
        total_rows = r_total * k
        # n_neighbors scales with replicates, not total rows — same formula as nmf.py consensus().
        # This measures within-cluster density: ~30% of same-program replicates per factor.
        # Using total_rows here would force cross-program comparisons and inflate the metric.
        n_neighbors = int(r_total * local_neighborhood_size)

        if n_neighbors < 2:
            ax.text(
                0.5,
                0.5,
                f"k={k}\ntoo few replicates\nfor neighborhood\n(r={r_total})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        # L2-normalize and flatten to (r_total*k, g), same as consensus()
        d_norm_mg = F.normalize(w_rkg.reshape(total_rows, -1), dim=-1, p=2)
        dist_mm = torch.cdist(d_norm_mg, d_norm_mg, p=2)
        dist_mm.fill_diagonal_(0.0)
        nearest_ml, _ = torch.topk(dist_mm, n_neighbors + 1, largest=False)
        mean_dist_m = nearest_ml[:, 1:].mean(dim=1).cpu().numpy()  # exclude self (col 0)

        pct_filtered = 100.0 * float((mean_dist_m > density_threshold).mean())
        ax.hist(mean_dist_m, bins=n_bins, color="steelblue", edgecolor="none")
        ax.axvline(density_threshold, color="red", linewidth=1.5, label=f"threshold={density_threshold}")
        ax.set_xlim(-0.05, 2.05)
        ax.set_xlabel(f"mean dist. to {n_neighbors} neighbors")
        ax.set_ylabel(f"factor rows  (r×k = {total_rows})")
        ax.set_title(f"k = {k}\n{pct_filtered:.1f}% filtered at threshold")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_k_selection(model: CNMFTransformer, k_values: list[int] | None = None, use_cross: bool = False) -> None:
    """
    Plot the measured stability / error trade-off, with error bars from the measurement batches.

    Args:
        model: A :class:`CNMFTransformer` whose measurement phase has been run.
        use_cross: Plot the cross-batch statistics (which fold in sampling variance) instead of the
            Kotliar-comparable within-batch ones.
    """
    from matplotlib import pyplot as plt

    curves = model.selection_curves(k_values=k_values if k_values is not None else model.k_values)
    stability_key = "stability_cross" if use_cross else "stability"
    error_key = "error_cross" if use_cross else "error"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(curves["k"], curves[stability_key], yerr=curves[f"{stability_key}_sem"], fmt="o-", color="b", capsize=2)
    ax.set_ylabel(f"Stability ({'cross-batch' if use_cross else 'within-batch'})", color="b")
    ax.set_xlabel("Number of components: k")
    ax.tick_params(axis="y", colors="b")
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.errorbar(curves["k"], curves[error_key], yerr=curves[f"{error_key}_sem"], fmt="o-", color="r", capsize=2)
    ax2.set_ylabel("Reconstruction error (per-entry MSE)", color="r")
    ax2.tick_params(axis="y", colors="r")
    ax2.grid(False)
    fig.tight_layout()
    plt.show()
