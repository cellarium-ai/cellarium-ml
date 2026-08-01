# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

# Must be set before torch is imported so that torch._dynamo picks it up: the FISTA helpers in
# cellarium.ml.models.nmf are torch.compile-decorated and compiling them for every k value in these
# tests would dominate the runtime.
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import anndata  # noqa: E402
import lightning.pytorch as pl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule  # noqa: E402
from cellarium.ml.models import CNMFTransformer  # noqa: E402
from cellarium.ml.models.cnmf_transformer import (  # noqa: E402
    LinearFactorDecoder,
    SlotAttentionBlock,
    align_factors,
    export_hot_start,
    frobenius_loss_trace,
    l1_normalize_rows,
    log_sinkhorn,
    match_stability,
    matched_distance,
    matched_silhouette,
    run_measurement_phase,
    sinkhorn_consensus,
)
from cellarium.ml.utilities.data import AnnDataField  # noqa: E402

# NOTE: nothing in this model is specific to scRNA-seq.  It solves NMF on any non-negative matrix,
# which is why the tests below use plain synthetic factorizations rather than realistic expression.


# -----------------------------------------------------------------------------------------------
# Synthetic data with a known number of factors
# -----------------------------------------------------------------------------------------------


def make_synthetic_nmf_data(
    n_cells: int,
    n_genes: int,
    k_true: int,
    seed: int = 0,
    noise: float = 0.0,
    programs_per_cell: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build ``X = H W`` with a known ``k_true`` and *disjoint* gene blocks per program.

    Disjoint blocks make the rank-``k_true`` non-negative factorization essentially unique, so
    "how many programs are in this data" has an unambiguous answer and the recovered ``k`` can be
    asserted on.

    Returns:
        ``(x_ng, w_kg, h_nk)`` as float32 numpy arrays.
    """
    rng = np.random.default_rng(seed)
    block = n_genes // k_true
    assert block >= 2, "need at least 2 genes per program"

    w_kg = np.zeros((k_true, n_genes), dtype=np.float32)
    for k in range(k_true):
        lo, hi = k * block, (k + 1) * block
        w_kg[k, lo:hi] = rng.gamma(shape=2.0, scale=1.0, size=hi - lo)
    w_kg /= w_kg.sum(axis=1, keepdims=True)

    h_nk = np.zeros((n_cells, k_true), dtype=np.float32)
    for n in range(n_cells):
        chosen = rng.choice(k_true, size=min(programs_per_cell, k_true), replace=False)
        h_nk[n, chosen] = rng.gamma(shape=2.0, scale=1.0, size=len(chosen))

    x_ng = h_nk @ w_kg
    if noise > 0:
        x_ng = np.clip(x_ng + noise * rng.standard_normal(x_ng.shape).astype(np.float32), 0.0, None)
    return x_ng.astype(np.float32), w_kg, h_nk


@pytest.fixture
def small_adata() -> anndata.AnnData:
    n, g, k_true = 600, 24, 4
    x_ng, _, _ = make_synthetic_nmf_data(n, g, k_true, seed=0)
    return anndata.AnnData(
        X=x_ng,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(g)]),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n)]),
    )


def make_model(n_genes: int, k_values: list[int], **kwargs) -> CNMFTransformer:
    """A tiny model with the schedules collapsed, suitable for unit tests."""
    defaults: dict = dict(
        latent_dim=16,
        n_iterations=2,
        n_self_attention_heads=2,
        ffn_mult=2,
        n_replicates=4,
        min_cells_per_split=8,
        fista_iterations_train=5,
        sinkhorn_iterations=20,
        curriculum_warmup_steps=0,
        stability_burn_in_steps=0,
        drift_eval_n_cells=32,
        drift_check_every_n_steps=0,
        r_store=4,
        measure_at_end=False,
        measurement_n_batches=3,
    )
    defaults.update(kwargs)
    return CNMFTransformer(
        var_names_g=[f"gene_{i}" for i in range(n_genes)],
        k_values=k_values,
        **defaults,
    )


def as_tensor(value: object) -> torch.Tensor:
    """Narrow a ``Tensor | None`` (or ``np.ndarray | Tensor``) dict value for the type checker."""
    assert isinstance(value, torch.Tensor)
    return value


# -----------------------------------------------------------------------------------------------
# log_sinkhorn / matching
# -----------------------------------------------------------------------------------------------


def test_log_sinkhorn_is_doubly_stochastic() -> None:
    torch.manual_seed(0)
    cost_kk = torch.rand(6, 6)
    plan = log_sinkhorn(cost_kk, epsilon=0.05, n_iterations=200)
    torch.testing.assert_close(plan.sum(dim=0), torch.ones(6), atol=1e-4, rtol=0)
    torch.testing.assert_close(plan.sum(dim=1), torch.ones(6), atol=1e-4, rtol=0)


def test_log_sinkhorn_recovers_a_known_permutation() -> None:
    k = 5
    permutation = torch.tensor([2, 0, 4, 1, 3])
    # Zero cost on the true match, unit cost elsewhere.
    cost_kk = torch.ones(k, k)
    cost_kk[torch.arange(k), permutation] = 0.0
    plan = log_sinkhorn(cost_kk, epsilon=0.02, n_iterations=200)
    assert torch.equal(plan.argmax(dim=1), permutation)
    # The plan should be close to a hard permutation matrix at small epsilon.
    assert float(plan.max(dim=1).values.min()) > 0.95


def test_log_sinkhorn_does_not_underflow_at_large_cost() -> None:
    """A naive exp(-C / 0.05) underflows here; the log-domain version must not."""
    cost_kk = torch.full((4, 4), 2.0)
    cost_kk.fill_diagonal_(0.0)
    plan = log_sinkhorn(cost_kk, epsilon=0.05, n_iterations=100)
    assert torch.isfinite(plan).all()
    torch.testing.assert_close(plan.sum(dim=1), torch.ones(4), atol=1e-4, rtol=0)


def test_log_sinkhorn_rejects_non_square_cost() -> None:
    with pytest.raises(ValueError, match="square cost matrix"):
        log_sinkhorn(torch.rand(3, 4))


def test_align_factors_undoes_a_permutation() -> None:
    torch.manual_seed(0)
    k, g = 5, 12
    reference_kg = torch.rand(k, g).abs()
    permutation = torch.randperm(k)
    w_rkg = reference_kg[permutation].unsqueeze(0)
    aligned_rkg, cosine_rk, _, _ = align_factors(w_rkg, reference_kg, epsilon=0.02, n_iterations=200)
    torch.testing.assert_close(aligned_rkg.squeeze(0), reference_kg, atol=1e-2, rtol=0)
    assert float(cosine_rk.min()) > 0.99


def test_matched_silhouette_removes_the_non_negative_cosine_baseline() -> None:
    """
    Raw cosine similarity is unusable as a stability metric for non-negative factors: independent
    random non-negative vectors already score ~0.75 pairwise, so the whole interesting range is
    squeezed into the last few percent.  The silhouette-style contrast must not have that floor.
    """
    torch.manual_seed(0)
    k, g = 5, 20
    # Disjoint gene blocks: distinct programs, perfectly reproduced -> score near 1.
    distinct_kg = torch.zeros(k, g)
    block = g // k
    for i in range(k):
        distinct_kg[i, i * block : (i + 1) * block] = torch.rand(block).abs() + 0.1
    _, _, plan, similarity = align_factors(distinct_kg.unsqueeze(0), distinct_kg, 0.02, 200)
    assert float(matched_silhouette(similarity, plan).mean()) > 0.9

    # Independent random non-negative factors -> score near 0 despite high raw cosine.
    random_kg = torch.rand(k, g).abs()
    other_kg = torch.rand(k, g).abs()
    _, cosine_rk, plan, similarity = align_factors(random_kg.unsqueeze(0), other_kg, 0.02, 200)
    assert float(cosine_rk.mean()) > 0.5, "raw cosine really does have a high floor here"
    assert float(matched_silhouette(similarity, plan).mean()) < 0.4


def test_matched_silhouette_handles_a_single_factor() -> None:
    w_kg = torch.rand(1, 8).abs()
    _, _, plan, similarity = align_factors(w_kg.unsqueeze(0), w_kg, 0.02, 50)
    score = matched_silhouette(similarity, plan)
    assert score.shape == (1, 1)
    assert bool(torch.isfinite(score).all()) and float(score) == pytest.approx(1.0)


# -----------------------------------------------------------------------------------------------
# sinkhorn_consensus
# -----------------------------------------------------------------------------------------------


def disjoint_programs(k: int, g: int, seed: int = 0) -> torch.Tensor:
    """``k`` L1-normalized programs on disjoint gene blocks, so distinct programs are orthogonal."""
    generator = torch.Generator().manual_seed(seed)
    w_kg = torch.zeros(k, g)
    block = g // k
    assert block >= 1
    for i in range(k):
        w_kg[i, i * block : (i + 1) * block] = torch.rand(block, generator=generator).abs() + 0.1
    return l1_normalize_rows(w_kg)


def test_consensus_recovers_shared_factors_from_permuted_replicates() -> None:
    """Replicates that agree up to a permutation should give a high stability and recover W."""
    torch.manual_seed(0)
    r, k, g = 8, 5, 20
    w_true_kg = disjoint_programs(k, g, seed=0)
    replicates = [
        l1_normalize_rows((w_true_kg[torch.randperm(k)] + 0.002 * torch.rand(k, g)).clamp(min=0)) for _ in range(r)
    ]
    w_rkg = torch.stack(replicates)

    out = sinkhorn_consensus(w_rkg, epsilon=0.02, n_iterations=200, n_refine=2)
    assert float(as_tensor(out["stability"])) > 0.9
    # The consensus should match the truth up to a permutation.
    assert float(match_stability(as_tensor(out["consensus_kg"]), w_true_kg, epsilon=0.02, n_iterations=200)) > 0.9


def test_consensus_stability_is_lower_for_disagreeing_replicates() -> None:
    torch.manual_seed(0)
    r, k, g = 8, 5, 20
    w_true_kg = disjoint_programs(k, g, seed=0)
    agreeing = torch.stack([l1_normalize_rows(w_true_kg[torch.randperm(k)]) for _ in range(r)])
    disagreeing = l1_normalize_rows(torch.rand(r, k, g).abs())

    stability_agree = float(as_tensor(sinkhorn_consensus(agreeing, epsilon=0.02, n_iterations=200)["stability"]))
    stability_disagree = float(as_tensor(sinkhorn_consensus(disagreeing, epsilon=0.02, n_iterations=200)["stability"]))
    assert stability_agree > 0.9
    assert stability_disagree < 0.5
    assert stability_agree > stability_disagree + 0.4


def test_consensus_rows_are_l1_normalized() -> None:
    torch.manual_seed(0)
    w_rkg = l1_normalize_rows(torch.rand(6, 4, 15).abs())
    consensus_kg = as_tensor(sinkhorn_consensus(w_rkg)["consensus_kg"])
    torch.testing.assert_close(consensus_kg.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)


def test_consensus_handles_single_replicate() -> None:
    w_rkg = torch.rand(1, 3, 8).abs()
    out = sinkhorn_consensus(w_rkg)
    assert out["plan_rkj"] is None
    assert float(as_tensor(out["stability"])) == pytest.approx(1.0)
    torch.testing.assert_close(as_tensor(out["consensus_kg"]), l1_normalize_rows(w_rkg[0]))


def test_consensus_is_not_anchored_on_replicate_zero() -> None:
    """
    An outlier placed at index 0 must not dominate the consensus.

    A naive implementation anchors on replicate 0 and gives it unpenalized privileged status; the
    detached barycenter refinement is what removes that asymmetry.
    """
    torch.manual_seed(0)
    r, k, g = 9, 4, 16
    w_true_kg = disjoint_programs(k, g, seed=1)
    replicates = [l1_normalize_rows(torch.rand(k, g).abs())]  # outlier first
    replicates += [l1_normalize_rows(w_true_kg[torch.randperm(k)]) for _ in range(r - 1)]
    w_rkg = torch.stack(replicates)

    out = sinkhorn_consensus(w_rkg, epsilon=0.02, n_iterations=200, n_refine=2, outlier_gamma=5.0)
    similarity = float(match_stability(as_tensor(out["consensus_kg"]), w_true_kg, epsilon=0.02, n_iterations=200))
    assert similarity > 0.8, f"outlier at index 0 dominated the consensus (similarity {similarity})"
    # And the outlier must be down-weighted relative to the agreeing replicates.
    weights_rk = as_tensor(out["weights_rk"])
    assert float(weights_rk[0].mean()) < float(weights_rk[1:].mean())


def test_consensus_gradient_flows_to_factors_but_not_through_plan() -> None:
    torch.manual_seed(0)
    w_rkg = l1_normalize_rows(torch.rand(4, 3, 10).abs()).requires_grad_(True)
    out = sinkhorn_consensus(w_rkg, detach_plan=True)
    as_tensor(out["consensus_kg"]).sum().backward()
    assert w_rkg.grad is not None
    assert torch.isfinite(w_rkg.grad).all()
    assert not as_tensor(out["plan_rkj"]).requires_grad


# -----------------------------------------------------------------------------------------------
# frobenius_loss_trace
# -----------------------------------------------------------------------------------------------


def test_frobenius_loss_trace_matches_materialized_loss() -> None:
    torch.manual_seed(0)
    r, n, k, g = 3, 40, 5, 12
    x_ng = torch.rand(n, g).abs()
    h_rnk = torch.rand(r, n, k).abs()
    w_rkg = torch.rand(r, k, g).abs()

    traced_r = frobenius_loss_trace(x_ng, h_rnk, w_rkg)
    naive_r = ((torch.einsum("rnk,rkg->rng", h_rnk, w_rkg) - x_ng) ** 2).sum(dim=(-2, -1))
    torch.testing.assert_close(traced_r, naive_r, atol=1e-2, rtol=1e-4)


def test_frobenius_loss_trace_is_zero_for_exact_factorization() -> None:
    x_ng, w_kg, h_nk = make_synthetic_nmf_data(60, 20, 4, seed=1)
    sse = frobenius_loss_trace(
        torch.from_numpy(x_ng), torch.from_numpy(h_nk).unsqueeze(0), torch.from_numpy(w_kg).unsqueeze(0)
    )
    assert float(sse) < 1e-4 * float((torch.from_numpy(x_ng) ** 2).sum())


def test_frobenius_loss_trace_gradient_only_touches_factors() -> None:
    torch.manual_seed(0)
    x_ng = torch.rand(30, 8).abs()
    h_rnk = torch.rand(2, 30, 3).abs()
    w_rkg = torch.rand(2, 3, 8).abs().requires_grad_(True)
    frobenius_loss_trace(x_ng, h_rnk, w_rkg).sum().backward()
    assert w_rkg.grad is not None and torch.isfinite(w_rkg.grad).all()
    assert not h_rnk.requires_grad


# -----------------------------------------------------------------------------------------------
# SlotAttentionBlock
# -----------------------------------------------------------------------------------------------


def test_slot_attention_competes_across_slots_not_cells() -> None:
    """The cross-attention softmax must be over k (competition), which is the anti-collapse device."""
    torch.manual_seed(0)
    e, k, n, r = 8, 5, 17, 3
    block = SlotAttentionBlock(e, n_self_attention_heads=2, ffn_mult=2)
    slots_rke = torch.randn(r, k, e)
    key_ne = torch.randn(n, e)
    query_rke = block.to_query(block.norm_cross_attention(slots_rke)) * block.scale
    attention_rkn = torch.einsum("rke,ne->rkn", query_rke, key_ne).softmax(dim=-2)
    torch.testing.assert_close(attention_rkn.sum(dim=-2), torch.ones(r, n), atol=1e-5, rtol=0)


def test_slot_attention_is_invariant_to_duplicating_cells() -> None:
    """
    Normalized-weighted-mean aggregation makes the block depend on the *distribution* of cells
    rather than their count, which is what allows the number of conditioning cells to change
    between training and inference.
    """
    torch.manual_seed(0)
    e, k, n, r = 8, 4, 11, 2
    block = SlotAttentionBlock(e, n_self_attention_heads=2, ffn_mult=2).eval()
    slots_rke = torch.randn(r, k, e)
    key_ne = torch.randn(n, e)
    value_ne = torch.randn(n, e)

    with torch.no_grad():
        once = block(slots_rke, key_ne, value_ne)
        twice = block(slots_rke, key_ne.repeat(2, 1), value_ne.repeat(2, 1))
    torch.testing.assert_close(once, twice, atol=1e-5, rtol=1e-4)


def test_slot_attention_preserves_shape() -> None:
    block = SlotAttentionBlock(16, n_self_attention_heads=4, ffn_mult=2)
    out = block(torch.randn(3, 7, 16), torch.randn(20, 16), torch.randn(20, 16))
    assert out.shape == (3, 7, 16)


def test_decoder_output_is_non_negative_and_l1_normalized() -> None:
    decoder = LinearFactorDecoder(16, 30)
    w_rkg = decoder(torch.randn(3, 5, 16))
    assert bool((w_rkg >= 0).all())
    torch.testing.assert_close(w_rkg.sum(dim=-1), torch.ones(3, 5), atol=1e-5, rtol=0)


# -----------------------------------------------------------------------------------------------
# construction, reset_parameters, curriculum
# -----------------------------------------------------------------------------------------------


def test_reset_parameters_from_meta_device() -> None:
    """Models are built under torch.device("meta"); reset_parameters must produce every value."""
    with torch.device("meta"):
        model = make_model(20, [2, 3, 4])
    model = model.to_empty(device="cpu")
    model.reset_parameters()

    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), f"non-finite parameter {name}"
    for name, buffer in model.named_buffers():
        if name.startswith("_measured_"):
            continue  # deliberately NaN until the measurement phase runs
        assert torch.isfinite(buffer).all(), f"non-finite buffer {name}"
    assert bool((model.drift_slot_noise_rke != 0).any()), "fixed drift noise was never generated"


def test_fixed_drift_noise_is_reproducible() -> None:
    a = make_model(20, [2, 3], noise_seed=7)
    b = make_model(20, [2, 3], noise_seed=7)
    c = make_model(20, [2, 3], noise_seed=8)
    torch.testing.assert_close(a.drift_slot_noise_rke, b.drift_slot_noise_rke)
    assert not torch.allclose(a.drift_slot_noise_rke, c.drift_slot_noise_rke)


def test_invalid_k_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_model(20, [])
    with pytest.raises(ValueError, match=">= 1"):
        make_model(20, [0, 2])
    with pytest.raises(ValueError, match="duplicates"):
        make_model(20, [2, 2, 3])
    with pytest.raises(ValueError, match="drift_k_values"):
        make_model(20, [2, 3], drift_k_values=[7])
    with pytest.raises(ValueError, match="store_replicates_k_values"):
        make_model(20, [2, 3], store_replicates_k_values=[9])
    with pytest.raises(ValueError, match="r_store"):
        model = make_model(20, [2, 3], r_store=8)
        run_measurement_phase(model, iter([{"x_ng": torch.rand(4, 20), "var_names_g": model.var_names_g}]), n_replicates=4)


def test_unsorted_k_values_are_sorted_with_a_warning() -> None:
    with pytest.warns(UserWarning, match="sorted ascending"):
        model = make_model(20, [5, 2, 3])
    assert model.k_values == [2, 3, 5]


def test_curriculum_expands_k_window_monotonically() -> None:
    model = make_model(20, [2, 3, 4, 5, 6], curriculum_warmup_steps=100, curriculum_initial_k_count=1)
    counts = [model.curriculum_k_count(step) for step in range(0, 141, 10)]
    assert counts[0] == 1
    assert counts[-1] == len(model.k_values)
    assert all(b >= a for a, b in zip(counts, counts[1:])), counts


def test_sampled_k_is_deterministic_in_the_step_and_respects_the_curriculum() -> None:
    """Every DDP rank must draw the same k, or the gradient all-reduce deadlocks on shape mismatch."""
    model = make_model(20, [2, 3, 4, 5, 6], curriculum_warmup_steps=100)
    for step in range(50):
        k = model.sample_k(step)
        assert k == model.sample_k(step), "k sampling is not deterministic in the step"
        assert k in model.k_values
        assert k <= model.k_values[model.curriculum_k_count(step) - 1]
    # And it does actually vary with the step.
    assert len({model.sample_k(step) for step in range(200, 260)}) > 1


# -----------------------------------------------------------------------------------------------
# forward / solve
# -----------------------------------------------------------------------------------------------


def test_solve_returns_valid_factors_and_loadings() -> None:
    torch.manual_seed(0)
    n, g, k, r = 40, 20, 4, 3
    model = make_model(g, [k])  # n_iterations=2 by default in make_model
    x_ng = torch.from_numpy(make_synthetic_nmf_data(n, g, k, seed=0)[0])
    noise_rke = torch.randn(r, k, model.latent_dim)

    out = model.solve(x_ng, k, noise_rke, n_iterations=10)
    assert out["w_rkg"].shape == (r, k, g)
    assert out["h_rnk"].shape == (r, n, k)
    assert bool((out["w_rkg"] >= 0).all()) and bool((out["h_rnk"] >= 0).all())
    torch.testing.assert_close(out["w_rkg"].sum(dim=-1), torch.ones(r, k), atol=1e-5, rtol=0)
    # H is a detached solver output: no gradient may flow through the FISTA polish.
    assert not out["h_rnk"].requires_grad
    assert out["w_rkg"].requires_grad
    # Per-iteration W: one per recurrent step, all with the right shape and L1-normalized rows.
    assert len(out["w_layers_rkg"]) == model.n_iterations
    for i, w_i in enumerate(out["w_layers_rkg"]):
        assert w_i.shape == (r, k, g), f"iteration {i} W shape mismatch"
        torch.testing.assert_close(w_i.sum(dim=-1), torch.ones(r, k), atol=1e-5, rtol=0)
    assert out["w_rkg"] is out["w_layers_rkg"][-1]


def test_hot_start_loadings_have_the_correct_total_scale() -> None:
    """L1-normalized factors imply sum_k H[n, k] == sum_g X[n, g] exactly."""
    torch.manual_seed(0)
    n, g, k, r = 25, 16, 3, 2
    model = make_model(g, [k])
    x_ng = torch.rand(n, g).abs()
    x_emb_ne = model.encoder(x_ng)
    slots_rke = torch.randn(r, k, model.latent_dim)
    h_rnk = model.hot_start_loadings(x_ng, x_emb_ne, slots_rke)
    torch.testing.assert_close(h_rnk.sum(dim=-1), x_ng.sum(dim=-1).expand(r, n), atol=1e-4, rtol=1e-4)


def test_forward_returns_finite_loss() -> None:
    torch.manual_seed(0)
    n, g, k_true = 80, 20, 4
    model = make_model(g, [2, 3, 4])
    x_ng = torch.from_numpy(make_synthetic_nmf_data(n, g, k_true, seed=0)[0])
    var_names_g = model.var_names_g

    out = model(x_ng=x_ng, var_names_g=var_names_g)
    assert set(out.keys()) == {"loss"}
    loss = as_tensor(out["loss"])
    assert torch.isfinite(loss) and float(loss) >= 0.0


def test_forward_backward_produces_gradients_everywhere() -> None:
    torch.manual_seed(0)
    model = make_model(20, [3])
    x_ng = torch.from_numpy(make_synthetic_nmf_data(80, 20, 3, seed=0)[0])
    as_tensor(model(x_ng=x_ng, var_names_g=model.var_names_g)["loss"]).backward()

    without_grad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not bool(parameter.grad.abs().sum() > 0)
    ]
    # k_embedding rows for k values not sampled this step legitimately receive no gradient.
    without_grad = [name for name in without_grad if "k_embedding" not in name]
    # slot_log_sigma is intentionally frozen (requires_grad=False) to prevent posterior collapse.
    without_grad = [name for name in without_grad if "slot_log_sigma" not in name]
    assert without_grad == [], f"parameters received no gradient: {without_grad}"


def test_forward_returns_only_loss_key() -> None:
    model = make_model(20, [3], cross_batch_stability=False)
    out = model(x_ng=torch.rand(40, 20).abs(), var_names_g=model.var_names_g)
    assert set(out.keys()) == {"loss"}


def test_split_is_skipped_for_small_batches() -> None:
    model = make_model(20, [3], min_cells_per_split=32)
    assert len(model._split_batch(torch.rand(40, 20))) == 1
    assert len(model._split_batch(torch.rand(80, 20))) == 2


def test_forward_rejects_mismatched_var_names() -> None:
    model = make_model(20, [3])
    with pytest.raises(ValueError, match="must match"):
        model(x_ng=torch.rand(40, 20).abs(), var_names_g=np.array([f"other_{i}" for i in range(20)]))


# -----------------------------------------------------------------------------------------------
# monitoring EMAs
# -----------------------------------------------------------------------------------------------


def test_ema_curve_is_nan_until_data_accumulates() -> None:
    model = make_model(20, [2, 3, 4])
    curve = model.ema_curve("stability")
    assert bool(torch.isnan(curve).all())

    model._update_ema(3, {"stability": torch.tensor(0.8)})
    curve = model.ema_curve("stability")
    assert float(curve[model._k_to_index[3]]) == pytest.approx(0.8, abs=1e-5)
    assert bool(torch.isnan(curve[model._k_to_index[2]]))


def test_ema_curve_is_bias_corrected_toward_the_true_mean() -> None:
    model = make_model(20, [2, 3], stability_ema_beta=0.5)
    for _ in range(30):
        model._update_ema(2, {"stability": torch.tensor(0.6)})
    assert float(model.ema_curve("stability")[model._k_to_index[2]]) == pytest.approx(0.6, abs=1e-3)


def test_ema_respects_burn_in() -> None:
    model = make_model(20, [2, 3], stability_burn_in_steps=10)
    model._step_cache = 0
    model._update_ema(2, {"stability": torch.tensor(0.9)})
    assert bool(torch.isnan(model.ema_curve("stability")).all())
    model._step_cache = 10
    model._update_ema(2, {"stability": torch.tensor(0.9)})
    assert not bool(torch.isnan(model.ema_curve("stability")[model._k_to_index[2]]))


def test_ema_curve_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        make_model(20, [2]).ema_curve("nonsense")


# -----------------------------------------------------------------------------------------------
# drift-based convergence
# -----------------------------------------------------------------------------------------------


def test_drift_is_zero_when_the_weights_do_not_change() -> None:
    """
    Drift is a deterministic function of the weights: fixed noise, fixed cells, fixed k.  With
    frozen weights it must be exactly zero, which is what makes the criterion noise-free.
    """
    torch.manual_seed(0)
    model = make_model(20, [3, 4], drift_check_every_n_steps=1, drift_eval_n_cells=32)
    x_ng = torch.from_numpy(make_synthetic_nmf_data(32, 20, 3, seed=0)[0])
    model._maybe_capture_drift_cells(x_ng)

    assert model.drift_check() is None  # nothing to compare against yet
    rate = model.drift_check()
    # The Sinkhorn matching can have a small numerical floor when factors are near-degenerate at
    # initialization; 1e-3 is still well below any real drift signal (drift_tol default: 5e-7/step).
    assert rate is not None and rate == pytest.approx(0.0, abs=1e-3)


def test_drift_is_positive_after_the_weights_move() -> None:
    torch.manual_seed(0)
    model = make_model(20, [3], drift_check_every_n_steps=1, drift_eval_n_cells=32)
    model._maybe_capture_drift_cells(torch.rand(32, 20).abs())
    model.drift_check()
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.add_(torch.randn_like(parameter))
    rate = model.drift_check()
    assert rate is not None and rate > 1e-6


def test_drift_survives_a_permutation_of_the_factors() -> None:
    """
    The drift metric re-aligns before comparing.  Without matching, one permutation flip in the
    slot ordering would read as catastrophic drift.
    """
    torch.manual_seed(0)
    k, g = 5, 16
    w_kg = disjoint_programs(k, g, seed=2)
    permuted_kg = w_kg[torch.randperm(k)]
    _, _, plan, similarity = align_factors(permuted_kg.unsqueeze(0), w_kg, epsilon=0.02, n_iterations=200)
    assert float(matched_distance(similarity, plan).mean()) == pytest.approx(0.0, abs=1e-6)


def test_matched_distance_is_exactly_zero_for_identical_factors() -> None:
    """
    The soft-aligned cosine has a nonzero floor from Sinkhorn's entropic blending, which would put
    a permanent floor under the drift metric.  The hard-assignment distance must not.
    """
    torch.manual_seed(0)
    w_kg = l1_normalize_rows(torch.rand(6, 20).abs())
    _, cosine_1k, plan, similarity = align_factors(w_kg.unsqueeze(0), w_kg, epsilon=0.05, n_iterations=20)
    soft_floor = float((1.0 - cosine_1k).mean())
    assert soft_floor > 1e-5, "expected the soft-aligned cosine to have a floor here"
    assert float(matched_distance(similarity, plan).mean()) == pytest.approx(0.0, abs=1e-6)


def test_convergence_requires_patience_and_a_finished_curriculum() -> None:
    model = make_model(
        20,
        [3],
        curriculum_warmup_steps=100,
        drift_settle_steps=50,
        drift_patience_checks=3,
        drift_check_every_n_steps=1,
        drift_eval_n_cells=16,
    )
    model._maybe_capture_drift_cells(torch.rand(16, 20).abs())
    model.drift_check()
    for _ in range(10):
        model.drift_check()
    assert int(model._drift_below_tol_count) >= 3
    # Still too early: the curriculum has not finished.
    model._step_cache = 100
    assert not model.converged
    model._step_cache = 150
    assert model.converged


def test_drift_counter_resets_when_drift_exceeds_tolerance() -> None:
    torch.manual_seed(0)
    model = make_model(20, [3], drift_check_every_n_steps=1, drift_eval_n_cells=16, drift_tol=1e-9)
    model._maybe_capture_drift_cells(torch.rand(16, 20).abs())
    model.drift_check()
    model.drift_check()
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.add_(torch.randn_like(parameter))
    model.drift_check()
    assert int(model._drift_below_tol_count) == 0


# -----------------------------------------------------------------------------------------------
# measurement phase, outputs and hand-off
# -----------------------------------------------------------------------------------------------


def _fake_dataloader(x_ng: np.ndarray, var_names_g: np.ndarray, batch_size: int, n_batches: int):
    rng = np.random.default_rng(0)
    for _ in range(n_batches):
        rows = rng.choice(x_ng.shape[0], size=batch_size, replace=False)
        yield {"x_ng": torch.from_numpy(x_ng[rows]), "var_names_g": var_names_g}


def test_measurement_phase_populates_curves_and_consensus() -> None:
    torch.manual_seed(0)
    n, g, k_true = 200, 20, 4
    k_values = [2, 3, 4, 5]
    model = make_model(g, k_values, store_replicates_k_values=[3], r_store=4)
    x_ng, _, _ = make_synthetic_nmf_data(n, g, k_true, seed=0)

    curves = run_measurement_phase(
        model,
        _fake_dataloader(x_ng, model.var_names_g, batch_size=50, n_batches=4),
        n_batches=4,
        n_replicates=4,
        verbose=False,
    )

    assert list(curves["k"]) == k_values
    for name in model.metric_names:
        assert np.isfinite(curves[name]).all(), f"{name} has non-finite entries"
        assert np.isfinite(curves[f"{name}_sem"]).all(), f"{name}_sem has non-finite entries"
    # SEM is meaningful only with more than one batch, and it is what distinguishes a real
    # discontinuity in the curve from measurement noise.
    assert (curves["stability_sem"] >= 0).all()
    assert int(model._measured_n_batches) == 4

    for k in k_values:
        consensus_kg = getattr(model, f"consensus_D_{k}_kg")
        assert bool((consensus_kg > 0).any())
        torch.testing.assert_close(consensus_kg.sum(dim=-1), torch.ones(k), atol=1e-4, rtol=0)
    assert bool((getattr(model, "D_3_rkg") != 0).any())


def test_measurement_phase_raises_on_an_empty_dataloader() -> None:
    model = make_model(20, [2, 3])
    with pytest.raises(RuntimeError, match="no usable batches"):
        run_measurement_phase(model, iter([]), n_batches=3, n_replicates=4, verbose=False)


def test_measurement_phase_rejects_unknown_k() -> None:
    model = make_model(20, [2, 3])
    with pytest.raises(ValueError, match="not in model.k_values"):
        run_measurement_phase(model, iter([]), k_values=[9], n_replicates=4, verbose=False)


def test_measurement_phase_restores_training_mode() -> None:
    torch.manual_seed(0)
    model = make_model(20, [2, 3]).train()
    x_ng, _, _ = make_synthetic_nmf_data(100, 20, 3, seed=0)
    run_measurement_phase(model, _fake_dataloader(x_ng, model.var_names_g, 40, 2), n_batches=2, n_replicates=4, verbose=False)
    assert model.training


def test_infer_loadings_and_reconstruction_error_after_measurement() -> None:
    torch.manual_seed(0)
    n, g, k_true = 150, 20, 4
    model = make_model(g, [3, 4])
    x_ng, _, _ = make_synthetic_nmf_data(n, g, k_true, seed=0)
    run_measurement_phase(model, _fake_dataloader(x_ng, model.var_names_g, 50, 3), n_batches=3, n_replicates=4, fista_iterations=30, verbose=False)

    x_batch = torch.from_numpy(x_ng[:50])
    alpha_nk = model.infer_loadings(x_batch, model.var_names_g, model.consensus_factors, k=4)
    assert alpha_nk.shape == (50, 4)
    assert bool((alpha_nk >= 0).all())

    errors = model.reconstruction_error(x_batch, model.var_names_g, model.consensus_factors)
    assert set(errors) == {3, 4}
    assert all(np.isfinite(v) and v >= 0 for v in errors.values())
    # More factors cannot fit worse.
    assert errors[4] <= errors[3] * 1.05


def test_infer_loadings_raises_before_the_measurement_phase() -> None:
    model = make_model(20, [3])
    with pytest.raises(ValueError, match="run_measurement_phase"):
        model.infer_loadings(torch.rand(10, 20).abs(), model.var_names_g, model.consensus_factors, k=3)


def test_factors_dict_shapes() -> None:
    model = make_model(20, [2, 3, 4], store_replicates_k_values=[3], r_store=4)
    factors = model.factors_dict
    assert set(factors) == {2, 3, 4}
    assert factors[3].shape == (4, 3, 20)  # genuine replicates
    assert factors[2].shape == (1, 2, 20)  # consensus fallback
    assert factors[4].shape == (1, 4, 20)


def test_predict_requires_a_k() -> None:
    torch.manual_seed(0)
    model = make_model(20, [3])
    x_ng, _, _ = make_synthetic_nmf_data(100, 20, 3, seed=0)
    run_measurement_phase(model, _fake_dataloader(x_ng, model.var_names_g, 40, 2), n_batches=2, n_replicates=4, verbose=False)
    with pytest.raises(ValueError, match="predict_k"):
        model.predict(torch.from_numpy(x_ng[:20]), model.var_names_g)
    model.predict_k = 3
    out = model.predict(torch.from_numpy(x_ng[:20]), model.var_names_g)
    alpha_nk = as_tensor(out["alpha_nk"])
    assert alpha_nk.shape == (20, 3)
    torch.testing.assert_close(alpha_nk.sum(dim=-1), torch.ones(20), atol=1e-4, rtol=0)  # normalize=True


def test_export_hot_start_shapes_and_normalization() -> None:
    torch.manual_seed(0)
    model = make_model(20, [3, 4])
    x_ng, _, _ = make_synthetic_nmf_data(100, 20, 4, seed=0)
    run_measurement_phase(model, _fake_dataloader(x_ng, model.var_names_g, 40, 2), n_batches=2, n_replicates=4, verbose=False)

    exported = export_hot_start(model, k_values=[4], r=3)
    assert set(exported) == {4}
    assert exported[4].shape == (3, 4, 20)
    # L1-normalized rows are required for compatibility with the amortized online NMF model.
    torch.testing.assert_close(exported[4].sum(dim=-1), torch.ones(3, 4), atol=1e-4, rtol=0)


def test_export_hot_start_raises_before_measurement() -> None:
    model = make_model(20, [3])
    with pytest.raises(ValueError, match="run_measurement_phase"):
        export_hot_start(model)


def test_export_hot_start_rejects_unknown_k() -> None:
    model = make_model(20, [3])
    with pytest.raises(ValueError, match="not in model.k_values"):
        export_hot_start(model, k_values=[11])


# -----------------------------------------------------------------------------------------------
# Lightning integration
# -----------------------------------------------------------------------------------------------


def _make_module(adata: anndata.AnnData, k_values: list[int], **kwargs) -> CellariumModule:
    model = make_model(adata.shape[1], k_values, **kwargs)
    return CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})


def _make_datamodule(adata: anndata.AnnData, batch_size: int) -> CellariumAnnDataDataModule:
    return CellariumAnnDataDataModule(
        dadc=adata,
        batch_size=batch_size,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=None),
            "var_names_g": AnnDataField(attr="var_names"),
            "obs_names_n": AnnDataField(attr="obs_names"),
        },
    )


def test_trains_for_one_epoch_under_lightning(small_adata: anndata.AnnData) -> None:
    dm = _make_datamodule(small_adata, batch_size=100)
    dm.setup(stage="fit")
    module = _make_module(small_adata, [2, 3, 4])
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)
    assert int(module.model._step) > 0
    assert module.model._step_cache == int(module.model._step)


def test_step_counter_survives_a_checkpoint_round_trip(small_adata: anndata.AnnData, tmp_path) -> None:
    dm = _make_datamodule(small_adata, batch_size=100)
    dm.setup(stage="fit")
    module = _make_module(small_adata, [2, 3, 4])
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1, default_root_dir=str(tmp_path))
    trainer.fit(module, dm)
    steps = int(module.model._step)

    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(str(path))
    reloaded = CellariumModule.load_from_checkpoint(str(path), map_location="cpu")
    assert int(reloaded.model._step) == steps
    # Convergence state is in buffers on purpose, so a resumed run continues the criterion
    # rather than restarting it.
    assert int(reloaded.model._drift_below_tol_count) == int(module.model._drift_below_tol_count)
    assert int(reloaded.model._drift_n_captured) == int(module.model._drift_n_captured)


def test_on_train_start_restores_python_side_caches(small_adata: anndata.AnnData) -> None:
    """
    The step counter and the drift-capture flag are cached in Python to keep `.item()` off the
    hot path, so ``on_train_start`` has to rehydrate them from their buffers or a resumed run
    silently restarts the curriculum and re-captures the drift cells.
    """
    model = make_model(small_adata.shape[1], [2, 3], drift_eval_n_cells=16)
    model._step.fill_(1234)
    model._drift_n_captured.fill_(16)

    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    model.on_train_start(trainer)

    assert model._step_cache == 1234
    assert model._drift_cells_full is True
    # And with a partially filled buffer the flag must stay False so capture resumes.
    model._drift_n_captured.fill_(8)
    model.on_train_start(trainer)
    assert model._drift_cells_full is False


def test_measure_at_end_runs_from_the_trainer_hook(small_adata: anndata.AnnData) -> None:
    dm = _make_datamodule(small_adata, batch_size=100)
    dm.setup(stage="fit")
    module = _make_module(small_adata, [2, 3], measure_at_end=True, measurement_n_batches=2)
    trainer = pl.Trainer(barebones=True, accelerator="cpu", devices=1, max_epochs=1)
    trainer.fit(module, dm)
    curves = module.model.selection_curves()
    assert np.isfinite(curves["stability"]).all()


# -----------------------------------------------------------------------------------------------
# End-to-end: recover a known number of factors from synthetic data
# -----------------------------------------------------------------------------------------------


def test_recovers_a_known_number_of_factors() -> None:
    """
    Train briefly on a clean synthetic factorization with ``k_true = 4`` disjoint gene blocks and
    check that the measured curves locate it.

    Two assertions, both of which follow from the structure of the data rather than from tuning:

    * reconstruction error falls steeply up to ``k_true`` and then flattens, because there is
      nothing left to explain;
    * stability is higher at ``k <= k_true`` than beyond it, because past ``k_true`` the extra
      factors have no reproducible structure to latch onto.
    """
    torch.manual_seed(0)
    n, g, k_true = 512, 24, 4
    k_values = [2, 3, 4, 5, 6, 7, 8]
    x_ng, _, _ = make_synthetic_nmf_data(n, g, k_true, seed=0, programs_per_cell=2)

    model = make_model(
        g,
        k_values,
        latent_dim=32,
        n_iterations=2,
        n_self_attention_heads=4,
        n_replicates=8,
        r_store=4,
        min_cells_per_split=32,
        fista_iterations_train=20,
        curriculum_warmup_steps=0,
        stability_burn_in_steps=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x_tensor = torch.from_numpy(x_ng)

    n_steps = 400
    batch_size = 128
    generator = torch.Generator().manual_seed(0)
    for step in range(n_steps):
        rows = torch.randperm(n, generator=generator)[:batch_size]
        loss = as_tensor(model(x_ng=x_tensor[rows], var_names_g=model.var_names_g)["loss"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model._step_cache = step + 1

    curves = run_measurement_phase(
        model,
        _fake_dataloader(x_ng, model.var_names_g, batch_size=128, n_batches=8),
        n_batches=8,
        n_replicates=8,
        fista_iterations=60,
        verbose=False,
    )

    k_array = curves["k"]
    error = curves["error"]
    stability = curves["stability"]
    report = "\n".join(
        f"  k={k:2d}  error={e:.5f} +/- {es:.5f}  stability={s:.4f} +/- {ss:.4f}"
        for k, e, es, s, ss in zip(k_array, error, curves["error_sem"], stability, curves["stability_sem"])
    )

    index_true = int(np.where(k_array == k_true)[0][0])

    # The error curve should have an elbow at k_true: substantial gains up to it, little after.
    gain_before = error[index_true - 1] - error[index_true]
    gain_after = error[index_true] - error[index_true + 1]
    assert gain_before > gain_after, f"no elbow in the error curve at k={k_true}\n{report}"

    # Stability should be higher within the true rank than beyond it.
    within = float(np.nanmean(stability[: index_true + 1]))
    beyond = float(np.nanmean(stability[index_true + 1 :]))
    assert within > beyond, f"stability did not drop past k={k_true}\n{report}"


def test_cross_batch_stability_tracks_within_batch_stability_on_clean_data() -> None:
    """
    On noiseless data with disjoint programs there is no sampling artifact to find, so cross-batch
    stability should not fall far below within-batch stability.  A large gap on real data is the
    signal that stability is being inflated by batch-specific structure.
    """
    torch.manual_seed(0)
    n, g, k_true = 400, 20, 4
    x_ng, _, _ = make_synthetic_nmf_data(n, g, k_true, seed=1)

    model = make_model(
        g,
        [k_true],
        latent_dim=32,
        n_replicates=8,
        r_store=4,
        min_cells_per_split=32,
        fista_iterations_train=20,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x_tensor = torch.from_numpy(x_ng)
    generator = torch.Generator().manual_seed(0)
    for _ in range(250):
        rows = torch.randperm(n, generator=generator)[:128]
        loss = as_tensor(model(x_ng=x_tensor[rows], var_names_g=model.var_names_g)["loss"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    curves = run_measurement_phase(
        model,
        _fake_dataloader(x_ng, model.var_names_g, batch_size=128, n_batches=6),
        n_batches=6,
        n_replicates=8,
        fista_iterations=60,
        verbose=False,
    )
    within = float(curves["stability"][0])
    cross = float(curves["stability_cross"][0])
    assert cross > within - 0.25, f"within={within:.4f} cross={cross:.4f}"
