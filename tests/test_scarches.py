# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scArches (single-cell Architectural Surgery)."""

import numpy as np
import pytest
import torch

from cellarium.ml.models import ScArches, SingleCellVariationalInference
from cellarium.ml.models.scarches import (
    ScArchesFinalAdditiveBiasLayer,
    ScArchesLinearSurgeryMixin,
    ScArchesLinearWithBatch,
    ScArchesLinearWithBatchAndCovariates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_GENES = 10
N_LATENT = 4
VAR_NAMES = np.array([f"gene_{i}" for i in range(N_GENES)])

LINEAR_BATCH_LAYER = {
    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
    "init_args": {"label_to_bias_hidden_layers": []},
}
LINEAR_BATCH_HIDDEN = {
    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
    "init_args": {"out_features": 16, "label_to_bias_hidden_layers": []},
}
LINEAR_BATCH_AND_COV_HIDDEN = {
    "class_path": "cellarium.ml.models.scvi.LinearWithBatchAndCovariates",
    "init_args": {"out_features": 16, "label_to_bias_hidden_layers": []},
}


def make_scvi(
    n_batch: int,
    *,
    batch_embedded: bool = False,
    n_latent_batch: int | None = None,
    final_additive_bias: bool = False,
    with_covariates: bool = False,
) -> SingleCellVariationalInference:
    hidden = LINEAR_BATCH_AND_COV_HIDDEN if with_covariates else LINEAR_BATCH_HIDDEN
    return SingleCellVariationalInference(
        var_names_g=VAR_NAMES,  # type: ignore[arg-type]
        n_batch=n_batch,
        n_latent=N_LATENT,
        batch_embedded=batch_embedded,
        n_latent_batch=n_latent_batch,
        use_size_factor_key=False,
        n_cats_per_cov=[3] if with_covariates else None,
        encoder={"hidden_layers": [hidden], "final_layer": LINEAR_BATCH_LAYER},
        decoder={
            "hidden_layers": [hidden],
            "final_layer": LINEAR_BATCH_LAYER,
            "final_additive_bias": final_additive_bias,
        },
    )


def rand_counts(n: int) -> torch.Tensor:
    return torch.poisson(torch.ones(n, N_GENES) * 2.0)


# ---------------------------------------------------------------------------
# Surgery structure tests
# ---------------------------------------------------------------------------


def test_surgery_replaces_linear_with_batch():
    scvi = make_scvi(n_batch=3)
    lw_before = [n for n, m in scvi.named_modules() if type(m).__name__ == "LinearWithBatch"]
    ScArches(scvi, n_new_batch=2)
    replaced = [n for n, m in scvi.named_modules() if isinstance(m, ScArchesLinearSurgeryMixin)]
    # every LinearWithBatch module must have been replaced
    assert set(lw_before) == set(replaced), f"Not all modules replaced: {lw_before} vs {replaced}"
    # replaced modules should be ScArchesLinearWithBatch
    for name in replaced:
        assert isinstance(scvi.get_submodule(name), ScArchesLinearWithBatch)


def test_surgery_replaces_linear_with_batch_and_covariates():
    scvi = make_scvi(n_batch=3, with_covariates=True)
    lw_and_cov_before = [n for n, m in scvi.named_modules() if type(m).__name__ == "LinearWithBatchAndCovariates"]
    assert len(lw_and_cov_before) > 0, "Expected at least one LinearWithBatchAndCovariates"
    ScArches(scvi, n_new_batch=2)
    for name in lw_and_cov_before:
        assert isinstance(scvi.get_submodule(name), ScArchesLinearWithBatchAndCovariates)


def test_surgery_replaces_final_additive_bias_layer():
    scvi = make_scvi(n_batch=3, final_additive_bias=True)
    assert scvi.decoder.final_additive_bias_layer is not None
    ScArches(scvi, n_new_batch=2)
    assert isinstance(scvi.decoder.final_additive_bias_layer, ScArchesFinalAdditiveBiasLayer)


def test_surgery_updates_n_batch_and_n_latent_batch():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref)
    ScArches(scvi, n_new_batch=n_new)
    assert scvi.n_batch == n_batch_ref + n_new
    assert scvi.n_latent_batch == n_batch_ref + n_new


# ---------------------------------------------------------------------------
# Trainable parameter tests
# ---------------------------------------------------------------------------


def test_only_new_params_are_trainable():
    scvi = make_scvi(n_batch=3)
    sa = ScArches(scvi, n_new_batch=2)
    trainable = {n for n, p in sa.named_parameters() if p.requires_grad}
    frozen = {n for n, p in sa.named_parameters() if not p.requires_grad}
    # All trainable names should refer to new surgery parameters
    assert all("new_batch_bias_weight" in n for n in trainable), f"Unexpected trainable: {trainable}"
    assert len(frozen) > 0, "Expected frozen parameters"
    assert len(trainable) > 0, "Expected trainable parameters"


def test_new_batch_bias_weight_shape():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref)
    ScArches(scvi, n_new_batch=n_new)
    for name, module in scvi.named_modules():
        if isinstance(module, ScArchesLinearSurgeryMixin):
            w = module.new_batch_bias_weight
            assert w.shape[0] == module.out_features  # type: ignore[attr-defined]
            assert w.shape[1] == n_new, f"Expected n_new={n_new} columns, got {w.shape}"


def test_only_embedding_rows_trainable_when_batch_embedded():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref, batch_embedded=True, n_latent_batch=8)
    sa = ScArches(scvi, n_new_batch=n_new)
    trainable = {n for n, p in sa.named_parameters() if p.requires_grad}
    assert trainable == {
        "scvi.batch_representation_mean_bd",
        "scvi.batch_representation_std_unconstrained_bd",
    }, f"Unexpected trainable params: {trainable}"
    assert scvi.batch_representation_mean_bd is not None
    assert scvi.batch_representation_std_unconstrained_bd is not None
    assert scvi.batch_representation_mean_bd.shape == (n_batch_ref + n_new, 8)
    assert scvi.batch_representation_std_unconstrained_bd.shape == (n_batch_ref + n_new, 8)


# ---------------------------------------------------------------------------
# Output invariant: old-batch cells are unaffected by surgery
# ---------------------------------------------------------------------------


def test_old_batch_output_unchanged_by_surgery():
    """
    Before surgery the new_batch_bias_decoder weights are zero, so old-batch
    cells should get exactly the same loss from the pretrained model and the
    ScArches-wrapped model.

    We reseed before each forward pass because ScArches.__init__ creates new
    nn.Linear instances that advance the RNG, causing z samples to differ
    without an explicit reseed.
    """
    import copy

    n_batch_ref = 3
    scvi_ref = make_scvi(n_batch=n_batch_ref)
    scvi_ref.eval()

    scvi_copy = copy.deepcopy(scvi_ref)
    sa = ScArches(scvi_copy, n_new_batch=2)
    sa.eval()

    x = rand_counts(8)
    batch_idx = torch.zeros(8, dtype=torch.long)  # batch 0 (old)

    torch.manual_seed(0)
    with torch.no_grad():
        out_ref = scvi_ref(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)

    torch.manual_seed(0)
    with torch.no_grad():
        out_sa = sa(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)

    torch.testing.assert_close(out_ref["loss"], out_sa["loss"])
    torch.testing.assert_close(out_ref["z_nk"], out_sa["z_nk"])


# ---------------------------------------------------------------------------
# New-batch forward passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_embedded,n_latent_batch,final_additive_bias",
    [
        (False, None, False),
        (False, None, True),
        (True, 8, False),
    ],
    ids=["one_hot", "one_hot_final_bias", "embedded"],
)
def test_new_batch_forward(batch_embedded, n_latent_batch, final_additive_bias):
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(
        n_batch=n_batch_ref,
        batch_embedded=batch_embedded,
        n_latent_batch=n_latent_batch,
        final_additive_bias=final_additive_bias,
    )
    sa = ScArches(scvi, n_new_batch=n_new)
    x = rand_counts(6)
    # Use both new-batch indices
    for new_idx in range(n_batch_ref, n_batch_ref + n_new):
        batch_idx = torch.full((6,), new_idx, dtype=torch.long)
        out = sa(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)
        assert "loss" in out
        assert isinstance(out["loss"], torch.Tensor)
        assert torch.isfinite(out["loss"]).all()


def test_mixed_old_and_new_batch_forward():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref)
    sa = ScArches(scvi, n_new_batch=n_new)
    x = rand_counts(10)
    # Mix old (0,1) and new (3,4) batch indices
    batch_idx = torch.tensor([0, 1, 0, 3, 4, 3, 1, 4, 0, 3], dtype=torch.long)
    out = sa(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)
    assert isinstance(out["loss"], torch.Tensor)
    assert torch.isfinite(out["loss"]).all()


# ---------------------------------------------------------------------------
# Gradient isolation for batch_embedded=True
# ---------------------------------------------------------------------------


def test_old_embedding_rows_frozen_on_backward():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref, batch_embedded=True, n_latent_batch=8)
    sa = ScArches(scvi, n_new_batch=n_new)

    x = rand_counts(6)
    batch_idx = torch.full((6,), n_batch_ref, dtype=torch.long)  # first new batch
    out = sa(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)
    assert isinstance(out["loss"], torch.Tensor)
    out["loss"].mean().backward()

    assert scvi.batch_representation_mean_bd is not None
    mean_grad = scvi.batch_representation_mean_bd.grad
    assert mean_grad is not None
    # Old rows must have zero gradient
    assert mean_grad[:n_batch_ref].abs().max().item() < 1e-9, "Old embedding rows should have zero gradient"
    # New rows must have nonzero gradient
    assert mean_grad[n_batch_ref:].abs().max().item() > 0, "New embedding rows should have nonzero gradient"


# ---------------------------------------------------------------------------
# reset_parameters
# ---------------------------------------------------------------------------


def test_reset_parameters_zeroes_new_weights():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref)
    sa = ScArches(scvi, n_new_batch=n_new)

    # Dirty the new weights
    for _, m in scvi.named_modules():
        if isinstance(m, ScArchesLinearSurgeryMixin):
            with torch.no_grad():
                m.new_batch_bias_weight.fill_(99.0)

    sa.reset_parameters()

    for _, m in scvi.named_modules():
        if isinstance(m, ScArchesLinearSurgeryMixin):
            torch.testing.assert_close(
                m.new_batch_bias_weight,
                torch.zeros_like(m.new_batch_bias_weight),
            )


# ---------------------------------------------------------------------------
# predict / var_names_g
# ---------------------------------------------------------------------------


def test_predict_returns_embeddings():
    scvi = make_scvi(n_batch=3)
    sa = ScArches(scvi, n_new_batch=2)
    sa.eval()
    x = rand_counts(6)
    batch_idx = torch.full((6,), 3, dtype=torch.long)
    with torch.no_grad():
        out = sa.predict(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)
    assert "x_ng" in out  # key used for embeddings (z_nk) in scVI predict
    assert out["x_ng"].shape[0] == 6


def test_var_names_g():
    scvi = make_scvi(n_batch=3)
    sa = ScArches(scvi, n_new_batch=2)
    np.testing.assert_array_equal(sa.var_names_g, VAR_NAMES)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_on_zero_new_batch():
    scvi = make_scvi(n_batch=3)
    with pytest.raises(ValueError, match="n_new_batch"):
        ScArches(scvi, n_new_batch=0)
