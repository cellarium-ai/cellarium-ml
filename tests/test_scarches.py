# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scArches (single-cell Architectural Surgery)."""

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cellarium.ml import CellariumModule
from cellarium.ml.models import SCANVI, ScArches, SingleCellVariationalInference
from cellarium.ml.models.scarches import (
    ScArchesFinalAdditiveBiasLayer,
    ScArchesLinearSurgeryMixin,
    ScArchesLinearWithBatch,
    ScArchesLinearWithBatchAndCovariates,
)
from cellarium.ml.models.scvi import LinearWithStructuredBias, compute_annealed_kl_weight
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDatasetSCVI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_GENES = 10
N_LATENT = 4
N_CATS = 3
VAR_NAMES = np.array([f"gene_{i}" for i in range(N_GENES)])

# names (last component) of the parameters that scArches adds, i.e. the only ones that may train
NEW_PARAM_NAMES = {
    "new_batch_bias_weight",
    "new_bias_weight",
    "batch_representation_mean_new_bd",
    "batch_representation_std_unconstrained_new_bd",
}


def is_new(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in NEW_PARAM_NAMES


def make_encoder_decoder_layers(with_covariates: bool, bias_hidden: list[int]) -> tuple[dict, dict]:
    hidden = {
        "class_path": "cellarium.ml.models.scvi."
        + ("LinearWithBatchAndCovariates" if with_covariates else "LinearWithBatch"),
        "init_args": {"out_features": 16, "label_to_bias_hidden_layers": bias_hidden},
    }
    final = {
        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
        "init_args": {"label_to_bias_hidden_layers": bias_hidden},
    }
    return hidden, final


def make_model_kwargs(
    n_batch: int,
    *,
    batch_embedded: bool = False,
    n_latent_batch: int | None = None,
    final_additive_bias: bool = False,
    with_covariates: bool = False,
    bias_hidden: list[int] | None = None,
) -> dict:
    hidden, final = make_encoder_decoder_layers(with_covariates, bias_hidden or [])
    return dict(
        var_names_g=VAR_NAMES,
        n_batch=n_batch,
        n_latent=N_LATENT,
        batch_embedded=batch_embedded,
        n_latent_batch=n_latent_batch,
        use_size_factor_key=False,
        n_cats_per_cov=[N_CATS] if with_covariates else None,
        encoder={"hidden_layers": [hidden], "final_layer": final},
        decoder={"hidden_layers": [hidden], "final_layer": final, "final_additive_bias": final_additive_bias},
    )


def make_scvi(n_batch: int, **kwargs) -> SingleCellVariationalInference:
    return SingleCellVariationalInference(**make_model_kwargs(n_batch, **kwargs))  # type: ignore[arg-type]


def rand_counts(n: int) -> torch.Tensor:
    return torch.poisson(torch.ones(n, N_GENES) * 2.0)


def tensor_output(out: dict, key: str) -> torch.Tensor:
    value = out[key]
    assert isinstance(value, torch.Tensor)
    return value


def make_inputs(batch_index_n: torch.Tensor, with_covariates: bool = False) -> dict:
    n = len(batch_index_n)
    inputs = dict(x_ng=rand_counts(n), var_names_g=VAR_NAMES, batch_index_n=batch_index_n)
    if with_covariates:
        inputs["categorical_covariate_index_nd"] = torch.randint(0, N_CATS, (n, 1))
    return inputs


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


def test_new_batch_bias_weight_shape():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref)
    ScArches(scvi, n_new_batch=n_new)
    for name, module in scvi.named_modules():
        if isinstance(module, ScArchesLinearSurgeryMixin):
            w = module.new_batch_bias_weight
            assert w.shape[0] == module.out_features  # type: ignore[attr-defined]
            assert w.shape[1] == n_new, f"Expected n_new={n_new} columns, got {w.shape}"


# ---------------------------------------------------------------------------
# Only the new parameters train; everything else (parameters and buffers) stays frozen
# ---------------------------------------------------------------------------

CONFIGS = {
    "one_hot": {},
    "one_hot_covariates_final_bias": dict(with_covariates=True, final_additive_bias=True),
    "one_hot_hidden_bias_decoder": dict(bias_hidden=[8]),
    "embedded": dict(batch_embedded=True, n_latent_batch=8),
}


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
@pytest.mark.parametrize(
    "optim_cls,optim_kwargs",
    [(torch.optim.Adam, {}), (torch.optim.AdamW, {"weight_decay": 0.1})],
    ids=["Adam", "AdamW_weight_decay"],
)
def test_training_only_changes_new_params(config, optim_cls, optim_kwargs):
    n_batch_ref, n_new = 3, 2
    with_covariates = config.get("with_covariates", False)
    sa = ScArches(make_scvi(n_batch=n_batch_ref, **config), n_new_batch=n_new, shift_batch_index=False)
    snapshot = {k: v.clone() for k, v in sa.state_dict().items()}

    # only the new parameters require grad
    trainable = {n for n, p in sa.named_parameters() if p.requires_grad}
    assert len(trainable) > 0
    assert all(is_new(n) for n in trainable), f"Unexpected trainable parameters: {trainable}"

    old_cells = make_inputs(torch.tensor([0, 1, 2, 0, 1, 2]), with_covariates)

    def predict_old_cells() -> torch.Tensor:
        sa.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            return sa.predict(**old_cells)["x_ng"]

    z_before = predict_old_cells()

    # train on a mix of new-batch and old-batch cells
    sa.train()
    optimizer = optim_cls(sa.parameters(), lr=1e-2, **optim_kwargs)
    for _ in range(3):
        optimizer.zero_grad()
        out = sa(**make_inputs(torch.tensor([3, 4, 3, 4, 3, 4, 0, 1]), with_covariates))
        tensor_output(out, "loss").mean().backward()
        optimizer.step()

    after = sa.state_dict()
    # all parameters *and buffers* (e.g. BatchNorm running statistics) other than the new parameters are bitwise equal
    for key, value in snapshot.items():
        if not is_new(key):
            assert torch.equal(value, after[key]), f"{key} changed during training"
    assert any(not torch.equal(v, after[k]) for k, v in snapshot.items() if is_new(k)), "No new parameter trained"

    # cells from the original batches get exactly the same output as before training
    torch.testing.assert_close(predict_old_cells(), z_before, rtol=0, atol=0)


def test_batchnorm_stays_in_eval_mode():
    sa = ScArches(make_scvi(n_batch=3), n_new_batch=2)
    batch_norms = [m for m in sa.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(batch_norms) > 0
    for set_mode in (sa.train, sa.eval, sa.train):  # type: ignore[attr-defined]
        set_mode()
        assert not any(bn.training for bn in batch_norms)
    assert sa.scvi.training  # other layers (e.g. dropout) do follow the mode


# ---------------------------------------------------------------------------
# Output invariant: old-batch cells are unaffected by surgery
# ---------------------------------------------------------------------------


def test_old_batch_output_unchanged_by_surgery():
    """
    Before surgery the new-batch weights are zero, so old-batch cells should get exactly the same
    loss from the pretrained model and the ScArches-wrapped model.

    We reseed before each forward pass because ScArches.__init__ creates new
    parameters that advance the RNG, causing z samples to differ
    without an explicit reseed.
    """
    n_batch_ref = 3
    scvi_ref = make_scvi(n_batch=n_batch_ref)
    scvi_ref.eval()

    scvi_copy = copy.deepcopy(scvi_ref)
    sa = ScArches(scvi_copy, n_new_batch=2, shift_batch_index=False)
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
# New-batch bias: no constant offset, covariates stay frozen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_covariates", [False, True], ids=["batch_only", "batch_and_covariates"])
def test_new_batch_cells_get_no_constant_offset(with_covariates):
    n_batch_ref, n_new, n = 3, 2, 5
    scvi = make_scvi(n_batch=n_batch_ref, bias_hidden=[8], with_covariates=with_covariates)
    # as in a trained model, the frozen bias decoders are not zero for an all-zero input
    for module in scvi.modules():
        if isinstance(module, LinearWithStructuredBias):
            for p in module.bias_decoder.parameters():
                torch.nn.init.normal_(p)
    sa = ScArches(scvi, n_new_batch=n_new, shift_batch_index=False)
    layer = next(
        m for m in sa.modules() if isinstance(m, (ScArchesLinearWithBatch, ScArchesLinearWithBatchAndCovariates))
    )

    cov = F.one_hot(torch.randint(0, N_CATS, (n,)), N_CATS).float() if with_covariates else None

    def frozen_decoder(batch_nb: torch.Tensor, cov_np: torch.Tensor | None) -> torch.Tensor:
        return layer.bias_decoder(batch_nb if cov_np is None else torch.cat([batch_nb, cov_np], dim=-1))

    zero_cov = None if cov is None else torch.zeros(1, N_CATS)
    baseline = frozen_decoder(torch.zeros(1, n_batch_ref), zero_cov)
    assert baseline.abs().max() > 1e-3, "the test needs a non-zero offset to be meaningful"

    new_nb = F.one_hot(torch.tensor([3, 4, 3, 4, 3]), n_batch_ref + n_new).float()
    old_nb = F.one_hot(torch.tensor([0, 1, 2, 0, 1]), n_batch_ref + n_new).float()

    # old-batch cells: bias is exactly that of the frozen decoder
    torch.testing.assert_close(
        layer.compute_bias(batch_nb=old_nb, categorical_covariate_np=cov),
        frozen_decoder(old_nb[:, :n_batch_ref], cov),
    )

    if cov is None:
        # new-batch cells start with no bias at all
        torch.testing.assert_close(layer.compute_bias(batch_nb=new_nb), torch.zeros(n, layer.out_features))

    # new-batch cells get their own bias plus (only if there are covariates) the frozen covariate effect
    with torch.no_grad():
        layer.new_batch_bias_weight.normal_()
    expected = (
        frozen_decoder(torch.zeros(n, n_batch_ref), cov)
        - baseline
        + F.linear(new_nb[:, n_batch_ref:], layer.new_batch_bias_weight)
    )
    torch.testing.assert_close(layer.compute_bias(batch_nb=new_nb, categorical_covariate_np=cov), expected)


def test_final_additive_bias_layer_surgery():
    n_batch_ref, n_new, n = 3, 2, 5
    scvi = make_scvi(n_batch=n_batch_ref, final_additive_bias=True, with_covariates=True)
    assert scvi.decoder.final_additive_bias_layer is not None
    original = copy.deepcopy(scvi.decoder.final_additive_bias_layer)
    sa = ScArches(scvi, n_new_batch=n_new, shift_batch_index=False)
    layer: torch.nn.Module | None = scvi.decoder.final_additive_bias_layer  # widened so that isinstance narrows
    assert isinstance(layer, ScArchesFinalAdditiveBiasLayer)
    # only the new-batch columns are new parameters: nothing for the covariates
    assert layer.new_bias_weight.shape == (N_GENES, n_new)
    assert {n for n, p in sa.named_parameters() if p.requires_grad and "final_additive" in n} == {
        "scvi.decoder.final_additive_bias_layer.new_bias_weight"
    }

    cov = F.one_hot(torch.randint(0, N_CATS, (n,)), N_CATS).float()
    old_idx, new_idx = torch.tensor([0, 1, 2, 0, 1]), torch.tensor([3, 4, 3, 4, 3])
    n_total = n_batch_ref + n_new

    # old-batch cells: identical to the original layer
    torch.testing.assert_close(
        layer(torch.cat([F.one_hot(old_idx, n_total).float(), cov], dim=-1)),
        original(torch.cat([F.one_hot(old_idx, n_batch_ref).float(), cov], dim=-1)),
        rtol=0,
        atol=0,
    )

    # new-batch cells: ReLU of the sum of the frozen pre-activation (covariates only) and the new-batch term
    with torch.no_grad():
        layer.new_bias_weight.normal_()
    new_nb = F.one_hot(new_idx, n_total).float()
    expected = F.relu(
        original[0](torch.cat([torch.zeros(n, n_batch_ref), cov], dim=-1))
        + F.linear(new_nb[:, n_batch_ref:], layer.new_bias_weight)
    )
    torch.testing.assert_close(layer(torch.cat([new_nb, cov], dim=-1)), expected)


def test_final_additive_bias_trains_for_new_batch_cells_without_covariates():
    # With a zero-initialised weight the ReLU would receive exactly zero pre-activation (and so zero gradient).
    sa = ScArches(make_scvi(n_batch=3, final_additive_bias=True), n_new_batch=2)
    out = sa(**make_inputs(torch.tensor([0, 1, 0, 1, 0, 1])))
    tensor_output(out, "loss").mean().backward()
    layer: torch.nn.Module | None = sa.scvi.decoder.final_additive_bias_layer  # widened so that isinstance narrows
    assert isinstance(layer, ScArchesFinalAdditiveBiasLayer)
    grad = layer.new_bias_weight.grad
    assert grad is not None and grad.abs().max() > 0


# ---------------------------------------------------------------------------
# New-batch forward passes and batch index handling
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
    n_new = 2
    scvi = make_scvi(
        n_batch=3,
        batch_embedded=batch_embedded,
        n_latent_batch=n_latent_batch,
        final_additive_bias=final_additive_bias,
    )
    sa = ScArches(scvi, n_new_batch=n_new)
    x = rand_counts(6)
    # query-local indices
    for new_idx in range(n_new):
        batch_idx = torch.full((6,), new_idx, dtype=torch.long)
        out = sa(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)
        assert "loss" in out
        assert isinstance(out["loss"], torch.Tensor)
        assert torch.isfinite(out["loss"]).all()


def test_mixed_old_and_new_batch_forward():
    n_batch_ref, n_new = 3, 2
    scvi = make_scvi(n_batch=n_batch_ref)
    sa = ScArches(scvi, n_new_batch=n_new, shift_batch_index=False)
    x = rand_counts(10)
    # Mix old (0,1) and new (3,4) batch indices
    batch_idx = torch.tensor([0, 1, 0, 3, 4, 3, 1, 4, 0, 3], dtype=torch.long)
    out = sa(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_idx)
    assert isinstance(out["loss"], torch.Tensor)
    assert torch.isfinite(out["loss"]).all()


def test_query_local_batch_index_is_shifted():
    n_batch_ref, n_new = 3, 2
    base = make_scvi(n_batch=n_batch_ref)
    sa_local = ScArches(copy.deepcopy(base), n_new_batch=n_new)
    sa_absolute = ScArches(copy.deepcopy(base), n_new_batch=n_new, shift_batch_index=False)
    # give both the same non-trivial new parameters
    absolute_params = dict(sa_absolute.named_parameters())
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in sa_local.named_parameters():
            if is_new(name):
                param.normal_(generator=generator)
                absolute_params[name].copy_(param)
    sa_local.eval()
    sa_absolute.eval()

    local_idx = torch.tensor([0, 1, 0, 1, 1, 0])
    x = rand_counts(6)

    def predict(sa: ScArches, batch_index_n: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(0)
        with torch.no_grad():
            return sa.predict(x_ng=x, var_names_g=VAR_NAMES, batch_index_n=batch_index_n)["x_ng"]

    z_local = predict(sa_local, local_idx)
    torch.testing.assert_close(z_local, predict(sa_absolute, local_idx + n_batch_ref))
    assert not torch.allclose(z_local, predict(sa_absolute, local_idx))  # those would be the old batches 0 and 1


@pytest.mark.parametrize("bad_index", [-1, 2], ids=["negative", "too_large"])
def test_query_local_batch_index_out_of_range_raises(bad_index):
    sa = ScArches(make_scvi(n_batch=3), n_new_batch=2)
    with pytest.raises(ValueError, match="query-local"):
        sa(x_ng=rand_counts(3), var_names_g=VAR_NAMES, batch_index_n=torch.tensor([0, 1, bad_index]))


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
    batch_idx = torch.ones(6, dtype=torch.long)
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


# ---------------------------------------------------------------------------
# scVI is unchanged by the batch embedding accessor used by scArches
# ---------------------------------------------------------------------------


def test_embedding_accessor_returns_original_table_without_surgery():
    scvi = make_scvi(n_batch=3, batch_embedded=True, n_latent_batch=8)
    assert getattr(scvi, "batch_representation_mean_new_bd", None) is None
    assert scvi._full_batch_representation_table("batch_representation_mean_bd") is scvi.batch_representation_mean_bd


# ---------------------------------------------------------------------------
# scANVI through ScArches (extra arguments such as labels reach the wrapped model)
# ---------------------------------------------------------------------------


def test_scanvi_through_scarches():
    n_new = 2
    scanvi = SCANVI(
        classifier_type="flat",
        cell_type_categories=["T", "B", "NK"],
        classifier_n_hidden=[16],
        secondary_n_hidden=[16],
        chunk_size=3,
        **make_model_kwargs(n_batch=3),
    )
    sa = ScArches(scanvi, n_new_batch=n_new)
    inputs = make_inputs(torch.tensor([0, 1, 0, 1, 0, 1]))
    labels = np.array(["T", "unknown", "B", "unknown", "NK", "unknown"], dtype=object)

    out_unlabeled = sa(**inputs)
    out_labeled = sa(**inputs, cell_type_labels_n=labels)
    assert tensor_output(out_unlabeled, "classification_loss").sum() == 0
    assert tensor_output(out_labeled, "classification_loss").sum() > 0

    # only the new-batch parameters receive gradients
    tensor_output(out_labeled, "loss").mean().backward()
    with_grad = {name for name, p in sa.named_parameters() if p.grad is not None}
    assert len(with_grad) > 0
    assert all(is_new(name) for name in with_grad), with_grad

    sa.eval()
    with torch.no_grad():
        prediction = sa.predict(**inputs)
    assert "y_probs_nc" in prediction


# ---------------------------------------------------------------------------
# The KL annealing schedule continues from the pretrained model's checkpoint
# ---------------------------------------------------------------------------


def test_kl_schedule_continues_from_pretrained_checkpoint(tmp_path: Path):
    n, batch_size, warmup_steps = 40, 8, 8
    rng = np.random.default_rng(0)
    dataset = BoringDatasetSCVI(
        data=rng.poisson(lam=2.0, size=(n, N_GENES)), batch_index_n=rng.integers(0, 3, size=n), var_names=VAR_NAMES
    )
    loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)
    scvi = SingleCellVariationalInference(
        **{**make_model_kwargs(n_batch=3), "kl_warmup_epochs": None, "kl_warmup_steps": warmup_steps}
    )
    module = CellariumModule(model=scvi, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=5, enable_checkpointing=False, logger=False)
    trainer.fit(module, train_dataloaders=loader)
    ckpt_path = tmp_path / "scvi.ckpt"
    trainer.save_checkpoint(ckpt_path)
    base_step = trainer.global_step

    # a model loaded from the checkpoint knows how far its training got (this is not part of the state dict)
    pretrained = CellariumModule.load_from_checkpoint(ckpt_path).model
    assert isinstance(pretrained, SingleCellVariationalInference)
    assert (pretrained.step, pretrained.epoch) == (base_step, trainer.current_epoch)

    # ScArches continues from there: the training position of its own trainer is added
    sa = ScArches(pretrained, n_new_batch=1)
    assert pretrained.step == base_step  # the first training step already uses the right KL weight
    for run_step in (0, 2, 4):
        sa.on_train_batch_end(cast(pl.Trainer, SimpleNamespace(global_step=run_step, current_epoch=0, logger=None)))
        assert (pretrained.step, pretrained.epoch) == (base_step + run_step, trainer.current_epoch)
    # past the warmup of the pretrained model (a restarted schedule would be at 4 / 8 here)
    weight = compute_annealed_kl_weight(
        epoch=pretrained.epoch,
        step=pretrained.step,
        n_epochs_kl_warmup=pretrained.kl_warmup_epochs,
        n_steps_kl_warmup=pretrained.kl_warmup_steps,
    )
    assert weight == 1.0


# ---------------------------------------------------------------------------
# Training with Lightning, checkpointing and training again from the checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [CONFIGS["one_hot"], CONFIGS["embedded"]], ids=["one_hot", "embedded"])
def test_frozen_state_survives_lightning_training_and_checkpoint_reload(config, tmp_path: Path):
    n, batch_size, n_new = 40, 8, 2  # batch_size must be > 1 for BatchNorm
    rng = np.random.default_rng(0)

    def make_loader() -> torch.utils.data.DataLoader:
        dataset = BoringDatasetSCVI(
            data=rng.poisson(lam=2.0, size=(n, N_GENES)),
            batch_index_n=rng.integers(0, n_new, size=n),  # query-local batch indices
            var_names=VAR_NAMES,
        )
        return torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)

    sa = ScArches(make_scvi(n_batch=3, **config), n_new_batch=n_new)
    frozen_before = {k: v.clone() for k, v in sa.state_dict().items() if not is_new(k)}
    module = CellariumModule(model=sa, optim_fn=torch.optim.AdamW, optim_kwargs={"lr": 1e-2, "weight_decay": 0.1})

    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=3, enable_checkpointing=False, logger=False)
    trainer.fit(module, train_dataloaders=make_loader(), val_dataloaders=make_loader())
    ckpt_path = tmp_path / "scarches.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded = CellariumModule.load_from_checkpoint(ckpt_path)
    assert isinstance(loaded.model, ScArches)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=3, enable_checkpointing=False, logger=False)
    trainer.fit(loaded, train_dataloaders=make_loader())

    frozen_after = {k: v for k, v in loaded.model.state_dict().items() if not is_new(k)}
    assert frozen_before.keys() == frozen_after.keys()
    for key, value in frozen_before.items():
        assert torch.equal(value, frozen_after[key].cpu()), f"{key} changed"
