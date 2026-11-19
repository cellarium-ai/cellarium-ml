# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
from typing import Final

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import SCANVI
from cellarium.ml.models.scanvi import compute_frontier
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDatasetSCVI

predict_logit_key: Final = "y_logits_nc"
predict_prob_key: Final = "y_probs_nc"
predict_cat_key: Final = "category_labels_c"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BoringDatasetSCANVI(BoringDatasetSCVI):
    """BoringDatasetSCVI extended with string cell-type labels."""

    def __init__(
        self,
        data: np.ndarray,
        batch_index_n: np.ndarray,
        cell_type_labels_n: np.ndarray,
        var_names: np.ndarray | None = None,
    ) -> None:
        super().__init__(data=data, batch_index_n=batch_index_n, var_names=var_names)
        self.cell_type_labels_n = cell_type_labels_n

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return super().__getitem__(idx) | {"cell_type_labels_n": self.cell_type_labels_n[idx, None]}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_ENCODER_CFG = {
    "hidden_layers": [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "label_to_bias_hidden_layers": []},
        },
    ],
    "final_layer": {
        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
        "init_args": {"label_to_bias_hidden_layers": []},
    },
}

_DECODER_CFG = {
    "hidden_layers": [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "label_to_bias_hidden_layers": []},
        },
    ],
    "final_layer": {
        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
        "init_args": {"label_to_bias_hidden_layers": []},
    },
    "final_additive_bias": False,
}


def _make_flat(var_names_g, *, n_batch=2, n_latent=8, cell_type_categories=("T", "B", "NK", "Mono", "DC"), **kwargs):
    kwargs.setdefault("secondary_n_hidden", [32])
    return SCANVI(
        classifier_type="flat",
        cell_type_categories=list(cell_type_categories),
        classifier_n_hidden=[32],
        chunk_size=3,
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=n_latent,
        encoder=_ENCODER_CFG,
        decoder=_DECODER_CFG,
        **kwargs,
    )


def _toy_ontology():
    """Return (cl_names, descendant_tensor) for R -> {T -> {CD4, CD8}, B, Mono}."""
    cl_names = ["R", "T", "B", "Mono", "CD4", "CD8", "unknown"]
    idx = {n: i for i, n in enumerate(cl_names)}
    c = len(cl_names)
    desc = torch.eye(c)
    # R is an ancestor of everything (except unknown)
    for child in ["T", "B", "Mono", "CD4", "CD8"]:
        desc[idx["R"], idx[child]] = 1.0
    # T is an ancestor of CD4, CD8
    desc[idx["T"], idx["CD4"]] = 1.0
    desc[idx["T"], idx["CD8"]] = 1.0
    return cl_names, desc


def _make_ontology(var_names_g, *, n_batch=2, n_latent=8, class_counts=None, **kwargs):
    cl_names, desc = _toy_ontology()
    if class_counts is None:
        class_counts = pd.Series({"CD4": 100.0, "CD8": 80.0, "B": 60.0, "Mono": 70.0, "T": 15.0})
    return SCANVI(
        classifier_type="ontology",
        descendant_tensor=desc,
        cl_names=cl_names,
        class_counts=class_counts,
        frontier_min_cells=50,
        classifier_n_hidden=[32],
        secondary_n_hidden=[32],
        chunk_size=2,
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=n_latent,
        cell_type_categories=cl_names,
        encoder=_ENCODER_CFG,
        decoder=_DECODER_CFG,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Frontier construction
# ---------------------------------------------------------------------------


def test_frontier_basic_cut():
    """A > 50 with children C > 50 and B = 5 -> frontier {B, C} (B added by coverage repair)."""
    cl = ["A", "B", "C", "unknown"]
    desc = torch.eye(4)
    desc[0, 1] = 1.0
    desc[0, 2] = 1.0
    counts = torch.tensor([0.0, 5.0, 100.0, 0.0])
    frontier, under = compute_frontier(desc, cl, counts, min_cells=50, excluded_names={"unknown"})
    assert set(frontier) == {"B", "C"}
    assert under == ["B"]


def test_frontier_deep_orphan_rolls_up_to_shallowest():
    """A deep orphan (A -> B -> B1, B1=5; A -> C, C=100) rolls up to B, not B1."""
    cl = ["A", "B", "B1", "C", "unknown"]
    desc = torch.eye(5)
    desc[0, 1] = desc[0, 2] = desc[0, 3] = 1.0  # A ancestor of B, B1, C
    desc[1, 2] = 1.0  # B ancestor of B1
    counts = torch.tensor([0.0, 0.0, 5.0, 100.0, 0.0])
    frontier, _ = compute_frontier(desc, cl, counts, min_cells=50, excluded_names={"unknown"})
    assert set(frontier) == {"B", "C"}


def test_frontier_and_active_set_on_model():
    var_names_g = np.array([f"g{i}" for i in range(20)])
    # T has only 15 direct cells but its subtree (CD4+CD8+T) is large, so T is coarse (not frontier).
    model = _make_ontology(var_names_g)
    # frontier = deepest nodes with >= 50 subtree support = {CD4, CD8, B, Mono}
    assert set(model.frontier_cl_names) == {"CD4", "CD8", "B", "Mono"}
    assert model.n_partition == 4
    # active set = frontier + ancestors {R, T}
    assert set(model.active_cl_names) == {"CD4", "CD8", "B", "Mono", "R", "T"}


# ---------------------------------------------------------------------------
# Flat mode
# ---------------------------------------------------------------------------


def test_flat_loss_structure():
    n, g = 16, 50
    n_labeled = 8
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    categories = ["T", "B", "NK", "Mono", "DC"]

    torch.manual_seed(0)
    model = _make_flat(var_names_g, cell_type_categories=categories)
    model.eval()

    x_ng = torch.poisson(torch.ones(n, g) * 2).float()
    batch_index_n = torch.zeros(n, dtype=torch.long)

    rng = np.random.default_rng(0)
    labels = np.array(["unknown"] * n, dtype=object)
    labels[:n_labeled] = rng.choice(categories, size=n_labeled)

    with torch.no_grad():
        out = model(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=batch_index_n, cell_type_labels_n=labels)

    ce = out["classification_loss"]
    kl_c = out["kl_divergence_c"]
    assert isinstance(ce, torch.Tensor)
    assert isinstance(kl_c, torch.Tensor)

    labeled = labels != "unknown"
    unlabeled = ~labeled
    assert (ce[labeled] > 0).all()
    assert ce[unlabeled].sum() == 0.0
    assert torch.allclose(kl_c[labeled], torch.zeros(int(labeled.sum())), atol=1e-5)
    assert (kl_c[unlabeled] > 0).all()
    assert out["loss"].isfinite()


def test_flat_requires_cell_type_categories():
    var_names_g = np.array([f"g{i}" for i in range(10)])
    with pytest.raises(ValueError, match="cell_type_categories"):
        SCANVI(
            classifier_type="flat",
            var_names_g=var_names_g,
            n_batch=1,
            encoder={"hidden_layers": [], "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}}},
            decoder={
                "hidden_layers": [],
                "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
                "final_additive_bias": False,
            },
        )


# ---------------------------------------------------------------------------
# Ontology mode
# ---------------------------------------------------------------------------


def test_ontology_loss_structure():
    """Frontier labels -> kl_c 0; coarse/unlabeled -> kl_c > 0; labeled -> CE > 0."""
    n, g = 18, 50
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    torch.manual_seed(0)
    model = _make_ontology(var_names_g)
    model.eval()

    x_ng = torch.poisson(torch.ones(n, g) * 2).float()
    batch_index_n = torch.zeros(n, dtype=torch.long)
    # CD4/CD8/B/Mono are frontier leaves; T is coarse (marginalize CD4,CD8); unknown is unlabeled
    labels = np.array(
        ["CD4", "CD8", "B", "Mono", "T", "unknown"] * 3,
        dtype=object,
    )

    with torch.no_grad():
        out = model(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=batch_index_n, cell_type_labels_n=labels)

    ce = out["classification_loss"]
    kl_c = out["kl_divergence_c"]
    assert isinstance(ce, torch.Tensor)
    assert isinstance(kl_c, torch.Tensor)

    frontier_mask = np.isin(labels, ["CD4", "CD8", "B", "Mono"])
    coarse_mask = labels == "T"
    unlabeled_mask = labels == "unknown"

    # frontier (known) labels: kl_c == 0 (restricted kl_c over a singleton set)
    assert torch.allclose(kl_c[frontier_mask], torch.zeros(int(frontier_mask.sum())), atol=1e-5)
    # coarse and unlabeled: kl_c > 0 (marginalize over a set of size > 1)
    assert (kl_c[coarse_mask] > 0).all()
    assert (kl_c[unlabeled_mask] > 0).all()
    # CE positive for any label, zero for unlabeled
    assert (ce[~unlabeled_mask] > 0).all()
    assert ce[unlabeled_mask].sum() == 0.0
    assert out["loss"].isfinite()


def test_ontology_finer_than_frontier_bins_up():
    """A label finer than the frontier is binned to its frontier ancestor (no error)."""
    n, g = 6, 30
    var_names_g = np.array([f"g{i}" for i in range(g)])
    # Add a finer node CD4mem under CD4 with too few cells to be its own frontier node.
    cl_names = ["R", "T", "B", "Mono", "CD4", "CD8", "CD4mem", "unknown"]
    idx = {nm: i for i, nm in enumerate(cl_names)}
    desc = torch.eye(len(cl_names))
    for child in ["T", "B", "Mono", "CD4", "CD8", "CD4mem"]:
        desc[idx["R"], idx[child]] = 1.0
    desc[idx["T"], idx["CD4"]] = desc[idx["T"], idx["CD8"]] = desc[idx["T"], idx["CD4mem"]] = 1.0
    desc[idx["CD4"], idx["CD4mem"]] = 1.0
    counts = pd.Series({"CD4": 100.0, "CD8": 80.0, "B": 60.0, "Mono": 70.0, "CD4mem": 3.0})

    torch.manual_seed(0)
    model = SCANVI(
        classifier_type="ontology",
        descendant_tensor=desc,
        cl_names=cl_names,
        class_counts=counts,
        frontier_min_cells=50,
        classifier_n_hidden=[16],
        secondary_n_hidden=[16],
        var_names_g=var_names_g,
        n_batch=1,
        n_latent=6,
        cell_type_categories=cl_names,
        encoder=_ENCODER_CFG,
        decoder=_DECODER_CFG,
    )
    # CD4mem should resolve to the CD4 frontier leaf
    assert model._label_to_active_idx["CD4mem"] == model._label_to_active_idx["CD4"]

    model.eval()
    x_ng = torch.poisson(torch.ones(n, g) * 2).float()
    labels = np.array(["CD4mem", "CD4", "CD8", "B", "Mono", "unknown"], dtype=object)
    with torch.no_grad():
        out = model(
            x_ng=x_ng,
            var_names_g=var_names_g,
            batch_index_n=torch.zeros(n, dtype=torch.long),
            cell_type_labels_n=labels,
        )
    # CD4mem behaves like CD4 (frontier leaf): kl_c == 0
    kl_c = out["kl_divergence_c"]
    assert isinstance(kl_c, torch.Tensor)
    assert torch.allclose(kl_c[0], torch.tensor(0.0), atol=1e-5)


def test_ontology_class_weights_shape_and_default():
    var_names_g = np.array([f"g{i}" for i in range(20)])
    model = _make_ontology(var_names_g, propagate_class_counts=True)
    assert model.class_weights is not None
    assert model.class_weights.shape == (model.n_active,)
    # data-frequency-weighted mean of weights ~ 1 is not asserted here (depends on propagation);
    # just check all weights are positive and finite
    assert bool((model.class_weights > 0).all())
    assert bool(torch.isfinite(model.class_weights).all())


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


def test_flat_predict_shapes():
    n, g, n_latent = 12, 50, 8
    categories = ["T", "B", "NK", "Mono", "DC"]
    var_names_g = np.array([f"g{i}" for i in range(g)])
    model = _make_flat(var_names_g, n_latent=n_latent, cell_type_categories=categories)
    model.eval()
    x_ng = torch.poisson(torch.ones(n, g) * 2).float()
    with torch.no_grad():
        pred = model.predict(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=torch.zeros(n, dtype=torch.long))
    assert pred["x_ng"].shape == (n, n_latent)
    assert pred[predict_prob_key].shape == (n, len(categories))
    assert pred[predict_logit_key].shape == (n, len(categories))
    assert torch.allclose(pred[predict_prob_key].sum(-1), torch.ones(n), atol=1e-5)


def test_ontology_predict_shapes():
    n, g, n_latent = 12, 50, 8
    var_names_g = np.array([f"g{i}" for i in range(g)])
    model = _make_ontology(var_names_g, n_latent=n_latent)
    model.eval()
    x_ng = torch.poisson(torch.ones(n, g) * 2).float()
    with torch.no_grad():
        pred = model.predict(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=torch.zeros(n, dtype=torch.long))
    assert pred["x_ng"].shape == (n, n_latent)
    # propagated probabilities over all active nodes
    assert pred[predict_prob_key].shape == (n, model.n_active)
    assert (pred[predict_prob_key] >= 0).all()


# ---------------------------------------------------------------------------
# Checkpoint round-trip (also exercises the meta-device path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["flat", "ontology"])
def test_checkpoint_roundtrip(mode, tmp_path: Path):
    n, g, n_batch = 40, 50, 2
    batch_size = 8
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    devices = int(os.environ.get("TEST_DEVICES", "1"))

    rng = np.random.default_rng(42)
    x = rng.poisson(lam=2.0, size=(n, g)).astype(np.float32)
    batch_idx = rng.integers(0, n_batch, size=n)

    if mode == "flat":
        categories = ["T", "B", "NK", "Mono", "DC"]
        pool = np.array(categories + ["unknown"], dtype=object)
        labels = rng.choice(pool, size=n)
        torch.manual_seed(0)
        model = _make_flat(
            var_names_g,
            n_batch=n_batch,
            cell_type_categories=categories,
            class_counts=pd.Series({"T": 5, "B": 4, "NK": 3, "Mono": 2, "DC": 1}),
            y_prior_probs="class_counts",
        )
    else:
        pool = np.array(["CD4", "CD8", "B", "Mono", "T", "unknown"], dtype=object)
        labels = rng.choice(pool, size=n)
        torch.manual_seed(0)
        model = _make_ontology(var_names_g, n_batch=n_batch)

    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCANVI(data=x, batch_index_n=batch_idx, cell_type_labels_n=labels, var_names=var_names_g),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    module = CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})
    trainer = pl.Trainer(accelerator="cpu", devices=devices, max_steps=2, enable_checkpointing=False)
    trainer.fit(module, train_dataloaders=train_loader)

    ckpt_path = tmp_path / "scanvi.ckpt"
    trainer.save_checkpoint(ckpt_path)

    x_t = torch.tensor(x)
    b_t = torch.tensor(batch_idx, dtype=torch.long)
    model.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        pre = model.predict(x_ng=x_t, var_names_g=var_names_g, batch_index_n=b_t, n_generative_samples=2)

    loaded = CellariumModule.load_from_checkpoint(ckpt_path).model
    assert isinstance(loaded, SCANVI)
    loaded.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        post = loaded.predict(x_ng=x_t, var_names_g=var_names_g, batch_index_n=b_t, n_generative_samples=2)

    torch.testing.assert_close(pre["x_ng"], post["x_ng"])
    torch.testing.assert_close(pre["y_probs_nc"], post["y_probs_nc"])
    torch.testing.assert_close(pre["generative_probs_nc"], post["generative_probs_nc"])
    torch.testing.assert_close(pre["in_distribution_score_n"], post["in_distribution_score_n"])
    # the label proportions of the training data, and the prior made from them, are kept
    torch.testing.assert_close(loaded.y_prior.cpu(), model.y_prior.cpu())
    reference = model.reference_class_proportions(space="partition")
    loaded_reference = loaded.reference_class_proportions(space="partition")
    assert reference is not None and loaded_reference is not None
    np.testing.assert_allclose(loaded_reference, reference)
    if mode == "flat":
        np.testing.assert_allclose(loaded.y_prior.cpu().numpy(), np.arange(5, 0, -1) / 15, rtol=1e-6)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_guard_use_flow_raises():
    var_names_g = np.array([f"gene_{i}" for i in range(10)])
    with pytest.raises(ValueError, match="use_flow"):
        SCANVI(
            classifier_type="flat",
            cell_type_categories=["A", "B", "C"],
            var_names_g=var_names_g,
            n_batch=1,
            use_flow=True,
            encoder={"hidden_layers": [], "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}}},
            decoder={
                "hidden_layers": [],
                "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
                "final_additive_bias": False,
            },
        )


def test_forward_uses_shared_encoder_input_augmentation():
    """scANVI's forward goes through the shared augmentation helper and respects train/eval mode."""
    n, g = 16, 50
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    torch.manual_seed(0)
    model = SCANVI(
        classifier_type="flat",
        cell_type_categories=["T", "B"],
        classifier_n_hidden=[32],
        secondary_n_hidden=[32],
        chunk_size=2,
        var_names_g=var_names_g,
        n_batch=2,
        n_latent=8,
        encoder=_ENCODER_CFG,
        decoder=_DECODER_CFG,
        input_gene_dropout_rate=0.1,
        binomial_downsampling_prob=1.0,
        binomial_downsampling_min_counts=100,
    )
    x_ng = torch.poisson(torch.full((n, g), 50.0))
    batch_index_n = torch.zeros(n, dtype=torch.long)

    seen_inputs = []
    original_inference = model.inference

    def spy(x_ng, **kwargs):
        seen_inputs.append(x_ng)
        return original_inference(x_ng=x_ng, **kwargs)

    model.inference = spy  # type: ignore[assignment]

    model.train()
    out = model(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=batch_index_n)
    assert isinstance(out["loss"], torch.Tensor)
    assert out["loss"].isfinite()
    out["loss"].backward()
    assert seen_inputs[-1].sum() < x_ng.sum()

    model.eval()
    with torch.no_grad():
        model(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=batch_index_n)
    assert torch.equal(seen_inputs[-1], x_ng)


def test_forward_zero_count_cell_is_finite():
    n, g = 8, 50
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    torch.manual_seed(0)
    model = _make_flat(var_names_g, cell_type_categories=["T", "B"], normalizetotal_variational=True)
    model.eval()
    x_ng = torch.poisson(torch.full((n, g), 20.0))
    x_ng[2] = 0.0
    batch_index_n = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        out = model(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=batch_index_n)
    assert out["loss"].isfinite()


# ---------------------------------------------------------------------------
# Label proportions of the training data, and the prior p(c)
# ---------------------------------------------------------------------------


def test_class_counts_give_the_reference_proportions_and_the_prior():
    var_names_g = np.array([f"gene_{i}" for i in range(10)])
    counts = pd.Series({"T": 50, "B": 30, "NK": 20, "unknown": 999})  # the unlabeled cells are left out
    model = _make_flat(
        var_names_g, cell_type_categories=["T", "B", "NK"], class_counts=counts, y_prior_probs="class_counts"
    )
    for space in ("partition", "classifier"):
        np.testing.assert_allclose(model.reference_class_proportions(space=space), [0.5, 0.3, 0.2])
    np.testing.assert_allclose(model.y_prior.numpy(), [0.5, 0.3, 0.2], rtol=1e-6)
    np.testing.assert_allclose(model.class_proportions(pd.Series({"NK": 2.0}), space="partition"), [0, 0, 1])
    with pytest.raises(ValueError, match="not classes of the model"):
        model.class_proportions(pd.Series({"Mono": 1.0}), space="partition")

    without = _make_flat(var_names_g, cell_type_categories=["T", "B", "NK"])
    assert without.reference_class_proportions(space="partition") is None
    with pytest.raises(ValueError, match="requires class_counts"):
        _make_flat(var_names_g, cell_type_categories=["T", "B", "NK"], y_prior_probs="class_counts")


def test_ontology_proportions_split_a_coarse_label_among_the_frontier_below_it():
    var_names_g = np.array([f"gene_{i}" for i in range(10)])
    model = _make_ontology(var_names_g)  # CD4 100, CD8 80, B 60, Mono 70, and the coarse T 15
    assert sorted(model.frontier_cl_names) == ["B", "CD4", "CD8", "Mono"]
    partition = model.reference_class_proportions(space="partition")
    classifier = model.reference_class_proportions(space="classifier")
    assert partition is not None and classifier is not None
    # T is split between CD4 and CD8 in proportion to their own counts
    expected = {"CD4": 100 + 15 * 100 / 180, "CD8": 80 + 15 * 80 / 180, "B": 60, "Mono": 70}
    np.testing.assert_allclose(partition, [expected[name] / 325 for name in model.frontier_cl_names])
    expected_classifier = {"CD4": 100, "CD8": 80, "B": 60, "Mono": 70, "T": 15, "R": 0}
    np.testing.assert_allclose(classifier, [expected_classifier[name] / 325 for name in model.active_cl_names])


# ---------------------------------------------------------------------------
# The prior of unlabeled cells (unlabeled_prior)
# ---------------------------------------------------------------------------


def _latent_inputs(m: int, n_latent: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(m, n_latent, generator=generator).requires_grad_()
    mean = torch.randn(m, n_latent, generator=generator).requires_grad_()
    std = (torch.rand(m, n_latent, generator=generator) + 0.5).requires_grad_()
    return z, mean, std


def test_the_mixture_prior_is_the_kl_to_the_mixture_with_its_gradient():
    var_names_g = np.array([f"gene_{i}" for i in range(10)])
    model = _make_flat(var_names_g, n_latent=4, y_prior_probs=[0.4, 0.3, 0.1, 0.15, 0.05], unlabeled_prior="mixture")
    model.eval()
    z, mean, std = _latent_inputs(6, 4)
    kl_weight = 0.7
    q_partition = torch.full((6, 5), 0.2)  # not used by the mixture prior

    torch.manual_seed(0)  # the same samples of u in both
    kl_z, kl_u, kl_c = model._unlabeled_kls(z, mean, std, q_partition, kl_weight=kl_weight)
    loss_n = kl_weight * (kl_z + kl_u) + kl_c
    torch.manual_seed(0)
    kl_z_np, kl_u_np = model._set_kls(z, mean, std, torch.arange(5))
    exact_n = -torch.logsumexp(model.y_prior.log() - kl_weight * (kl_z_np + kl_u_np), dim=-1)

    # the loss is -log sum_c p(c) exp(-β K_c), and its gradient (with the posterior over classes held fixed) is that
    # of -log sum_c p(c) exp(-β K_c) differentiated through everything
    torch.testing.assert_close(loss_n, exact_n)
    for grad, exact_grad in zip(
        torch.autograd.grad(loss_n.sum(), [z, mean, std]), torch.autograd.grad(exact_n.sum(), [z, mean, std])
    ):
        torch.testing.assert_close(grad, exact_grad)


def test_the_standard_normal_prior_is_that_of_scvi():
    var_names_g = np.array([f"gene_{i}" for i in range(10)])
    model = _make_flat(var_names_g, n_latent=4, unlabeled_prior="standard_normal")
    z, mean, std = _latent_inputs(6, 4)
    kl_z, kl_u, kl_c = model._unlabeled_kls(z, mean, std, torch.full((6, 5), 0.2), kl_weight=1.0)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean, std), torch.distributions.Normal(0.0, 1.0)
    ).sum(dim=-1)
    torch.testing.assert_close(kl_z, expected)
    assert (kl_u == 0).all() and (kl_c == 0).all()


@pytest.mark.parametrize(
    "unlabeled_prior,classifier_has_gradient", [("scanvi", True), ("mixture", False), ("standard_normal", False)]
)
def test_only_the_scanvi_prior_gives_the_classifier_gradients_from_unlabeled_cells(
    unlabeled_prior, classifier_has_gradient
):
    n, g = 8, 50
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    torch.manual_seed(0)
    model = _make_flat(var_names_g, unlabeled_prior=unlabeled_prior)
    model.train()
    x_ng = torch.poisson(torch.full((n, g), 2.0))
    out = model(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=torch.zeros(n, dtype=torch.long))  # unlabeled
    out["loss"].backward()
    has_gradient = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cell_type_classifier.parameters())
    assert has_gradient == classifier_has_gradient
    # the encoder is trained in every case
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.z_encoder.parameters())


def test_unlabeled_prior_must_be_known():
    with pytest.raises(ValueError, match="unlabeled_prior"):
        _make_flat(np.array(["g0", "g1"]), unlabeled_prior="uniform")


# ---------------------------------------------------------------------------
# The generative class posterior and the in-distribution score
# ---------------------------------------------------------------------------


def _gaussian_mixture_scanvi(mu_kc: torch.Tensor, y_prior: list[float]) -> SCANVI:
    """
    An scANVI model whose class-conditional prior is ``p(z|u,c) = N(mu_c, (1 + var_eps) I)`` whatever ``u``, and whose
    ``q(u|z,c) = N(0, (1 + var_eps) I)``, so the generative class posterior is Bayes' rule for a Gaussian mixture.
    """
    n_latent, n_classes = mu_kc.shape
    model = _make_flat(
        np.array([f"gene_{i}" for i in range(10)]),
        n_latent=n_latent,
        cell_type_categories=[f"c{i}" for i in range(n_classes)],
        secondary_n_hidden=[],
        y_prior_probs=y_prior,
    )
    with torch.no_grad():
        for encoder in (model.u_encoder, model.z_prior_decoder):
            for head in (encoder.mean_head, encoder.log_var_head):
                head.weight.zero_()
                head.bias.zero_()
        model.z_prior_decoder.mean_head.weight[:, n_latent:] = mu_kc
    model.eval()
    return model


def test_the_generative_class_posterior_is_bayes_rule_with_the_class_priors(monkeypatch):
    n_latent, n = 3, 5
    mu_kc = torch.tensor([[0.0, 3.0, -2.0, 0.5], [0.0, 1.0, 2.0, -3.0], [0.0, -1.0, 0.0, 2.0]])
    y_prior = [0.5, 0.2, 0.2, 0.1]
    model = _gaussian_mixture_scanvi(mu_kc, y_prior)
    var_eps = model.z_prior_decoder.var_eps
    mean_nk = torch.randn(n, n_latent) * 2
    std_nk = torch.rand(n, n_latent) + 0.2

    # the encoder gives this posterior q(z|x)
    qz = torch.distributions.Normal(mean_nk, std_nk)
    monkeypatch.setattr(model, "inference", lambda **kwargs: {"z": qz.rsample(), "qz": qz})
    prediction = _predict(model, n=n, n_generative_samples=3)

    # K_c = KL(q(z|x) || N(mu_c, σ²)) + KL(N(0, σ²) || N(0, 1)) with σ² = 1 + var_eps
    var = 1.0 + var_eps
    diff_nkc = mean_nk.unsqueeze(-1) - mu_kc.unsqueeze(0)
    kl_z_nc = 0.5 * np.log(var) - std_nk.log().unsqueeze(-1) + (std_nk.unsqueeze(-1) ** 2 + diff_nkc**2) / (2 * var)
    kl_z_nc = (kl_z_nc - 0.5).sum(dim=1)
    kl_u = n_latent * 0.5 * (var - 1.0 - np.log(var))
    log_joint_nc = torch.tensor(y_prior).log() - (kl_z_nc + kl_u)
    torch.testing.assert_close(prediction["generative_probs_nc"], torch.softmax(log_joint_nc, dim=-1))
    # Bayes' rule for the Gaussian mixture (in the terms that depend on the class)
    bayes_nc = torch.softmax(torch.tensor(y_prior).log() - (diff_nkc**2).sum(dim=1) / (2 * var), dim=-1)
    torch.testing.assert_close(prediction["generative_probs_nc"], bayes_nc)

    # the score is log sum_c p(c) exp(E_q[log N(z; mu_c, σ²)]) - KL_u: the entropy of q(z|x) is not in it
    expected_log_density_nc = -0.5 * np.log(2 * np.pi * var) - (std_nk.unsqueeze(-1) ** 2 + diff_nkc**2) / (2 * var)
    expected_score_n = torch.logsumexp(torch.tensor(y_prior).log() + expected_log_density_nc.sum(dim=1) - kl_u, dim=-1)
    torch.testing.assert_close(prediction["in_distribution_score_n"], expected_score_n)


def test_the_generative_class_posterior_is_only_computed_when_asked():
    n, g = 6, 50
    var_names_g = np.array([f"g{i}" for i in range(g)])
    model = _make_flat(var_names_g)
    model.eval()
    x_ng = torch.poisson(torch.ones(n, g) * 2)
    with torch.no_grad():
        prediction = model.predict(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=torch.zeros(n, dtype=torch.long))
    assert "generative_probs_nc" not in prediction and "in_distribution_score_n" not in prediction


def test_the_ontology_generative_class_posterior_is_propagated_over_the_active_nodes():
    n, g = 6, 50
    var_names_g = np.array([f"g{i}" for i in range(g)])
    model = _make_ontology(var_names_g)
    model.eval()
    x_ng = torch.poisson(torch.ones(n, g) * 2)
    with torch.no_grad():
        prediction = model.predict(
            x_ng=x_ng, var_names_g=var_names_g, batch_index_n=torch.zeros(n, dtype=torch.long), n_generative_samples=2
        )
    probs_na = prediction["generative_probs_nc"]
    assert probs_na.shape == (n, model.n_active)
    names = model.active_cl_names
    # the root is above every class, and T is the sum of CD4 and CD8
    torch.testing.assert_close(probs_na[:, names.index("R")], torch.ones(n))
    torch.testing.assert_close(
        probs_na[:, names.index("T")], probs_na[:, names.index("CD4")] + probs_na[:, names.index("CD8")]
    )


# ---------------------------------------------------------------------------
# Classifier prior correction (Saerens et al.)
# ---------------------------------------------------------------------------


def _predict(model: SCANVI, n: int = 6, **kwargs) -> dict:
    g = len(model.var_names_g)
    x_ng = torch.poisson(torch.ones(n, g) * 2, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return dict(
            model.predict(
                x_ng=x_ng, var_names_g=model.var_names_g, batch_index_n=torch.zeros(n, dtype=torch.long), **kwargs
            )
        )


def test_the_classifier_prior_correction_of_a_flat_classifier():
    var_names_g = np.array([f"g{i}" for i in range(50)])
    model = _make_flat(
        var_names_g, cell_type_categories=["T", "B", "NK"], class_counts=pd.Series({"T": 50, "B": 30, "NK": 20})
    )
    model.eval()
    plain = _predict(model)
    # to the proportions of the training labels: no change, since that is what the classifier was trained with
    reference = torch.as_tensor(model.reference_class_proportions(space="classifier"), dtype=torch.float32)
    to_reference = _predict(model, classifier_target_proportions_c=reference)
    torch.testing.assert_close(to_reference["y_probs_nc"], plain["y_probs_nc"])
    # to other proportions: q(c|z) target(c) / reference(c), renormalized
    target = torch.tensor([0.2, 0.2, 0.6])
    expected = plain["y_probs_nc"] * target / reference
    expected = expected / expected.sum(dim=-1, keepdim=True)
    to_target = _predict(model, classifier_target_proportions_c=target)
    torch.testing.assert_close(to_target["y_probs_nc"], expected)
    torch.testing.assert_close(to_target["y_logits_nc"], plain["y_logits_nc"])  # the logits are not changed


def test_the_classifier_prior_correction_of_an_ontology_classifier_undoes_the_class_balancing():
    var_names_g = np.array([f"g{i}" for i in range(50)])
    model = _make_ontology(var_names_g)
    model.eval()
    assert model.class_weights is not None
    reference = torch.as_tensor(model.reference_class_proportions(space="classifier"), dtype=torch.float32)
    corrected = _predict(model, classifier_target_proportions_c=reference)
    # the implicit prior is the reference proportions times the class weights, so the ratio is 1 / weight (and 1 for
    # the root, which has no training labels)
    ratio = torch.where(reference > 0, 1.0 / model.class_weights, torch.ones_like(reference))
    base = torch.softmax(corrected["y_logits_nc"], dim=-1) * ratio
    expected = model._propagate_probs(base / base.sum(dim=-1, keepdim=True))
    torch.testing.assert_close(corrected["y_probs_nc"], expected)


def test_the_classifier_prior_correction_needs_class_counts():
    model = _make_flat(np.array([f"g{i}" for i in range(50)]), cell_type_categories=["T", "B"])
    with pytest.raises(ValueError, match="class_counts"):
        _predict(model, classifier_target_proportions_c=torch.tensor([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Models pickled before the new attributes existed
# ---------------------------------------------------------------------------


def test_a_model_pickled_before_the_new_attributes_works_as_before():
    n, g = 6, 50
    var_names_g = np.array([f"g{i}" for i in range(g)])
    torch.manual_seed(0)
    model = _make_flat(var_names_g)
    del model.unlabeled_prior  # as in a model pickled before these existed
    del model._class_counts
    assert model.reference_class_proportions(space="partition") is None
    x_ng = torch.poisson(torch.ones(n, g) * 2)
    batch_index_n = torch.zeros(n, dtype=torch.long)
    out = model(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=batch_index_n)  # unlabeled: the scanvi prior
    assert out["loss"].isfinite() and (out["kl_divergence_c"] > 0).all()
    model.eval()
    assert "generative_probs_nc" in _predict(model, n_generative_samples=1)
