# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

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


def _make_flat(var_names_g, *, n_batch=2, n_latent=8, cell_type_categories=("T", "B", "NK", "Mono", "DC")):
    return SCANVI(
        classifier_type="flat",
        cell_type_categories=list(cell_type_categories),
        classifier_n_hidden=[32],
        secondary_n_hidden=[32],
        chunk_size=3,
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=n_latent,
        encoder=_ENCODER_CFG,
        decoder=_DECODER_CFG,
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
    assert pred["cell_type_probs_nc"].shape == (n, len(categories))
    assert torch.allclose(pred["cell_type_probs_nc"].sum(-1), torch.ones(n), atol=1e-5)


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
    assert pred["cell_type_probs_nc"].shape == (n, model.n_active)
    assert (pred["cell_type_probs_nc"] >= 0).all()


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
        model = _make_flat(var_names_g, n_batch=n_batch, cell_type_categories=categories)
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
    with torch.no_grad():
        pre = model.predict(x_ng=x_t, var_names_g=var_names_g, batch_index_n=b_t)

    loaded = CellariumModule.load_from_checkpoint(ckpt_path).model
    assert isinstance(loaded, SCANVI)
    loaded.eval()
    with torch.no_grad():
        post = loaded.predict(x_ng=x_t, var_names_g=var_names_g, batch_index_n=b_t)

    torch.testing.assert_close(pre["x_ng"], post["x_ng"])
    torch.testing.assert_close(pre["cell_type_probs_nc"], post["cell_type_probs_nc"])


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
