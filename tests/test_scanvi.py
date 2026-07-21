# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import SCANVI
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDatasetSCVI

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BoringDatasetSCANVI(BoringDatasetSCVI):
    """BoringDatasetSCVI extended with cell_type_index_n."""

    def __init__(
        self,
        data: np.ndarray,
        batch_index_n: np.ndarray,
        cell_type_index_n: np.ndarray,
        var_names: np.ndarray | None = None,
    ) -> None:
        super().__init__(data=data, batch_index_n=batch_index_n, var_names=var_names)
        self.cell_type_index_n = cell_type_index_n

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return super().__getitem__(idx) | {"cell_type_index_n": self.cell_type_index_n[idx, None]}


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


def _make_scanvi(var_names_g, *, n_batch: int = 2, n_latent: int = 8, n_classes: int = 5) -> SCANVI:
    return SCANVI(
        n_classes=n_classes,
        classifier_n_hidden=[32],
        secondary_n_hidden=[32],
        chunk_size=3,  # forces multiple chunks with n_classes=5
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=n_latent,
        encoder=_ENCODER_CFG,
        decoder=_DECODER_CFG,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loss_structure() -> None:
    """Labeled and unlabeled cells contribute distinct loss terms."""
    n, g, n_batch, n_classes = 16, 50, 2, 5
    n_labeled = 8
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    torch.manual_seed(0)
    model = _make_scanvi(var_names_g, n_batch=n_batch, n_classes=n_classes)
    model.eval()

    x_ng = torch.poisson(torch.ones(n, g) * 2).float()
    batch_index_n = torch.zeros(n, dtype=torch.long)

    # First n_labeled cells are labeled; the rest use the unlabeled sentinel (-1)
    cell_type_index_n = torch.full((n,), -1, dtype=torch.long)
    cell_type_index_n[:n_labeled] = torch.randint(0, n_classes, (n_labeled,))

    with torch.no_grad():
        out = model(
            x_ng=x_ng,
            var_names_g=var_names_g,
            batch_index_n=batch_index_n,
            cell_type_index_n=cell_type_index_n,
        )

    ce = out["classification_loss"]
    kl_c = out["kl_divergence_c"]
    kl_z = out["kl_divergence_z"]
    kl_u = out["kl_divergence_u"]
    loss = out["loss"]
    assert isinstance(ce, torch.Tensor)
    assert isinstance(kl_c, torch.Tensor)
    assert isinstance(kl_z, torch.Tensor)
    assert isinstance(kl_u, torch.Tensor)
    assert isinstance(loss, torch.Tensor)

    # Cross-entropy is positive for labeled cells, exactly zero for unlabeled
    assert (ce[:n_labeled] > 0).all(), "Expected positive CE loss for labeled cells"
    assert ce[n_labeled:].sum() == 0.0, "Expected zero CE loss for unlabeled cells"

    # kl_c is exactly zero for labeled cells, positive for unlabeled
    assert kl_c[:n_labeled].sum() == 0.0, "Expected zero kl_c for labeled cells"
    assert (kl_c[n_labeled:] > 0).all(), "Expected positive kl_c for unlabeled cells"

    # KL terms are non-negative for all cells
    assert (kl_z >= 0).all(), "kl_z must be non-negative"
    assert (kl_u >= 0).all(), "kl_u must be non-negative"

    assert loss.isfinite(), "Scalar loss must be finite"


def test_predict_shapes() -> None:
    """predict() returns embeddings and normalized class probabilities."""
    n, g, n_batch, n_classes, n_latent = 16, 50, 2, 5, 8
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    model = _make_scanvi(var_names_g, n_batch=n_batch, n_latent=n_latent, n_classes=n_classes)
    model.eval()

    x_ng = torch.poisson(torch.ones(n, g) * 2).float()
    batch_index_n = torch.zeros(n, dtype=torch.long)

    with torch.no_grad():
        pred = model.predict(x_ng=x_ng, var_names_g=var_names_g, batch_index_n=batch_index_n)

    assert pred["x_ng"].shape == (n, n_latent), f"Expected [{n}, {n_latent}], got {pred['x_ng'].shape}"
    assert pred["cell_type_probs_nc"].shape == (n, n_classes), (
        f"Expected [{n}, {n_classes}], got {pred['cell_type_probs_nc'].shape}"
    )
    assert torch.allclose(pred["cell_type_probs_nc"].sum(dim=-1), torch.ones(n), atol=1e-5), (
        "Cell-type probabilities must sum to 1"
    )
    assert (pred["cell_type_probs_nc"] >= 0).all(), "Probabilities must be non-negative"


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Saving and reloading a checkpoint preserves predict() outputs exactly."""
    n, g, n_batch, n_classes = 32, 50, 2, 5
    batch_size = 8
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    devices = int(os.environ.get("TEST_DEVICES", "1"))

    rng = np.random.default_rng(42)
    x = rng.poisson(lam=2.0, size=(n, g)).astype(np.float32)
    batch_idx = rng.integers(0, n_batch, size=n)
    # ~half labeled, half unlabeled
    labeled_mask = rng.random(n) < 0.5
    cell_type_idx = np.where(labeled_mask, rng.integers(0, n_classes, size=n), -1)

    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCANVI(
            data=x,
            batch_index_n=batch_idx,
            cell_type_index_n=cell_type_idx,
            var_names=var_names_g,
        ),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    torch.manual_seed(0)
    model = _make_scanvi(var_names_g, n_batch=n_batch, n_classes=n_classes)
    module = CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=devices,
        max_steps=2,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=train_loader)

    ckpt_path = tmp_path / "scanvi.ckpt"
    trainer.save_checkpoint(ckpt_path)

    # Capture predict() output from the original model
    x_t = torch.tensor(x)
    b_t = torch.tensor(batch_idx, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pre = model.predict(x_ng=x_t, var_names_g=var_names_g, batch_index_n=b_t)

    # Reload and verify outputs are identical
    loaded_model = CellariumModule.load_from_checkpoint(ckpt_path).model
    assert isinstance(loaded_model, SCANVI)
    loaded_model.eval()
    with torch.no_grad():
        post = loaded_model.predict(x_ng=x_t, var_names_g=var_names_g, batch_index_n=b_t)

    torch.testing.assert_close(pre["x_ng"], post["x_ng"])
    torch.testing.assert_close(pre["cell_type_probs_nc"], post["cell_type_probs_nc"])


def test_guard_use_flow_raises() -> None:
    """SCANVI must reject use_flow=True at construction time."""
    var_names_g = np.array([f"gene_{i}" for i in range(10)])
    with pytest.raises(ValueError, match="use_flow"):
        SCANVI(
            n_classes=3,
            var_names_g=var_names_g,
            n_batch=1,
            use_flow=True,
            encoder={
                "hidden_layers": [],
                "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            },
            decoder={
                "hidden_layers": [],
                "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
                "final_additive_bias": False,
            },
        )
