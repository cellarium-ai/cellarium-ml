# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import StreamingGeometricSketch
from cellarium.ml.utilities.data import collate_fn


class GeometricSketchDataset(torch.utils.data.Dataset):
    """Minimal dataset providing x_ng, var_names_g, and obs_names_n."""

    def __init__(self, data: np.ndarray, var_names: np.ndarray, obs_names: np.ndarray) -> None:
        self.data = data
        self.var_names = var_names
        self.obs_names = obs_names

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "x_ng": self.data[idx, None],
            "var_names_g": self.var_names,
            "obs_names_n": self.obs_names[idx, None],
        }


def _make_loader(n: int = 30, g: int = 6) -> tuple[torch.utils.data.DataLoader, np.ndarray]:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, g)).astype(np.float32)
    var_names = np.array([f"gene_{i}" for i in range(g)])
    obs_names = np.array([f"cell_{i}" for i in range(n)])
    dataset = GeometricSketchDataset(data, var_names, obs_names)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
    return loader, var_names


def test_geometric_sketch_fit(tmp_path):
    loader, var_names = _make_loader()
    model = StreamingGeometricSketch(var_names, n_bits=4, max_cells_per_bucket=10)
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    assert model.total_cells > 0
    assert 0 < model.num_filled_buckets <= model.num_buckets

    res = model.get_reservoir(return_cell_data=True)

    assert "obs_names" in res
    assert isinstance(res["obs_names"], np.ndarray)
    assert len(res["obs_names"]) == model.total_cells

    assert "x_ng" in res
    assert res["x_ng"].layout == torch.sparse_csr  # type: ignore[union-attr]
    assert res["x_ng"].shape == (model.total_cells, len(var_names))


def test_geometric_sketch_no_cell_data(tmp_path):
    loader, var_names = _make_loader()
    model = StreamingGeometricSketch(var_names, n_bits=4, max_cells_per_bucket=10, store_cell_data=False)
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    res = model.get_reservoir(return_cell_data=False)
    assert "obs_names" in res
    assert "x_ng" not in res
    assert len(res["obs_names"]) == model.total_cells

    with pytest.raises(ValueError, match="store_cell_data=False"):
        model.get_reservoir(return_cell_data=True)


def test_geometric_sketch_multi_device_raises():
    var_names = np.array(["g0", "g1", "g2"])
    model = StreamingGeometricSketch(var_names, n_bits=4, max_cells_per_bucket=10)

    class _MockTrainer:
        world_size = 2

    with pytest.raises(RuntimeError, match="single-device"):
        model.on_train_start(_MockTrainer())  # type: ignore[arg-type]
