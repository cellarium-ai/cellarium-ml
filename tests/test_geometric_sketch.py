# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import StreamingHyperplaneGeometricSketch, StreamingPlaidGeometricSketch
from cellarium.ml.utilities.data import collate_fn


class GeometricSketchDataset(torch.utils.data.Dataset):
    """Minimal dataset providing x_ng, var_names_g, obs_names_n, and optionally metadata_n."""

    def __init__(
        self,
        data: np.ndarray,
        var_names: np.ndarray,
        obs_names: np.ndarray,
        metadata: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.var_names = var_names
        self.obs_names = obs_names
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        item = {
            "x_ng": self.data[idx, None],
            "var_names_g": self.var_names,
            "obs_names_n": self.obs_names[idx, None],
        }
        if self.metadata is not None:
            item["metadata_n"] = self.metadata[idx, None]
        return item


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
    model = StreamingHyperplaneGeometricSketch(var_names, n_bits=4, max_cells_per_bucket=10)
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
    model = StreamingHyperplaneGeometricSketch(var_names, n_bits=4, max_cells_per_bucket=10, store_cell_data=False)
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    res = model.get_reservoir(return_cell_data=False)
    assert "obs_names" in res
    assert "x_ng" not in res
    assert len(res["obs_names"]) == model.total_cells

    with pytest.raises(ValueError, match="store_cell_data=False"):
        model.get_reservoir(return_cell_data=True)


def test_geometric_sketch_sampling_is_reproducible():
    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.standard_normal((40, 6)).astype(np.float32))
    var_names = np.array([f"gene_{i}" for i in range(6)])
    obs_names = np.array([f"cell_{i}" for i in range(40)])

    def run(seed: int) -> dict[int, list[str]]:
        model = StreamingHyperplaneGeometricSketch(var_names, n_bits=2, max_cells_per_bucket=2, seed=seed)
        model._lazy_init(data)
        # Perturbing the global RNG must not affect a seeded model.
        torch.manual_seed(seed + 1000)
        torch.rand(seed + 1)
        model.update(data, obs_names)
        return model._bucket_obs_names

    assert run(0) == run(0)
    assert any(run(seed) != run(0) for seed in range(1, 10))


def test_geometric_sketch_multi_device_raises():
    var_names = np.array(["g0", "g1", "g2"])
    model = StreamingHyperplaneGeometricSketch(var_names, n_bits=4, max_cells_per_bucket=10)

    class _MockTrainer:
        world_size = 2

    with pytest.raises(RuntimeError, match="single-device"):
        model.on_train_start(_MockTrainer())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# StreamingPlaidGeometricSketch
# ----------------------------------------------------------------------


def _make_plaid_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cells in three known unit voxels: (0, 0) x6, (2, 2) x6, and (5, 5) x2.

    Metadata is constant within the (0, 0) voxel and mixed within the (2, 2) voxel.
    """
    offsets = np.linspace(0.1, 0.9, 6, dtype=np.float32)
    data = np.concatenate(
        [
            np.stack([offsets, offsets], axis=1),  # voxel (0, 0)
            np.stack([offsets + 2.0, offsets + 2.0], axis=1),  # voxel (2, 2)
            np.array([[5.1, 5.1], [5.2, 5.2]], dtype=np.float32),  # voxel (5, 5)
        ]
    )
    metadata = np.array([0] * 6 + [0, 0, 0, 1, 1, 1] + [0, 0])
    var_names = np.array(["gene_0", "gene_1"])
    obs_names = np.array([f"cell_{i}" for i in range(len(data))])
    return data, var_names, obs_names, metadata


def _make_plaid_loader(
    with_metadata: bool = False, batch_size: int = 5
) -> tuple[torch.utils.data.DataLoader, np.ndarray]:
    data, var_names, obs_names, metadata = _make_plaid_data()
    dataset = GeometricSketchDataset(data, var_names, obs_names, metadata if with_metadata else None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return loader, var_names


def test_plaid_sketch_fit(tmp_path):
    loader, var_names = _make_plaid_loader()
    model = StreamingPlaidGeometricSketch(
        var_names,
        initial_voxel_size=1.0,
        max_cells_per_bucket=2,
        min_cells_per_voxel=5,
    )
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    # The (5, 5) voxel saw only 2 cells and is pruned at the end of the epoch.
    assert set(model._bucket_obs_names) == {(0, 0), (2, 2)}
    assert model.num_filled_buckets == 2
    assert model.total_cells == 4  # 2 voxels x max_cells_per_bucket

    res = model.get_reservoir(return_cell_data=True)
    assert isinstance(res["obs_names"], np.ndarray)
    assert len(res["obs_names"]) == 4
    assert res["x_ng"].layout == torch.sparse_csr  # type: ignore[union-attr]
    assert res["x_ng"].shape == (4, len(var_names))

    # Retained cells really do live in the voxel they were assigned to.
    x_dense = res["x_ng"].to_dense()  # type: ignore[union-attr]
    coords = torch.floor(x_dense / model.voxel_size).long()
    assert {tuple(c.tolist()) for c in coords} == {(0, 0), (2, 2)}


def test_plaid_sketch_reservoir_caps_cells_per_voxel():
    data, var_names, obs_names, _ = _make_plaid_data()
    model = StreamingPlaidGeometricSketch(var_names, initial_voxel_size=1.0, max_cells_per_bucket=3)

    model.update(torch.from_numpy(data), obs_names)

    assert set(model._bucket_obs_names) == {(0, 0), (2, 2), (5, 5)}
    assert model._bucket_total_seen == {(0, 0): 6, (2, 2): 6, (5, 5): 2}
    assert [len(v) for v in model._bucket_obs_names.values()] == [3, 3, 2]
    # No cell is retained twice, and all retained cells came from the stream.
    all_obs = [name for names in model._bucket_obs_names.values() for name in names]
    assert len(set(all_obs)) == len(all_obs)
    assert set(all_obs) <= set(obs_names)


def test_plaid_sketch_coarsening():
    var_names = np.array(["gene_0"])
    data = torch.tensor([[0.5], [1.5], [2.5], [3.5]])
    obs_names = np.array([f"cell_{i}" for i in range(4)])
    model = StreamingPlaidGeometricSketch(
        var_names,
        target_voxels=2,
        initial_voxel_size=1.0,
        max_cells_per_bucket=1,
    )

    model.update(data, obs_names)

    # 4 occupied voxels exceeds target_voxels=2, so the grid doubles and merges pairs.
    assert model.voxel_size.item() == pytest.approx(2.0)
    assert set(model._bucket_obs_names) == {(0,), (1,)}
    assert sum(model._bucket_total_seen.values()) == 4
    assert model.total_cells == 2

    # Subsequent cells are bucketed on the coarsened grid.
    model.update(torch.tensor([[0.1]]), np.array(["cell_4"]))
    assert model._bucket_total_seen[(0,)] == 3


def test_plaid_sketch_metadata_diversity_pruning(tmp_path):
    loader, var_names = _make_plaid_loader(with_metadata=True)
    model = StreamingPlaidGeometricSketch(
        var_names,
        initial_voxel_size=1.0,
        max_cells_per_bucket=2,
        min_cells_per_voxel=1,
        min_metadata_diversity=2,
    )
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    # Only the (2, 2) voxel saw more than one metadata category.
    assert set(model._bucket_obs_names) == {(2, 2)}
    assert model.total_cells == 2


def test_plaid_sketch_requires_metadata_when_diversity_enforced():
    data, var_names, obs_names, _ = _make_plaid_data()
    model = StreamingPlaidGeometricSketch(var_names, min_metadata_diversity=2)

    with pytest.raises(ValueError, match="metadata_n must be provided"):
        model(torch.from_numpy(data), var_names, obs_names)


def test_plaid_sketch_no_cell_data(tmp_path):
    loader, var_names = _make_plaid_loader()
    model = StreamingPlaidGeometricSketch(
        var_names,
        initial_voxel_size=1.0,
        max_cells_per_bucket=2,
        min_cells_per_voxel=1,
        store_cell_data=False,
    )
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    res = model.get_reservoir(return_cell_data=False)
    assert "x_ng" not in res
    assert len(res["obs_names"]) == model.total_cells > 0
    assert model._bucket_cells == {}

    with pytest.raises(ValueError, match="store_cell_data=False"):
        model.get_reservoir(return_cell_data=True)


def test_plaid_sketch_sparse_input_matches_dense():
    data, var_names, obs_names, _ = _make_plaid_data()
    x_dense = torch.from_numpy(data)

    dense_model = StreamingPlaidGeometricSketch(var_names, initial_voxel_size=1.0, max_cells_per_bucket=2)
    sparse_model = StreamingPlaidGeometricSketch(var_names, initial_voxel_size=1.0, max_cells_per_bucket=2)
    dense_model.update(x_dense, obs_names)
    sparse_model.update(x_dense.to_sparse(), obs_names)

    assert dense_model._bucket_total_seen == sparse_model._bucket_total_seen
    assert dense_model._bucket_obs_names == sparse_model._bucket_obs_names
    torch.testing.assert_close(
        dense_model.get_reservoir()["x_ng"].to_dense(),  # type: ignore[union-attr]
        sparse_model.get_reservoir()["x_ng"].to_dense(),  # type: ignore[union-attr]
    )


def test_plaid_sketch_projector_defines_voxel_space():
    data, var_names, obs_names, _ = _make_plaid_data()
    projector = torch.nn.Linear(len(var_names), 3, bias=False)
    model = StreamingPlaidGeometricSketch(var_names, initial_voxel_size=1.0, projector=projector)

    model.update(torch.from_numpy(data), obs_names)

    assert not any(p.requires_grad for p in projector.parameters())
    # Voxel coordinates live in the projector's output space, not gene space.
    assert all(len(coord) == 3 for coord in model._bucket_obs_names)


def test_plaid_sketch_sampling_is_reproducible():
    data, var_names, obs_names, _ = _make_plaid_data()
    x = torch.from_numpy(data)

    def run(seed: int) -> dict[tuple[int, ...], list[str]]:
        model = StreamingPlaidGeometricSketch(var_names, initial_voxel_size=1.0, max_cells_per_bucket=2, seed=seed)
        # Perturbing the global RNG must not affect a seeded model.
        torch.manual_seed(seed + 1000)
        torch.rand(seed + 1)
        model.update(x, obs_names)
        return model._bucket_obs_names

    assert run(0) == run(0)
    assert any(run(seed) != run(0) for seed in range(1, 10))


def test_plaid_sketch_coarsening_is_reproducible():
    var_names = np.array(["gene_0"])
    x = torch.arange(0.5, 8.5, 1.0).unsqueeze(1)
    obs_names = np.array([f"cell_{i}" for i in range(len(x))])

    def run() -> tuple[dict[tuple[int, ...], list[str]], float]:
        model = StreamingPlaidGeometricSketch(
            var_names, target_voxels=2, initial_voxel_size=1.0, max_cells_per_bucket=1, seed=3
        )
        torch.manual_seed(999)  # must not leak into the merge
        for i in range(len(x)):
            model.update(x[i, None], obs_names[i, None])
        return model._bucket_obs_names, model.voxel_size.item()

    first = run()
    torch.rand(17)
    assert run() == first


def test_plaid_sketch_reset_parameters_restores_reproducibility():
    data, var_names, obs_names, _ = _make_plaid_data()
    x = torch.from_numpy(data)
    model = StreamingPlaidGeometricSketch(var_names, initial_voxel_size=1.0, max_cells_per_bucket=2, seed=5)

    model.update(x, obs_names)
    first = {k: list(v) for k, v in model._bucket_obs_names.items()}

    model.reset_parameters()
    model.update(x, obs_names)

    assert model._bucket_obs_names == first


def test_plaid_sketch_reset_parameters():
    data, var_names, obs_names, _ = _make_plaid_data()
    model = StreamingPlaidGeometricSketch(var_names, initial_voxel_size=1.0, max_cells_per_bucket=2)
    model.update(torch.from_numpy(data), obs_names)
    assert model.total_cells > 0

    model.reset_parameters()

    assert model.total_cells == 0
    assert model.num_filled_buckets == 0
    assert model._bucket_total_seen == {}
    assert model._bucket_metadata == {}
    assert model._batches_seen == 0
    assert len(model.get_reservoir(return_cell_data=True)["obs_names"]) == 0


def test_plaid_sketch_multi_device_raises():
    var_names = np.array(["g0", "g1", "g2"])
    model = StreamingPlaidGeometricSketch(var_names)

    class _MockTrainer:
        world_size = 2

    with pytest.raises(RuntimeError, match="single-device"):
        model.on_train_start(_MockTrainer())  # type: ignore[arg-type]
