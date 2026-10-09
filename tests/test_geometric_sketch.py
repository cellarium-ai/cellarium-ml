# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import StreamingHyperplaneGeometricSketch, StreamingPlaidGeometricSketch
from cellarium.ml.models import geometric_sketch as geometric_sketch_module
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

    def run(seed: int) -> dict[tuple[int, ...], list[str]]:
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


def _pin_voxel_grid(
    model: StreamingPlaidGeometricSketch,
    dim: int,
    voxel_size: float = 1.0,
    voxel_offset: float = 0.0,
    module: CellariumModule | None = None,
) -> None:
    """Fix a `StreamingPlaidGeometricSketch`'s grid at a known size and origin.

    Tests built on `_make_plaid_data()` assert on its hand-picked, unit-spaced, zero-anchored
    voxel coordinates (e.g. `(0, 0)`, `(2, 2)`). Without this, `StreamingPlaidGeometricSketch`'s
    data-driven `_lazy_init` picks an arbitrary `voxel_offset`/`voxel_size`, shifting/rescaling
    those coordinates without changing the underlying grouping. Registering both buffers ahead
    of time makes `_lazy_init`'s `hasattr(self, "voxel_offset")` guard skip recomputing either.

    Pass `module` when the model is wrapped in a `CellariumModule` driven by `trainer.fit()`:
    its `configure_model()` hook calls `model.reset_parameters()` again on the first training
    step unless `is_initialized` is already set, which would otherwise silently clear this pin.
    """
    model.register_buffer("voxel_offset", torch.full((dim,), voxel_offset))
    model.register_buffer("voxel_size", torch.full((dim,), voxel_size))
    if module is not None:
        module.hparams["is_initialized"] = True


def test_plaid_sketch_fit(tmp_path):
    loader, var_names = _make_plaid_loader()
    model = StreamingPlaidGeometricSketch(
        var_names,
        max_cells_per_bucket=2,
        min_cells_per_bucket=5,
        store_cell_data=True,
    )
    module = CellariumModule(model=model)
    _pin_voxel_grid(model, dim=len(var_names), module=module)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    # The (5, 5) voxel saw only 2 cells and is pruned at the end of the epoch.
    assert set(model._voxel_summary()) == {(0, 0), (2, 2)}
    assert model.num_filled_buckets == 2
    assert model.total_cells == 4  # 2 voxels x max_cells_per_bucket

    res = model.get_reservoir(return_cell_data=True)
    assert isinstance(res["obs_names"], np.ndarray)
    assert len(res["obs_names"]) == 4
    assert res["x_ng"].layout == torch.sparse_csr  # type: ignore[union-attr]
    assert res["x_ng"].shape == (4, len(var_names))

    # Retained cells really do live in the voxel they were assigned to.
    x_dense = res["x_ng"].to_dense()  # type: ignore[union-attr]
    coords = torch.floor((x_dense - model.voxel_offset) / model.voxel_size).long()
    assert {tuple(c.tolist()) for c in coords} == {(0, 0), (2, 2)}


def test_plaid_sketch_reservoir_caps_cells_per_voxel():
    data, var_names, obs_names, _ = _make_plaid_data()
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=3, store_cell_data=True)
    _pin_voxel_grid(model, dim=len(var_names))

    model.update(torch.from_numpy(data), obs_names)

    voxels = model._voxel_summary()
    assert set(voxels) == {(0, 0), (2, 2), (5, 5)}
    assert {k: v["seen"] for k, v in voxels.items()} == {(0, 0): 6, (2, 2): 6, (5, 5): 2}
    assert {k: len(v["obs_names"]) for k, v in voxels.items()} == {(0, 0): 3, (2, 2): 3, (5, 5): 2}
    # No cell is retained twice, and all retained cells came from the stream.
    all_obs = [name for v in voxels.values() for name in v["obs_names"]]
    assert len(set(all_obs)) == len(all_obs)
    assert set(all_obs) <= set(obs_names)


def test_plaid_sketch_coarsening():
    var_names = np.array(["gene_0"])
    data = torch.tensor([[0.5], [1.5], [2.5], [3.5]])
    obs_names = np.array([f"cell_{i}" for i in range(4)])
    model = StreamingPlaidGeometricSketch(
        var_names,
        target_voxels=2,
        max_cells_per_bucket=1,
        store_cell_data=True,
    )
    _pin_voxel_grid(model, dim=1)

    model.update(data, obs_names)

    # 4 occupied voxels exceeds target_voxels=2, so the grid doubles and merges pairs.
    assert model.voxel_size.item() == pytest.approx(2.0)
    voxels = model._voxel_summary()
    assert set(voxels) == {(0,), (1,)}
    assert sum(v["seen"] for v in voxels.values()) == 4
    assert model.total_cells == 2

    # Subsequent cells are bucketed on the coarsened grid.
    model.update(torch.tensor([[0.1]]), np.array(["cell_4"]))
    assert model._voxel_summary()[(0,)]["seen"] == 3


def test_plaid_sketch_coarsening_is_one_axis_at_a_time():
    """Each coarsening step doubles a single axis: the one with the smallest voxel size.

    Doubling all axes at once could merge far more than pairs of voxels. Here axis 1 starts
    with a larger voxel size than axis 0, so it must be left alone until axis 0 catches up.
    """
    var_names = np.array(["gene_0", "gene_1"])
    model = StreamingPlaidGeometricSketch(var_names, target_voxels=2, max_cells_per_bucket=1)
    model.register_buffer("voxel_offset", torch.zeros(2))
    model.register_buffer("voxel_size", torch.tensor([0.25, 1.0]))

    # 4 distinct values along axis 0, all sharing the same axis-1 coordinate: 4 occupied
    # voxels driven purely by axis 0's resolution, exceeding target_voxels=2.
    data = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]])
    obs_names = np.array([f"cell_{i}" for i in range(4)])

    model.update(data, obs_names)

    # Axis 0 doubled (0.25 -> 0.5 -> 1.0 -> 2.0) until at most 2 voxels remained; the whole
    # time axis 1 was never the smallest-voxel axis strictly before axis 0 reached 1.0,
    # and the loop stopped as soon as the target was met.
    assert model.voxel_size.tolist() == [2.0, 1.0]
    assert model.num_filled_buckets == 2


def test_plaid_sketch_coarsening_axes_take_turns():
    """With equal voxel sizes, ties go to the lowest axis, then the next one, and so on."""
    var_names = np.array(["gene_0", "gene_1", "gene_2"])
    model = StreamingPlaidGeometricSketch(var_names, target_voxels=1, max_cells_per_bucket=1)
    model.register_buffer("voxel_offset", torch.zeros(3))
    model.register_buffer("voxel_size", torch.ones(3))

    # Eight cells filling the corners of a cube: every axis must be doubled for them to merge.
    corners = torch.tensor([[x, y, z] for x in (0.5, 1.5) for y in (0.5, 1.5) for z in (0.5, 1.5)])
    model.update(corners, np.array([f"cell_{i}" for i in range(8)]))

    assert model.num_filled_buckets == 1
    assert model.voxel_size.tolist() == [2.0, 2.0, 2.0]


def test_plaid_sketch_coarsening_lands_between_half_and_all_of_target_in_high_dimension():
    """Regression test: a coarsening event must not overshoot far below ``target_voxels``.

    In moderate-to-high dimension the occupied voxel count of a grid is a cliff-like function
    of the voxel size: doubling every axis at once can drop it from many times ``target_voxels``
    to a single voxel. Doubling one axis at a time can drop it by at most a factor of 2 per step.
    """
    torch.manual_seed(0)
    n, d, target = 6000, 30, 400
    data = torch.randn(n, d) * (3.0 / torch.arange(1, d + 1).float().sqrt())
    var_names = np.array([f"gene_{i}" for i in range(d)])
    obs_names = np.array([f"cell_{i}" for i in range(n)])
    model = StreamingPlaidGeometricSketch(var_names, target_voxels=target, max_cells_per_bucket=1)

    n_events_before = 0
    for start in range(0, n, 100):
        model.update(data[start : start + 100], obs_names[start : start + 100])
        # The target is never exceeded after a batch ...
        assert model.num_filled_buckets <= target
        # ... and a batch that triggered coarsening leaves more than half of it.
        if model._coarsen_event_count > n_events_before:
            assert model.num_filled_buckets > target // 2
        n_events_before = model._coarsen_event_count

    assert model._coarsen_event_count > 0
    assert model.total_cells == model.num_filled_buckets  # one cell per voxel


def test_plaid_sketch_metadata_diversity_pruning(tmp_path):
    loader, var_names = _make_plaid_loader(with_metadata=True)
    model = StreamingPlaidGeometricSketch(
        var_names,
        max_cells_per_bucket=2,
        min_cells_per_bucket=1,
        min_metadata_diversity=2,
        store_cell_data=True,
    )
    module = CellariumModule(model=model)
    _pin_voxel_grid(model, dim=len(var_names), module=module)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    # Only the (2, 2) voxel saw more than one metadata category.
    assert set(model._voxel_summary()) == {(2, 2)}
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
        max_cells_per_bucket=2,
        min_cells_per_bucket=1,
        store_cell_data=False,
    )
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    res = model.get_reservoir(return_cell_data=False)
    assert "x_ng" not in res
    assert len(res["obs_names"]) == model.total_cells > 0
    assert model._pool.state_dict()["csr"] is None  # no cell data is kept

    with pytest.raises(ValueError, match="store_cell_data=False"):
        model.get_reservoir(return_cell_data=True)


def test_plaid_sketch_sparse_input_matches_dense():
    data, var_names, obs_names, _ = _make_plaid_data()
    x_dense = torch.from_numpy(data)

    dense_model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2, store_cell_data=True)
    sparse_model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2, store_cell_data=True)
    dense_model.update(x_dense, obs_names)
    sparse_model.update(x_dense.to_sparse(), obs_names)

    assert dense_model._voxel_summary() == sparse_model._voxel_summary()
    torch.testing.assert_close(
        dense_model.get_reservoir(return_cell_data=True)["x_ng"].to_dense(),  # type: ignore[union-attr]
        sparse_model.get_reservoir(return_cell_data=True)["x_ng"].to_dense(),  # type: ignore[union-attr]
    )


def test_plaid_sketch_coarsening_escapes_zero_centered_degeneracy():
    """Zero-centered data (e.g. PCA output) must not get stuck at ~1 cell/voxel forever.

    Regression test: a grid floored around a fixed origin of zero has a permanent boundary
    at zero, which bisects zero-centered, symmetric data down the middle on every axis no
    matter how large voxel_size grows. The lazily-estimated voxel_offset exists to move that
    permanent boundary away from the bulk of the data so coarsening can actually converge.

    voxel_size is forced artificially fine right after lazy-init (simulating a badly
    miscalibrated starting guess) to stress-test that repeated coarsening still converges;
    voxel_offset is left at its natural, data-driven (non-zero) value, since that — not
    voxel_size — is what this regression test is actually about.
    """
    torch.manual_seed(0)
    n, d = 2000, 20
    data = torch.randn(n, d) * torch.empty(d).uniform_(1.0, 20.0)
    var_names = np.array([f"gene_{i}" for i in range(d)])
    obs_names = np.array([f"cell_{i}" for i in range(n)])

    model = StreamingPlaidGeometricSketch(var_names, target_voxels=10, max_cells_per_bucket=1_000)
    model._lazy_init(data[:20])
    model.voxel_size.fill_(0.01)

    batch_size = 20
    for start in range(0, n, batch_size):
        model.update(data[start : start + batch_size], obs_names[start : start + batch_size])

    # Coarsening must actually reduce occupied buckets well below the cell count, converging
    # near (or under) target_voxels rather than plateauing at ~1 cell/bucket.
    assert model.num_filled_buckets < n // 10


def test_plaid_sketch_projector_defines_voxel_space():
    data, var_names, obs_names, _ = _make_plaid_data()
    projector = torch.nn.Linear(len(var_names), 3, bias=False)
    model = StreamingPlaidGeometricSketch(var_names, projector=projector)

    model.update(torch.from_numpy(data), obs_names)

    assert not any(p.requires_grad for p in projector.parameters())
    # Voxel coordinates live in the projector's output space, not gene space.
    assert all(len(coord) == 3 for coord in model._voxel_summary())


def _retained(model: StreamingPlaidGeometricSketch) -> dict[tuple[int, ...], list[str]]:
    return {k: v["obs_names"] for k, v in model._voxel_summary().items()}


def test_plaid_sketch_sampling_is_reproducible():
    data, var_names, obs_names, _ = _make_plaid_data()
    x = torch.from_numpy(data)

    def run(seed: int) -> dict[tuple[int, ...], list[str]]:
        model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2, seed=seed)
        _pin_voxel_grid(model, dim=len(var_names))
        # Perturbing the global RNG must not affect a seeded model.
        torch.manual_seed(seed + 1000)
        torch.rand(seed + 1)
        model.update(x, obs_names)
        return _retained(model)

    assert run(0) == run(0)
    assert any(run(seed) != run(0) for seed in range(1, 10))


def test_plaid_sketch_coarsening_is_reproducible():
    var_names = np.array(["gene_0"])
    x = torch.arange(0.5, 8.5, 1.0).unsqueeze(1)
    obs_names = np.array([f"cell_{i}" for i in range(len(x))])

    def run() -> tuple[dict[tuple[int, ...], list[str]], float]:
        model = StreamingPlaidGeometricSketch(var_names, target_voxels=2, max_cells_per_bucket=1, seed=3)
        torch.manual_seed(999)  # must not leak into the merge
        for i in range(len(x)):
            model.update(x[i, None], obs_names[i, None])
        return _retained(model), model.voxel_size.item()

    first = run()
    torch.rand(17)
    assert run() == first


def test_plaid_sketch_reset_parameters_restores_reproducibility():
    data, var_names, obs_names, _ = _make_plaid_data()
    x = torch.from_numpy(data)
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2, seed=5)

    model.update(x, obs_names)
    first = _retained(model)

    model.reset_parameters()
    model.update(x, obs_names)

    assert _retained(model) == first


def test_plaid_sketch_reset_parameters():
    data, var_names, obs_names, _ = _make_plaid_data()
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2, store_cell_data=True)
    model.update(torch.from_numpy(data), obs_names)
    assert model.total_cells > 0
    assert hasattr(model, "voxel_size")

    model.reset_parameters()

    assert model.total_cells == 0
    assert model.num_filled_buckets == 0
    assert model._voxel_summary() == {}
    assert model._batches_seen == 0
    assert model._coarsen_event_count == 0
    assert len(model.get_reservoir(return_cell_data=True)["obs_names"]) == 0
    assert model.get_reservoir(return_cell_data=True)["x_ng"].shape == (0, len(var_names))  # type: ignore[union-attr]

    # voxel_size and voxel_offset are data-driven; reset_parameters clears them entirely
    # rather than restoring a fixed value, deferring to the next _lazy_init.
    assert not hasattr(model, "voxel_size")
    assert not hasattr(model, "voxel_offset")

    # A subsequent update() re-triggers _lazy_init and works normally.
    model.update(torch.from_numpy(data), obs_names)
    assert model.total_cells > 0


def test_plaid_sketch_multi_device_raises():
    var_names = np.array(["g0", "g1", "g2"])
    model = StreamingPlaidGeometricSketch(var_names, store_cell_data=True)

    class _MockTrainer:
        world_size = 2

    with pytest.raises(RuntimeError, match="single-device"):
        model.on_train_start(_MockTrainer())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# StreamingPlaidGeometricSketch: tensor implementation vs. a slow reference
# ----------------------------------------------------------------------


class _ReferencePlaid:
    """Obviously-correct dict implementation of the voxel bookkeeping of a Plaid sketch.

    It keeps every cell of every voxel (so it can check which cells the model may retain), and coarsens by
    literally doubling one axis at a time (the axis with the smallest voxel size, ties to the lowest index)
    until at most ``target_voxels`` voxels are occupied.
    """

    def __init__(self, target_voxels: int, offset: torch.Tensor, size: torch.Tensor) -> None:
        self.target_voxels = target_voxels
        self.offset = offset.clone()
        self.size = size.clone()
        self.voxels: dict[tuple[int, ...], dict] = {}

    def update(self, z: torch.Tensor, names: np.ndarray, metadata: np.ndarray | None) -> None:
        coords = torch.floor((z - self.offset) / self.size).long().numpy()
        for i, coord in enumerate(coords):
            v = self.voxels.setdefault(tuple(int(c) for c in coord), {"seen": 0, "names": [], "metadata": set()})
            v["seen"] += 1
            v["names"].append(str(names[i]))
            if metadata is not None:
                v["metadata"].add(int(metadata[i]))
        while len(self.voxels) > self.target_voxels:
            axis = int(torch.argmin(self.size))
            self.size[axis] *= 2.0
            merged: dict[tuple[int, ...], dict] = {}
            for coord, v in self.voxels.items():
                new = coord[:axis] + (coord[axis] // 2,) + coord[axis + 1 :]
                m = merged.setdefault(new, {"seen": 0, "names": [], "metadata": set()})
                m["seen"] += v["seen"]
                m["names"] += v["names"]
                m["metadata"] |= v["metadata"]
            self.voxels = merged


@pytest.mark.parametrize("small_index_delta", [False, True])
@pytest.mark.parametrize("dim, slots", [(2, 1), (10, 2), (50, 2)])
def test_plaid_sketch_matches_reference_implementation(dim, slots, small_index_delta, monkeypatch):
    if small_index_delta:  # exercise merging the lookup index's delta run into its main run
        monkeypatch.setattr(geometric_sketch_module._VoxelStore, "_MIN_DELTA", 4)
    rng = np.random.default_rng(dim)
    n, target = 2000, 150
    z = torch.from_numpy((rng.standard_normal((n, dim)) * 3 / np.sqrt(np.arange(1, dim + 1))).astype(np.float32))
    names = np.array([f"cell_{i}" for i in range(n)])
    metadata = rng.integers(0, 4, n)
    var_names = np.array([f"d{i}" for i in range(dim)])

    model = StreamingPlaidGeometricSketch(var_names, target_voxels=target, max_cells_per_bucket=slots)
    model._lazy_init(z[:100])
    reference = _ReferencePlaid(target, model.voxel_offset, model.voxel_size)
    for start in range(0, n, 100):
        batch = slice(start, start + 100)
        model.update(z[batch], names[batch], metadata[batch])
        reference.update(z[batch], names[batch], metadata[batch])

    assert model._coarsen_event_count > 0
    assert model.voxel_size.tolist() == reference.size.tolist()
    voxels = model._voxel_summary()
    assert set(voxels) == set(reference.voxels)
    for coord, v in voxels.items():
        ref = reference.voxels[coord]
        assert v["seen"] == ref["seen"]
        assert v["metadata"] == ref["metadata"]
        # The retained cells are distinct cells that landed in this voxel.
        assert len(v["obs_names"]) == len(set(v["obs_names"])) == min(slots, ref["seen"])
        assert set(v["obs_names"]) <= set(ref["names"])


def _chi_square_uniform(counts: list[int]) -> float:
    expected = sum(counts) / len(counts)
    return sum((c - expected) ** 2 / expected for c in counts)


def test_plaid_sketch_retained_cell_is_uniform_within_voxel_and_after_merge():
    """The cell kept by a voxel is uniform over the cells it saw, also when voxels merge (random-tag sampling)."""
    var_names = np.array(["gene_0"])
    names = np.array([f"cell_{i}" for i in range(10)])
    x = torch.tensor([[0.1], [0.2], [0.3]] + [[1.0 + 0.1 * j] for j in range(1, 8)])  # 3 cells in (0,), 7 in (1,)
    n_runs = 800

    def kept(seed: int, target_voxels: int) -> dict[tuple[int, ...], list[str]]:
        model = StreamingPlaidGeometricSketch(var_names, target_voxels=target_voxels, seed=seed)
        _pin_voxel_grid(model, dim=1)
        model.update(x[:3], names[:3])  # voxel (0,)
        model.update(x[3:], names[3:])  # voxel (1,), merged into (0,) if target_voxels == 1
        return _retained(model)

    separate = [kept(seed, target_voxels=2) for seed in range(n_runs)]
    for coord, cells in ((0,), names[:3]), ((1,), names[3:]):
        counts = [sum(run[coord][0] == name for run in separate) for name in cells]
        assert _chi_square_uniform(counts) < 25, counts

    merged = [kept(seed, target_voxels=1) for seed in range(n_runs)]
    assert all(list(run) == [(0,)] for run in merged)
    counts = [sum(run[(0,)][0] == name for run in merged) for name in names]
    assert _chi_square_uniform(counts) < 30, counts


def _cell_index(names: np.ndarray) -> list[int]:
    return [int(name.split("_")[1]) for name in names]


def test_plaid_sketch_get_reservoir_max_cells_coarsens_a_copy_of_the_grid():
    """With more voxels than ``max_cells``, the grid is coarsened (as a copy) to the coarsest one with enough voxels."""
    var_names = np.array(["gene_0"])
    x = torch.arange(0.5, 8.5, 1.0).unsqueeze(1)  # cells 0..7, one per unit voxel
    names = np.array([f"cell_{i}" for i in range(8)])
    model = StreamingPlaidGeometricSketch(var_names, target_voxels=8)
    _pin_voxel_grid(model, dim=1)
    model.update(x, names)
    before = model._voxel_summary()

    for seed in range(5):
        reservoir_obs_names = model.get_reservoir(max_cells=3, seed=seed)["obs_names"]
        assert isinstance(reservoir_obs_names, np.ndarray)
        picked = _cell_index(reservoir_obs_names)
        # Voxel sizes 1, 2 and 4 give 8, 4 and 2 voxels: the size-2 grid pairs up neighbours {0,1}, {2,3}, ...
        assert len(picked) == 3
        assert len({i // 2 for i in picked}) == 3

    # Nothing was modified.
    assert model.voxel_size.item() == 1.0
    assert model._voxel_summary() == before


def test_plaid_sketch_get_reservoir_max_cells_tops_up_from_second_cells():
    """With fewer voxels than ``max_cells`` the voxels are visited round by round, as in the paper."""
    data, var_names, obs_names, _ = _make_plaid_data()  # voxels (0, 0), (2, 2), (5, 5) with 6, 6 and 2 cells
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2)
    _pin_voxel_grid(model, dim=len(var_names))
    model.update(torch.from_numpy(data), obs_names)
    voxels = model._voxel_summary()
    voxel_of = {name: coord for coord, v in voxels.items() for name in v["obs_names"]}

    # 3 voxels hold 6 cells; 5 are wanted: every voxel gives its first cell, two give their second.
    for seed in range(5):
        picked = model.get_reservoir(max_cells=5, seed=seed)["obs_names"]
        assert len(picked) == len(set(picked)) == 5
        per_voxel = np.bincount([sorted(voxels).index(voxel_of[name]) for name in picked], minlength=3)
        assert per_voxel.min() == 1 and per_voxel.max() == 2

    # A sketch no larger than max_cells is returned whole, in every order of voxels.
    assert len(model.get_reservoir(max_cells=6)["obs_names"]) == 6
    assert len(model.get_reservoir(max_cells=100)["obs_names"]) == 6


def test_plaid_sketch_get_reservoir_max_cells_is_seeded():
    rng = np.random.default_rng(0)
    z = torch.from_numpy(rng.standard_normal((500, 4)).astype(np.float32))
    names = np.array([f"cell_{i}" for i in range(500)])
    model = StreamingPlaidGeometricSketch(np.array(list("abcd")), target_voxels=200, seed=7)
    model.update(z, names)
    assert model.num_filled_buckets > 50

    default = model.get_reservoir(max_cells=50)["obs_names"]
    np.testing.assert_array_equal(default, model.get_reservoir(max_cells=50)["obs_names"])  # the model's seed
    other = model.get_reservoir(max_cells=50, seed=1)["obs_names"]
    assert len(other) == 50
    assert not np.array_equal(default, other)


def test_plaid_sketch_cell_data_follows_cells_through_coarsening_and_compaction():
    rng = np.random.default_rng(0)
    n, g = 1500, 5
    x = torch.from_numpy(
        (rng.standard_normal((n, g)) * np.arange(1, g + 1)).astype(np.float32) * (rng.random((n, g)) < 0.6)
    )  # sparse-ish, and every row distinct
    names = np.array([f"cell_{i}" for i in range(n)])
    model = StreamingPlaidGeometricSketch(
        np.array([f"d{i}" for i in range(g)]), target_voxels=60, max_cells_per_bucket=2, store_cell_data=True
    )
    for start in range(0, n, 100):
        model.update(x[start : start + 100], names[start : start + 100])
    assert model._coarsen_event_count > 0

    def check() -> None:
        res = model.get_reservoir(return_cell_data=True)
        assert len(res["obs_names"]) == model.total_cells
        assert isinstance(res["obs_names"], np.ndarray)
        torch.testing.assert_close(res["x_ng"].to_dense(), x[_cell_index(res["obs_names"])])  # type: ignore[union-attr]

    check()
    model._compact_pool()
    assert len(model._pool) == model.total_cells  # only live cells remain
    check()


def test_plaid_sketch_target_n_cells_validated():
    var_names = np.array(["gene_0"])
    with pytest.raises(ValueError, match="can never be reached"):
        StreamingPlaidGeometricSketch(var_names, target_voxels=10, max_cells_per_bucket=2, target_n_cells=21)
    StreamingPlaidGeometricSketch(var_names, target_voxels=10, max_cells_per_bucket=2, target_n_cells=20)


def test_plaid_sketch_fit_with_target_n_cells(tmp_path):
    loader, var_names = _make_loader(n=300, g=6)
    model = StreamingPlaidGeometricSketch(var_names, target_voxels=120, max_cells_per_bucket=2, target_n_cells=60)
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    assert model.total_cells > 60
    assert len(model.sketch_obs_names) == len(set(model.sketch_obs_names)) == 60
    # The selection is deterministic, so asking again (as the api tool does) gives the same cells.
    np.testing.assert_array_equal(model.sketch_obs_names, model.get_reservoir(max_cells=60)["obs_names"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plaid_sketch_cuda_matches_cpu():
    rng = np.random.default_rng(0)
    n, dim = 3000, 20
    z = torch.from_numpy((rng.standard_normal((n, dim)) * 3 / np.sqrt(np.arange(1, dim + 1))).astype(np.float32))
    names = np.array([f"cell_{i}" for i in range(n)])
    metadata = rng.integers(0, 3, n)
    var_names = np.array([f"d{i}" for i in range(dim)])

    def run(device: str) -> tuple[dict, np.ndarray]:
        model = StreamingPlaidGeometricSketch(var_names, target_voxels=200, max_cells_per_bucket=2, seed=1).to(device)
        for start in range(0, n, 100):
            batch = slice(start, start + 100)
            model.update(z[batch].to(device), names[batch], metadata[batch])
        reservoir_obs_names = model.get_reservoir(max_cells=100)["obs_names"]
        assert isinstance(reservoir_obs_names, np.ndarray)
        return model._voxel_summary(), reservoir_obs_names

    cpu_voxels, cpu_cells = run("cpu")
    cuda_voxels, cuda_cells = run("cuda")
    assert cpu_voxels == cuda_voxels
    np.testing.assert_array_equal(cpu_cells, cuda_cells)


# ----------------------------------------------------------------------
# Shared base-class behaviour: get_reservoir max_cells cap
# ----------------------------------------------------------------------


def test_get_reservoir_max_cells_cap():
    data, var_names, obs_names, _ = _make_plaid_data()
    # Use a large-enough bucket cap so all 14 cells are retained.
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=10, store_cell_data=True)
    model.update(torch.from_numpy(data), obs_names)

    full = model.get_reservoir(return_cell_data=True)
    assert len(full["obs_names"]) == model.total_cells

    capped = model.get_reservoir(return_cell_data=True, max_cells=5, seed=42)
    assert len(capped["obs_names"]) == 5
    assert capped["x_ng"].shape[0] == 5  # type: ignore[union-attr]

    # Seeded calls are reproducible.
    capped2 = model.get_reservoir(return_cell_data=True, max_cells=5, seed=42)
    np.testing.assert_array_equal(capped["obs_names"], capped2["obs_names"])

    # No downsampling when total <= max_cells.
    nocap = model.get_reservoir(max_cells=1000)
    assert len(nocap["obs_names"]) == model.total_cells


# ----------------------------------------------------------------------
# Shared base-class behaviour: apply_bucket_filters
# ----------------------------------------------------------------------


def test_apply_bucket_filters_density():
    data, var_names, obs_names, _ = _make_plaid_data()
    # voxels: (0, 0) → 6 seen, (2, 2) → 6 seen, (5, 5) → 2 seen
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=6)
    _pin_voxel_grid(model, dim=len(var_names))
    model.update(torch.from_numpy(data), obs_names)

    assert set(model._voxel_summary()) == {(0, 0), (2, 2), (5, 5)}

    model.apply_bucket_filters(min_cells_per_bucket=5)

    assert set(model._voxel_summary()) == {(0, 0), (2, 2)}

    # The lookup index was rebuilt: new cells still find the surviving voxels (and the pruned one is empty again).
    model.update(torch.tensor([[0.5, 0.5], [5.5, 5.5]]), np.array(["late_a", "late_b"]))
    assert {k: v["seen"] for k, v in model._voxel_summary().items()} == {(0, 0): 7, (2, 2): 6, (5, 5): 1}


def test_apply_bucket_filters_diversity():
    data, var_names, obs_names, metadata = _make_plaid_data()
    # metadata: (0,0) → only category 0; (2,2) → categories 0 and 1; (5,5) → only 0
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=6)
    _pin_voxel_grid(model, dim=len(var_names))
    model.update(torch.from_numpy(data), obs_names, metadata)

    model.apply_bucket_filters(min_metadata_diversity=2)

    voxels = model._voxel_summary()
    assert set(voxels) == {(2, 2)}
    assert voxels[(2, 2)]["metadata"] == {0, 1}


def test_apply_bucket_filters_no_metadata_raises():
    data, var_names, obs_names, _ = _make_plaid_data()
    model = StreamingPlaidGeometricSketch(var_names)
    model.update(torch.from_numpy(data), obs_names)  # no metadata_n

    with pytest.raises(ValueError, match="no metadata was tracked"):
        model.apply_bucket_filters(min_metadata_diversity=2)


# ----------------------------------------------------------------------
# Hyperplane: metadata support
# ----------------------------------------------------------------------


def test_hyperplane_metadata_tracked():
    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.standard_normal((20, 6)).astype(np.float32))
    var_names = np.array([f"gene_{i}" for i in range(6)])
    obs_names = np.array([f"cell_{i}" for i in range(20)])
    # Two categories: cells 0-9 → 0, cells 10-19 → 1
    metadata = np.array([0] * 10 + [1] * 10)

    model = StreamingHyperplaneGeometricSketch(var_names, n_bits=2, max_cells_per_bucket=10)
    model._lazy_init(data)
    model.update(data, obs_names, metadata)

    # Every occupied bucket should have a metadata set.
    assert all(isinstance(v, set) for v in model._bucket_metadata.values())
    # At least one bucket should have seen both categories (since data spans all buckets with n_bits=2).
    assert any(len(v) > 0 for v in model._bucket_metadata.values())


def test_hyperplane_apply_bucket_filters_diversity():
    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.standard_normal((40, 6)).astype(np.float32))
    var_names = np.array([f"gene_{i}" for i in range(6)])
    obs_names = np.array([f"cell_{i}" for i in range(40)])
    # Only category 0 — diversity filter should prune all buckets.
    metadata = np.zeros(40, dtype=int)

    model = StreamingHyperplaneGeometricSketch(var_names, n_bits=2, max_cells_per_bucket=10)
    model._lazy_init(data)
    model.update(data, obs_names, metadata)

    model.apply_bucket_filters(min_metadata_diversity=2)
    assert model.total_cells == 0


# ----------------------------------------------------------------------
# sketch_obs_names and checkpoint round-trip
# ----------------------------------------------------------------------


def test_sketch_obs_names_populated_after_training(tmp_path):
    loader, var_names = _make_plaid_loader()
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2, min_cells_per_bucket=5)
    module = CellariumModule(model=model)
    _pin_voxel_grid(model, dim=len(var_names), module=module)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    # sketch_obs_names is set at end of training after filtering.
    assert isinstance(model.sketch_obs_names, np.ndarray)
    assert len(model.sketch_obs_names) == model.total_cells > 0
    np.testing.assert_array_equal(model.sketch_obs_names, model.get_reservoir()["obs_names"])


def test_checkpoint_contains_sketch_obs_names(tmp_path):
    loader, var_names = _make_plaid_loader()
    model = StreamingPlaidGeometricSketch(var_names, max_cells_per_bucket=2, min_cells_per_bucket=5)
    module = CellariumModule(model=model)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(module, train_dataloaders=loader)

    ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    raw = torch.load(ckpt_path, weights_only=False)

    assert "sketch_state" in raw
    assert "sketch_obs_names" in raw["sketch_state"]
    np.testing.assert_array_equal(raw["sketch_state"]["sketch_obs_names"], model.sketch_obs_names)


def test_checkpoint_restores_bucket_state_and_sketch_obs_names(tmp_path):
    loader, var_names = _make_plaid_loader()
    model = StreamingPlaidGeometricSketch(
        var_names,
        max_cells_per_bucket=2,
        min_cells_per_bucket=5,
        store_cell_data=True,
    )
    module = CellariumModule(model=model)
    _pin_voxel_grid(model, dim=len(var_names), module=module)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        default_root_dir=tmp_path,
        enable_checkpointing=True,
    )
    trainer.fit(module, train_dataloaders=loader)

    assert isinstance(model.sketch_obs_names, np.ndarray)
    obs_names_after_training = model.sketch_obs_names.copy()
    total_cells_after_training = model.total_cells

    # Load the checkpoint into a fresh model.
    ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    model2 = StreamingPlaidGeometricSketch(
        var_names, max_cells_per_bucket=2, min_cells_per_bucket=5, store_cell_data=True
    )
    module2 = CellariumModule.load_from_checkpoint(ckpt_path, model=model2)

    torch.testing.assert_close(model2.voxel_size, model.voxel_size)
    torch.testing.assert_close(model2.voxel_offset, model.voxel_offset)

    assert module2.model.total_cells == total_cells_after_training
    np.testing.assert_array_equal(module2.model.sketch_obs_names, obs_names_after_training)
    np.testing.assert_array_equal(module2.model.get_reservoir()["obs_names"], obs_names_after_training)


def test_plaid_checkpoint_resumes_training_identically():
    rng = np.random.default_rng(0)
    n, dim = 1200, 8
    z = torch.from_numpy((rng.standard_normal((n, dim)) * 3 / np.sqrt(np.arange(1, dim + 1))).astype(np.float32))
    names = np.array([f"cell_{i}" for i in range(n)])
    metadata = rng.integers(0, 3, n)
    var_names = np.array([f"d{i}" for i in range(dim)])

    def make() -> StreamingPlaidGeometricSketch:
        return StreamingPlaidGeometricSketch(
            var_names, target_voxels=100, max_cells_per_bucket=2, store_cell_data=True, seed=3
        )

    def feed(model: StreamingPlaidGeometricSketch, start: int, stop: int) -> None:
        for s in range(start, stop, 100):
            model.update(z[s : s + 100], names[s : s + 100], metadata[s : s + 100])

    original = make()
    feed(original, 0, 600)
    checkpoint: dict = {}
    original.on_save_checkpoint(checkpoint)
    checkpoint["state_dict"] = original.state_dict()

    restored = make()
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"])
    assert restored._voxel_summary() == original._voxel_summary()

    # Both continue the stream (coarsening further) and stay identical, down to the cell data.
    feed(original, 600, n)
    feed(restored, 600, n)
    assert original._coarsen_event_count > 0
    assert restored._voxel_summary() == original._voxel_summary()
    torch.testing.assert_close(restored.voxel_size, original.voxel_size)
    torch.testing.assert_close(
        restored.get_reservoir(return_cell_data=True)["x_ng"].to_dense(),  # type: ignore[union-attr]
        original.get_reservoir(return_cell_data=True)["x_ng"].to_dense(),  # type: ignore[union-attr]
    )
