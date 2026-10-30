# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from cellarium.ml import CellariumAnnDataDataModule
from cellarium.ml.api import _rebatch
from cellarium.ml.api._rebatch import (
    ShardChunkDataModule,
    ShardChunkDataset,
    concatenate_batches,
    slice_batch,
)
from cellarium.ml.api._view_collections import ViewCollection
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.utilities.data import to_torch_sparse_csr

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`

SCATTERED = np.arange(0, 40, 3)  # 14 cells, in every shard of the data


def _batch(x: np.ndarray, first_name: int) -> dict:
    return {
        "x_ng": to_torch_sparse_csr(sp.csr_matrix(x)),
        "dense_n": torch.as_tensor(x.sum(axis=1)),
        "names_n": np.array([f"cell{first_name + i}" for i in range(len(x))]),
        "obs": {"codes_n": np.arange(len(x)) + first_name},
        "var_names_g": np.array(["a", "b", "c"]),
    }


def _dense(x: torch.Tensor) -> np.ndarray:
    return x.to_dense().numpy() if x.layout != torch.strided else x.numpy()


def test_batches_are_put_together_and_cut_along_the_cells():
    rng = np.random.default_rng(0)
    x = (rng.random((9, 3)) < 0.5) * rng.integers(1, 5, size=(9, 3)).astype(np.float32)
    batches = [_batch(x[:2], 0), _batch(x[2:7], 2), _batch(x[7:], 7)]

    whole = concatenate_batches(batches)
    np.testing.assert_array_equal(_dense(whole["x_ng"]), x)
    np.testing.assert_array_equal(whole["dense_n"].numpy(), x.sum(axis=1))
    assert whole["names_n"].tolist() == [f"cell{i}" for i in range(9)]
    assert whole["obs"]["codes_n"].tolist() == list(range(9))
    assert whole["var_names_g"].tolist() == ["a", "b", "c"]  # the same for every cell, so not concatenated

    for start, stop in [(0, 9), (0, 4), (3, 4), (2, 9), (5, 5)]:
        part = slice_batch(whole, start, stop)
        np.testing.assert_array_equal(_dense(part["x_ng"]), x[start:stop])
        assert part["names_n"].tolist() == [f"cell{i}" for i in range(start, stop)]
        assert part["obs"]["codes_n"].tolist() == list(range(start, stop))
        assert part["var_names_g"].tolist() == ["a", "b", "c"]


def test_a_single_batch_is_returned_as_it_is():
    batch = _batch(np.eye(3, dtype=np.float32), 0)
    assert concatenate_batches([batch]) is batch


def test_scipy_sparse_lists_and_coo_fields_are_put_together_too():
    first = {"x_ng": sp.csr_matrix(np.eye(2)), "tags_n": ["a", "b"]}
    second = {"x_ng": sp.csr_matrix(np.ones((3, 2))), "tags_n": ["c", "d", "e"]}
    whole = concatenate_batches([first, second])
    assert whole["x_ng"].shape == (5, 2) and whole["tags_n"] == ["a", "b", "c", "d", "e"]
    part = slice_batch(whole, 1, 4)
    assert part["x_ng"].shape == (3, 2) and part["tags_n"] == ["b", "c", "d"]

    coo = torch.sparse_coo_tensor([[0, 2, 3], [1, 0, 1]], [1.0, 2.0, 3.0], size=(4, 2))
    sliced = slice_batch({"x_ng": coo}, 2, 4)["x_ng"]
    np.testing.assert_array_equal(sliced.to_dense().numpy(), [[2.0, 0.0], [0.0, 3.0]])


def _datamodule(cdata, cls, positions=SCATTERED, **kwargs):
    view = cdata[positions]
    settings = dict(
        dadc=view.datamodule.dadc,
        batch_keys=dict(view.datamodule.batch_keys),
        batch_size=8,
        iteration_strategy="same_order",
        shuffle=False,
        shuffle_seed=3,
    )
    settings.update(kwargs)
    datamodule = cls(**settings)
    datamodule.prepare_data()
    datamodule.setup("fit")
    return view, datamodule


def _read(datamodule):
    batches = list(datamodule.train_dataloader())
    x = [_dense(batch["x_ng"]) for batch in batches]
    names = [name for batch in batches for name in batch["obs_names_n"]]
    return [len(part) for part in x], np.concatenate(x), np.array(names)


@pytest.mark.parametrize("shuffle", [False, True])
def test_a_view_of_scattered_cells_gives_the_batches_of_the_plain_dataset(cdata, shuffle):
    view, chunked = _datamodule(cdata, ShardChunkDataModule, shuffle=shuffle)
    assert isinstance(view.datamodule.dadc, ViewCollection) and chunked.reads_by_shard()
    assert isinstance(chunked.train_dataset, ShardChunkDataset)
    _, plain = _datamodule(cdata, CellariumAnnDataDataModule, shuffle=shuffle)

    sizes, x, names = _read(chunked)
    plain_sizes, plain_x, plain_names = _read(plain)
    assert sizes == plain_sizes == [8, 6]  # exact batch sizes, with the rest in the last one
    np.testing.assert_array_equal(x, plain_x)
    assert names.tolist() == plain_names.tolist()
    assert len(chunked.train_dataloader()) == 2


def test_the_incomplete_batch_can_be_dropped(cdata):
    _, chunked = _datamodule(cdata, ShardChunkDataModule, drop_incomplete_batch=True)
    sizes, _, _ = _read(chunked)
    assert sizes == [8] and len(chunked.train_dataloader()) == 1


def test_epochs_are_reshuffled_and_each_covers_every_cell_once(cdata):
    view, chunked = _datamodule(cdata, ShardChunkDataModule, shuffle=True)
    loader = chunked.train_dataloader()
    orders = []
    for epoch in range(2):
        loader.dataset.set_epoch(epoch)
        orders.append([name for batch in loader for name in batch["obs_names_n"]])
    assert sorted(orders[0]) == sorted(orders[1]) == sorted(view.obs_names.tolist())
    assert orders[0] != orders[1]


@pytest.mark.parametrize("iteration_strategy", ["same_order", "cache_efficient"])
def test_worker_processes_read_each_cell_once(cdata, iteration_strategy):
    view, chunked = _datamodule(
        cdata, ShardChunkDataModule, iteration_strategy=iteration_strategy, num_workers=2, shuffle=True
    )
    sizes, x, names = _read(chunked)
    assert sizes == [8, 6]
    assert sorted(names.tolist()) == sorted(view.obs_names.tolist())
    expected = np.asarray(view.datamodule.dadc[np.arange(view.n_obs)].X.todense())
    order = [view.obs_names.get_loc(name) for name in names]
    np.testing.assert_array_equal(x, expected[order])


def test_only_views_are_read_by_shard(cdata):
    _, root = _datamodule(cdata, ShardChunkDataModule, positions=np.arange(40))
    assert root.reads_by_shard()  # a view, even of every cell

    plain = CellariumAnnDataDataModule(
        dadc=cdata.datamodule.dadc, batch_keys=dict(cdata.datamodule.batch_keys), batch_size=8
    )
    plain.setup("fit")
    assert not isinstance(plain.train_dataset, ShardChunkDataset)
    assert not isinstance(cdata.datamodule, ShardChunkDataModule)

    in_ram = cdata[SCATTERED].to_ram()  # the cells are in memory: no shards to read
    assert not isinstance(in_ram.datamodule.train_dataset, ShardChunkDataset)


def test_the_datamodule_of_a_view_reads_by_shard_and_predict_batches_cover_the_cells(cdata):
    view = cdata[SCATTERED]
    view.datamodule.batch_size = 8
    assert isinstance(view.datamodule, ShardChunkDataModule)
    positions, names = [], []
    for batch in view.predict_batches(batch_size=8):
        positions.extend(batch["row_indices_n"].tolist())
        names.extend(batch["obs_names_n"])
    # `row_indices_n` are positions in the view, and name the cells that they were read for
    assert sorted(positions) == list(range(view.n_obs))
    assert names == [view.obs_names[p] for p in positions]


def test_the_chunked_dataset_is_for_one_process(cdata, monkeypatch):
    _, chunked = _datamodule(cdata, ShardChunkDataModule)
    monkeypatch.setattr(_rebatch, "get_rank_and_num_replicas", lambda: (0, 2))
    with pytest.raises(NotImplementedError, match="single process"):
        next(iter(chunked.train_dataset))


def _read_last_cell_in_forked_process(collection, queue) -> None:
    queue.put(float(collection[[collection.n_obs - 1]].X.sum()))


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs fork")
def test_a_forked_process_can_read_after_the_parent_has_read(deltacells_uri, deltacells_kwargs):
    # the parent's fetch threads exist once it has read a tile; a dataloader worker forked from it must not wait for
    # them. The worker reads a tile that the parent has not (a tile in the parent's cache needs no fetch)
    cdata = CellariumData.from_deltacells(deltacells_uri, deltacells_kwargs=deltacells_kwargs)
    collection = cdata.datamodule.dadc
    expected = float(collection[[collection.n_obs - 1]].X.sum())
    collection.reset_cache()
    collection[[1, 2]].X  # starts the thread pool in this process

    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_read_last_cell_in_forked_process, args=(collection, queue))
    process.start()
    process.join(timeout=60)
    hung = process.is_alive()
    if hung:
        process.kill()
        process.join()
    assert not hung, "the forked process hung reading a tile"
    assert queue.get(timeout=10) == expected
