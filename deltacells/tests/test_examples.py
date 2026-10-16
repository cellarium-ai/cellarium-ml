# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import sys

import numpy as np
import pytest
from conftest import make_counts

torch = pytest.importorskip("torch")

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load(name):
    sys.path.insert(0, EXAMPLES)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(EXAMPLES, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(EXAMPLES)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from deltacells import DatasetWriter

    root = str(tmp_path_factory.mktemp("ex") / "ds")
    full = make_counts(470, 60, density=0.2, seed=60)
    with DatasetWriter(root, n_genes=60, tile_size=100, level=3, n_chunks=2) as w:
        for lo in range(0, 470, 100):
            w.add_tile(full[lo : lo + 100])
    return root, full


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_reference_loader_covers_every_cell_exactly_once(built, shuffle, drop_last):
    from deltacells import open_dataset

    root, full = built
    loader_mod = _load("reference_loader")
    ds = open_dataset(root, max_cached_tiles=1)
    it = loader_mod.TileShuffleDataset(ds, 64, shuffle=shuffle, seed=3, drop_last=drop_last, shared_memory=False)
    rows = []
    for b in it:
        csr = loader_mod.to_torch_csr(b, 60)
        rows.append(csr.to_dense().numpy())
    got = np.vstack(rows)
    assert len(got) == (470 // 64) * 64 if drop_last else len(got) == 470
    dense = full.toarray()
    if not drop_last:
        # every original row appears exactly once (rows are unique with overwhelming probability)
        key = lambda a: sorted(map(bytes, a.astype(np.float32)))  # noqa: E731
        assert key(got) == key(dense)
    if not shuffle and not drop_last:
        assert np.array_equal(got, dense)


def test_reference_loader_epochs_differ_and_are_reproducible(built):
    from deltacells import open_dataset

    root, _ = built
    loader_mod = _load("reference_loader")

    def first_batch(epoch):
        ds = open_dataset(root, max_cached_tiles=1)
        it = loader_mod.TileShuffleDataset(ds, 32, seed=1, shared_memory=False)
        it.set_epoch(epoch)
        return next(iter(it))["values"].numpy().copy()

    assert np.array_equal(first_batch(0), first_batch(0))
    assert not np.array_equal(first_batch(0), first_batch(1))


def test_reference_loader_splits_tiles_across_replicas(built):
    from deltacells import open_dataset

    root, _ = built
    loader_mod = _load("reference_loader")
    ds = open_dataset(root)
    parts = [loader_mod.TileShuffleDataset(ds, 8, seed=2, num_replicas=3, rank=r).my_tiles() for r in range(3)]
    assert sorted(np.concatenate(parts).tolist()) == list(range(ds.n_tiles))


def test_example_script_runs(capsys):
    mod = _load("dataloader_example")
    assert mod.main(["--workers", "0", "--max-batches", "3", "--batch-size", "50"]) == 0
    out = capsys.readouterr().out
    assert "2000 cells x 500 genes in 4 tiles of 500" in out and "first batch: (50, 500)" in out
