# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import time
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils._pytree import tree_leaves
from torch.utils.data import DataLoader

from cellarium.ml.api import cellariumdata
from cellarium.ml.api.tools import pca
from cellarium.ml.api.utils import shutdown_trainer_workers

psutil = pytest.importorskip("psutil")

SCATTERED = np.arange(0, 40, 3)  # 14 cells, in every shard of the data


class FakeIterator:
    """What a dataloader with persistent workers keeps after an epoch."""

    def __init__(self) -> None:
        self.shutdowns = 0

    def _shutdown_workers(self) -> None:
        self.shutdowns += 1


def fake_loader(iterator: FakeIterator | None = None, inner: DataLoader | None = None) -> DataLoader:
    """A dataloader that has ``iterator`` (as one with persistent workers does), and wraps ``inner`` in its dataset (as
    the dataloaders of a view do: the workers are those of the loader inside)."""
    loader: DataLoader = DataLoader(SimpleNamespace(loader=inner))  # type: ignore[arg-type]
    loader._iterator = iterator  # type: ignore[assignment]
    return loader


def test_shutdown_trainer_workers_stops_the_workers_of_the_loaders_of_the_trainer_and_of_the_loaders_inside():
    plain, inside = FakeIterator(), FakeIterator()
    plain_loader = fake_loader(plain)
    inner_loader = fake_loader(inside)
    wrapping_loader = fake_loader(None, inner=inner_loader)
    trainer = SimpleNamespace(
        train_dataloader=[plain_loader, {"view": wrapping_loader}], val_dataloaders=wrapping_loader
    )

    shutdown_trainer_workers(trainer)  # type: ignore[arg-type]

    assert plain.shutdowns == 1 and inside.shutdowns == 1  # once each, though the loaders are listed twice
    assert plain_loader._iterator is None  # the loaders start new workers if they are iterated again
    assert inner_loader._iterator is None

    shutdown_trainer_workers(trainer)  # type: ignore[arg-type]
    assert plain.shutdowns == 1 and inside.shutdowns == 1  # nothing left to stop


def test_shutdown_trainer_workers_ignores_what_it_does_not_know():
    shutdown_trainer_workers(SimpleNamespace())  # type: ignore[arg-type]
    shutdown_trainer_workers(SimpleNamespace(train_dataloader=None, val_dataloaders=[object()]))  # type: ignore[arg-type]
    shutdown_trainer_workers(SimpleNamespace(train_dataloader=fake_loader(None)))  # type: ignore[arg-type]

    class NoShutdown:
        pass

    loader = fake_loader(NoShutdown())  # type: ignore[arg-type]
    shutdown_trainer_workers(SimpleNamespace(train_dataloader=loader))  # type: ignore[arg-type]
    assert loader._iterator is None  # still forgotten

    class Broken:
        @property
        def train_dataloader(self):
            raise RuntimeError("not available")

    shutdown_trainer_workers(Broken())  # type: ignore[arg-type]


def _worker_pids(loader: DataLoader) -> set[int]:
    """The processes of the persistent workers of a dataloader (or of the one inside it), if they have been started."""
    inner = getattr(loader.dataset, "loader", None)
    if isinstance(inner, DataLoader):
        return _worker_pids(inner)
    iterator = getattr(loader, "_iterator", None)
    return set() if iterator is None else {worker.pid for worker in iterator._workers}


def _alive(pids: set[int], timeout: float = 10.0) -> set[int]:
    """The processes of ``pids`` that have not exited after up to ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    alive = pids
    while alive and time.monotonic() < deadline:
        alive = {pid for pid in alive if psutil.pid_exists(pid) and psutil.Process(pid).status() != "zombie"}
        time.sleep(0.05)
    return alive


def test_a_fit_over_a_view_stops_the_workers_of_its_dataloaders(h5ad_cdata, monkeypatch):
    # Lightning stops the persistent workers of the dataloaders that it holds, but a view's dataloader holds another
    # one (with the workers) inside, which it does not know of
    view = h5ad_cdata[SCATTERED]
    view.datamodule.num_workers = 1
    view.datamodule.persistent_workers = True
    seen: list[tuple[set[int], set[int]]] = []  # the workers before the fit's loaders are shut down, those alive then
    shutdown = cellariumdata.shutdown_trainer_workers

    def spy(trainer) -> None:
        workers = {pid for loader in tree_leaves(trainer.train_dataloader) for pid in _worker_pids(loader)}
        seen.append((workers, {pid for pid in workers if psutil.pid_exists(pid)}))
        shutdown(trainer)

    monkeypatch.setattr(cellariumdata, "shutdown_trainer_workers", spy)

    pca(view, n_components=3, zscore=False, accelerator="cpu")

    ((workers, alive_before),) = seen
    assert len(workers) == 1 and alive_before == workers  # the fit left its worker running
    assert _alive(workers) == set()
