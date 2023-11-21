# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from anndata import AnnData

from cellarium.ml import CellariumModule
from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.models import CellariumModel, GatherLayer
from cellarium.ml.utilities.data import AnnDataField, collate_fn, get_rank_and_num_replicas

# RuntimeError: Too many open files. Communication with the workers is no longer possible.
# Please increase the limit using `ulimit -n` in the shell or change the sharing strategy
# by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
torch.multiprocessing.set_sharing_strategy("file_system")


class BoringModel(CellariumModel):
    """
    This model appends a batch input to an :attr:`iter_data` list at each iteration.
    Its intended use is for testing purposes where batch inputs can be inspected after
    iteration over the dataset with ``Trainer.fit()``. Batch input would typically contain
    feature counts, worker info, torch.distributed info, cache info, etc.
    """

    def __init__(self) -> None:
        super().__init__()
        self.iter_data: list = []
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        return (), tensor_dict

    def forward(self, **batch: torch.Tensor) -> None:
        _, num_replicas = get_rank_and_num_replicas()
        if num_replicas > 1:
            for key, value in batch.items():
                batch[key] = torch.cat(GatherLayer.apply(value), dim=0)
        self.iter_data.append(batch)


@pytest.fixture(params=[[3, 6, 9, 12], [4, 8, 12], [4, 8, 11]])  # limits
def dadc(tmp_path: Path, request: pytest.FixtureRequest):
    limits = request.param
    n_cell = limits[-1]
    g_gene = 1

    X = np.arange(n_cell).reshape(n_cell, g_gene)
    adata = AnnData(X, dtype=X.dtype)
    for i, limit in enumerate(zip([0] + limits, limits)):
        sliced_adata = adata[slice(*limit)]
        sliced_adata.write(os.path.join(tmp_path, f"adata.00{i}.h5ad"))

    # distributed anndata
    filenames = str(os.path.join(tmp_path, f"adata.{{000..00{len(limits)-1}}}.h5ad"))
    dadc = DistributedAnnDataCollection(
        filenames,
        limits,
        max_cache_size=2,
        cache_size_strictly_enforced=True,
    )
    return dadc


@pytest.mark.parametrize("shuffle", [False, True], ids=["no shuffle", "shuffle"])
@pytest.mark.parametrize("num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"])
@pytest.mark.parametrize("batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"])
def test_iterable_dataset(dadc: DistributedAnnDataCollection, shuffle: bool, num_workers: int, batch_size: int):
    n_obs = len(dadc)
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={"X": AnnDataField("X")},
        batch_size=batch_size,
        shuffle=shuffle,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    miss_counts = list(int(i) for batch in data_loader for i in batch["miss_count"])

    if num_workers > 1:
        worker_ids = list(int(i) for batch in data_loader for i in batch["worker_id"])
        for worker in range(num_workers):
            miss_count = max(c for c, w in zip(miss_counts, worker_ids) if w == worker)
            assert miss_count == math.ceil(len(dadc.limits) / num_workers)
    else:
        miss_count = max(miss_counts)
        assert miss_count == len(dadc.limits)

    actual_idx = list(int(i) for batch in data_loader for i in batch["X"])
    expected_idx = list(range(n_obs))

    # assert entire dataset is sampled
    assert len(expected_idx) == len(actual_idx)
    assert set(expected_idx) == set(actual_idx)


@pytest.mark.parametrize("shuffle", [False, True], ids=["no shuffle", "shuffle"])
@pytest.mark.parametrize("num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"])
@pytest.mark.parametrize("batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"])
@pytest.mark.parametrize("drop_last", [False, True], ids=["no drop last", "drop last"])
def test_iterable_dataset_multi_device(
    dadc: DistributedAnnDataCollection,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    drop_last: bool,
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    n_obs = len(dadc)
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={"X": AnnDataField("X")},
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # fit
    model = BoringModel()
    module = CellariumModule(model)
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,  # one pass
    )
    trainer.fit(module, data_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    actual_idx = list(int(i) for batch in model.iter_data for i in batch["X"])
    expected_idx = list(range(n_obs))

    # assert entire dataset is sampled
    if drop_last and n_obs % devices != 0:
        expected_len = (n_obs // devices) * devices
        assert expected_len == len(actual_idx)
        assert set(actual_idx).issubset(expected_idx)
    else:
        expected_len = math.ceil(n_obs / devices) * devices
        assert expected_len == len(actual_idx)
        assert set(expected_idx) == set(actual_idx)


@pytest.mark.parametrize(
    "num_workers,persistent_workers",
    [(0, False), (1, False), (1, True), (2, False), (2, True)],
    ids=[
        "zero workers",
        "one not persistent worker",
        "one persistent worker",
        "two not persistent workers",
        "two persistent workers",
    ],
)
@pytest.mark.parametrize("epochs", [2, 3], ids=["two epochs", "three epochs"])
def test_iterable_dataset_set_epoch_multi_device(
    dadc: DistributedAnnDataCollection,
    num_workers: int,
    persistent_workers: bool,
    epochs: int,
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={"X": AnnDataField("X")},
        batch_size=1,
        shuffle=True,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
    )

    # fit
    model = BoringModel()
    module = CellariumModule(model)
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=epochs,
        strategy="ddp",
    )
    trainer.fit(module, data_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    iter_data = model.iter_data

    actual_epochs = set(int(i) for batch in iter_data for i in batch["epoch"])
    expected_epochs = set(range(epochs))

    assert set(expected_epochs) == set(actual_epochs)
