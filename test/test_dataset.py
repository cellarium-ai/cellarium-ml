import math
import os

import common
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from anndata import AnnData

from scvid.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
    collate_fn,
)
from scvid.train import DummyTrainingPlan


@pytest.fixture(params=[[3, 6, 9, 12], [4, 8, 12], [4, 8, 11]])  # limits
def dadc(tmp_path, request):
    limits = request.param
    n_cell = limits[-1]
    g_gene = 1

    X = np.arange(n_cell).reshape(n_cell, g_gene)
    adata = AnnData(X, dtype=np.float32)
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
@pytest.mark.parametrize(
    "num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"]
)
@pytest.mark.parametrize(
    "batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"]
)
def test_iterable_dataset(dadc, shuffle, num_workers, batch_size):
    n_obs = len(dadc)
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc, batch_size=batch_size, shuffle=shuffle, test_mode=True
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


@pytest.mark.parametrize(
    "shuffle", [False, True][:1], ids=["no shuffle", "shuffle"][:1]
)
@pytest.mark.parametrize(
    "num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"]
)
@pytest.mark.parametrize(
    "batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"]
)
def test_iterable_dataset_multigpu(dadc, shuffle, num_workers, batch_size):
    n_obs = len(dadc)
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc, batch_size=batch_size, shuffle=shuffle, test_mode=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # fit
    model = common.TestModule()
    training_plan = DummyTrainingPlan(model)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        max_epochs=1,  # one pass
        log_every_n_steps=1,  # to suppress logger warnings
    )
    trainer.fit(training_plan, train_dataloaders=data_loader)
    import pdb

    pdb.set_trace()

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
