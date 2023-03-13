import math
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from anndata import AnnData

from scvid.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from scvid.data.util import collate_fn, get_rank_and_num_replicas
from scvid.module import GatherLayer
from scvid.train import DummyTrainingPlan

# RuntimeError: Too many open files. Communication with the workers is no longer possible.
# Please increase the limit using `ulimit -n` in the shell or change the sharing strategy
# by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
torch.multiprocessing.set_sharing_strategy("file_system")


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.iter_data = []

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Iterable, dict]:
        return (), tensor_dict

    def forward(self, **batch):
        num_replicas = get_rank_and_num_replicas()[1]
        if num_replicas > 1:
            for key, value in batch.items():
                batch[key] = torch.cat(GatherLayer.apply(value), dim=0)
        self.iter_data.append(batch)


@pytest.fixture(params=[[3, 6, 9, 12], [4, 8, 12], [4, 8, 11]])  # limits
def dadc(tmp_path, request):
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


@pytest.mark.parametrize("shuffle", [False, True], ids=["no shuffle", "shuffle"])
@pytest.mark.parametrize(
    "num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"]
)
@pytest.mark.parametrize(
    "batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"]
)
@pytest.mark.parametrize("drop_last", [False, True], ids=["no drop last", "drop last"])
def test_iterable_dataset_multi_device(
    dadc, shuffle, num_workers, batch_size, drop_last
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    n_obs = len(dadc)
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
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
    model = TestModule()
    training_plan = DummyTrainingPlan(model)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=devices,
        max_epochs=1,  # one pass
        log_every_n_steps=1,  # to suppress logger warnings
        strategy="ddp",
    )
    trainer.fit(training_plan, data_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    data_loader = model.iter_data

    actual_idx = list(int(i) for batch in data_loader for i in batch["X"])
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
