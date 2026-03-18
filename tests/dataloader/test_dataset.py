# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes, collate_fn
from tests.common import BoringModel

# RuntimeError: Too many open files. Communication with the workers is no longer possible.
# Please increase the limit using `ulimit -n` in the shell or change the sharing strategy
# by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
torch.multiprocessing.set_sharing_strategy("file_system")


@pytest.fixture()
def obs() -> pd.DataFrame:
    n_cell = 20
    obs = pd.DataFrame(
        data={
            "batch": np.concatenate(
                [
                    np.zeros(3),
                    np.ones(
                        n_cell,
                    ),
                ]
            ).astype(int)[:n_cell],
            "assay": np.array(["10x", "dropseq"] * (n_cell // 2 + 1))[:n_cell],
        }
    )
    obs["batch"] = obs["batch"].astype("category")
    obs["assay"] = obs["assay"].astype("category")
    return obs


@pytest.fixture(params=[[3, 6, 9, 12], [4, 8, 12], [4, 8, 11]])  # limits
def dadc(tmp_path: Path, obs: pd.DataFrame, request: pytest.FixtureRequest) -> DistributedAnnDataCollection:
    limits = request.param
    n_cell = limits[-1]
    g_gene = 1

    assert n_cell <= len(obs), "the pytest fixture called obs() is too small a dataframe. increase its n_cell value"

    X = np.arange(n_cell).reshape(n_cell, g_gene)
    # first 3 "batch" 0, rest 1; even "assay" 10x, odd dropseq
    adata = AnnData(X, dtype=X.dtype, obs=obs[:n_cell])
    for i, limit in enumerate(zip([0] + limits, limits)):
        sliced_adata = adata[slice(*limit)]
        # the following two lines exist because of https://github.com/scverse/anndata/issues/1710
        sliced_adata.obs["batch"] = sliced_adata.obs["batch"].cat.set_categories(adata.obs["batch"].cat.categories)
        sliced_adata.obs["assay"] = sliced_adata.obs["assay"].cat.set_categories(adata.obs["assay"].cat.categories)
        sliced_adata.write(os.path.join(tmp_path, f"adata.00{i}.h5ad"))

    # distributed anndata
    filenames = str(os.path.join(tmp_path, f"adata.{{000..00{len(limits) - 1}}}.h5ad"))

    dadc = DistributedAnnDataCollection(
        filenames,
        limits,
        max_cache_size=3,
        cache_size_strictly_enforced=True,
    )
    return dadc


def test_iterable_dataset_anndatafields(dadc: DistributedAnnDataCollection, obs: pd.DataFrame):
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={
            "x_ng": AnnDataField("X"),
            "batch_n": AnnDataField("obs", key="batch", convert_fn=categories_to_codes),
            "assay_n": AnnDataField("obs", key="assay", convert_fn=categories_to_codes),
            "batch_assay_n2": AnnDataField("obs", key=["batch", "assay"], convert_fn=categories_to_codes),
        },
        batch_size=1,
        shuffle=False,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
    )

    for i, batch in enumerate(data_loader):
        assert "x_ng" in batch
        assert batch["x_ng"].shape == (1, 1)
        assert batch["x_ng"].dtype == torch.int64
        assert batch["batch_n"] == torch.tensor([obs["batch"].cat.codes[i]])
        assert batch["assay_n"] == torch.tensor([obs["assay"].cat.codes[i]])
        torch.testing.assert_close(
            torch.cat([batch["batch_n"], batch["assay_n"]]).unsqueeze(0),
            batch["batch_assay_n2"],
        )


@pytest.mark.parametrize("iteration_strategy", ["same_order", "cache_efficient"])
@pytest.mark.parametrize("shuffle", [False, True], ids=["no shuffle", "shuffle"])
@pytest.mark.parametrize("num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"])
@pytest.mark.parametrize("batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"])
@pytest.mark.parametrize(
    "drop_incomplete_batch", [False, True], ids=["no drop incomplete batch", "drop incomplete batch"]
)
@pytest.mark.parametrize("start_idx", [0, 2])
@pytest.mark.parametrize("end_idx", [10, None])
def test_iterable_dataset(
    dadc: DistributedAnnDataCollection,
    iteration_strategy: Literal["same_order", "cache_efficient"],
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    drop_incomplete_batch: bool,
    start_idx: int | None,
    end_idx: int | None,
):
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        iteration_strategy=iteration_strategy,
        batch_keys={"x_ng": AnnDataField("X")},
        batch_size=batch_size,
        shuffle=shuffle,
        drop_incomplete_batch=drop_incomplete_batch,
        start_idx=start_idx,
        end_idx=end_idx,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    all_batches = list(data_loader)
    miss_counts = list(int(batch["miss_count"]) for batch in all_batches for _ in batch["x_ng"])
    actual_idx = list(int(i) for batch in all_batches for i in batch["x_ng"])

    worker_ids = list(int(batch["worker_id"]) for batch in all_batches for _ in batch["x_ng"])
    adatas_oidx = np.searchsorted([0] + dadc.limits, actual_idx, side="right")
    for worker in set(worker_ids):
        miss_count = max(c for c, w in zip(miss_counts, worker_ids) if w == worker)
        assert miss_count == len(set([o for o, w in zip(adatas_oidx, worker_ids) if w == worker]))

    n_obs = dataset.end_idx - dataset.start_idx
    expected_idx = list(range(dataset.start_idx, dataset.end_idx))
    expected_len = n_obs
    if drop_incomplete_batch and n_obs % batch_size != 0:
        expected_len = n_obs // batch_size * batch_size
    assert expected_len == len(actual_idx)

    # assert entire dataset is sampled
    if not shuffle and iteration_strategy == "same_order":
        assert expected_idx[:expected_len] == actual_idx
    else:
        if drop_incomplete_batch and n_obs % batch_size != 0:
            assert len(set(expected_idx) - set(actual_idx)) < batch_size
        else:
            assert set(expected_idx) == set(actual_idx)


@pytest.mark.parametrize("iteration_strategy", ["same_order", "cache_efficient"])
@pytest.mark.parametrize("shuffle", [False, True], ids=["no shuffle", "shuffle"])
@pytest.mark.parametrize("num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"])
@pytest.mark.parametrize("batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"])
@pytest.mark.parametrize("drop_last_indices", [False, True], ids=["no drop last indices", "drop last indices"])
@pytest.mark.parametrize(
    "drop_incomplete_batch", [False, True], ids=["no drop incomplete batch", "drop incomplete batch"]
)
@pytest.mark.parametrize("start_idx", [0, 2])
@pytest.mark.parametrize("end_idx", [10, None])
def test_iterable_dataset_multi_device(
    dadc: DistributedAnnDataCollection,
    iteration_strategy: Literal["same_order", "cache_efficient"],
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    drop_last_indices: bool,
    drop_incomplete_batch: bool,
    start_idx: int | None,
    end_idx: int | None,
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        iteration_strategy=iteration_strategy,
        batch_keys={"x_ng": AnnDataField("X")},
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last_indices=drop_last_indices,
        drop_incomplete_batch=drop_incomplete_batch,
        start_idx=start_idx,
        end_idx=end_idx,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # fit
    model = BoringModel()
    module = CellariumModule(model=model)
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

    actual_idx = list(int(i) for batch in model.iter_data for i in batch["x_ng"])
    n_obs = dataset.end_idx - dataset.start_idx
    expected_idx = list(range(dataset.start_idx, dataset.start_idx + n_obs))

    # assert entire dataset is sampled
    if drop_last_indices and n_obs % devices != 0:
        expected_len_per_replica = n_obs // devices
        if drop_incomplete_batch:
            expected_len_per_replica = expected_len_per_replica // batch_size * batch_size
            assert len(set(expected_idx) - set(actual_idx)) < devices * (batch_size + 1)
        else:
            assert len(set(expected_idx) - set(actual_idx)) < devices
    else:
        expected_len_per_replica = math.ceil(n_obs / devices)
        if drop_incomplete_batch:
            expected_len_per_replica = expected_len_per_replica // batch_size * batch_size
            assert len(set(expected_idx) - set(actual_idx)) < (devices * batch_size - devices + 1)
        else:
            assert set(expected_idx) == set(actual_idx)
    expected_len = expected_len_per_replica * devices
    assert expected_len == len(actual_idx)


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
        batch_keys={"x_ng": AnnDataField("X")},
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
    module = CellariumModule(model=model)
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


@pytest.mark.parametrize("iteration_strategy", ["same_order", "cache_efficient"])
@pytest.mark.parametrize("shuffle", [False, True], ids=["no shuffle", "shuffle"])
@pytest.mark.parametrize("num_workers", [0, 1, 2], ids=["zero workers", "one worker", "two workers"])
@pytest.mark.parametrize("batch_size", [1, 2, 3], ids=["batch size 1", "batch size 2", "batch size 3"])
@pytest.mark.parametrize(
    "drop_incomplete_batch", [False, True], ids=["no drop incomplete batch", "drop incomplete batch"]
)
@pytest.mark.parametrize("persistent_workers", [False, True], ids=["non-persistent workers", "persistent workers"])
@pytest.mark.parametrize("resume_step", [1, 4, 5])
def test_load_from_checkpoint(
    dadc: DistributedAnnDataCollection,
    iteration_strategy: Literal["same_order", "cache_efficient"],
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
    batch_size: int,
    drop_incomplete_batch: bool,
    tmp_path: Path,
    resume_step: int,
):
    if persistent_workers and num_workers == 0:
        pytest.skip("persistent_workers requires num_workers > 0")

    datamodule1 = CellariumAnnDataDataModule(
        dadc=dadc,
        batch_keys={"x_ng": AnnDataField("X")},
        batch_size=batch_size,
        iteration_strategy=iteration_strategy,
        shuffle=shuffle,
        drop_incomplete_batch=drop_incomplete_batch,
        test_mode=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    module1 = CellariumModule(model=BoringModel())
    trainer1 = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,
        logger=False,
        callbacks=[pl.callbacks.ModelCheckpoint(every_n_train_steps=1, save_top_k=-1)],
        default_root_dir=tmp_path,
    )
    trainer1.fit(module1, datamodule1)

    # resume from checkpoint
    datamodule2 = CellariumAnnDataDataModule(
        dadc=dadc,
        batch_keys={"x_ng": AnnDataField("X")},
        batch_size=batch_size,
        iteration_strategy=iteration_strategy,
        shuffle=shuffle,
        drop_incomplete_batch=drop_incomplete_batch,
        test_mode=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    module2 = CellariumModule(model=BoringModel())
    trainer2 = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,
        logger=False,
    )
    try:
        ckpt_path = tmp_path / f"checkpoints/epoch=0-step={resume_step}.ckpt"
        trainer2.fit(module2, datamodule2, ckpt_path=ckpt_path)
    except FileNotFoundError:
        ckpt_path = tmp_path / f"checkpoints/epoch=1-step={resume_step}.ckpt"
        trainer2.fit(module2, datamodule2, ckpt_path=ckpt_path)

    iter_data1 = collate_fn(module1.model.iter_data)
    iter_data2 = collate_fn(module2.model.iter_data)

    torch.testing.assert_close(iter_data1["x_ng"], iter_data2["x_ng"])
