# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from anndata import AnnData

from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.data.util import collate_fn
from cellarium.ml.module import TDigest, TDigestFromCLI
from cellarium.ml.train import TrainingPlan
from cellarium.ml.transforms import NormalizeTotal

from .common import TestDataset, requires_crick


@pytest.fixture
def adata():
    n_cell, g_gene = 1000, 5
    rng = np.random.default_rng(1465)
    X = rng.integers(100, size=(n_cell, g_gene))
    return AnnData(X, dtype=X.dtype)


@pytest.fixture
def dadc(adata: AnnData, tmp_path: Path):
    # save anndata files
    limits = [200, 500, 1000]
    for i, limit in enumerate(zip([0] + limits, limits)):
        sliced_adata = adata[slice(*limit)]
        sliced_adata.write(os.path.join(tmp_path, f"adata.00{i}.h5ad"))

    # distributed anndata
    filenames = str(os.path.join(tmp_path, "adata.{000..002}.h5ad"))
    return DistributedAnnDataCollection(
        filenames,
        limits,
        max_cache_size=2,  # allow max_cache_size=2 for IterableDistributedAnnDataCollectionDataset
        cache_size_strictly_enforced=True,
    )


@requires_crick
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("batch_size", [10, 100])
@pytest.mark.parametrize(
    "transform",
    [NormalizeTotal(target_count=10_000, eps=0), None],
)
def test_tdigest_multi_device(
    adata: AnnData,
    dadc: DistributedAnnDataCollection,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    transform: torch.nn.Module | None,
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    batch_size = batch_size // devices

    # prepare dataloader
    dataset = IterableDistributedAnnDataCollectionDataset(dadc, batch_size=batch_size, shuffle=shuffle)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # fit
    model = TDigest(g_genes=dadc.n_vars, transform=transform)
    training_plan = TrainingPlan(model)
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,  # one pass
    )
    trainer.fit(training_plan, train_dataloaders=data_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    # actual median
    actual_median_g = model.median_g

    # expected median
    x = torch.from_numpy(adata.X)
    if transform is not None:
        x = transform(x)
    for i in range(x.shape[1]):
        mask = torch.nonzero(x[:, i])
        expected_median = torch.median(x[mask, i])
        actual_median = actual_median_g[i]
        # assert within 1% accuracy
        np.testing.assert_allclose(expected_median, actual_median, rtol=0.01)


@requires_crick
def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 4, 3
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        TestDataset(np.arange(n * g).reshape(n, g)),
        collate_fn=collate_fn,
    )
    # model
    init_args = {"g_genes": g, "target_count": 10}
    model = TDigestFromCLI(**init_args)
    config = {
        "model": {
            "module": {
                "class_path": "cellarium.ml.module.TDigestFromCLI",
                "init_args": init_args,
            }
        }
    }
    training_plan = TrainingPlan(model, config=config)
    # trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=devices,
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    # load model from checkpoint
    ckpt_path = tmp_path / f"lightning_logs/version_0/checkpoints/epoch=0-step={n}.ckpt"
    assert ckpt_path.is_file()
    loaded_model: TDigestFromCLI = TrainingPlan.load_from_checkpoint(ckpt_path).module
    # assert
    assert isinstance(model.transform, NormalizeTotal)
    assert isinstance(loaded_model.transform, NormalizeTotal)
    assert model.transform.target_count == loaded_model.transform.target_count
    np.testing.assert_allclose(model.median_g, loaded_model.median_g)
