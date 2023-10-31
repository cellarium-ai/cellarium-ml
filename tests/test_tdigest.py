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
from cellarium.ml.models import TDigest
from cellarium.ml.transforms import NormalizeTotal
from cellarium.ml.utilities.data import AnnDataField, collate_fn, pandas_to_numpy
from tests.common import BoringDataset, requires_crick


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
    "transforms",
    [[NormalizeTotal(target_count=10_000, eps=0)], []],
)
def test_tdigest_multi_device(
    adata: AnnData,
    dadc: DistributedAnnDataCollection,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    transforms: list[torch.nn.Module],
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    batch_size = batch_size // devices

    # prepare dataloader
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={
            "x_ng": AnnDataField("X"),
            "feature_g": AnnDataField("var_names", convert_fn=pandas_to_numpy),
        },
        batch_size=batch_size,
        shuffle=shuffle,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # fit
    model = TDigest(feature_schema=dadc.var_names)
    module = CellariumModule(model, transforms=transforms)
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,  # one pass
    )
    trainer.fit(module, train_dataloaders=data_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    # actual median
    actual_median_g = model.median_g

    # expected median
    batch = {"x_ng": torch.from_numpy(adata.X)}
    for transform in transforms:
        batch = transform(**batch)
    x = batch["x_ng"]
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
        BoringDataset(
            np.random.randn(n, g),
            np.array([f"gene_{i}" for i in range(g)]),
        ),
        collate_fn=collate_fn,
    )
    # model
    init_args = {"feature_schema": [f"gene_{i}" for i in range(g)]}
    model = TDigest(**init_args)  # type: ignore[arg-type]
    config = {
        "model": {
            "model": {
                "class_path": "cellarium.ml.models.TDigest",
                "init_args": init_args,
            },
        }
    }
    module = CellariumModule(model, config=config)
    # trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=devices,
        max_epochs=1,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(module, train_dataloaders=train_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    # load model from checkpoint
    ckpt_path = tmp_path / f"lightning_logs/version_0/checkpoints/epoch=0-step={math.ceil(n / devices)}.ckpt"
    assert ckpt_path.is_file()
    loaded_model = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    assert isinstance(loaded_model, TDigest)
    np.testing.assert_allclose(model.median_g, loaded_model.median_g)
