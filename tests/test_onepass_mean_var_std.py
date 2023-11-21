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
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml import CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
from cellarium.ml.models import OnePassMeanVarStd, OnePassMeanVarStdFromCLI
from cellarium.ml.transforms import Log1p, NormalizeTotal
from cellarium.ml.utilities.data import AnnDataField, collate_fn
from tests.common import BoringDataset


@pytest.fixture
def adata():
    n_cell, g_gene = 10, 5
    rng = np.random.default_rng(1465)
    X = rng.integers(10, size=(n_cell, g_gene))
    return AnnData(X, dtype=X.dtype)


@pytest.fixture
def dadc(adata: AnnData, tmp_path: Path):
    # save anndata files
    limits = [2, 5, 10]
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


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_onepass_mean_var_std_multi_device(
    adata: AnnData,
    dadc: DistributedAnnDataCollection,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # prepare dataloader
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={"X": AnnDataField("X")},
        batch_size=batch_size,
        shuffle=shuffle,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    transform = torch.nn.Sequential(NormalizeTotal(target_count=10_000), Log1p())

    # fit
    model = OnePassMeanVarStd(g_genes=dadc.n_vars, transform=transform)
    module = CellariumModule(model)
    strategy = DDPStrategy(broadcast_buffers=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,  # one pass
        strategy=strategy,  # type: ignore[arg-type]
    )
    trainer.fit(module, train_dataloaders=data_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    # actual mean, var, and std
    actual_mean = model.mean_g
    actual_var = model.var_g
    actual_std = model.std_g

    # expected mean, var, and std
    x = transform(torch.from_numpy(adata.X))
    expected_mean = torch.mean(x, dim=0)
    expected_var = torch.var(x, dim=0, unbiased=False)
    expected_std = torch.std(x, dim=0, unbiased=False)

    np.testing.assert_allclose(expected_mean, actual_mean, atol=1e-5)
    np.testing.assert_allclose(expected_var, actual_var, atol=1e-4)
    np.testing.assert_allclose(expected_std, actual_std, atol=1e-4)


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 3, 2
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(np.arange(n * g).reshape(n, g)),
        collate_fn=collate_fn,
    )
    # model
    init_args = {"g_genes": g, "target_count": 10}
    model = OnePassMeanVarStdFromCLI(**init_args)
    config = {
        "model": {
            "model": {
                "class_path": "cellarium.ml.models.OnePassMeanVarStdFromCLI",
                "init_args": init_args,
            }
        }
    }
    module = CellariumModule(model, config=config)
    # trainer
    strategy = DDPStrategy(broadcast_buffers=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="cpu",
        strategy=strategy,  # type: ignore[arg-type]
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
    loaded_model: OnePassMeanVarStdFromCLI = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    assert isinstance(model.transform, torch.nn.Sequential) and len(model.transform) == 2
    assert isinstance(loaded_model.transform, torch.nn.Sequential) and len(loaded_model.transform) == 2
    assert isinstance(model.transform[0], NormalizeTotal)
    assert isinstance(loaded_model.transform[0], NormalizeTotal)
    assert model.transform[0].target_count == loaded_model.transform[0].target_count
    np.testing.assert_allclose(model.mean_g, loaded_model.mean_g)
    np.testing.assert_allclose(model.var_g, loaded_model.var_g)
    np.testing.assert_allclose(model.std_g, loaded_model.std_g)
