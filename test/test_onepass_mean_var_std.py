# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from anndata import AnnData
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.callbacks import ModuleCheckpoint
from cellarium.ml.data import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
from cellarium.ml.data.util import collate_fn
from cellarium.ml.module import OnePassMeanVarStd, OnePassMeanVarStdFromCLI
from cellarium.ml.train import TrainingPlan
from cellarium.ml.transforms import ZScoreLog1pNormalize

from .common import TestDataset


@pytest.fixture
def adata():
    n_cell, g_gene = (10, 5)
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
    dataset = IterableDistributedAnnDataCollectionDataset(dadc, batch_size=batch_size, shuffle=shuffle)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    transform = ZScoreLog1pNormalize(mean_g=0, std_g=None, perform_scaling=False, target_count=10_000)

    # fit
    model = OnePassMeanVarStd(g_genes=dadc.n_vars, transform=transform)
    training_plan = TrainingPlan(model)
    strategy = DDPStrategy(broadcast_buffers=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,  # one pass
        strategy=strategy,  # type: ignore[arg-type]
    )
    trainer.fit(training_plan, train_dataloaders=data_loader)

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


@pytest.mark.parametrize(
    "checkpoint_kwargs",
    [
        {
            "save_on_train_end": True,
            "save_on_train_epoch_end": False,
            "save_on_train_batch_end": False,
        },
        {
            "save_on_train_end": False,
            "save_on_train_epoch_end": True,
            "save_on_train_batch_end": False,
        },
        {
            "save_on_train_end": False,
            "save_on_train_epoch_end": False,
            "save_on_train_batch_end": True,
        },
    ],
)
def test_module_checkpoint(tmp_path: Path, checkpoint_kwargs: dict):
    # dataloader
    train_loader = torch.utils.data.DataLoader(TestDataset(np.arange(3)))
    # model
    model = OnePassMeanVarStdFromCLI(g_genes=1, target_count=10)
    training_plan = TrainingPlan(model)
    # trainer
    checkpoint_kwargs["dirpath"] = tmp_path
    module_checkpoint = ModuleCheckpoint(**checkpoint_kwargs)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        callbacks=[module_checkpoint],
        log_every_n_steps=1,
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)
    # load model from checkpoint
    assert os.path.exists(os.path.join(tmp_path, "module_checkpoint.pt"))
    loaded_model: OnePassMeanVarStdFromCLI = torch.load(os.path.join(tmp_path, "module_checkpoint.pt"))
    # assert
    assert isinstance(model.transform, ZScoreLog1pNormalize)
    assert isinstance(loaded_model.transform, ZScoreLog1pNormalize)
    assert model.transform.target_count == loaded_model.transform.target_count
    np.testing.assert_allclose(model.mean_g, loaded_model.mean_g)
    np.testing.assert_allclose(model.var_g, loaded_model.var_g)
    np.testing.assert_allclose(model.std_g, loaded_model.std_g)
