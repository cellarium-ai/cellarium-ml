# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from anndata import AnnData

from scvid.data import (
    DistributedAnnDataCollection,
    DistributedAnnDataCollectionDataset,
    DistributedAnnDataCollectionSingleConsumerSampler,
    IterableDistributedAnnDataCollectionDataset,
    collate_fn,
)
from scvid.module import OnePassMeanVarStd
from scvid.train import DummyTrainingPlan
from scvid.transforms import ZScoreLog1pNormalize


@pytest.fixture
def adata():
    n_cell, g_gene = (10, 5)
    rng = np.random.default_rng(1465)
    X = rng.integers(10, size=(n_cell, g_gene))
    return AnnData(X, dtype=X.dtype)


@pytest.fixture
def dadc(adata, tmp_path):
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
def test_onepass_mean_var_std(adata, dadc, shuffle, num_workers, batch_size):
    # prepare dataloader
    dataset = DistributedAnnDataCollectionDataset(dadc)
    sampler = DistributedAnnDataCollectionSingleConsumerSampler(
        limits=dadc.limits, shuffle=shuffle
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    transform = ZScoreLog1pNormalize(
        mean_g=0, std_g=None, perform_scaling=False, target_count=10_000
    )

    # fit
    model = OnePassMeanVarStd(transform=transform)
    training_plan = DummyTrainingPlan(model)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,  # one pass
        log_every_n_steps=1,  # to suppress logger warnings
    )
    trainer.fit(training_plan, train_dataloaders=data_loader)

    # actual mean, var, and std
    actual_mean = model.mean
    actual_var = model.var
    actual_std = model.std

    # expected mean, var, and std
    x = transform(torch.from_numpy(adata.X))
    expected_mean = torch.mean(x, dim=0)
    expected_var = torch.var(x, dim=0, unbiased=False)
    expected_std = torch.std(x, dim=0, unbiased=False)

    np.testing.assert_allclose(expected_mean, actual_mean, atol=1e-5)
    np.testing.assert_allclose(expected_var, actual_var, atol=1e-4)
    np.testing.assert_allclose(expected_std, actual_std, atol=1e-4)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_onepass_mean_var_std_iterable_dataset(
    adata, dadc, shuffle, num_workers, batch_size
):
    # prepare dataloader
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc, batch_size=batch_size, shuffle=shuffle
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    transform = ZScoreLog1pNormalize(
        mean_g=0, std_g=None, perform_scaling=False, target_count=10_000
    )

    # fit
    model = OnePassMeanVarStd(transform=transform)
    training_plan = DummyTrainingPlan(model)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,  # one pass
        log_every_n_steps=1,  # to suppress logger warnings
    )
    trainer.fit(training_plan, train_dataloaders=data_loader)

    # actual mean, var, and std
    actual_mean = model.mean
    actual_var = model.var
    actual_std = model.std

    # expected mean, var, and std
    x = transform(torch.from_numpy(adata.X))
    expected_mean = torch.mean(x, dim=0)
    expected_var = torch.var(x, dim=0, unbiased=False)
    expected_std = torch.std(x, dim=0, unbiased=False)

    np.testing.assert_allclose(expected_mean, actual_mean, atol=1e-5)
    np.testing.assert_allclose(expected_var, actual_var, atol=1e-4)
    np.testing.assert_allclose(expected_std, actual_std, atol=1e-4)
