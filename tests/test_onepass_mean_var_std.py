# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from anndata import AnnData
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml import CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
from cellarium.ml.models import OnePassMeanVarStd
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
@pytest.mark.parametrize("algorithm", ["naive", "shifted_data"])
def test_onepass_mean_var_std_multi_device(
    adata: AnnData,
    dadc: DistributedAnnDataCollection,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    algorithm: Literal["naive", "shifted_data"],
):
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # prepare dataloader
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={
            "x_ng": AnnDataField("X"),
            "var_names_g": AnnDataField("var_names"),
        },
        batch_size=batch_size,
        shuffle=shuffle,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    transforms = [NormalizeTotal(target_count=10_000), Log1p()]

    # fit
    model = OnePassMeanVarStd(var_names_g=dadc.var_names, algorithm=algorithm)
    module = CellariumModule(transforms=transforms, model=model)
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
    batch = {"x_ng": torch.from_numpy(adata.X)}
    for transform in transforms:
        batch = transform(**batch)
    x = batch["x_ng"]
    expected_mean = torch.mean(x, dim=0)
    expected_var = torch.var(x, dim=0, unbiased=False)
    expected_std = torch.std(x, dim=0, unbiased=False)

    np.testing.assert_allclose(expected_mean, actual_mean, atol=1e-5)
    np.testing.assert_allclose(expected_var, actual_var, atol=1e-4)
    np.testing.assert_allclose(expected_std, actual_std, atol=1e-4)


@pytest.mark.parametrize("algorithm", ["naive", "shifted_data"])
def test_load_from_checkpoint_multi_device(tmp_path: Path, algorithm: Literal["naive", "shifted_data"]):
    n, g = 3, 2
    var_names_g = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.random.randn(n, g),
            np.array(var_names_g),
        ),
        collate_fn=collate_fn,
    )
    # model
    model = OnePassMeanVarStd(var_names_g=var_names_g, algorithm=algorithm)
    module = CellariumModule(model=model)
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
    loaded_model: OnePassMeanVarStd = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    np.testing.assert_allclose(model.mean_g, loaded_model.mean_g, atol=1e-6)
    np.testing.assert_allclose(model.var_g, loaded_model.var_g, atol=1e-6)
    np.testing.assert_allclose(model.std_g, loaded_model.std_g, atol=1e-6)
    if algorithm == "shifted_data":
        assert model.x_shift is not None and loaded_model.x_shift is not None
        np.testing.assert_allclose(model.x_shift, loaded_model.x_shift, atol=1e-6)
    assert model.algorithm == loaded_model.algorithm


@pytest.mark.parametrize("mean", [1, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("algorithm", ["naive", "shifted_data"])
def test_accuracy(mean: float, dtype: torch.dtype, algorithm: Literal["naive", "shifted_data"]):
    n_trials = 5_000_000
    std = 0.1
    x = mean + std * torch.randn(n_trials, dtype=dtype)

    onepass = OnePassMeanVarStd(var_names_g=["x"], algorithm=algorithm)
    for chunk in x.split(1000):
        onepass(x_ng=chunk[:, None], var_names_g=["x"])

    mean_expected = x.mean().item()
    mean_actual = onepass.mean_g[0].item()
    assert mean_actual == pytest.approx(mean_expected, rel=1e-5)

    var_expected = x.var(correction=0).item()
    var_actual = onepass.var_g[0].item()
    if algorithm == "naive" and dtype == torch.float32 and mean == 100:
        with pytest.raises(AssertionError):
            assert var_actual == pytest.approx(var_expected, rel=1e-3)
    else:
        assert var_actual == pytest.approx(var_expected, rel=1e-3)
