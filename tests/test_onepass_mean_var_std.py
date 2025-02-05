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
from scipy.stats import pearsonr, spearmanr

from cellarium.ml import CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
from cellarium.ml.models import OnePassMeanVarStd, WelfordOnlineGeneGeneStats, WelfordOnlineGeneStats
from cellarium.ml.models.onepass_gene_stats import compute_ranks
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


@pytest.mark.parametrize("ModelClass", [OnePassMeanVarStd, WelfordOnlineGeneStats], ids=["naive", "welford"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("algorithm", ["naive", "shifted_data"])
def test_onepass_mean_var_std_multi_device(
    adata: AnnData,
    dadc: DistributedAnnDataCollection,
    ModelClass,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    algorithm: Literal["naive", "shifted_data"],
):
    if ModelClass == WelfordOnlineGeneStats and algorithm == "shifted_data":
        pytest.skip("WelfordOnlineGeneStats does not support shifted_data algorithm.")
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
    kwargs: dict[str, str | np.ndarray] = {"var_names_g": dadc.var_names}
    if ModelClass == OnePassMeanVarStd:
        kwargs |= {"algorithm": algorithm}
    model = ModelClass(**kwargs)
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


@pytest.mark.parametrize("ModelClass", [OnePassMeanVarStd, WelfordOnlineGeneStats], ids=["naive", "welford"])
@pytest.mark.parametrize("algorithm", ["naive", "shifted_data"])
def test_load_from_checkpoint_multi_device(tmp_path: Path, ModelClass, algorithm: Literal["naive", "shifted_data"]):
    if ModelClass == WelfordOnlineGeneStats and algorithm == "shifted_data":
        pytest.skip("WelfordOnlineGeneStats does not support shifted_data algorithm.")
    n, g = 3, 2
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.random.randn(n, g),
            var_names_g,
        ),
        collate_fn=collate_fn,
    )
    # model
    kwargs: dict[str, str | np.ndarray] = {"var_names_g": var_names_g}
    if ModelClass == OnePassMeanVarStd:
        kwargs |= {"algorithm": algorithm}
    model = ModelClass(**kwargs)
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
    loaded_model = CellariumModule.load_from_checkpoint(ckpt_path).model
    assert isinstance(loaded_model, ModelClass)
    # assert
    np.testing.assert_allclose(model.mean_g, loaded_model.mean_g, atol=1e-6)
    np.testing.assert_allclose(model.var_g, loaded_model.var_g, atol=1e-6)
    np.testing.assert_allclose(model.std_g, loaded_model.std_g, atol=1e-6)
    if algorithm == "shifted_data":
        assert model.x_shift_g is not None and loaded_model.x_shift_g is not None
        np.testing.assert_allclose(model.x_shift_g, loaded_model.x_shift_g, atol=1e-6)
    if ModelClass == OnePassMeanVarStd:
        assert model.algorithm == loaded_model.algorithm


@pytest.mark.parametrize("ModelClass", [OnePassMeanVarStd, WelfordOnlineGeneStats], ids=["naive", "welford"])
@pytest.mark.parametrize("mean", [1, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("algorithm", ["naive", "shifted_data"])
def test_accuracy(ModelClass, mean: float, dtype: torch.dtype, algorithm: Literal["naive", "shifted_data"]):
    if ModelClass == WelfordOnlineGeneStats and algorithm == "shifted_data":
        pytest.skip("WelfordOnlineGeneStats does not support shifted_data algorithm.")
    n_trials = 5_000_000
    std = 0.1
    x = mean + std * torch.randn(n_trials, dtype=dtype)

    kwargs: dict[str, str | np.ndarray] = {"var_names_g": np.array(["x"])}
    if ModelClass == OnePassMeanVarStd:
        kwargs |= {"algorithm": algorithm}
    onepass = ModelClass(**kwargs)
    for chunk in x.split(1000):
        onepass(x_ng=chunk[:, None], var_names_g=["x"])

    mean_expected = x.mean().item()
    mean_actual = onepass.mean_g[0].item()
    assert mean_actual == pytest.approx(mean_expected, rel=1e-5)

    var_expected = x.var(correction=0).item()
    var_actual = onepass.var_g[0].item()
    if ModelClass == OnePassMeanVarStd and algorithm == "naive" and dtype == torch.float32 and mean == 100:
        with pytest.raises(AssertionError):
            assert var_actual == pytest.approx(var_expected, rel=1e-3)
    else:
        assert var_actual == pytest.approx(var_expected, rel=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 100], ids=["batch1", "batch2", "batch3", "fullbatch"])
@pytest.mark.parametrize("use_rank", [False, True], ids=["raw", "ranks"])
def test_welford_covariance(use_rank: bool, batch_size: int):
    # Simulated test data: shape [cells, genes]
    x_ng = torch.randn(100, 5)

    var_names_g = [str(i) for i in range(x_ng.shape[1])]
    model = WelfordOnlineGeneGeneStats(var_names_g=var_names_g, use_rank=use_rank)

    # Feed rows (cells) into the Welford stats one by one
    for x_mg in torch.split(x_ng, batch_size, dim=0):
        model.forward(x_mg, var_names_g=var_names_g)

    # Expected covariance matrix
    if use_rank:
        # Compute rank-transformed covariance matrix
        ranked_x_ng = compute_ranks(x_ng)
        expected_cov_matrix_gg = np.cov(ranked_x_ng.T, bias=True)
    else:
        expected_cov_matrix_gg = np.cov(x_ng.T, bias=True)

    # Assert covariance matrix values are correct
    computed_cov_matrix_gg = model.covariance_gg
    tol = 1e-6
    print('expected:')
    print(expected_cov_matrix_gg)
    print('computed:')
    print(computed_cov_matrix_gg)
    assert np.allclose(computed_cov_matrix_gg, expected_cov_matrix_gg, atol=tol), \
        f"Expected covariance matrix:\n{expected_cov_matrix_gg}\nGot:\n{computed_cov_matrix_gg}"


@pytest.mark.parametrize("batch_size", [1, 2, 3, 75, 100], ids=["batch1", "batch2", "batch3", "batch75", "fullbatch"])
@pytest.mark.parametrize("use_rank", [False, True], ids=["raw", "ranks"])
def test_welford_correlation(use_rank: bool, batch_size: int):
    # Simulated test data: shape [cells, genes]
    x_ng = torch.randn(100, 5)

    n_genes = x_ng.shape[1]
    var_names_g = [str(i) for i in range(x_ng.shape[1])]
    model = WelfordOnlineGeneGeneStats(var_names_g=var_names_g, use_rank=use_rank)

    # Feed rows (cells) into the Welford stats one by one
    for x_mg in torch.split(x_ng, batch_size, dim=0):
        model.forward(x_mg, var_names_g=var_names_g)

    # Expected correlation matrix
    expected_corr_matrix_gg = np.zeros((n_genes, n_genes))
    if use_rank:
        x_ng = compute_ranks(x_ng)
    for i in range(n_genes):
        for j in range(n_genes):
            x = x_ng[:, i]
            y = x_ng[:, j]
            if use_rank:
                expected_corr_matrix_gg[i, j], _ = spearmanr(x, y)
            else:
                expected_corr_matrix_gg[i, j], _ = pearsonr(x, y)

    # Assert correlation matrix values are correct
    computed_corr_matrix_gg = model.correlation_gg
    tol = 1e-6
    assert np.allclose(computed_corr_matrix_gg, expected_corr_matrix_gg, atol=tol), \
        f"Expected correlation matrix:\n{expected_corr_matrix_gg}\nGot:\n{computed_corr_matrix_gg}"


def test_compute_ranks():
    x = torch.tensor([1, 2, 3, 4, 5])
    ranks = compute_ranks(x)
    expected_ranks = torch.tensor([0, 1, 2, 3, 4]) + 1
    assert torch.all(ranks == expected_ranks)

    x = torch.tensor([5, 4, 3, 2, 1])
    ranks = compute_ranks(x)
    expected_ranks = torch.tensor([4, 3, 2, 1, 0]) + 1
    assert torch.all(ranks == expected_ranks)

    x = torch.tensor([1, 2, 3, 3, 5])
    ranks = compute_ranks(x)
    expected_ranks = torch.tensor([0, 1, 2, 3, 4]) + 1
    assert torch.all(ranks == expected_ranks)

    x = torch.tensor([1, 2, 3, 3, 3])
    ranks = compute_ranks(x)
    expected_ranks = torch.tensor([0, 1, 2, 3, 4]) + 1
    assert torch.all(ranks == expected_ranks)
