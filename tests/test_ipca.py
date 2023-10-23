# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml import CellariumModule
from cellarium.ml.models import IncrementalPCA, IncrementalPCAFromCLI
from cellarium.ml.transforms import NormalizeTotal
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


@pytest.fixture
def x_ng():
    n, g = 10000, 100
    rng = torch.Generator()
    rng.manual_seed(1465)
    mean_g = torch.randn((g,), generator=rng)
    x_ng = torch.randn((n, g), generator=rng) + mean_g
    return x_ng


@pytest.mark.parametrize("perform_mean_correction", [False, True])
@pytest.mark.parametrize("batch_size", [10_000, 5000, 1000, 500, 250])
@pytest.mark.parametrize("k", [30, 50, 80])
def test_incremental_pca_multi_device(x_ng: torch.Tensor, perform_mean_correction: bool, batch_size: int, k: int):
    n, g = x_ng.shape
    x_ng_centered = x_ng - x_ng.mean(dim=0)
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    batch_size = batch_size // devices

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset((x_ng if perform_mean_correction else x_ng_centered).numpy()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # model
    ipca = IncrementalPCA(
        g_genes=g,
        k_components=k,
        perform_mean_correction=perform_mean_correction,
    )
    module = CellariumModule(ipca)
    # trainer
    strategy = DDPStrategy(broadcast_buffers=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,
        strategy=strategy,  # type: ignore[arg-type]
    )
    # fit
    trainer.fit(module, train_dataloaders=train_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    # actual approximation error
    x_diff = torch.linalg.matrix_norm(x_ng_centered - x_ng_centered @ ipca.V_kg.T @ ipca.V_kg, ord="fro")

    # optimal rank-k approximation error
    _, _, V_gg = torch.linalg.svd(x_ng_centered, full_matrices=False)
    V_kg = V_gg[:k]
    x_diff_rank_k = torch.linalg.matrix_norm(x_ng_centered - x_ng_centered @ V_kg.T @ V_kg, ord="fro")

    assert x_diff < x_diff_rank_k * 1.06
    assert ipca.x_size == n
    np.testing.assert_allclose(
        ipca.x_mean_g,
        (x_ng if perform_mean_correction else x_ng_centered).mean(dim=0),
        atol=1e-5,
    )


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 3, 2
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(np.arange(n * g).reshape(n, g)),
        collate_fn=collate_fn,
    )
    # model
    init_args = {"g_genes": g, "k_components": 1, "perform_mean_correction": True, "target_count": 10}
    model = IncrementalPCAFromCLI(**init_args)  # type: ignore[arg-type]
    config = {
        "model": {
            "model": {
                "class_path": "cellarium.ml.models.IncrementalPCAFromCLI",
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
    loaded_model: IncrementalPCAFromCLI = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    assert isinstance(model.transform, torch.nn.Sequential) and len(model.transform) == 2
    assert isinstance(loaded_model.transform, torch.nn.Sequential) and len(loaded_model.transform) == 2
    assert isinstance(model.transform[0], NormalizeTotal)
    assert isinstance(loaded_model.transform[0], NormalizeTotal)
    assert model.transform[0].target_count == loaded_model.transform[0].target_count
    np.testing.assert_allclose(model.V_kg.detach(), loaded_model.V_kg.detach())
    np.testing.assert_allclose(model.S_k.detach(), loaded_model.S_k.detach())
    np.testing.assert_allclose(model.x_size.detach(), loaded_model.x_size.detach())
    np.testing.assert_allclose(model.x_mean_g.detach(), loaded_model.x_mean_g.detach())
