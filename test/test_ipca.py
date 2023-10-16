# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.module import IncrementalPCA, IncrementalPCAFromCLI
from cellarium.ml.train import TrainingPlan
from cellarium.ml.transforms import ZScoreLog1pNormalize

from .common import TestDataset

n, g = 10000, 100


@pytest.fixture
def x_ng():
    rng = torch.Generator()
    rng.manual_seed(1465)
    mean_g = torch.randn((g,), generator=rng)
    x_ng = torch.randn((n, g), generator=rng) + mean_g
    return x_ng


@pytest.mark.parametrize("perform_mean_correction", [False, True])
@pytest.mark.parametrize("batch_size", [10_000, 5000, 1000, 500, 250])
@pytest.mark.parametrize("k", [30, 50, 80])
def test_incremental_pca_multi_device(x_ng: np.ndarray, perform_mean_correction: bool, batch_size: int, k: int):
    n, g = x_ng.shape
    x_ng_centered = x_ng - x_ng.mean(axis=0)
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    batch_size = batch_size // devices

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        TestDataset(x_ng if perform_mean_correction else x_ng_centered),
        batch_size=batch_size,
        shuffle=False,
    )
    # model
    ipca = IncrementalPCA(
        g_genes=g,
        k_components=k,
        perform_mean_correction=perform_mean_correction,
    )
    training_plan = TrainingPlan(ipca)
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
    trainer.fit(training_plan, train_dataloaders=train_loader)

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
        (x_ng if perform_mean_correction else x_ng_centered).mean(axis=0),
        atol=1e-5,
    )


def test_module_checkpoint(tmp_path: Path):
    # dataloader
    train_loader = torch.utils.data.DataLoader(TestDataset(np.arange(6).reshape(3, 2)))
    # model
    init_args = {"g_genes": 2, "k_components": 1, "perform_mean_correction": True, "target_count": 10}
    model = IncrementalPCAFromCLI(**init_args)
    training_plan = TrainingPlan(model)
    config = {
        "model": {
            "module": {
                "class_path": "cellarium.ml.module.IncrementalPCAFromCLI",
                "init_args": init_args,
            }
        }
    }
    training_plan._set_hparams(config)
    # trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)
    # load model from checkpoint
    ckpt_path = os.path.join(tmp_path, "lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt")
    assert os.path.exists(ckpt_path)
    loaded_model: IncrementalPCAFromCLI = TrainingPlan.load_from_checkpoint(ckpt_path).module
    # assert
    assert isinstance(model.transform, ZScoreLog1pNormalize)
    assert isinstance(loaded_model.transform, ZScoreLog1pNormalize)
    assert model.transform.target_count == loaded_model.transform.target_count
    np.testing.assert_allclose(model.V_kg.detach(), loaded_model.V_kg.detach())
    np.testing.assert_allclose(model.S_k.detach(), loaded_model.S_k.detach())
    np.testing.assert_allclose(model.x_size.detach(), loaded_model.x_size.detach())
    np.testing.assert_allclose(model.x_mean_g.detach(), loaded_model.x_mean_g.detach())
