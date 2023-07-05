# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from scvid.module import IncrementalPCA
from scvid.train import TrainingPlan

from .common import TestDataset

n, g = 10000, 100


@pytest.fixture
def x_ng():
    rng = torch.Generator()
    rng.manual_seed(1465)
    mean_g = torch.randn((g,), generator=rng)
    x_ng = torch.randn((n, g), generator=rng) + mean_g
    return x_ng


@pytest.mark.parametrize("mean_correct", [False, True])
@pytest.mark.parametrize("batch_size", [10_000, 5000, 1000, 500, 100])
@pytest.mark.parametrize("k", [30, 50, 80])
def test_incremental_pca(x_ng: np.ndarray, mean_correct: bool, batch_size: int, k: int):
    n, g = x_ng.shape
    x_ng_centered = x_ng - x_ng.mean(axis=0)

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        TestDataset(x_ng if mean_correct else x_ng_centered),
        batch_size=batch_size,
        shuffle=False,
    )
    # model
    ipca = IncrementalPCA(
        g_genes=g,
        k_components=k,
        mean_correct=mean_correct,
    )
    training_plan = TrainingPlan(ipca)
    # trainer
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)

    # actual approximation error
    x_diff = torch.linalg.matrix_norm(
        x_ng_centered - x_ng_centered @ ipca.V_kg.T @ ipca.V_kg, ord="fro"
    )

    # optimal rank-k approximation error
    _, _, V_gg = torch.linalg.svd(x_ng_centered, full_matrices=False)
    V_kg = V_gg[:k]
    x_diff_rank_k = torch.linalg.matrix_norm(
        x_ng_centered - x_ng_centered @ V_kg.T @ V_kg, ord="fro"
    )

    assert x_diff < x_diff_rank_k * 1.05
    assert ipca.x_size == n
    np.testing.assert_allclose(
        ipca.x_mean_g,
        (x_ng if mean_correct else x_ng_centered).mean(axis=0),
        atol=1e-5,
    )
