# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import lightning.pytorch as pl
import numpy as np
import pyro
import pytest
import torch

from scvid.callbacks import VarianceMonitor
from scvid.module import IncrementalPCA
from scvid.train import TrainingPlan

from .common import TestDataset

n, g, k = 10000, 50, 30


@pytest.fixture
def x_ng():
    rng = np.random.default_rng(0)
    z_nk = rng.standard_normal(size=(n, k), dtype=np.float32)
    w_kg = rng.standard_normal(size=(k, g), dtype=np.float32)
    sigma = 0.6
    noise_ng = sigma * rng.standard_normal(size=(n, g), dtype=np.float32)
    mean_g = rng.standard_normal(size=(g,), dtype=np.float32)
    x_ng = mean_g + z_nk @ w_kg# + noise_ng
    return x_ng


@pytest.mark.parametrize("mean_correct", [False])
@pytest.mark.parametrize("low_rank", [-5, -5, -10])
def test_incremental_pca(x_ng: np.ndarray, mean_correct: bool, low_rank: int):
    n, g = x_ng.shape
    d = 20
    if not mean_correct:
        x_ng = x_ng - x_ng.mean(axis=0)

    # dataloader
    batch_size = 1000
    train_loader = torch.utils.data.DataLoader(
        TestDataset(x_ng),
        batch_size=batch_size,
        shuffle=False,
    )
    # model
    total_var = np.var(x_ng, axis=0).sum()
    ipca = IncrementalPCA(
        g_genes=g,
        k_components=d,
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

    # pca fit
    x_ng_centered = x_ng - x_ng.mean(axis=0)
    _, S_g, V_gg = np.linalg.svd(x_ng_centered, full_matrices=False)
    # x_cov_gg = x_ng_centered.T @ x_ng_centered / n
    # L_g, U_gg = np.linalg.eig(x_cov_gg)
    L_g = S_g ** 2 / n

    # variance explained be each PC
    expected_explained_var = L_g[:d]
    actual_explained_var = ipca.L_k[:d]
    print(actual_explained_var / expected_explained_var)
    import pdb;pdb.set_trace()
    np.testing.assert_allclose(expected_explained_var, actual_explained_var, rtol=1e-3)

    # absolute cosine similarity between expected and actual PCs
    abs_cos_sim = torch.abs(
        torch.nn.functional.cosine_similarity(
            ipca.U_gk[:, :d],
            torch.as_tensor(V_gg[:d].T),
            dim=0,
        )
    )
    np.testing.assert_allclose(np.ones(d), abs_cos_sim, rtol=1e-3)


#  def test_variance_monitor(x_ng: np.ndarray):
#      # dataloader
#      train_loader = torch.utils.data.DataLoader(TestDataset(x_ng), batch_size=n // 2)
#      # model
#      ppca = ProbabilisticPCA(n, g, k, "marginalized")
#      training_plan = TrainingPlan(ppca)
#      # trainer
#      var_monitor = VarianceMonitor(total_variance=np.var(x_ng, axis=0).sum())
#      trainer = pl.Trainer(
#          accelerator="cpu",
#          devices=1,
#          max_steps=2,
#          callbacks=[var_monitor],
#          log_every_n_steps=1,
#      )
#      # fit
#      trainer.fit(training_plan, train_dataloaders=train_loader)
