# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import lightning.pytorch as pl
import numpy as np
import pyro
import pytest
import torch
from sklearn.decomposition import PCA

from scvid.callbacks import VarianceMonitor
from scvid.module import ProbabilisticPCA
from scvid.train import TrainingPlan
from scvid.callbacks.swa import StochasticWeightAveraging

from .common import TestDataset

n, g, k = 10000, 10, 3


@pytest.fixture
def x_ng():
    rng = np.random.default_rng(0)
    z_nk = rng.standard_normal(size=(n, k), dtype=np.float32)
    w_kg = rng.standard_normal(size=(k, g), dtype=np.float32)
    sigma = 0.6
    noise = sigma * rng.standard_normal(size=(n, g), dtype=np.float32)
    x_ng = z_nk @ w_kg + noise
    return x_ng


@pytest.fixture
def pca_fit(x_ng: np.ndarray):
    pca = PCA(n_components=k)
    pca.fit(x_ng)
    return pca


@pytest.mark.parametrize("ppca_flavor", ["marginalized", "linear_vae"])
@pytest.mark.parametrize("learn_mean", [False, True])
@pytest.mark.parametrize("minibatch", [False, True], ids=["fullbatch", "minibatch"])
def test_probabilistic_pca_multi_device(
    x_ng: np.ndarray, pca_fit: PCA, minibatch: bool, ppca_flavor: str, learn_mean: bool
):
    n, g = x_ng.shape
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    if learn_mean:
        x_mean_g = None
    else:
        x_mean_g = torch.as_tensor(x_ng.mean(axis=0))

    # dataloader
    # batch_size = n // 2 if minibatch else n
    batch_size = 1000
    train_loader = torch.utils.data.DataLoader(
        TestDataset(x_ng),
        batch_size=batch_size,
        shuffle=True,
    )
    # model
    pyro.clear_param_store()
    total_var = np.var(x_ng, axis=0).sum()
    w = np.sqrt(0.5 * total_var / (g * k))
    s = np.sqrt(0.5 * total_var / g)
    ppca = ProbabilisticPCA(
        n_cells=n,
        g_genes=g,
        k_components=k,
        ppca_flavor=ppca_flavor,
        mean_g=x_mean_g,
        W_init_scale=w,
        sigma_init_scale=s,
    )
    training_plan = TrainingPlan(
        ppca,
        optim_fn=torch.optim.Adam,
        optim_kwargs={"lr": 0.03},
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs={"T_max": 10000},  # one cycle
    )
    # trainer
    trainer = pl.Trainer(
        # barebones=True,
        accelerator="cpu",
        devices=devices,
        max_steps=10000,
        log_every_n_steps=1,
        callbacks=[var_monitor, lr_monitor],
        default_root_dir=f"tests/results/{ppca_flavor}_cos10000_lr0.03_bs{batch_size}",
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)

    # total variance
    expected_total_var = np.var(x_ng, axis=0).sum()
    actual_total_var = ppca.W_variance + ppca.sigma_variance
    np.testing.assert_allclose(expected_total_var, actual_total_var, rtol=0.01)

    # variance explained be each PC
    expected_explained_var = pca_fit.explained_variance_
    actual_explained_var = ppca.L_k
    np.testing.assert_allclose(expected_explained_var, actual_explained_var, rtol=0.01)

    # absolute cosine similarity between expected and actual PCs
    abs_cos_sim = torch.abs(
        torch.nn.functional.cosine_similarity(
            ppca.U_gk.T, torch.as_tensor(pca_fit.components_)
        )
    )
    np.testing.assert_allclose(np.ones(k), abs_cos_sim, rtol=0.01)


def test_variance_monitor(x_ng: np.ndarray):
    # dataloader
    train_loader = torch.utils.data.DataLoader(TestDataset(x_ng), batch_size=n // 2)
    # model
    ppca = ProbabilisticPCA(n, g, k, "marginalized")
    training_plan = TrainingPlan(ppca)
    # trainer
    var_monitor = VarianceMonitor(total_variance=np.var(x_ng, axis=0).sum())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        callbacks=[var_monitor],
        log_every_n_steps=1,
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)
