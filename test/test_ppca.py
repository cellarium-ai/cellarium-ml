# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pyro
import pytest
import torch

from cellarium.ml.callbacks import ModuleCheckpoint, VarianceMonitor
from cellarium.ml.module import ProbabilisticPCA, ProbabilisticPCAFromCLI
from cellarium.ml.train import TrainingPlan
from cellarium.ml.transforms import ZScoreLog1pNormalize

from .common import TestDataset

n, g, k = 1000, 10, 3


@pytest.fixture
def x_ng():
    rng = np.random.default_rng(0)
    z_nk = rng.standard_normal(size=(n, k), dtype=np.float32)
    w_kg = rng.standard_normal(size=(k, g), dtype=np.float32)
    sigma = 0.6
    noise = sigma * rng.standard_normal(size=(n, g), dtype=np.float32)
    x_ng = z_nk @ w_kg + noise
    return x_ng


@pytest.mark.parametrize("ppca_flavor", ["marginalized", "linear_vae"])
@pytest.mark.parametrize("learn_mean", [False, True])
@pytest.mark.parametrize("minibatch", [False, True], ids=["fullbatch", "minibatch"])
def test_probabilistic_pca_multi_device(x_ng: np.ndarray, minibatch: bool, ppca_flavor: str, learn_mean: bool):
    n, g = x_ng.shape
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    if learn_mean:
        x_mean_g = None
    else:
        x_mean_g = torch.as_tensor(x_ng.mean(axis=0))

    # dataloader
    batch_size = n // 2 if minibatch else n
    train_loader = torch.utils.data.DataLoader(
        TestDataset(x_ng),
        batch_size=batch_size,
        shuffle=False,
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
        optim_kwargs={"lr": 3e-2},
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs={"T_max": 1000},  # one cycle
    )
    # trainer
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_steps=1000,
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)

    # pca fit
    x_ng_centered = x_ng - x_ng.mean(axis=0)
    x_cov_gg = x_ng_centered.T @ x_ng_centered / n
    L_g, U_gg = np.linalg.eig(x_cov_gg)

    # total variance
    expected_total_var = np.var(x_ng, axis=0).sum()
    actual_total_var = ppca.W_variance + ppca.sigma_variance
    np.testing.assert_allclose(expected_total_var, actual_total_var, rtol=1e-3)

    # variance explained be each PC
    expected_explained_var = L_g[:k]
    actual_explained_var = ppca.L_k
    np.testing.assert_allclose(expected_explained_var, actual_explained_var, rtol=1e-3)

    # absolute cosine similarity between expected and actual PCs
    abs_cos_sim = torch.abs(
        torch.nn.functional.cosine_similarity(
            ppca.U_gk,
            torch.as_tensor(U_gg[:, :k]),
            dim=0,
        )
    )
    np.testing.assert_allclose(np.ones(k), abs_cos_sim, rtol=1e-3)


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
    model = ProbabilisticPCAFromCLI(3, 1, 1, "marginalized", target_count=10)
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
    loaded_model: ProbabilisticPCAFromCLI = torch.load(os.path.join(tmp_path, "module_checkpoint.pt"))
    # assert
    assert isinstance(model.transform, ZScoreLog1pNormalize)
    assert isinstance(loaded_model.transform, ZScoreLog1pNormalize)
    assert model.transform.target_count == loaded_model.transform.target_count
    np.testing.assert_allclose(model.W_kg.detach(), loaded_model.W_kg.detach())
    np.testing.assert_allclose(model.sigma.detach(), loaded_model.sigma.detach())
    np.testing.assert_allclose(model.mean_g.detach(), loaded_model.mean_g.detach())
