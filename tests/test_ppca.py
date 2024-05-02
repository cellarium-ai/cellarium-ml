# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pyro
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.callbacks import VarianceMonitor
from cellarium.ml.models import ProbabilisticPCA
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


@pytest.fixture
def x_ng():
    n, g, k = 1000, 10, 3
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
def test_probabilistic_pca_multi_device(
    x_ng: np.ndarray, minibatch: bool, ppca_flavor: Literal["marginalized", "linear_vae"], learn_mean: bool
):
    n, g = x_ng.shape
    k = 3
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    if learn_mean:
        x_mean_g = None
    else:
        x_mean_g = torch.as_tensor(x_ng.mean(axis=0))

    # dataloader
    batch_size = n // 2 if minibatch else n
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            x_ng,
            np.array([f"gene_{i}" for i in range(g)]),
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # model
    pyro.clear_param_store()
    total_var = np.var(x_ng, axis=0).sum()
    w = np.sqrt(0.5 * total_var / (g * k))
    s = np.sqrt(0.5 * total_var / g)
    ppca = ProbabilisticPCA(
        n_obs=n,
        var_names_g=[f"gene_{i}" for i in range(g)],
        n_components=k,
        ppca_flavor=ppca_flavor,
        mean_g=x_mean_g,
        W_init_scale=w,
        sigma_init_scale=s,
    )
    module = CellariumModule(
        model=ppca,
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
    trainer.fit(module, train_dataloaders=train_loader)

    # pca fit
    x_ng_centered = x_ng - x_ng.mean(axis=0)
    x_cov_gg = x_ng_centered.T @ x_ng_centered / n
    L_g, U_gg = np.linalg.eig(x_cov_gg)

    # total variance
    expected_total_var = np.var(x_ng, axis=0).sum()
    actual_total_var = ppca.W_variance + ppca.sigma_variance
    np.testing.assert_allclose(expected_total_var, actual_total_var, rtol=1e-3)

    # variance explained by each PC
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
    n, g = x_ng.shape
    k = 3
    var_names_g = [f"gene_{i}" for i in range(g)]
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            x_ng,
            np.array(var_names_g),
        ),
        batch_size=n // 2,
        collate_fn=collate_fn,
    )
    # model
    ppca = ProbabilisticPCA(n, var_names_g, k, "marginalized")
    module = CellariumModule(model=ppca, optim_fn=torch.optim.Adam)
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
    trainer.fit(module, train_dataloaders=train_loader)


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 3, 2
    var_names_g = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.random.randn(n, g).astype(np.float32),
            np.array(var_names_g),
        ),
        collate_fn=collate_fn,
    )
    # model
    pyro.clear_param_store()
    model = ProbabilisticPCA(
        n_obs=n,
        var_names_g=var_names_g,
        n_components=1,
        ppca_flavor="marginalized",
    )
    module = CellariumModule(model=model, optim_fn=torch.optim.Adam)
    # trainer
    trainer = pl.Trainer(
        accelerator="cpu",
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
    loaded_model: ProbabilisticPCA = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    np.testing.assert_allclose(model.W_kg.detach(), loaded_model.W_kg.detach())
    np.testing.assert_allclose(model.sigma.detach(), loaded_model.sigma.detach())  # type: ignore[attr-defined]
    np.testing.assert_allclose(model.mean_g.detach(), loaded_model.mean_g.detach())
