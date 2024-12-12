# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.callbacks import ComputeNorm, LossScaleMonitor
from cellarium.ml.models import LogisticRegression
from cellarium.ml.utilities.data import collate_fn
from tests.common import USE_CUDA, BoringDataset


@pytest.mark.skipif(not USE_CUDA, reason="requires_cuda")
def test_loss_scale_monitor(tmp_path: Path):
    n, g = 4, 3
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    y_categories = np.array(["a", "b"])
    y = np.array([0, 1, 0, 1])
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.random.randn(n, g).astype(np.float32),
            var_names=var_names_g,
            y=y,
            y_categories=y_categories,
        ),
        collate_fn=collate_fn,
    )
    # model
    model = LogisticRegression(
        n_obs=n,
        var_names_g=var_names_g,
        y_categories=y_categories,
        log_metrics=False,
    )
    module = CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})
    # trainer
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        precision="16-mixed",
        callbacks=[LossScaleMonitor()],
        max_epochs=1,
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(module, train_dataloaders=train_loader)


def test_compute_norm(tmp_path: Path):
    n, g = 4, 3
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    y_categories = np.array(["a", "b"])
    y = np.array([0, 1, 0, 1])
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.random.randn(n, g).astype(np.float32),
            var_names=var_names_g,
            y=y,
            y_categories=y_categories,
        ),
        collate_fn=collate_fn,
    )
    # model
    model = LogisticRegression(
        n_obs=n,
        var_names_g=var_names_g,
        y_categories=y_categories,
        log_metrics=False,
    )
    module = CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})
    # trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        callbacks=[ComputeNorm()],
        max_epochs=1,
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(module, train_dataloaders=train_loader)
