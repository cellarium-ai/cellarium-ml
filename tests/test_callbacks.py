# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.strategies import FSDPStrategy

from cellarium.ml import CellariumModule
from cellarium.ml.callbacks import ComputeNorm, LossScaleMonitor
from cellarium.ml.models import LogisticRegression
from cellarium.ml.utilities.data import collate_fn
from cellarium.ml.utilities.testing import PandasLogger
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


def test_compute_norm_multi_device(tmp_path: Path):
    devices = int(os.environ.get("TEST_DEVICES", "1"))

    # dataset
    n, g = 6, 3
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    y_categories = np.array(["a", "b"])
    y = np.array([0, 1, 0, 1, 1, 0])
    dataset = BoringDataset(
        np.ones((n, g)).astype(np.float32),
        var_names=var_names_g,
        y=y,
        y_categories=y_categories,
    )
    # model
    model = LogisticRegression(
        n_obs=n,
        var_names_g=var_names_g,
        y_categories=y_categories,
        log_metrics=False,
    )
    orig_module = CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})

    # single device
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=devices,
        shuffle=False,
        collate_fn=collate_fn,
    )
    module = copy.deepcopy(orig_module)
    logger_single_device = PandasLogger()
    # trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger_single_device,
        callbacks=[ComputeNorm()],
        max_epochs=1,
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(module, train_dataloaders=train_loader)

    # multi device
    class DataModule(pl.LightningDataModule):
        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=torch.utils.data.DistributedSampler(dataset, shuffle=False),
                batch_size=1,
                collate_fn=collate_fn,
            )

    datamodule = DataModule()
    module = copy.deepcopy(orig_module)
    logger_multi_device = PandasLogger()
    # trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=FSDPStrategy(sharding_strategy="NO_SHARD"),
        devices=devices,
        logger=logger_multi_device,
        callbacks=[ComputeNorm()],
        max_epochs=1,
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(module, datamodule)

    if trainer.global_rank != 0:
        return

    assert logger_single_device.df.equals(logger_multi_device.df)
