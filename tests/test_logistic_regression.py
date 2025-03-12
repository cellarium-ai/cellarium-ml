# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import LogisticRegression
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 4, 3
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    y_categories = np.array(["a", "b"])
    y = np.array([0, 1, 0, 1])
    devices = int(os.environ.get("TEST_DEVICES", "1"))
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
    assert isinstance(loaded_model, LogisticRegression)
    # assert
    assert np.array_equal(model.var_names_g, loaded_model.var_names_g)
    assert np.array_equal(model.y_categories, loaded_model.y_categories)
