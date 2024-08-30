# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import ContrastiveMLP
from cellarium.ml.transforms import Duplicate
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 4, 3
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(np.arange(n * g).reshape(n, g).astype("float32")),
        collate_fn=collate_fn,
    )
    # model
    model = ContrastiveMLP(
        n_obs=3,
        embed_dim=2,
        hidden_size=[2],
        temperature=1.0,
    )
    module = CellariumModule(
        transforms=[Duplicate()],
        model=model,
        optim_fn=torch.optim.Adam,
        optim_kwargs={"lr": 1e-3},
    )
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
    assert isinstance(loaded_model, ContrastiveMLP)
