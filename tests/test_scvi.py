# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml import CellariumModule
from cellarium.ml.models import SingleCellVariationalInference
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDatasetSCVI


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 100, 50
    batch_size = 8  # must be > 1 for BatchNorm
    n_batch = 2  # number of "batches" for scvi to batch-correct
    var_names_g = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCVI(
            data=np.random.poisson(lam=2., size=(n, g)),
            batch_index_n=np.random.randint(0, n_batch, size=n),
            var_names=np.array(var_names_g),
        ),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    # model
    model = SingleCellVariationalInference(
        var_names_g=var_names_g,
        n_batch=n_batch,
        encoder={
            "layers":[{
                "class_path":"cellarium.ml.models.common.nn.LinearWithBatch",
                "init_args":{
                    "out_features": 32
                }
            }],
            "output_bias": True,
        },
        decoder={
            "layers":[{
                "class_path":"cellarium.ml.models.common.nn.LinearWithBatch",
                "init_args":{
                    "out_features": 32
                }
            }],
            "output_bias": True,
        },
    )
    module = CellariumModule(
        model=model,
        optim_fn=torch.optim.Adam,
        optim_kwargs={"lr": 1e-3},
    )
    # trainer
    strategy = DDPStrategy(broadcast_buffers=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="cpu",
        strategy=strategy,
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
    ckpt_path = tmp_path / f"lightning_logs/version_0/checkpoints/epoch=0-step={math.ceil(n / batch_size / devices)}.ckpt"
    assert ckpt_path.is_file()
    loaded_model: SingleCellVariationalInference = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    assert np.array_equal(model.var_names_g, loaded_model.var_names_g)
    torch.testing.assert_close(
        model.z_encoder.fully_connected.module_list[0].layer.weight,
        loaded_model.z_encoder.fully_connected.module_list[0].layer.weight,
    )
