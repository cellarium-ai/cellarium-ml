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

'''
@pytest.fixture
def x_ng():
    n, k = 10000, 100
    rng = np.random.default_rng(0)
    x_ng = rng.standard_normal(size=(n, k), dtype=np.float32)
    return x_ng

@pytest.fixture
def batch_index_n():
    n = 10000
    rng = np.random.default_rng(0)
    batch_index_n  = rng.integers(low=0, high=1, size=n).astype(np.float32)
    return batch_index_n


@pytest.mark.parametrize("batch_size", [10000, 5000, 1000, 500, 250])
def test_scvi_multi_device(x_ng: np.array, batch_index_n: np.array, batch_size: int):
    n, g = x_ng.shape
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    batch_size = batch_size // devices

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCVI(
            x_ng,
            batch_index_n,
            np.array([f"gene_{i}" for i in range(g)])
        ),
        # batch_size=batch_size,
        # shuffle=False,
        collate_fn=collate_fn,
    )
    # model
    scvi = SingleCellVariationalInference(
        var_names_g=[f"gene_{i}" for i in range(g)],
        n_batch=2,
        encoder={
            "layers":[{
                "class_path":"cellarium.ml.models.common.nn.LinearWithBatch",
                "init_args":{
                    "out_features": 128,
                }
            }],
            "output_bias": True,
        },
        decoder={
            "layers":[{
                "class_path":"cellarium.ml.models.common.nn.LinearWithBatch",
                "init_args":{
                    "out_features": 128
                }
            }],
            "output_bias": True,
        },
    )
    module = CellariumModule(
        model=scvi,
        optim_fn=torch.optim.Adam,
        optim_kwargs={"lr": 1e-3},
    )
    # trainer
    strategy = DDPStrategy(broadcast_buffers=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_epochs=1,
        strategy=strategy,
    )
    # fit
    trainer.fit(module, train_dataloaders=train_loader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return
'''

def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 4, 2
    var_names_g = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCVI(
            np.random.default_rng(0).standard_normal(size=(n, g), dtype=np.float32), # np.random.randn(n, g),
            np.asarray([1.,0.,0.,1.]),
            np.array(var_names_g)
        ),
        collate_fn=collate_fn,
    )
    # model
    model = SingleCellVariationalInference(
        var_names_g=var_names_g,
        n_batch=2,
        encoder={
            "layers":[{
                "class_path":"cellarium.ml.models.common.nn.LinearWithBatch",
                "init_args":{
                    "out_features": 128
                }
            }],
            "output_bias": True,
        },
        decoder={
            "layers":[{
                "class_path":"cellarium.ml.models.common.nn.LinearWithBatch",
                "init_args":{
                    "out_features": 128
                }
            }],
            "output_bias": True,
        },
    )
    module = CellariumModule(model=model)
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
    ckpt_path = tmp_path / f"lightning_logs/version_0/checkpoints/epoch=0-step={math.ceil(n / devices)}.ckpt"
    assert ckpt_path.is_file()
    loaded_model: SingleCellVariationalInference = CellariumModule.load_from_checkpoint(ckpt_path).model
