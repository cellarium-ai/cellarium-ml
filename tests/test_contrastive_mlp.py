# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.models import ContrastiveMLP
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset

from cellarium.ml.utilities.data import AnnDataField

import anndata

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def test_multi_gpu_consistency(tmp_path: Path):
    class SimpleShift(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_ng):
            shift = torch.zeros_like(x_ng)
            shift[x_ng.shape[0] // 2:, :] = 0.5

            return x_ng + shift
    
    class SimpleGaussian(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_ng):
            shift = torch.randn_like(x_ng) / 4

            return x_ng + shift

    n_vars = 4
    hidden_size = [2]
    embed_dim = 2
    data_size = 40
    full_batch_size = 8
    max_steps = 4000
    
    # dataset
    rng = np.random.default_rng(123)
    counts = rng.integers(1, 5, size=(data_size, n_vars)).astype("float32")
    np.save('/home/jupyter/bw-bican-data/toy_ds.npy', counts)
    adata = anndata.AnnData(counts)
    adata.write('/home/jupyter/bw-bican-data/toy_ds.h5ad')
    
    logger.debug(counts)
    
    # re-run test after switching to 1 or 2
    n_gpu = 2
        
    init_args = {
        "g_genes": n_vars,
        "hidden_size": hidden_size,
        "embed_dim": embed_dim,
        "batch_size": full_batch_size // n_gpu,
        "world_size": n_gpu,
        # "augment": SimpleShift(),
        "augment": SimpleGaussian(),
    }
    torch.manual_seed(1)
    model = ContrastiveMLP(**init_args)  # type: ignore[arg-type]
    config = {
        "model": {
            "model": {
                "class_path": "cellarium.ml.models.ContrastiveMLP",
                "init_args": init_args,
            },
            'optim_fn': torch.optim.SGD,
            'optim_kwargs': {
                'lr': 0.001,
            },
        }
    }
    module = CellariumModule(model, config=config)
    trainer = pl.Trainer(
        strategy='ddp',
        accelerator="gpu",
        devices=n_gpu,
        max_steps=max_steps,
        default_root_dir=tmp_path,
    )
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(counts),
        collate_fn=collate_fn,
        batch_size=full_batch_size // n_gpu,
        shuffle=False,
        num_workers=2,
    )
    data_module = CellariumAnnDataDataModule(
        filenames='/home/jupyter/bw-bican-data/toy_ds.h5ad',
        shard_size=data_size,
        last_shard_size=data_size,
        max_cache_size=1,
        num_workers=1,
        batch_keys={'X': AnnDataField(attr='X')},
        batch_size=full_batch_size // n_gpu,
        shuffle=False,
        drop_last=False)
    
    for p in module.model.parameters():
        logger.debug(p.data)
    
    # fit
    # trainer.fit(module, train_dataloaders=train_loader)
    trainer.fit(module, datamodule=data_module)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return
    
#     logger.debug('----------------AFTER TRAINING----------------')

    for p in module.model.parameters():
        logger.debug(p.data)

    assert False
