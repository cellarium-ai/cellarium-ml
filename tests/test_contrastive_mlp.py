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
from cellarium.ml.transforms import Duplicate
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset

from cellarium.ml.utilities.data import AnnDataField, get_rank_and_num_replicas

import anndata

import hashlib
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def test_multi_gpu_consistency(tmp_path: Path):
    class BasicNet(torch.nn.Module):
        def __init__(self):
            super(BasicNet, self).__init__()
            self.net = torch.nn.Linear(4, 2)
        
        def forward(self, x_ng: torch.Tensor):
            rank, num_replicas = get_rank_and_num_replicas()
            
            logger.debug(f'in {rank}/{num_replicas}')
            logger.debug(x_ng)
            out = self.net(x_ng)
            logger.debug(f'out {rank}/{num_replicas}')
            logger.debug(out)
            loss_n = torch.norm(out, dim=1)
            logger.debug(f'loss_n {rank}/{num_replicas}')
            logger.debug(loss_n)
            return {'loss': loss_n.mean()}

    class SimpleShift(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.shift = shift

        def forward(self, x_ng: torch.Tensor):
            shift = torch.zeros_like(x_ng)
            shift[x_ng.shape[0] // 2:, :] = self.shift

            return {'x_ng': x_ng + shift}
    
    class SimpleGaussian(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_ng: torch.Tensor):
            shift = torch.randn_like(x_ng) / 4

            return {'x_ng': x_ng + shift}
    
    def get_cpu_state_dict(state_dict: dict) -> dict:
        cpu_state_dict = dict()
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_state_dict[k] = v.detach().cpu().numpy()
            elif isinstance(v, dict):
                cpu_state_dict[k] = get_cpu_state_dict(v)
            else:
                cpu_state_dict[k] = v
        return cpu_state_dict

    def print_hash(
            model,
            optimizer,
            scheduler):
        
        logger.debug('optimizer full')
        logger.debug(optimizer)

        model_md5_hash = hashlib.md5(pickle.dumps(get_cpu_state_dict(module.state_dict()))).hexdigest()

        logger.debug(f'model MD5 hash: {model_md5_hash}')
        
        if optimizer is not None:
            optimizer_md5_hash = hashlib.md5(pickle.dumps(get_cpu_state_dict(optimizer.state_dict()))).hexdigest()
            logger.debug(f'optimizer MD5 hash: {optimizer_md5_hash}')
        else:
            logger.debug(f'optimizer: None')
        
        if scheduler is not None:
            scheduler_md5_hash = hashlib.md5(pickle.dumps(get_cpu_state_dict(scheduler.state_dict()))).hexdigest()
            logger.debug(f'scheduler MD5 hash: {scheduler_md5_hash}')
        else:
            logger.debug(f'scheduler: None')

    n_vars = 4
    hidden_size = [2]
    embed_dim = 2
    data_size = 4
    full_batch_size = 4
    max_steps = 10
    
    # dataset
    rng = np.random.default_rng(123)
    counts = rng.integers(1, 5, size=(data_size, n_vars)).astype("float32")
    np.save('/home/jupyter/bw-bican-data/toy_ds.npy', counts)
    adata = anndata.AnnData(counts)
    adata.write('/home/jupyter/bw-bican-data/toy_ds.h5ad')
    
    # logger.debug(counts)
    
    # re-run test after switching to 1 or 2
    n_gpu = 2

    init_args = {
        "g_genes": n_vars,
        "hidden_size": hidden_size,
        "embed_dim": embed_dim,
        "batch_size": full_batch_size // n_gpu,
        "world_size": n_gpu,
    }
    seed = 5
    pl.seed_everything(seed)
    # model = ContrastiveMLP(**init_args)  # type: ignore[arg-type]
    model = BasicNet()
    config = {
        "model": {
            "model": {
                "class_path": "cellarium.ml.models.ContrastiveMLP",
                "init_args": init_args,
            },
            'optim_fn': torch.optim.Adam,
            'optim_kwargs': {
                'lr': 0.002,
            },
            "transforms": [
                {
                    "class_path": "cellarium.ml.transforms.Duplicate"
                },
            ],
        }
    }
    path = f'/home/jupyter/bw-bican-data/gpu_experiments/ds-{data_size}__bs-{full_batch_size}__seed-{seed}__gpu-{n_gpu}'
    
    module = CellariumModule(transforms=[Duplicate()], model=model, optim_fn=config['model']['optim_fn'], optim_kwargs=config['model']['optim_kwargs'], config=config)
    # module = CellariumModule(transforms=[Duplicate(), SimpleShift(0.5)], model=model, optim_fn=config['model']['optim_fn'], optim_kwargs=config['model']['optim_kwargs'], config=config)
    # module = CellariumModule(model=model, optim_fn=config['model']['optim_fn'], optim_kwargs=config['model']['optim_kwargs'])
    
    trainer = pl.Trainer(
        strategy='ddp',
        accelerator="cpu",
        devices=n_gpu,
        max_steps=max_steps,
        default_root_dir=path,
    )
    data_module = CellariumAnnDataDataModule(
        filenames='/home/jupyter/bw-bican-data/toy_ds.h5ad',
        shard_size=data_size,
        last_shard_size=data_size,
        max_cache_size=1,
        num_workers=1,
        batch_keys={'x_ng': AnnDataField(attr='X')},
        batch_size=full_batch_size // n_gpu,
        shuffle=False,
        drop_last=False)
    
    # for p in module.model.parameters():
    #     logger.debug(p.data)
    
    # print_hash(model, None, None)
    
    # fit
    # trainer.fit(module, train_dataloaders=train_loader)
    trainer.fit(module, datamodule=data_module)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return
    
#     logger.debug('----------------AFTER TRAINING----------------')

    # for p in module.model.parameters():
    #     logger.debug(p.data)

    assert False
