# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models.cellarium_gpt import CellariumGPT
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


def test_load_from_checkpoint_multi_device(tmp_path: Path) -> None:
    n, g = 4, 3
    var_names = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    data = np.arange(n * g).reshape(n, g).astype(np.float32)
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            data,
            var_names=np.array(var_names),
            total_mrna_umis=data.sum(-1),
        ),
        collate_fn=collate_fn,
    )
    # model
    model = CellariumGPT(
        var_names_g=var_names,
        d_model=2,
        d_ffn=4,
        n_heads=1,
        n_blocks=1,
        dropout_p=0.0,
        use_bias=False,
        max_prefix_len=2,
        suffix_len=None,
        context_len=3,
        attn_mult=math.sqrt(2),
        input_mult=2.0,
        output_mult=1.0,
        initializer_range=0.02,
        attn_backend="math",
    )
    module = CellariumModule(model=model, optim_fn=torch.optim.AdamW)
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
    loaded_model: CellariumGPT = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    assert np.array_equal(model.var_names_g, loaded_model.var_names_g)
    assert model.transformer.input_mult == loaded_model.transformer.input_mult
    torch.testing.assert_close(model.transformer.Et.weight, loaded_model.transformer.Et.weight)
