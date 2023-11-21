# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import GeneformerFromCLI
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 4, 3
    var_names = np.array([f"gene_{i}" for i in range(g)])
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.arange(n * g).reshape(n, g),
            var_names=var_names,
        ),
        collate_fn=collate_fn,
    )
    # model
    init_args = {
        "feature_schema": var_names,
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": 4,
        "max_position_embeddings": 2,
    }
    model = GeneformerFromCLI(**init_args)  # type: ignore[arg-type]
    config = {
        "model": {
            "model": {
                "class_path": "cellarium.ml.models.GeneformerFromCLI",
                "init_args": init_args,
            }
        }
    }
    module = CellariumModule(model, config=config)
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
    loaded_model: GeneformerFromCLI = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    assert np.array_equal(model.feature_schema, loaded_model.feature_schema)
    np.testing.assert_allclose(model.feature_ids, loaded_model.feature_ids)
