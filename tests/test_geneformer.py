# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models import Geneformer
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    n, g = 4, 3
    var_names_g = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.arange(n * g).reshape(n, g),
            var_names=np.array(var_names_g),
        ),
        collate_fn=collate_fn,
    )
    # model
    model = Geneformer(
        var_names_g=var_names_g,
        hidden_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=4,
        max_position_embeddings=2,
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
    loaded_model: Geneformer = CellariumModule.load_from_checkpoint(ckpt_path).model
    # assert
    assert np.array_equal(model.var_names_g, loaded_model.var_names_g)
    np.testing.assert_allclose(model.feature_ids, loaded_model.feature_ids)


@pytest.mark.parametrize("perturb", ["activation", "deletion", "map", "none"])
def test_tokenize_with_perturbations(perturb: str):
    var_names_g = ["a", "b", "c", "d"]
    geneformer = Geneformer(var_names_g=var_names_g)
    x_ng = torch.tensor([[4, 3, 2, 1]])  # sort order will be [a,b,c,d] and tokens will be [2,3,4,5]

    # test that we get the expected output for a well-formed set of input args
    match perturb:
        case "none":
            input_ids, _ = geneformer.tokenize_with_perturbations(x_ng)
            expected_input_ids = torch.tensor([[2, 3, 4, 5]])
        case "activation":
            input_ids, _ = geneformer.tokenize_with_perturbations(x_ng, feature_activation=["d"])
            expected_input_ids = torch.tensor([[5, 2, 3, 4]])
        case "deletion":
            input_ids, _ = geneformer.tokenize_with_perturbations(x_ng, feature_deletion=["c"])
            expected_input_ids = torch.tensor([[2, 3, 5, 0]])
        case "map":
            input_ids, _ = geneformer.tokenize_with_perturbations(x_ng, feature_map={"a": 1, "b": 1})
            expected_input_ids = torch.tensor([[1, 1, 4, 5]])

    print(f"Expected input_ids:\n{expected_input_ids}")
    print(f"Actual input_ids:\n{input_ids}")
    torch.testing.assert_close(input_ids, expected_input_ids)

    # test that we raise an AssertionError if we try to perturb something outside schema
    if perturb == "none":
        return
    with pytest.raises(ValueError):
        match perturb:
            case "activation":
                geneformer.tokenize_with_perturbations(x_ng, feature_activation=["e"])
            case "deletion":
                geneformer.tokenize_with_perturbations(x_ng, feature_deletion=["e"])
            case "map":
                geneformer.tokenize_with_perturbations(x_ng, feature_map={"a": 1, "e": 1})
