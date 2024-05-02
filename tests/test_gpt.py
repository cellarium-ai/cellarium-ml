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
from cellarium.ml.models.cellarium_gpt import CellariumGPT, ScaledDotProductAttention
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


@pytest.mark.parametrize("backend", ["keops", "torch", "math"])
@pytest.mark.parametrize("attention_type", ["block", "block_diagonal", "full"])
def test_scaled_dot_product_attention(backend, attention_type):
    batch_size = 2
    query_len = 3
    seq_len = 4
    dim = 5
    queries_nqd = torch.rand(batch_size, query_len, dim)
    keys_nsd = torch.rand(batch_size, seq_len, dim)
    values_nsv = torch.rand(batch_size, seq_len, dim)
    prefix_len_n = torch.tensor([2, 1])
    if attention_type == "block":
        block_mask_nqs = torch.arange(seq_len).expand([query_len, seq_len]) < prefix_len_n[:, None, None]
        attention_mask_nqs = block_mask_nqs
    elif attention_type == "block_diagonal":
        block_mask_nqs = torch.arange(seq_len).expand([query_len, seq_len]) < prefix_len_n[:, None, None]
        diag_mask_qs = torch.arange(query_len)[:, None] == torch.arange(seq_len)
        attention_mask_nqs = block_mask_nqs | diag_mask_qs
    elif attention_type == "full":
        attention_mask_nqs = None

    attn = ScaledDotProductAttention(dropout=0, attn_mult=math.sqrt(dim), backend=backend)
    actual = attn(queries_nqd, keys_nsd, values_nsv, prefix_len_n, attention_type)
    expected = torch.nn.functional.scaled_dot_product_attention(
        queries_nqd.unsqueeze(0),
        keys_nsd.unsqueeze(0),
        values_nsv.unsqueeze(0),
        attention_mask_nqs,
        scale=1 / math.sqrt(dim),
    ).squeeze(0)
    assert actual.size() == (batch_size, query_len, dim)
    assert torch.allclose(actual, expected, atol=1e-6)


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
        dropout=0.0,
        use_bias=False,
        n_context=3,
        attn_mult=math.sqrt(2),
        input_mult=2.0,
        output_mult=1.0,
        initializer_range=0.02,
        backend="torch",
        log_metrics=False,
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
    assert model.gpt_model.input_mult == loaded_model.gpt_model.input_mult
    torch.testing.assert_close(model.gpt_model.id_embedding.weight, loaded_model.gpt_model.id_embedding.weight)
