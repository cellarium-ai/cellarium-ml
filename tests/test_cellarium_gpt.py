# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.data import PyTreeDataset, read_h5ad_file
from cellarium.ml.models import CellariumGPT
from cellarium.ml.utilities.data import categories_to_codes, collate_fn


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    adata = read_h5ad_file("https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad")
    n = adata.n_obs
    s = 4  # number of subsampled genes
    c = 5  # context size
    batch_size = 2
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    prompt_mask_nc = np.random.choice([True, False], size=(n, c), p=[0.5, 0.5])
    query_mask_nc = (~prompt_mask_nc).astype(np.float32)
    token_type_nc = np.full((n, c), fill_value=15, dtype=np.int64)  # 0111 to select gene tokens
    token_type_nc[:, s:] = 16  # 1000 to select cell type token
    X = adata.X[:, :s].toarray()

    cell_type = categories_to_codes(adata.obs["cell_type"])[:, None]
    data = {
        "token_nc_dict": {
            "gene_id": np.broadcast_to(np.arange(c), (n, c)),
            "gene_value": np.concatenate([X, np.zeros((n, 1), dtype=np.float32)], axis=1),
            "gene_query_mask": query_mask_nc,
            "total_mrna_umis": np.broadcast_to(
                np.asarray(adata.obs["total_mrna_umis"], dtype=np.float32)[:, None], (n, c)
            ),
            "cell_type": np.broadcast_to(cell_type, (n, c)),
        },
        # use random numbers between 0 and 31
        "token_type_nc": token_type_nc,
        "prompt_mask_nc": prompt_mask_nc,
        "label_nc_dict": {
            "gene_value": np.concatenate([X, np.zeros((n, 1))], axis=1),
            "cell_type": np.concatenate([np.zeros((n, s)), cell_type], axis=1),
        },
        "label_weight_nc_dict": {
            "gene_value": (np.broadcast_to(np.arange(c), (n, c)) < s).astype(np.float32),
            "cell_type": (np.broadcast_to(np.arange(c), (n, c)) == s).astype(np.float32),
        },
    }
    dataset = PyTreeDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )
    # model
    model = CellariumGPT(
        gene_vocab_sizes={"gene_id": c, "gene_value": 100},
        metadata_vocab_sizes={"cell_type": adata.obs["cell_type"].cat.categories.size},
        d_model=3,
        d_ffn=6,
        n_heads=1,
        n_blocks=1,
        dropout_p=0,
        use_bias=False,
        attention_backend="torch",
        attention_softmax_fp32=True,
        loss_scales={"gene_value": 0.8, "cell_type": 0.2},
        attention_logits_scale=1,
        mup_base_d_model=2,
        mup_base_d_ffn=4,
    )
    module = CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3, "eps": 1e-8})
    # trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=devices,
        max_steps=2,
        default_root_dir=tmp_path,
    )
    # fit
    trainer.fit(module, dataloader)

    # run tests only for rank 0
    if trainer.global_rank != 0:
        return

    # load model from checkpoint
    ckpt_path = tmp_path / "lightning_logs/version_0/checkpoints/epoch=0-step=2.ckpt"
    assert ckpt_path.is_file()
    loaded_model = CellariumModule.load_from_checkpoint(ckpt_path).model
    assert isinstance(loaded_model, CellariumGPT)
    # assert
    assert model.attention_backend == loaded_model.attention_backend
    assert model.embeddings_scale == loaded_model.embeddings_scale
    assert model.attention_logits_scale == loaded_model.attention_logits_scale
    assert model.output_logits_scale == loaded_model.output_logits_scale
