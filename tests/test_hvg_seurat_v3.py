# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import lightning.pytorch as pl
import numpy as np
import pytest
import scipy.sparse
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.cli import main
from cellarium.ml.models import HVGSeuratV3
from cellarium.ml.utilities.data import collate_fn

N_TOP_GENES = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dense(x):
    return x.toarray() if scipy.sparse.issparse(x) else np.asarray(x)


class _HVGDataset(torch.utils.data.Dataset):
    """
    In-memory dataset that wraps a raw-count matrix for HVGSeuratV3 tests.

    Each item is a dict with:
        ``x_ng``        - single-row float32 count slice, shape ``(1, n_genes)``
        ``var_names_g`` - gene-name array (same for every item)
        ``<batch_key>`` - (optional) int64 batch-index slice, shape ``(1,)``
    """

    def __init__(
        self,
        x: np.ndarray,
        var_names: np.ndarray,
        batch_idx: np.ndarray | None = None,
        batch_key: str = "batch_idx_n",
    ) -> None:
        self.x = x.astype(np.float32)
        self.var_names = var_names
        self.batch_idx = batch_idx
        self.batch_key = batch_key

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        batch_index = self.batch_idx[idx : idx + 1] if self.batch_idx is not None else np.zeros(1, dtype=np.int64)
        return {
            "x_ng": self.x[idx : idx + 1],  # shape (1, n_genes)
            "var_names_g": self.var_names,
            "batch_index_n": batch_index,
        }


def _run_model(
    x: np.ndarray,
    var_names: np.ndarray,
    n_batch: int,
    batch_idx: np.ndarray | None,
    batch_key: str | None,
    output_path: str,
    batch_size: int = 512,
) -> HVGSeuratV3:
    """Instantiate, fit (2 epochs), and return the HVGSeuratV3 model."""
    model = HVGSeuratV3(
        var_names_g=var_names,
        n_top_genes=N_TOP_GENES,
        n_batch=n_batch,
        output_path=output_path,
    )
    module = CellariumModule(model=model)

    dataset = _HVGDataset(x, var_names, batch_idx=batch_idx, batch_key=batch_key or "batch_idx_n")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=loader)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _load_adata():
    import scanpy as sc

    adata, batch_column = sc.datasets.ebi_expression_atlas("E-MTAB-10137"), "Sample Characteristic[individual]"
    adata.obs[batch_column] = adata.obs[batch_column].astype("category")
    return adata, batch_column


@pytest.mark.parametrize(
    "use_batch_key",
    [False, True],
    ids=["no_batch", "with_batch"],
)
def test_hvg_seurat_v3_matches_scanpy(tmp_path, use_batch_key: bool):
    """
    End-to-end test: run HVGSeuratV3 through pl.Trainer(max_epochs=2) and
    compare the selected HVG set with scanpy.pp.highly_variable_genes
    (flavor='seurat_v3').

    seurat_v3 expects *raw counts* (no log-transform), so we feed the count
    matrix directly to both our model and Scanpy.
    """
    import pandas as pd
    import scanpy as sc

    adata, batch_col = _load_adata()
    x = _to_dense(adata.X)
    var_names = np.asarray(adata.var_names)

    if use_batch_key:
        batch_cats = list(adata.obs[batch_col].cat.categories)
        n_batch = len(batch_cats)
        cat_to_idx = {c: i for i, c in enumerate(batch_cats)}
        batch_idx = np.array([cat_to_idx[c] for c in adata.obs[batch_col]], dtype=np.int64)
        sc_batch_key = batch_col
        model_batch_key = "batch_idx_n"
    else:
        n_batch = 1
        batch_idx = None
        sc_batch_key = None
        model_batch_key = None

    # ---- Scanpy reference (seurat_v3 uses raw counts) --------------------
    sc_df = sc.pp.highly_variable_genes(
        adata.copy(),
        flavor="seurat_v3",
        n_top_genes=N_TOP_GENES,
        batch_key=sc_batch_key,
        inplace=False,
    )

    # ---- Our model -------------------------------------------------------
    output_file = str(tmp_path / "hvg_seurat_v3.csv")
    model = _run_model(
        x=x,
        var_names=var_names,
        n_batch=n_batch,
        batch_idx=batch_idx,
        batch_key=model_batch_key,
        output_path=output_file,
    )

    # ---- Assertions: output artefacts ------------------------------------
    hvg_df = model.hvg_df
    assert hvg_df is not None, "HVGSeuratV3.hvg_df was not populated after training"
    assert os.path.exists(output_file), "on_train_epoch_end did not write the output CSV"

    # The saved file should round-trip cleanly
    saved_df = pd.read_csv(output_file, index_col=0)
    assert set(saved_df.columns).issuperset({"highly_variable"}), (
        f"Saved CSV is missing expected columns; got {list(saved_df.columns)}"
    )

    # ---- Assertions: HVG count ------------------------------------------
    assert "highly_variable" in hvg_df.columns
    assert hvg_df["highly_variable"].sum() == N_TOP_GENES, (
        f"Expected {N_TOP_GENES} HVGs, got {hvg_df['highly_variable'].sum()}"
    )

    # ---- Assertions: overlap with Scanpy --------------------------------
    our_hvg = set(hvg_df[hvg_df["highly_variable"]].index)
    sc_hvg = set(sc_df[sc_df["highly_variable"]].index)
    union = our_hvg | sc_hvg
    jaccard = len(our_hvg & sc_hvg) / len(union)

    # Multi-batch ranking introduces extra ordering variation; allow slightly
    # more slack there.
    threshold = 0.98 if use_batch_key else 0.98
    assert jaccard >= threshold, (
        f"Jaccard similarity {jaccard:.4f} is below {threshold} for seurat_v3 "
        f"{'with' if use_batch_key else 'without'} batch_key"
    )

    # ---- Assertions: batch-specific columns ------------------------------
    if use_batch_key:
        assert "highly_variable_nbatches" in hvg_df.columns, (
            "highly_variable_nbatches column missing from batch-mode output"
        )
        assert hvg_df["highly_variable_nbatches"].between(0, n_batch).all(), (
            "highly_variable_nbatches contains out-of-range values"
        )


def test_hvg_seurat_v3_cli_matches_scanpy(tmp_path):
    """
    End-to-end CLI integration test: run HVGSeuratV3 via the Lightning CLI
    (``main()``) backed by a :class:`~cellarium.ml.CellariumAnnDataDataModule`
    and compare the output CSV against scanpy's seurat_v3 HVG results.

    seurat_v3 requires raw counts – no log-normalisation is applied.
    The batch column is encoded as integer codes via ``batch_index_n``;
    the number of batches is wired automatically by the CLI through the
    ``batch_n_categories`` batch key.
    """
    import pandas as pd
    import scanpy as sc

    adata, batch_col = _load_adata()
    adata_path = tmp_path / "test.h5ad"
    adata.write_h5ad(adata_path)
    batch_cats = list(adata.obs[batch_col].cat.categories)
    n_batch = len(batch_cats)
    output_file = str(tmp_path / "hvg_cli.csv")

    config = {
        "model_name": "hvg_seurat_v3",
        "subcommand": "fit",
        "fit": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.HVGSeuratV3",
                    "init_args": {
                        "n_top_genes": str(N_TOP_GENES),
                        "n_batch": None,  # wired by CLI from batch_n_categories
                        "output_path": output_file,
                    },
                },
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": str(adata_path),
                        "shard_size": str(adata.n_obs),
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [batch_col],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                    "batch_index_n": {
                        "attr": "obs",
                        "key": batch_col,
                        "convert_fn": "cellarium.ml.utilities.data.categories_to_codes",
                    },
                    "batch_n_categories": {
                        "attr": "obs",
                        "key": batch_col,
                        "convert_fn": "cellarium.ml.utilities.data.get_categories",
                    },
                },
                "batch_size": "512",
                "num_workers": "0",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": "1",
                "default_root_dir": str(tmp_path),
                "enable_checkpointing": "false",
                "enable_progress_bar": "false",
                "enable_model_summary": "false",
            },
        },
    }

    main(config)

    # ---- Output file was written by on_train_epoch_end -------------------
    assert os.path.exists(output_file), "CLI did not write the HVG output CSV"
    hvg_df = pd.read_csv(output_file, index_col=0)

    # ---- Basic structural checks ----------------------------------------
    assert "highly_variable" in hvg_df.columns
    assert hvg_df["highly_variable"].sum() == N_TOP_GENES, (
        f"Expected {N_TOP_GENES} HVGs in CSV, got {hvg_df['highly_variable'].sum()}"
    )
    assert "highly_variable_nbatches" in hvg_df.columns, "highly_variable_nbatches column missing from batch-mode CSV"
    assert hvg_df["highly_variable_nbatches"].between(0, n_batch).all()

    # ---- Scanpy reference (seurat_v3 uses raw counts) --------------------
    sc_df = sc.pp.highly_variable_genes(
        adata.copy(),
        flavor="seurat_v3",
        n_top_genes=N_TOP_GENES,
        batch_key=batch_col,
        inplace=False,
    )

    our_hvg = set(hvg_df[hvg_df["highly_variable"]].index)
    sc_hvg = set(sc_df[sc_df["highly_variable"]].index)
    jaccard = len(our_hvg & sc_hvg) / len(our_hvg | sc_hvg)
    assert jaccard >= 0.98, f"CLI Jaccard similarity {jaccard:.4f} is below 0.98 for seurat_v3 with batch_key"
