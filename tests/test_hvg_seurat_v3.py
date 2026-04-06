# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pytest
import scipy.sparse
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.cli import main
from cellarium.ml.models import HVGSeuratV3
from cellarium.ml.models.hvg_seurat_v3 import _fit_loess_with_jitter
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
        batch_key: str | None = None,
    ) -> None:
        self.x = x.astype(np.float32)
        self.var_names = var_names
        self.batch_idx = batch_idx
        self.batch_key = batch_key

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        batch_index = self.batch_idx[idx : idx + 1] if self.batch_idx is not None else np.zeros(1, dtype=np.int64)
        batch = {
            "x_ng": self.x[idx : idx + 1],  # shape (1, n_genes)
            "var_names_g": self.var_names,
        }
        if self.batch_key is not None:
            batch[self.batch_key] = batch_index  # shape (1,)
        return batch


def _run_model(
    x: np.ndarray,
    var_names: np.ndarray,
    n_batch: int,
    batch_idx: np.ndarray | None,
    batch_key: str | None,
    output_path: str,
    batch_size: int = 512,
    flavor: Literal["seurat_v3", "seurat_v3_paper"] = "seurat_v3",
) -> HVGSeuratV3:
    """Instantiate, fit (2 epochs), and return the HVGSeuratV3 model."""
    model = HVGSeuratV3(
        var_names_g=var_names,
        n_top_genes=N_TOP_GENES,
        n_batch=n_batch,
        use_batch_key=(batch_key is not None),
        flavor=flavor,
        output_path=output_path,
    )
    module = CellariumModule(model=model)

    dataset = _HVGDataset(x, var_names, batch_idx=batch_idx, batch_key=batch_key)
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
# Unit tests (no network required)
# ---------------------------------------------------------------------------


def test_hvg_seurat_v3_sort_order():
    """
    Verify the sort order for ``flavor='seurat_v3'`` in _compute_hvg_df matches
    Scanpy's seurat_v3 flavor:
      - Primary  key: highly_variable_rank ASCENDING (lower rank = more variable)
      - Secondary key: highly_variable_nbatches DESCENDING (tiebreaker)

    Two assertions are made:
    1. A gene with median_rank=0 in one batch is preferred over a gene with
       median_rank=2 in both batches — rank is primary, not nbatches.
    2. Among genes with equal median_rank=2, the gene in two batches (nbatches=2)
       is selected over the gene in only one batch (nbatches=1).
    """
    # 8 genes, 2 batches, n_top_genes=4
    #
    # Batch 0 norm_var ordering (best→worst): g0, g1(=gene_B), g2(=gene_A), g3, g4, g5, g6, g7
    # Batch 1 norm_var ordering (best→worst): g4, g5, g6, g1(=gene_B), g0, g2(=gene_A), g3, g7
    #
    # With n_top_genes=4 (ranks 0-3 qualify):
    #   g0:      b0-rank=0 (✓), b1-rank=4 (✗) → nbatches=1, median_rank=0
    #   gene_B(g1): b0-rank=1 (✓), b1-rank=3 (✓) → nbatches=2, median_rank=2.0
    #   gene_A(g2): b0-rank=2 (✓), b1-rank=5 (✗) → nbatches=1, median_rank=2.0  ← same rank as gene_B!
    #   g3:      b0-rank=3 (✓), b1-rank=6 (✗) → nbatches=1, median_rank=3
    #   g4:      b0-rank=4 (✗), b1-rank=0 (✓) → nbatches=1, median_rank=0
    #   g5:      b0-rank=5 (✗), b1-rank=1 (✓) → nbatches=1, median_rank=1
    #   g6:      b0-rank=6 (✗), b1-rank=2 (✓) → nbatches=1, median_rank=2.0  ← same rank as gene_B!
    #   g7:      b0-rank=7 (✗), b1-rank=7 (✗) → nbatches=0, median_rank=nan
    #
    # Correct top-4 (rank primary, nbatches tiebreaker):
    #   rank=0: g0, g4 → both selected
    #   rank=1: g5       → selected
    #   rank=2 tie: gene_B(nbatches=2) > gene_A(nbatches=1), gene_B > g6(nbatches=1) → gene_B selected
    # → selected = {g0, g4, g5, gene_B}; gene_A NOT selected despite having same rank as gene_B.
    n_genes = 8
    n_top = 4
    N = 100
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    model = HVGSeuratV3(var_names_g=gene_names, n_top_genes=n_top, n_batch=2, flavor="seurat_v3")

    # Norm-var values chosen so that double-argsort gives the ranks listed above.
    # With x_sums_bg=0 and reg_std=1: nv = sq_counts_sum / (N-1)
    nv_b0 = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]  # g0..g7 in batch 0
    nv_b1 = [60.0, 70.0, 50.0, 40.0, 100.0, 90.0, 80.0, 30.0]  # g0..g7 in batch 1

    model.x_size_b = torch.full((2,), float(N))
    model.reg_std_bg = torch.ones(2, n_genes)
    model.x_sums_bg = torch.zeros(2, n_genes)
    model.counts_sum_bg = torch.zeros(2, n_genes)
    model.sq_counts_sum_bg = torch.stack(
        [
            torch.tensor([v * (N - 1) for v in nv_b0]),
            torch.tensor([v * (N - 1) for v in nv_b1]),
        ]
    )

    df = model._compute_hvg_df()

    assert df["highly_variable"].sum() == n_top

    # Assertion 1: rank is primary — g0 (rank=0, nbatches=1) must be selected
    # even though gene_B (nbatches=2) has more batch coverage.
    assert df.loc["gene_0", "highly_variable"], (
        "gene_0 (rank=0, nbatches=1) should be selected — median rank is the primary sort key"
    )

    # Assertion 2: nbatches is the tiebreaker at equal median_rank=2 —
    # gene_1(gene_B, nbatches=2) must be selected over gene_2(gene_A, nbatches=1).
    assert df.loc["gene_1", "highly_variable"], (
        "gene_1 (nbatches=2, median_rank=2) must be selected as the tiebreaker winner "
        "over gene_2 and gene_6 which also have median_rank=2 but nbatches=1"
    )
    assert not df.loc["gene_2", "highly_variable"], (
        "gene_2 (nbatches=1, median_rank=2) should NOT be selected — gene_1 (nbatches=2, same rank) wins the tiebreaker"
    )


@pytest.mark.parametrize(
    "flavor,expected_hvg,excluded_hvg",
    [
        # seurat_v3: rank primary → genes with median_rank=0 fill the top-2 slots;
        # gene_1 (nbatches=3, rank=1) loses to gene_0 and gene_2 (both rank=0).
        ("seurat_v3", {"gene_0", "gene_2"}, {"gene_1"}),
        # seurat_v3_paper: nbatches primary → gene_1 (nbatches=3) takes the first
        # slot regardless of rank; gene_0 (rank=0) takes the second slot.
        # gene_2 (nbatches=1, rank=0) is displaced by gene_1.
        ("seurat_v3_paper", {"gene_0", "gene_1"}, {"gene_2"}),
    ],
    ids=["seurat_v3", "seurat_v3_paper"],
)
def test_hvg_seurat_v3_flavor_sort_order(flavor, expected_hvg, excluded_hvg):
    """
    Verify that the two flavors produce *different* selected gene sets on data
    specifically constructed so the flavors diverge.

    Setup: 4 genes, 3 batches, n_top_genes=2.

    Batch norm_var ordering (ranks derived from values below):
      Batch 0: gene_0 > gene_1 > gene_2 > gene_3
      Batch 1: gene_2 > gene_1 > gene_0 > gene_3
      Batch 2: gene_3 > gene_1 > gene_0 > gene_2

    Resulting nbatches/median_rank (n_top=2 per batch):
      gene_0: nbatches=1, median_rank=0  (top in batch 0 only)
      gene_1: nbatches=3, median_rank=1  (rank 1 in every batch)
      gene_2: nbatches=1, median_rank=0  (top in batch 1 only)
      gene_3: nbatches=1, median_rank=0  (top in batch 2 only)

    seurat_v3   (rank primary):    top-2 = {gene_0, gene_2}  — gene_1 rank=1 loses
    seurat_v3_paper (nbatches primary): top-2 = {gene_0, gene_1}  — gene_2 displaced
    """
    n_genes = 4
    n_top = 2
    N = 100
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    model = HVGSeuratV3(var_names_g=gene_names, n_top_genes=n_top, n_batch=3, flavor=flavor)

    # With x_sums_bg=0 and reg_std=1:
    #   nv = sq_counts_sum / (N-1)  →  set sq_counts_sum = nv * (N-1)
    nv_b0 = [100.0, 90.0, 80.0, 70.0]  # batch 0 ordering: gene_0>gene_1>gene_2>gene_3
    nv_b1 = [80.0, 90.0, 100.0, 70.0]  # batch 1 ordering: gene_2>gene_1>gene_0>gene_3
    nv_b2 = [80.0, 90.0, 70.0, 100.0]  # batch 2 ordering: gene_3>gene_1>gene_0>gene_2

    model.x_size_b = torch.full((3,), float(N))
    model.reg_std_bg = torch.ones(3, n_genes)
    model.x_sums_bg = torch.zeros(3, n_genes)
    model.counts_sum_bg = torch.zeros(3, n_genes)
    model.sq_counts_sum_bg = torch.stack(
        [
            torch.tensor([v * (N - 1) for v in nv_b0]),
            torch.tensor([v * (N - 1) for v in nv_b1]),
            torch.tensor([v * (N - 1) for v in nv_b2]),
        ]
    )

    df = model._compute_hvg_df()

    assert df["highly_variable"].sum() == n_top
    for gene in expected_hvg:
        assert df.loc[gene, "highly_variable"], (
            f"{gene} should be selected under flavor={flavor!r} "
            f"but was not. Selected: {set(df[df['highly_variable']].index)}"
        )
    for gene in excluded_hvg:
        assert not df.loc[gene, "highly_variable"], (
            f"{gene} should NOT be selected under flavor={flavor!r} "
            f"but was. Selected: {set(df[df['highly_variable']].index)}"
        )


def test_hvg_seurat_v3_invalid_batch_clip_not_zero():
    """
    Regression test: a batch with N < 2 must get clip_val=+inf, not 0.

    Before the fix, _compute_clip_val skipped invalid batches and left
    clip_val_bg[b]=0.  Epoch 1 then called torch.minimum(x_ng, 0), clipping
    every count in that batch down to zero and corrupting the sums.
    """
    n_genes = 30
    N = 100
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    model = HVGSeuratV3(var_names_g=gene_names, n_top_genes=5, n_batch=2)

    model.x_size_b = torch.tensor([float(N), 1.0])  # batch 1 has N=1 → invalid

    # Give batch 0 genes with varied means so LOESS can fit (overdispersed model):
    # mean_j = (j+1)*0.5, var_j = mean_j * 2
    means = torch.arange(1, n_genes + 1, dtype=torch.float32) * 0.5
    vars_ = means * 2.0
    model.x_sums_bg[0] = means * N
    model.x_squared_sums_bg[0] = (vars_ + means**2) * N

    model._compute_clip_val()

    assert (model.clip_val_bg[1] == float("inf")).all(), (
        "Invalid batch (N<2) must have clip_val=+inf so epoch-1 clipping is a no-op"
    )
    assert (model.clip_val_bg[0] > 0).all()
    assert not (model.clip_val_bg[0] == float("inf")).any()


def test_fit_loess_with_jitter_jitter_recovery():
    """
    Verify that _fit_loess_with_jitter retries with increasing jitter when LOESS
    fails on the first attempt.

    All-zeros x causes a rank-deficient design matrix that skmisc.loess rejects.
    The retry loop must add enough jitter to create numerical spread and eventually
    return a fitted-values array of the correct shape.
    """
    n = 50
    x = np.zeros(n)
    y = np.linspace(1.0, 2.0, n)
    result = _fit_loess_with_jitter(x, y, span=0.5)  # default max_jitter=1e-6
    assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}"


def test_fit_loess_with_jitter_raises_on_exhaustion():
    """
    Verify that _fit_loess_with_jitter raises ValueError when jitter is capped
    at a value so small that LOESS never succeeds.

    Setting max_jitter=0.0 means: the while loop runs once (0.0 <= 0.0), the
    first attempt fails on all-zeros x, jitter is then set to initial_jitter
    which exceeds max_jitter, so the loop exits and ValueError is raised.
    """
    x = np.zeros(50)
    y = np.linspace(1.0, 2.0, 50)
    with pytest.raises(ValueError, match="LOESS fit failed"):
        _fit_loess_with_jitter(x, y, span=0.5, max_jitter=0.0)


def test_hvg_seurat_v3_invalid_batch_excluded_from_ranking():
    """
    Regression test: a batch with N < 2 must not contribute to
    highly_variable_nbatches or median rank in _compute_hvg_df.

    Before the fix, all-zero norm_vars for an invalid batch were still ranked,
    causing the first n_top_genes to each gain an extra spurious nbatches count.
    """
    n_genes = 6
    n_top = 3
    N = 100
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    model = HVGSeuratV3(var_names_g=gene_names, n_top_genes=n_top, n_batch=2, flavor="seurat_v3")

    model.x_size_b = torch.tensor([float(N), 1.0])  # batch 1 invalid
    model.reg_std_bg = torch.ones(2, n_genes)
    model.x_sums_bg = torch.zeros(2, n_genes)
    model.counts_sum_bg = torch.zeros(2, n_genes)
    # Give batch 0 distinct norm_vars; batch 1 buffers stay zero (N=1, skipped)
    nv_b0 = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]
    model.sq_counts_sum_bg[0] = torch.tensor([v * (N - 1) for v in nv_b0])

    df = model._compute_hvg_df()

    assert df["highly_variable"].sum() == n_top
    assert df["highly_variable_nbatches"].max() == 1, (
        "Invalid batch (N<2) should not be counted in highly_variable_nbatches; "
        f"max was {df['highly_variable_nbatches'].max()} but expected 1"
    )


# ---------------------------------------------------------------------------
# Integration tests (require network to download E-MTAB-10137)
# ---------------------------------------------------------------------------


def _load_adata():
    import scanpy as sc

    adata, batch_column = sc.datasets.ebi_expression_atlas("E-MTAB-10137"), "Sample Characteristic[individual]"
    adata.obs[batch_column] = adata.obs[batch_column].astype("category")
    return adata, batch_column


def test_hvg_seurat_v3_paper_matches_scanpy(tmp_path):
    """
    End-to-end test: compare HVGSeuratV3 (``flavor='seurat_v3_paper'``) against
    Scanpy in multi-batch mode.

    ``seurat_v3`` with batch_key is already covered by
    ``test_hvg_seurat_v3_matches_scanpy[with_batch]``.  This test validates the
    ``seurat_v3_paper`` sort order (nbatches-primary) against Scanpy's own
    implementation of that flavor.
    """
    import scanpy as sc

    flavor: Literal["seurat_v3", "seurat_v3_paper"] = "seurat_v3_paper"
    adata, batch_col = _load_adata()
    x = _to_dense(adata.X)
    var_names = np.asarray(adata.var_names)

    batch_cats = list(adata.obs[batch_col].cat.categories)
    n_batch = len(batch_cats)
    cat_to_idx = {c: i for i, c in enumerate(batch_cats)}
    batch_idx = np.array([cat_to_idx[c] for c in adata.obs[batch_col]], dtype=np.int64)

    sc_df = sc.pp.highly_variable_genes(
        adata.copy(),
        flavor=flavor,
        n_top_genes=N_TOP_GENES,
        batch_key=batch_col,
        inplace=False,
    )

    output_file = str(tmp_path / f"hvg_{flavor}.csv")
    model = _run_model(
        x=x,
        var_names=var_names,
        n_batch=n_batch,
        batch_idx=batch_idx,
        batch_key="batch_index_n",
        output_path=output_file,
        flavor=flavor,
    )

    hvg_df = model.hvg_df
    assert hvg_df is not None
    assert hvg_df["highly_variable"].sum() == N_TOP_GENES

    our_hvg = set(hvg_df[hvg_df["highly_variable"]].index)
    sc_hvg = set(sc_df[sc_df["highly_variable"]].index)
    jaccard = len(our_hvg & sc_hvg) / len(our_hvg | sc_hvg)
    assert jaccard >= 0.98, f"Jaccard {jaccard:.4f} below 0.98 for flavor={flavor!r} with batch_key"


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
        model_batch_key = "batch_index_n"
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

    # Jaccard threshold
    threshold = 0.98
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
                        "use_batch_key": True,
                        "flavor": "seurat_v3",
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
