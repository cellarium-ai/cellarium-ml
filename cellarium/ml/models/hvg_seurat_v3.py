# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

try:
    import skmisc.loess  # noqa: F401
except (ImportError, ModuleNotFoundError):
    warnings.warn("HVGSeuratV3 requires scikit-misc: pip install scikit-misc", UserWarning)
    raise


class HVGSeuratV3(CellariumModel):
    """
    Compute highly variable genes using the Seurat v3 method in two Lightning epochs.

    **Epoch 0** — streams data to accumulate per-batch mean and variance.

    Between epochs — fits a LOESS model of ``log10(var) ~ log10(mean)`` per batch
    to estimate a regularized standard deviation and per-cell clip value.

    **Epoch 1** — streams data again, clips counts per cell at the batch-level
    ``clip_val``, and accumulates clipped sums.

    After epoch 1 (``on_train_epoch_end``) — computes normalized variance per
    batch, ranks genes, and writes ``self.hvg_df`` (and optionally a CSV/Parquet
    file at ``output_path``).

    Usage::

        model = HVGSeuratV3(var_names_g=gene_names, n_top_genes=2000, n_batch=4,
                            batch_key="batch_idx_n", span=0.3)
        trainer = pl.Trainer(max_epochs=2)
        trainer.fit(module, datamodule)
        df = model.hvg_df  # pandas DataFrame, Scanpy-compatible columns

    Args:
        var_names_g:
            Array of gene names, length ``n_genes``.
        n_top_genes:
            Number of highly variable genes to select.
        n_batch:
            Number of batches (use 1 when no batch information is given).
        span:
            LOESS span (fraction of data used per local fit).  Default 0.3.
        output_path:
            If given, the result DataFrame is written to this path after training.
            The format is inferred from the extension (``.csv`` or ``.parquet``).
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        n_top_genes: int,
        n_batch: int = 1,
        span: float = 0.3,
        output_path: str | None = None,
    ) -> None:
        super().__init__()
        self.var_names_g = var_names_g
        n_vars = len(var_names_g)
        self.n_vars = n_vars
        self.n_top_genes = n_top_genes
        self.n_batch = n_batch
        self.span = span
        self.output_path = output_path

        self._current_epoch: int = 0

        # Epoch-0 buffers: shape (n_batch, n_vars)
        self.x_sums_bg: torch.Tensor
        self.x_squared_sums_bg: torch.Tensor
        self.x_size_b: torch.Tensor
        self.register_buffer("x_sums_bg", torch.zeros(n_batch, n_vars))
        self.register_buffer("x_squared_sums_bg", torch.zeros(n_batch, n_vars))
        self.register_buffer("x_size_b", torch.zeros(n_batch))

        # Set between epochs by on_train_epoch_end after epoch 0: shape (n_batch, n_vars)
        self.clip_val_bg: torch.Tensor
        self.reg_std_bg: torch.Tensor
        self.register_buffer("clip_val_bg", torch.zeros(n_batch, n_vars))
        self.register_buffer("reg_std_bg", torch.zeros(n_batch, n_vars))

        # Epoch-1 buffers: shape (n_batch, n_vars)
        self.counts_sum_bg: torch.Tensor
        self.sq_counts_sum_bg: torch.Tensor
        self.register_buffer("counts_sum_bg", torch.zeros(n_batch, n_vars))
        self.register_buffer("sq_counts_sum_bg", torch.zeros(n_batch, n_vars))

        # Dummy parameter so Lightning treats this as a trainable module
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self._dummy_param.data.zero_()

        self.hvg_df: pd.DataFrame | None = None

    def reset_parameters(self) -> None:
        self.x_sums_bg.zero_()
        self.x_squared_sums_bg.zero_()
        self.x_size_b.zero_()
        self.clip_val_bg.zero_()
        self.reg_std_bg.zero_()
        self.counts_sum_bg.zero_()
        self.sq_counts_sum_bg.zero_()
        self._dummy_param.data.zero_()

    # ------------------------------------------------------------------
    # Lightning hook: cache current epoch so forward() can branch on it
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: pl.Trainer) -> None:
        self._current_epoch = trainer.current_epoch

    # ------------------------------------------------------------------
    # Forward: dispatches on epoch
    # ------------------------------------------------------------------

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        if batch_index_n is None:
            batch_index_n = torch.zeros(x_ng.shape[0], dtype=torch.long, device=x_ng.device)
        else:
            batch_index_n = batch_index_n.long()  # needed for scatter_add_

        if self._current_epoch == 0:
            self._accumulate_epoch0(x_ng, batch_index_n)
        elif self._current_epoch == 1:
            self._accumulate_epoch1(x_ng, batch_index_n)
        else:
            raise RuntimeError(f"HVGSeuratV3 expects max_epochs=2, but got epoch {self._current_epoch}.")

        return {}

    def _accumulate_epoch0(self, x_ng: torch.Tensor, batch_idx_n: torch.Tensor) -> None:
        n_cells = x_ng.shape[0]
        idx_exp = batch_idx_n.unsqueeze(1).expand(n_cells, self.n_vars)
        sums_contrib = torch.zeros(self.n_batch, self.n_vars, dtype=x_ng.dtype, device=x_ng.device)
        sq_sums_contrib = torch.zeros(self.n_batch, self.n_vars, dtype=x_ng.dtype, device=x_ng.device)
        sums_contrib.scatter_add_(0, idx_exp, x_ng)
        sq_sums_contrib.scatter_add_(0, idx_exp, x_ng**2)
        self.x_sums_bg = self.x_sums_bg + sums_contrib
        self.x_squared_sums_bg = self.x_squared_sums_bg + sq_sums_contrib
        self.x_size_b = self.x_size_b + torch.bincount(batch_idx_n, minlength=self.n_batch)

    def _accumulate_epoch1(self, x_ng: torch.Tensor, batch_idx_n: torch.Tensor) -> None:
        n_cells = x_ng.shape[0]
        # Per-cell clip value: shape (n_cells, n_vars)
        per_cell_clip = self.clip_val_bg[batch_idx_n].to(x_ng.dtype)  # (n_cells, n_vars)
        x_clipped = torch.minimum(x_ng, per_cell_clip)
        idx_exp = batch_idx_n.unsqueeze(1).expand(n_cells, self.n_vars)
        sums_contrib = torch.zeros(self.n_batch, self.n_vars, dtype=x_ng.dtype, device=x_ng.device)
        sq_sums_contrib = torch.zeros(self.n_batch, self.n_vars, dtype=x_ng.dtype, device=x_ng.device)
        sums_contrib.scatter_add_(0, idx_exp, x_clipped)
        sq_sums_contrib.scatter_add_(0, idx_exp, x_clipped**2)
        self.counts_sum_bg = self.counts_sum_bg + sums_contrib
        self.sq_counts_sum_bg = self.sq_counts_sum_bg + sq_sums_contrib

    # ------------------------------------------------------------------
    # on_train_start: validate DDP config
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), "HVGSeuratV3 requires the DDP strategy."
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is False, (
                "HVGSeuratV3 requires broadcast_buffers=False."
            )

    # ------------------------------------------------------------------
    # on_train_epoch_end: reduce → LOESS (after epoch 0) or finalize (after epoch 1)
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        if trainer.current_epoch == 0:
            self._finish_epoch0(trainer)
        elif trainer.current_epoch == 1:
            self._finish_epoch1(trainer)

    def _finish_epoch0(self, trainer: pl.Trainer) -> None:
        # 1. Reduce epoch-0 buffers to rank 0
        if trainer.world_size > 1:
            dist.reduce(self.x_sums_bg, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(self.x_squared_sums_bg, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(self.x_size_b, dst=0, op=dist.ReduceOp.SUM)

        # 2. Rank 0: compute mean/var per batch; fit LOESS; set clip_val_bg and reg_std_bg
        if trainer.global_rank == 0:
            self._compute_clip_val()

        # 3. Broadcast clip_val_bg to all ranks so epoch-1 can use it
        if trainer.world_size > 1:
            dist.broadcast(self.clip_val_bg, src=0)
            dist.broadcast(self.reg_std_bg, src=0)

    def _compute_clip_val(self) -> None:
        n_vars = self.n_vars
        for b in range(self.n_batch):
            N = self.x_size_b[b].item()
            if N < 2:
                continue
            mean_g = (self.x_sums_bg[b] / N).cpu().numpy().astype(np.float64)
            var_g = (self.x_squared_sums_bg[b] / N - (self.x_sums_bg[b] / N) ** 2).cpu().numpy().astype(np.float64)

            not_const = var_g > 0
            estimated_var = np.zeros(n_vars, dtype=np.float64)

            if not_const.any():
                x = np.log10(mean_g[not_const])
                y = np.log10(var_g[not_const])
                jitter = 0.0
                max_jitter = 1e-6
                while True:
                    try:
                        _x = x + np.random.default_rng(0).uniform(-jitter, jitter, x.shape[0]) if jitter > 0 else x
                        model = skmisc.loess.loess(_x, y, span=self.span, degree=2)
                        model.fit()
                        estimated_var[not_const] = model.outputs.fitted_values
                        break
                    except ValueError:
                        jitter = 1e-18 if jitter == 0 else jitter * 10
                        if jitter > max_jitter:
                            raise

            reg_std = np.sqrt(10**estimated_var)  # shape (n_vars,)
            clip_val = reg_std * np.sqrt(N) + mean_g

            self.reg_std_bg[b] = torch.tensor(reg_std, dtype=torch.float32)
            self.clip_val_bg[b] = torch.tensor(clip_val, dtype=torch.float32)

    def _finish_epoch1(self, trainer: pl.Trainer) -> None:
        # 1. Reduce epoch-1 buffers to rank 0
        if trainer.world_size > 1:
            dist.reduce(self.counts_sum_bg, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(self.sq_counts_sum_bg, dst=0, op=dist.ReduceOp.SUM)

        # 2. Rank 0: compute norm_var, rank, select top genes, build DataFrame
        if trainer.global_rank == 0:
            self.hvg_df = self._compute_hvg_df()
            if self.output_path is not None:
                self._save(self.hvg_df)

    def _compute_hvg_df(self) -> pd.DataFrame:
        n_vars = self.n_vars
        norm_gene_vars = np.zeros((self.n_batch, n_vars), dtype=np.float64)

        for b in range(self.n_batch):
            N = self.x_size_b[b].item()
            if N < 2:
                continue
            mean_bg = (self.x_sums_bg[b] / N).cpu().numpy().astype(np.float64)
            reg_std = self.reg_std_bg[b].cpu().numpy().astype(np.float64)
            sum_bg = self.counts_sum_bg[b].cpu().numpy().astype(np.float64)
            sq_sum_bg = self.sq_counts_sum_bg[b].cpu().numpy().astype(np.float64)

            denom = (N - 1) * reg_std**2
            with np.errstate(divide="ignore", invalid="ignore"):
                nv = (N * mean_bg**2 + sq_sum_bg - 2 * mean_bg * sum_bg) / denom
            nv[np.isnan(nv)] = 0.0
            norm_gene_vars[b] = nv

        # Rank genes within each batch
        ranked = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1).astype(np.float32)
        num_batches_high_var = (ranked < self.n_top_genes).sum(axis=0).astype(int)
        ranked[ranked >= self.n_top_genes] = np.nan
        ma = np.ma.masked_invalid(ranked)
        median_ranked = np.ma.median(ma, axis=0).filled(np.nan)

        variances_norm = norm_gene_vars.mean(axis=0)

        df = pd.DataFrame(
            index=pd.Index(self.var_names_g, name="gene"),
            data={
                "highly_variable_nbatches": num_batches_high_var,
                "highly_variable_rank": median_ranked,
                "variances_norm": variances_norm,
            },
        )

        # Use integer-position sort so duplicate gene names don't cause
        # df.loc to mark more than n_top_genes rows as highly variable.
        rank_vals = df["highly_variable_rank"].fillna(np.inf).values
        sort_positions = np.lexsort((-df["highly_variable_nbatches"].values, rank_vals))
        hvg_flags = np.zeros(len(df), dtype=bool)
        hvg_flags[sort_positions[: self.n_top_genes]] = True
        df["highly_variable"] = hvg_flags

        if self.n_batch == 1:
            df = df.drop(columns=["highly_variable_nbatches"])

        return df

    def _save(self, df: pd.DataFrame) -> None:
        assert self.output_path is not None
        if self.output_path.endswith(".parquet"):
            df.to_parquet(self.output_path)
        else:
            df.to_csv(self.output_path)
