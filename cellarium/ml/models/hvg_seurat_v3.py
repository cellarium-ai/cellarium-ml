# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import skmisc.loess
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.core.datamodule import CellariumAnnDataDataModule
from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.distributed import get_rank_and_num_replicas
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


def _fit_loess_with_jitter(
    x: np.ndarray,
    y: np.ndarray,
    span: float,
    max_jitter: float = 1e-6,
    initial_jitter: float = 1e-18,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jitter = 0.0

    while jitter <= max_jitter:
        x_fit = x if jitter == 0 else x + rng.uniform(-jitter, jitter, size=x.shape[0])

        try:
            model = skmisc.loess.loess(x_fit, y, span=span, degree=2)
            model.fit()
            return model.outputs.fitted_values
        except ValueError:
            jitter = initial_jitter if jitter == 0 else jitter * 10

    raise ValueError(f"LOESS fit failed after retrying with jitter up to {max_jitter}.")


class HVGSeuratV3(CellariumModel):
    """
    Compute highly variable genes using the Seurat v3 method in two Lightning epochs.

    **Epoch 0** — streams data to accumulate per-batch mean and variance. The sums are of the counts minus a
    per-batch, per-gene shift (the mean of the first cells seen from that batch), so that the variance is not
    computed as a difference of two large, nearly equal numbers.

    Between epochs — fits a LOESS model of ``log10(var) ~ log10(mean)`` per batch
    to estimate a regularized standard deviation and per-cell clip value.

    **Epoch 1** — streams data again, clips counts per cell at the batch-level
    ``clip_val``, and accumulates the sum of squared deviations of the clipped counts from the batch mean.

    After epoch 1 (``on_train_epoch_end``) — computes normalized variance per
    batch, ranks genes, and writes ``self.hvg_dfs`` (and optionally a CSV
    file at ``output_path``).

    Both epochs use only sums of positive terms or of shifted values, so the accumulators can be float32
    (see ``accumulator_dtype``) with little loss of accuracy.

    The ``flavor`` argument controls how genes are ranked across batches when
    ``n_batch > 1``, matching the two multi-batch modes in Scanpy:

    * ``"seurat_v3"`` *(default)* — sorts by ``highly_variable_rank`` ascending first, then
      by ``highly_variable_nbatches`` descending as tiebreaker.  Genes with the
      lowest median rank across batches are preferred.  This matches Scanpy's
      ``flavor='seurat_v3'``.

    * ``"seurat_v3_paper"`` — sorts by ``highly_variable_nbatches``
      descending first, then by ``highly_variable_rank`` ascending as tiebreaker.
      Genes that are highly variable in more batches are preferred, regardless
      of their exact rank.  This matches Scanpy's ``flavor='seurat_v3_paper'``.

    Usage::

        model = HVGSeuratV3(var_names_g=gene_names, n_top_genes=2000, n_batch=4,
                            flavor="seurat_v3_paper", span=0.3)
        trainer = pl.Trainer(max_epochs=2)
        trainer.fit(module, datamodule)
        df = model.hvg_df  # pandas DataFrame, Scanpy-compatible columns

    Args:
        var_names_g: Array of gene names, length ``n_genes``.
        n_top_genes: Number of highly variable genes to select. Can be a list of ints
            to produce multiple gene sets in a single training run.
        n_batch: Number of batches (use 1 when no batch information is given).
        flavor: Multi-batch gene ranking strategy.  ``"seurat_v3_paper"`` (default)
            prioritises batch consistency; ``"seurat_v3"`` prioritises median rank.
        use_batch_key: Whether to expect a ``batch_index_n`` batch key in the dataloader output.
            If False, the model will ignore batch keys and treat all data as a single batch.
        span: LOESS span (fraction of data used per local fit).  Default 0.3.
        output_path: If given, the result DataFrame is written to this CSV filepath after training.
            (Ends with ``.csv``). If ``None``, no file is written, but you really want to
            write this output, as it would require manually calling _compute_hvg_df() later.
        batch_n_cell_minimum: Minimum number of cells required for a batch to be considered valid.
        n_batch_minimum: Minimum number of batches in which the gene is in n_top_genes highly variable
            for a gene to make the final list.
        accumulator_dtype: The dtype of the accumulators, ``"float64"`` (default) or ``"float32"``. Use
            ``"float32"`` on devices without float64 support (MPS).
    """

    _VALID_FLAVORS = frozenset({"seurat_v3", "seurat_v3_paper"})

    def __init__(
        self,
        var_names_g: np.ndarray,
        n_top_genes: int | list[int],
        n_batch: int = 1,
        flavor: Literal["seurat_v3", "seurat_v3_paper"] = "seurat_v3",
        use_batch_key: bool = False,
        span: float = 0.3,
        output_path: str | None = "hvg_seurat_v3_output.csv",
        batch_n_cell_minimum: int = 2,
        n_batch_minimum: int = 1,
        accumulator_dtype: Literal["float32", "float64"] = "float64",
    ) -> None:
        super().__init__()
        if flavor not in self._VALID_FLAVORS:
            raise ValueError(f"flavor must be one of {sorted(self._VALID_FLAVORS)}, got {flavor!r}")
        if accumulator_dtype not in ("float32", "float64"):
            raise ValueError(f"accumulator_dtype must be 'float32' or 'float64', got {accumulator_dtype!r}")
        self.accumulator_dtype = accumulator_dtype
        dtype = getattr(torch, accumulator_dtype)
        self.var_names_g = var_names_g
        n_vars = len(var_names_g)
        self.n_vars = n_vars
        if isinstance(n_top_genes, int):
            self.n_top_genes_list = [n_top_genes]
        else:
            self.n_top_genes_list = sorted(set(n_top_genes))
        self.n_batch = n_batch
        self.flavor = flavor
        self.use_batch_key = use_batch_key
        if use_batch_key and n_batch < 2:
            raise ValueError(
                "n_batch must be at least 2 when use_batch_key is True. This error may also be triggered "
                "if your dataloader is not providing the expected `batch_index_n` key: check your dataloader "
                "batch_keys and ensure `batch_index_n` is included when use_batch_key=True --"
                "data:"
                "  ..."
                "  batch_keys:"
                "    ..."
                "    batch_index_n:"
                "      attr: obs"
                "      key: my_categorical_batch_column"
                "      convert_fn: cellarium.ml.utilities.data.categories_to_codes"
            )
        self.span = span
        if (output_path is not None) and (not output_path.endswith(".csv")):
            raise ValueError("output_path must end with .csv")
        self.output_path = output_path
        self.batch_n_cell_minimum = batch_n_cell_minimum
        if n_batch_minimum > n_batch:
            raise ValueError(f"n_batch_minimum ({n_batch_minimum}) cannot exceed n_batch ({n_batch}).")
        self.n_batch_minimum = n_batch_minimum

        self._current_epoch: int = 0

        # Epoch-0 buffers: shape (n_batch, n_vars). The sums are of ``x - x_shift_bg``, not of ``x``.
        # The cell count is int64 so that it stays exact past 2**24 cells (a float32 count stops incrementing there).
        self.x_shift_bg: torch.Tensor
        self.x_shift_set_b: torch.Tensor
        self.x_sums_bg: torch.Tensor
        self.x_squared_sums_bg: torch.Tensor
        self.x_size_b: torch.Tensor
        self.register_buffer("x_shift_bg", torch.zeros(n_batch, n_vars, dtype=dtype))
        self.register_buffer("x_shift_set_b", torch.zeros(n_batch, dtype=torch.bool))
        self.register_buffer("x_sums_bg", torch.zeros(n_batch, n_vars, dtype=dtype))
        self.register_buffer("x_squared_sums_bg", torch.zeros(n_batch, n_vars, dtype=dtype))
        self.register_buffer("x_size_b", torch.zeros(n_batch, dtype=torch.int64))
        # whether every batch has a shift (a cache, to skip the work of finding shifts once they all are set)
        self._shifts_complete = False

        # Set between epochs by on_train_epoch_end after epoch 0: shape (n_batch, n_vars)
        self.mean_bg: torch.Tensor
        self.clip_val_bg: torch.Tensor
        self.reg_std_bg: torch.Tensor
        self.register_buffer("mean_bg", torch.zeros(n_batch, n_vars, dtype=dtype))
        self.register_buffer("clip_val_bg", torch.zeros(n_batch, n_vars, dtype=dtype))
        self.register_buffer("reg_std_bg", torch.zeros(n_batch, n_vars, dtype=dtype))

        # Epoch-1 buffer: shape (n_batch, n_vars), the sum over cells of (clipped count - batch mean) ** 2
        self.clipped_sq_dev_sum_bg: torch.Tensor
        self.register_buffer("clipped_sq_dev_sum_bg", torch.zeros(n_batch, n_vars, dtype=dtype))

        # Dummy parameter so Lightning treats this as a trainable module
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self._dummy_param.data.zero_()

        self.hvg_dfs: dict[int, pd.DataFrame] | None = None

    def reset_parameters(self) -> None:
        self.x_shift_bg.zero_()
        self.x_shift_set_b.zero_()
        self.x_sums_bg.zero_()
        self.x_squared_sums_bg.zero_()
        self.x_size_b.zero_()
        self.mean_bg.zero_()
        self.clip_val_bg.zero_()
        self.reg_std_bg.zero_()
        self.clipped_sq_dev_sum_bg.zero_()
        self._dummy_param.data.zero_()
        self._shifts_complete = False

    def on_train_epoch_start(self, trainer: pl.Trainer) -> None:
        """Cache the current epoch number so forward() can branch on it."""
        self._current_epoch = trainer.current_epoch

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        # help users avoid unintentional errors in configuring the use of batch_key
        if self.use_batch_key and (batch_index_n is None):
            raise ValueError(
                "batch_index_n is required when use_batch_key is True: add `batch_index_n` to your "
                "dataloader batch_keys."
            )
        if (not self.use_batch_key) and (batch_index_n is not None):
            raise ValueError(
                "batch_index_n was given but use_batch_key is False: either set use_batch_key=True or remove "
                "batch_index_n from your dataloader batch_keys."
            )

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

    def _set_new_shifts(self, x_ng: torch.Tensor, batch_idx_n: torch.Tensor) -> None:
        """
        Set the shift of every batch that has cells in ``x_ng`` and no shift yet to the mean of those cells (over
        all processes, so that every process has the same shifts). The shift only has to be near the mean of the
        batch: any constant gives the same answer, but a near one avoids the loss of precision.
        """
        if self._shifts_complete:
            return
        idx_exp_ng = batch_idx_n.unsqueeze(1).expand_as(x_ng)
        sums_bg = torch.zeros(self.n_batch, self.n_vars, dtype=x_ng.dtype, device=x_ng.device)
        sums_bg.scatter_add_(0, idx_exp_ng, x_ng)
        counts_b = torch.bincount(batch_idx_n, minlength=self.n_batch)
        _, world_size = get_rank_and_num_replicas()
        if world_size > 1:
            dist.all_reduce(sums_bg)
            dist.all_reduce(counts_b)
        new_b = (counts_b > 0) & ~self.x_shift_set_b
        mean_bg = sums_bg / counts_b.clamp(min=1).unsqueeze(1)
        self.x_shift_bg = torch.where(new_b.unsqueeze(1), mean_bg, self.x_shift_bg)
        self.x_shift_set_b = self.x_shift_set_b | new_b
        self._shifts_complete = bool(self.x_shift_set_b.all())

    def _accumulate_epoch0(self, x_ng: torch.Tensor, batch_idx_n: torch.Tensor) -> None:
        x_ng = x_ng.to(self.x_sums_bg.dtype)
        self._set_new_shifts(x_ng, batch_idx_n)
        dev_ng = x_ng - self.x_shift_bg[batch_idx_n]
        idx_exp_ng = batch_idx_n.unsqueeze(1).expand_as(dev_ng)
        sums_contrib_bg = torch.zeros_like(self.x_sums_bg)
        sq_sums_contrib_bg = torch.zeros_like(self.x_squared_sums_bg)
        sums_contrib_bg.scatter_add_(0, idx_exp_ng, dev_ng)
        sq_sums_contrib_bg.scatter_add_(0, idx_exp_ng, dev_ng**2)
        self.x_sums_bg = self.x_sums_bg + sums_contrib_bg
        self.x_squared_sums_bg = self.x_squared_sums_bg + sq_sums_contrib_bg
        self.x_size_b = self.x_size_b + torch.bincount(batch_idx_n, minlength=self.n_batch)

    def _accumulate_epoch1(self, x_ng: torch.Tensor, batch_idx_n: torch.Tensor) -> None:
        x_ng = x_ng.to(self.clipped_sq_dev_sum_bg.dtype)
        # Per-cell clip value and batch mean: shape (n_cells, n_vars)
        x_clipped_ng = torch.minimum(x_ng, self.clip_val_bg[batch_idx_n])
        dev_ng = x_clipped_ng - self.mean_bg[batch_idx_n]
        idx_exp_ng = batch_idx_n.unsqueeze(1).expand_as(dev_ng)
        sq_dev_contrib_bg = torch.zeros_like(self.clipped_sq_dev_sum_bg)
        sq_dev_contrib_bg.scatter_add_(0, idx_exp_ng, dev_ng**2)
        self.clipped_sq_dev_sum_bg = self.clipped_sq_dev_sum_bg + sq_dev_contrib_bg

    def on_train_start(self, trainer: pl.Trainer) -> None:
        """Validation: if in a distributed setting, use DDP with broadcast_buffers=False."""
        if trainer.world_size > 1:
            if not isinstance(trainer.strategy, DDPStrategy):
                raise ValueError("HVGSeuratV3 requires the DDP strategy.")
            if trainer.strategy._ddp_kwargs.get("broadcast_buffers") is not False:
                raise ValueError("HVGSeuratV3 requires broadcast_buffers=False.")

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        """Implement the two-step Seurat v3 HVG method using hooks at the end of first and second epochs.
        After epoch 0, reduce buffers and fit LOESS to set clip_val_bg and reg_std_bg.
        After epoch 1, reduce buffers and compute hvg_df.
        """
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

        # 3. Broadcast mean_bg, clip_val_bg and reg_std_bg to all ranks so epoch-1 can use them
        if trainer.world_size > 1:
            dist.broadcast(self.mean_bg, src=0)
            dist.broadcast(self.clip_val_bg, src=0)
            dist.broadcast(self.reg_std_bg, src=0)

    def _compute_clip_val(self) -> None:
        # Default to +inf so any batch skipped below (N < batch_n_cell_minimum) acts as a no-op clip.
        self.clip_val_bg.fill_(float("inf"))
        n_vars = self.n_vars
        for b in range(self.n_batch):
            N = self.x_size_b[b].item()
            if N < self.batch_n_cell_minimum:
                continue
            # the statistics are computed on the host in float64, from sums of the shifted counts
            shift_g = self.x_shift_bg[b].cpu().numpy().astype(np.float64)
            sums = self.x_sums_bg[b].cpu().numpy().astype(np.float64)
            sq_sums = self.x_squared_sums_bg[b].cpu().numpy().astype(np.float64)
            mean_g = shift_g + sums / N
            # Unbiased (sample) variance with Bessel's correction, matching Scanpy's correction=1
            var_g = (sq_sums - sums**2 / N) / (N - 1)

            not_const = var_g > 0
            estimated_var = np.zeros(n_vars, dtype=np.float64)

            if not_const.any():
                x = np.log10(mean_g[not_const])
                y = np.log10(var_g[not_const])
                estimated_var[not_const] = _fit_loess_with_jitter(x, y, span=self.span)

            reg_std = np.sqrt(10**estimated_var)  # shape (n_vars,)
            clip_val = reg_std * np.sqrt(N) + mean_g

            dtype, device = self.reg_std_bg.dtype, self.reg_std_bg.device
            self.mean_bg[b] = torch.as_tensor(mean_g, dtype=dtype, device=device)
            self.reg_std_bg[b] = torch.as_tensor(reg_std, dtype=dtype, device=device)
            self.clip_val_bg[b] = torch.as_tensor(clip_val, dtype=dtype, device=device)

    def _finish_epoch1(self, trainer: pl.Trainer) -> None:
        trainer.should_stop = True

        # 1. Reduce epoch-1 buffers to rank 0
        if trainer.world_size > 1:
            dist.reduce(self.clipped_sq_dev_sum_bg, dst=0, op=dist.ReduceOp.SUM)

        # 2. Rank 0: compute norm_var, rank, select top genes, build DataFrame and optionally save to CSV
        if trainer.global_rank == 0:
            # Retrieve adata.var annotation columns to enrich the output DataFrame.
            var_df: pd.DataFrame | None = None
            datamodule = getattr(trainer, "datamodule", None)
            if isinstance(datamodule, CellariumAnnDataDataModule):
                var_df = datamodule.dadc.var
            else:
                warnings.warn(
                    "HVGSeuratV3: trainer.datamodule is not a CellariumAnnDataDataModule; "
                    "adata.var annotations will not be added to the output DataFrame.",
                    UserWarning,
                    stacklevel=2,
                )
            self.hvg_dfs = {}
            for n in self.n_top_genes_list:
                df = self._compute_hvg_df(n_top_genes=n, var_df=var_df)
                self.hvg_dfs[n] = df
                if self.output_path is not None:
                    path = self.output_path.replace(".csv", f"__top{n}.csv")
                    self._save(df=df, output_path=path)

    def _compute_hvg_df(self, n_top_genes: int, var_df: pd.DataFrame | None = None) -> pd.DataFrame:
        n_vars = self.n_vars
        norm_gene_var_bg = np.zeros((self.n_batch, n_vars), dtype=np.float64)

        for b in range(self.n_batch):
            N = self.x_size_b[b].item()
            if N < self.batch_n_cell_minimum:
                continue
            reg_std_g = self.reg_std_bg[b].cpu().numpy().astype(np.float64)
            sq_dev_sum_g = self.clipped_sq_dev_sum_bg[b].cpu().numpy().astype(np.float64)

            denom_g = (N - 1) * reg_std_g**2
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_gene_var_g = sq_dev_sum_g / denom_g
            norm_gene_var_g[np.isnan(norm_gene_var_g)] = 0.0
            norm_gene_var_bg[b] = norm_gene_var_g

        # Only rank over batches with sufficient data (N >= batch_n_cell_minimum); invalid batches
        # have norm_gene_vars == 0 and would otherwise pollute num_batches_high_var
        # and median_ranked with arbitrary rankings of equal values.
        valid_b = np.array([self.x_size_b[b].item() >= self.batch_n_cell_minimum for b in range(self.n_batch)])
        norm_gene_vars_valid_vg = norm_gene_var_bg[valid_b]

        # Rank genes within each valid batch v
        # (genes with identical counts in a batch tie exactly; a stable sort breaks the tie by gene order, instead of
        # by whatever the unstable sort does with the rounding noise of the other genes)
        hvg_rank_vg = np.argsort(np.argsort(-norm_gene_vars_valid_vg, axis=1, kind="stable"), axis=1).astype(np.float32)
        num_batches_high_var_v = (hvg_rank_vg < n_top_genes).sum(axis=0).astype(int)
        hvg_rank_vg[hvg_rank_vg >= n_top_genes] = np.nan
        masked_hvg_rank_vg = np.ma.masked_invalid(hvg_rank_vg)
        median_hvg_rank_g = np.ma.median(masked_hvg_rank_vg, axis=0).filled(np.nan)

        variances_norm_g = norm_gene_vars_valid_vg.mean(axis=0)

        df = pd.DataFrame(
            index=pd.Index(self.var_names_g, name="gene"),
            data={
                "highly_variable_nbatches": num_batches_high_var_v,
                "highly_variable_rank": median_hvg_rank_g,
                "variances_norm": variances_norm_g,
            },
        )

        # Use integer-position sort so duplicate gene names don't cause
        # df.loc to mark more than n_top_genes rows as highly variable.
        rank_vals_g = df["highly_variable_rank"].fillna(np.inf).values
        nbatches_vals_g = df["highly_variable_nbatches"].values
        # np.lexsort: LAST key = primary sort key.
        if self.flavor == "seurat_v3":
            # Primary: rank ascending; tiebreaker: nbatches descending.
            sort_positions_g = np.lexsort((-nbatches_vals_g, rank_vals_g))
        else:  # seurat_v3_paper
            # Primary: nbatches descending; tiebreaker: rank ascending.
            sort_positions_g = np.lexsort((rank_vals_g, -nbatches_vals_g))
        eligible_g = df["highly_variable_nbatches"].values >= self.n_batch_minimum
        eligible_in_sort_order_g = sort_positions_g[eligible_g[sort_positions_g]]
        hvg_flags = np.zeros(len(df), dtype=bool)
        hvg_flags[eligible_in_sort_order_g[:n_top_genes]] = True
        df["highly_variable"] = hvg_flags

        if self.n_batch == 1:
            df = df.drop(columns=["highly_variable_nbatches"])

        # Append any extra columns from adata.var (e.g. gene_id, gene_name).
        # df.index values are var_names_g; var_df.index is adata.var_names.
        # pandas .join() aligns on label values, so index *name* differences
        # are harmless, and genes absent from var_df receive NaN.
        if var_df is not None:
            extra_cols = var_df.columns.difference(df.columns)
            if len(extra_cols) > 0:
                df = df.join(var_df[extra_cols], how="left")

        df = df.iloc[sort_positions_g]  # reorder rows by rank
        return df

    def _save(self, df: pd.DataFrame, output_path: str) -> None:
        df.to_csv(output_path)
        hvg_df = df[df["highly_variable"]]
        hvg_df.to_csv(output_path.replace(".csv", "__hvg_only.csv"), index=True, header=True)
