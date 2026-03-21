# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.distributed import get_rank_and_num_replicas
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class OnePassMeanVarStd(CellariumModel):
    """
    Calculate the mean, variance, and standard deviation of the data in one pass (epoch)
    using running sums and running squared sums.

    Tracks per-batch statistics. Use ``n_batch=1`` when there is no meaningful batch
    structure. After training, ``batch_mean_bg`` and ``batch_var_bg`` give per-batch
    per-gene statistics suitable for passing to ``get_highly_variable_genes``.

    **References:**

    1. `Algorithms for calculating variance
       <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        algorithm:
            ``"naive"`` (default) or ``"shifted_data"`` (numerically stable).
        n_batch:
            Number of batches. Use 1 to reproduce ``batch_key``=None behavior.
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        algorithm: Literal["naive", "shifted_data"] = "naive",
        n_batch: int = 1,
    ) -> None:
        super().__init__()
        self.var_names_g = var_names_g
        n_vars = len(self.var_names_g)
        self.n_vars = n_vars
        self.algorithm = algorithm
        self.n_batch = n_batch

        self.x_shift: torch.Tensor | None
        if self.algorithm == "shifted_data":
            self.register_buffer("x_shift", torch.empty(n_vars))
        else:
            self.register_buffer("x_shift", None)

        self.x_sums_bg: torch.Tensor
        self.x_squared_sums_bg: torch.Tensor
        self.x_size_b: torch.Tensor
        self.register_buffer("x_sums_bg", torch.empty(self.n_batch, n_vars))
        self.register_buffer("x_squared_sums_bg", torch.empty(self.n_batch, n_vars))
        self.register_buffer("x_size_b", torch.empty(self.n_batch))

        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.x_shift is not None:
            self.x_shift.zero_()
        self.x_sums_bg.zero_()
        self.x_squared_sums_bg.zero_()
        self.x_size_b.zero_()
        self._dummy_param.data.zero_()

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng: Gene counts matrix.
            var_names_g: Variable names in the input data.
            batch_index_n: Optional batch indices for each cell, required if ``n_batch`` > 1.

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        if batch_index_n is None:
            batch_index_n = torch.zeros(x_ng.shape[0], dtype=torch.long, device=x_ng.device)

        if self.algorithm == "naive":
            x_for_sum = x_ng
        elif self.algorithm == "shifted_data":
            assert self.x_shift is not None
            if (self.x_shift == 0).all():
                _, world_size = get_rank_and_num_replicas()
                if world_size > 1:
                    gathered_x_ng = torch.zeros(
                        world_size * x_ng.shape[0], x_ng.shape[1], dtype=x_ng.dtype, device=x_ng.device
                    )
                    dist.all_gather_into_tensor(gathered_x_ng, x_ng)
                    x_shift = gathered_x_ng.mean(dim=0)
                else:
                    x_shift = x_ng.mean(dim=0)
                self.x_shift = x_shift
            x_for_sum = x_ng - self.x_shift
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        n_cells = x_ng.shape[0]
        idx_expanded = batch_index_n.unsqueeze(1).expand(n_cells, self.n_vars)
        sums_contrib = torch.zeros(self.n_batch, self.n_vars, dtype=x_for_sum.dtype, device=x_ng.device)
        sq_sums_contrib = torch.zeros(self.n_batch, self.n_vars, dtype=x_for_sum.dtype, device=x_ng.device)
        sums_contrib.scatter_add_(0, idx_expanded, x_for_sum)
        sq_sums_contrib.scatter_add_(0, idx_expanded, x_for_sum**2)
        self.x_sums_bg = self.x_sums_bg + sums_contrib
        self.x_squared_sums_bg = self.x_squared_sums_bg + sq_sums_contrib
        self.x_size_b = self.x_size_b + torch.bincount(batch_index_n, minlength=self.n_batch)

        return {}

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "OnePassMeanVarStd requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is False, (
                "OnePassMeanVarStd requires that broadcast_buffers is set to False."
            )

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        if trainer.world_size == 1:
            return
        dist.reduce(self.x_sums_bg, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_squared_sums_bg, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_size_b, dst=0, op=dist.ReduceOp.SUM)

    @property
    def mean_g(self) -> torch.Tensor:
        mean_g = self.x_sums_bg.sum(0) / self.x_size_b.sum()
        if self.algorithm == "shifted_data":
            assert isinstance(self.x_shift, torch.Tensor)
            mean_g = mean_g + self.x_shift
        return mean_g

    @property
    def var_g(self) -> torch.Tensor:
        x_sums_g = self.x_sums_bg.sum(0)
        x_squared_sums_g = self.x_squared_sums_bg.sum(0)
        x_size = self.x_size_b.sum()
        return x_squared_sums_g / x_size - (x_sums_g / x_size) ** 2

    @property
    def std_g(self) -> torch.Tensor:
        return torch.sqrt(self.var_g)

    @property
    def batch_mean_bg(self) -> torch.Tensor:
        """Per-batch mean, shape ``(n_batch, n_genes)``."""
        mean_bg = self.x_sums_bg / self.x_size_b.unsqueeze(1)
        if self.algorithm == "shifted_data":
            assert isinstance(self.x_shift, torch.Tensor)
            mean_bg = mean_bg + self.x_shift
        return mean_bg

    @property
    def batch_var_bg(self) -> torch.Tensor:
        """Per-batch population variance, shape ``(n_batch, n_genes)``."""
        mean_bg = self.x_sums_bg / self.x_size_b.unsqueeze(1)
        return self.x_squared_sums_bg / self.x_size_b.unsqueeze(1) - mean_bg**2
