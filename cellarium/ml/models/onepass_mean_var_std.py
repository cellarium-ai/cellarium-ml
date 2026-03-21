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

    Optionally tracks per-batch statistics when ``batch_key`` and ``n_batch`` are set.
    After training, ``batch_mean_bg`` and ``batch_var_bg`` give per-batch per-gene
    statistics suitable for passing to ``get_highly_variable_genes``.

    **References:**

    1. `Algorithms for calculating variance
       <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        algorithm:
            ``"naive"`` (default) or ``"shifted_data"`` (numerically stable).
        batch_key:
            Name of the batch-index field in the batch dict. The field must contain
            integer tensors of shape ``(n_cells,)`` with values in ``[0, n_batch)``.
            ``None`` disables per-batch tracking.
        n_batch:
            Number of batches. Required when ``batch_key`` is not ``None``.
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        algorithm: Literal["naive", "shifted_data"] = "naive",
        batch_key: str | None = None,
        n_batch: int | None = None,
    ) -> None:
        super().__init__()
        self.var_names_g = var_names_g
        n_vars = len(self.var_names_g)
        self.n_vars = n_vars
        self.algorithm = algorithm
        self.batch_key = batch_key

        if batch_key is not None and n_batch is None:
            raise ValueError("`n_batch` must be provided when `batch_key` is set.")
        self._n_batch = n_batch if batch_key is not None else 0

        self.x_sums: torch.Tensor
        self.x_squared_sums: torch.Tensor
        self.x_size: torch.Tensor
        self.x_shift: torch.Tensor | None
        self.register_buffer("x_sums", torch.empty(n_vars))
        self.register_buffer("x_squared_sums", torch.empty(n_vars))
        self.register_buffer("x_size", torch.empty(()))
        if self.algorithm == "shifted_data":
            self.register_buffer("x_shift", torch.empty(n_vars))
        else:
            self.register_buffer("x_shift", None)

        # Per-batch accumulators (only allocated when batch_key is set)
        self.x_sums_bg: torch.Tensor | None
        self.x_squared_sums_bg: torch.Tensor | None
        self.x_size_b: torch.Tensor | None
        if self._n_batch > 0:
            self.register_buffer("x_sums_bg", torch.empty(self._n_batch, n_vars))
            self.register_buffer("x_squared_sums_bg", torch.empty(self._n_batch, n_vars))
            self.register_buffer("x_size_b", torch.empty(self._n_batch))
        else:
            self.register_buffer("x_sums_bg", None)
            self.register_buffer("x_squared_sums_bg", None)
            self.register_buffer("x_size_b", None)

        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.x_sums.zero_()
        self.x_squared_sums.zero_()
        self.x_size.zero_()
        if self.x_shift is not None:
            self.x_shift.zero_()
        if self._n_batch > 0:
            assert self.x_sums_bg is not None
            assert self.x_squared_sums_bg is not None
            assert self.x_size_b is not None
            self.x_sums_bg.zero_()
            self.x_squared_sums_bg.zero_()
            self.x_size_b.zero_()
        self._dummy_param.data.zero_()

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray, **kwargs) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng: Gene counts matrix.
            var_names_g: Variable names in the input data.
            **kwargs: Additional batch fields. Must include ``batch_key`` field
                containing int tensor ``(n_cells,)`` when ``batch_key`` is set.

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        if self.algorithm == "naive":
            self.x_sums = self.x_sums + x_ng.sum(dim=0)
            self.x_squared_sums = self.x_squared_sums + (x_ng**2).sum(dim=0)
            self.x_size = self.x_size + x_ng.shape[0]
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
            self.x_sums = self.x_sums + (x_ng - self.x_shift).sum(dim=0)
            self.x_squared_sums = self.x_squared_sums + ((x_ng - self.x_shift) ** 2).sum(dim=0)
            self.x_size = self.x_size + x_ng.shape[0]
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Per-batch accumulation (scatter_add for vectorized update)
        if self.batch_key is not None:
            if self.batch_key not in kwargs:
                raise ValueError(
                    f"batch_key '{self.batch_key}' not found in batch. Available: {list(kwargs.keys())}"
                )
            assert self.x_sums_bg is not None
            assert self.x_squared_sums_bg is not None
            assert self.x_size_b is not None
            batch_idx_n = kwargs[self.batch_key].long()  # (n_cells,)
            n_cells = x_ng.shape[0]
            idx_expanded = batch_idx_n.unsqueeze(1).expand(n_cells, self.n_vars)
            self.x_sums_bg.scatter_add_(0, idx_expanded, x_ng.float())
            self.x_squared_sums_bg.scatter_add_(0, idx_expanded, x_ng.float() ** 2)
            self.x_size_b.scatter_add_(0, batch_idx_n, torch.ones(n_cells, device=x_ng.device))

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
        dist.reduce(self.x_sums, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_squared_sums, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_size, dst=0, op=dist.ReduceOp.SUM)
        if self._n_batch > 0:
            assert self.x_sums_bg is not None
            assert self.x_squared_sums_bg is not None
            assert self.x_size_b is not None
            dist.reduce(self.x_sums_bg, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(self.x_squared_sums_bg, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(self.x_size_b, dst=0, op=dist.ReduceOp.SUM)

    @property
    def mean_g(self) -> torch.Tensor:
        mean_g = self.x_sums / self.x_size
        if self.algorithm == "shifted_data":
            assert isinstance(self.x_shift, torch.Tensor)
            mean_g = mean_g + self.x_shift
        return mean_g

    @property
    def var_g(self) -> torch.Tensor:
        return self.x_squared_sums / self.x_size - (self.x_sums / self.x_size) ** 2

    @property
    def std_g(self) -> torch.Tensor:
        return torch.sqrt(self.var_g)

    @property
    def batch_mean_bg(self) -> torch.Tensor:
        """Per-batch mean, shape ``(n_batch, n_genes)``. Requires ``batch_key`` to be set."""
        if self._n_batch == 0:
            raise RuntimeError("`batch_mean_bg` requires `batch_key` to be set.")
        assert self.x_sums_bg is not None and self.x_size_b is not None
        return self.x_sums_bg / self.x_size_b.unsqueeze(1)

    @property
    def batch_var_bg(self) -> torch.Tensor:
        """Per-batch population variance, shape ``(n_batch, n_genes)``. Requires ``batch_key``."""
        if self._n_batch == 0:
            raise RuntimeError("`batch_var_bg` requires `batch_key` to be set.")
        assert self.x_sums_bg is not None and self.x_squared_sums_bg is not None and self.x_size_b is not None
        mean_bg = self.x_sums_bg / self.x_size_b.unsqueeze(1)
        return self.x_squared_sums_bg / self.x_size_b.unsqueeze(1) - mean_bg**2
