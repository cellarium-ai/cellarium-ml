# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel


class OnePassMeanVarStd(CellariumModel):
    """
    Calculate the mean, variance, and standard deviation of the data in one pass (epoch)
    using running sums and running squared sums.

    **References:**

    1. `Algorithms for calculating variance
       <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

    Args:
        g_genes:
            Number of genes.
        transform:
            If not ``None`` is used to transform the input data.
    """

    def __init__(self, g_genes: int, transform: nn.Module | None = None) -> None:
        super().__init__()
        self.g_genes = g_genes
        self.transform = transform
        self.x_sums: torch.Tensor
        self.x_squared_sums: torch.Tensor
        self.x_size: torch.Tensor
        self.register_buffer("x_sums", torch.zeros(g_genes))
        self.register_buffer("x_squared_sums", torch.zeros(g_genes))
        self.register_buffer("x_size", torch.tensor(0))
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng: torch.Tensor) -> None:
        if self.transform is not None:
            x_ng = self.transform(x_ng)

        self.x_sums = self.x_sums + x_ng.sum(dim=0)
        self.x_squared_sums = self.x_squared_sums + (x_ng**2).sum(dim=0)
        self.x_size = self.x_size + x_ng.shape[0]

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(
                trainer.strategy, DDPStrategy
            ), "OnePassMeanVarStd requires that the trainer uses the DDP strategy."
            assert (
                trainer.strategy._ddp_kwargs["broadcast_buffers"] is False
            ), "OnePassMeanVarStd requires that broadcast_buffers is set to False."

    def on_epoch_end(self, trainer: pl.Trainer) -> None:
        # no need to merge if only one process
        if trainer.world_size == 1:
            return

        # merge the running sums
        dist.all_reduce(self.x_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.x_squared_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.x_size, op=dist.ReduceOp.SUM)

    @property
    def mean_g(self) -> torch.Tensor:
        return self.x_sums / self.x_size

    @property
    def var_g(self) -> torch.Tensor:
        return self.x_squared_sums / self.x_size - self.mean_g**2

    @property
    def std_g(self) -> torch.Tensor:
        return torch.sqrt(self.var_g)
