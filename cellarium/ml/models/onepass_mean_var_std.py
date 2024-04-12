# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.data import get_rank_and_num_replicas
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class OnePassMeanVarStd(CellariumModel):
    """
    Calculate the mean, variance, and standard deviation of the data in one pass (epoch)
    using running sums and running squared sums.

    **References:**

    1. `Algorithms for calculating variance
       <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

    Args:
        var_names_g: The variable names schema for the input data validation.
    """

    def __init__(self, var_names_g: Sequence[str], algorithm: Literal["naive", "shifted_data"] = "naive") -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        n_vars = len(self.var_names_g)
        self.n_vars = n_vars
        self.algorithm = algorithm

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
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.x_sums.zero_()
        self.x_squared_sums.zero_()
        self.x_size.zero_()
        if self.x_shift is not None:
            self.x_shift.zero_()
        self._dummy_param.data.zero_()

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.

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

        return {}

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
        dist.reduce(self.x_sums, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_squared_sums, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_size, dst=0, op=dist.ReduceOp.SUM)

    @property
    def mean_g(self) -> torch.Tensor:
        """
        Mean of the data.
        """
        mean_g = self.x_sums / self.x_size
        if self.algorithm == "shifted_data":
            mean_g = mean_g + self.x_shift
        return mean_g

    @property
    def var_g(self) -> torch.Tensor:
        """
        Variance of the data.
        """
        return self.x_squared_sums / self.x_size - (self.x_sums / self.x_size) ** 2

    @property
    def std_g(self) -> torch.Tensor:
        """
        Standard deviation of the data.
        """
        return torch.sqrt(self.var_g)
