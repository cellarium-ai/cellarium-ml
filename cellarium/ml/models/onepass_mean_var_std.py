# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel, CellariumPipelineUpdatable
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class OnePassMeanVarStd(CellariumPipelineUpdatable, CellariumModel):
    """
    Calculate the mean, variance, and standard deviation of the data in one pass (epoch)
    using running sums and running squared sums.

    **References:**

    1. `Algorithms for calculating variance
       <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

    """

    def __init__(self, feature_schema: Sequence[str]) -> None:
        super().__init__()
        self.feature_schema = np.array(feature_schema)
        g_genes = len(self.feature_schema)
        self.g_genes = g_genes
        self.x_sums: torch.Tensor
        self.x_squared_sums: torch.Tensor
        self.x_size: torch.Tensor
        self.register_buffer("x_sums", torch.zeros(g_genes))
        self.register_buffer("x_squared_sums", torch.zeros(g_genes))
        self.register_buffer("x_size", torch.tensor(0))
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def update_input_tensors_from_previous_module(self, batch: dict[str, np.ndarray | torch.Tensor]) -> None:
        mask = np.isin(element=self.feature_schema, test_elements=batch["feature_g"])
        mask_tensor = torch.Tensor(mask)
        mask_tensor.to(self.x_sums.device)

        self.feature_schema = self.feature_schema[mask]
        self.g_genes = len(self.feature_schema)

        self.x_sums = self.x_sums[mask_tensor]
        self.x_squared_sums = self.x_squared_sums[mask_tensor]

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                Gene counts matrix.
            feature_g:
                The list of the variable names in the input data.

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        self.x_sums = self.x_sums + x_ng.sum(dim=0)
        self.x_squared_sums = self.x_squared_sums + (x_ng**2).sum(dim=0)
        self.x_size = self.x_size + x_ng.shape[0]
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
        dist.all_reduce(self.x_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.x_squared_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.x_size, op=dist.ReduceOp.SUM)

    @property
    def mean_g(self) -> torch.Tensor:
        """
        Mean of the data.
        """
        return self.x_sums / self.x_size

    @property
    def var_g(self) -> torch.Tensor:
        """
        Variance of the data.
        """
        return self.x_squared_sums / self.x_size - self.mean_g**2

    @property
    def std_g(self) -> torch.Tensor:
        """
        Standard deviation of the data.
        """
        return torch.sqrt(self.var_g)
