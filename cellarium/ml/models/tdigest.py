# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import crick
import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class TDigest(CellariumModel):
    """
    Compute an approximate non-zero histogram of the distribution of each gene in a batch of
    cells using t-digests.

    **References**:

    1. `Computing Extremely Accurate Quantiles Using T-Digests (Dunning et al.)
       <https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf>`_.

    Args:
        var_names_g: The variable names schema for the input data validation.
    """

    def __init__(self, var_names_g: np.ndarray) -> None:
        super().__init__()
        self.var_names_g = var_names_g
        n_vars = len(self.var_names_g)
        self.n_vars = n_vars
        self.tdigests = [crick.tdigest.TDigest() for _ in range(self.n_vars)]
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
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

        for i, tdigest in enumerate(self.tdigests):
            x_n = x_ng[:, i]
            nonzero_mask = torch.nonzero(x_n)
            if len(nonzero_mask) > 0:
                tdigest.update(x_n[nonzero_mask])
        return {}

    def on_epoch_end(self, trainer: pl.Trainer) -> None:
        # no need to merge if only one process
        if trainer.world_size == 1:
            return

        tdigests_gather_list: list | None = (
            [None for _ in range(trainer.world_size)] if trainer.global_rank == 0 else None
        )
        dist.gather_object(self.tdigests, tdigests_gather_list, dst=0)

        if trainer.global_rank != 0:
            return

        # merge tdigests
        assert tdigests_gather_list is not None
        for new_tdigests in tdigests_gather_list[1:]:  # iterate over processes
            assert new_tdigests is not None
            for j in range(len(self.tdigests)):  # iterate over genes
                self.tdigests[j].merge(new_tdigests[j])

    @property
    def median_g(self) -> torch.Tensor:
        """
        Median of the data.
        """
        return torch.as_tensor([tdigest.quantile(0.5) for tdigest in self.tdigests])

    def get_extra_state(self) -> dict[str, Any]:
        return {"tdigests": self.tdigests}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.tdigests = state["tdigests"]
