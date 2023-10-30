# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from numpy.typing import ArrayLike

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)
from cellarium.ml.utilities.types import BatchDict


class TDigest(CellariumModel):
    """
    Compute an approximate non-zero histogram of the distribution of each gene in a batch of
    cells using t-digests.

    **References**:

    1. `Computing Extremely Accurate Quantiles Using T-Digests (Dunning et al.)
       <https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf>`_.

    Args:
        feature_schema: The variable names schema for the input data validation.
    """

    def __init__(self, feature_schema: ArrayLike) -> None:
        import crick

        super().__init__()
        self.feature_schema = np.array(feature_schema)
        self.g_genes = len(self.feature_schema)
        self.tdigests = [crick.tdigest.TDigest() for _ in range(self.g_genes)]
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray) -> BatchDict:
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

        # save tdigests
        dirpath = self._resolve_ckpt_dir(trainer)
        torch.save(
            self.tdigests,
            os.path.join(dirpath, f"tdigests_{trainer.global_rank}.pt"),
        )
        # wait for all processes to save
        dist.barrier()

        if trainer.global_rank != 0:
            return

        # merge tdigests
        for i in range(1, trainer.world_size):  # iterate over processes
            new_tdigests = torch.load(os.path.join(dirpath, f"tdigests_{i}.pt"))
            for j in range(len(self.tdigests)):  # iterate over genes
                self.tdigests[j].merge(new_tdigests[j])

    @property
    def median_g(self) -> torch.Tensor:
        return torch.as_tensor([tdigest.quantile(0.5) for tdigest in self.tdigests])

    @staticmethod
    def _resolve_ckpt_dir(trainer: pl.Trainer) -> Path | str:
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        os.makedirs(ckpt_path, exist_ok=True)

        return ckpt_path

    def get_extra_state(self) -> dict[str, Any]:
        return {"tdigests": self.tdigests}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.tdigests = state["tdigests"]