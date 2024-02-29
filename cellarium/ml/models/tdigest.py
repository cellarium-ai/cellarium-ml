# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Sequence
from pathlib import Path
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

    def __init__(self, var_names_g: Sequence[str]) -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
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

        # save tdigests
        dirpath = self._resolve_ckpt_dir(trainer)
        filepath = os.path.join(dirpath, f"tdigests_{trainer.global_rank}.pt")
        trainer.strategy.checkpoint_io.save_checkpoint(self.tdigests, filepath)  # type: ignore[arg-type]
        # wait for all processes to save
        dist.barrier()

        if trainer.global_rank != 0:
            return

        # merge tdigests
        for i in range(1, trainer.world_size):  # iterate over processes
            filepath = os.path.join(dirpath, f"tdigests_{i}.pt")
            new_tdigests: list = trainer.strategy.checkpoint_io.load_checkpoint(filepath)  # type: ignore[assignment]
            for j in range(len(self.tdigests)):  # iterate over genes
                self.tdigests[j].merge(new_tdigests[j])

    @property
    def median_g(self) -> torch.Tensor:
        """
        Median of the data.
        """
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
