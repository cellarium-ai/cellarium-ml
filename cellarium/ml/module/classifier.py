# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .base_module import BasePyroModule


class Classifier(BasePyroModule):
    def __init__(
        self,
        n_cells: int,
        g_genes: int,
        k_cell_types: int,
        transform: torch.nn.Module | None = None,
        elbo: pyro.infer.ELBO | None = None,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.g_genes = g_genes
        self.k_cell_types = k_cell_types
        self.fc = pyro.nn.PyroModule[torch.nn.Linear](g_genes, k_cell_types)
        self.transform = transform
        self.elbo = elbo or pyro.infer.Trace_ELBO()

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        y = tensor_dict["cell_types"]
        return (x, y), {}

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.elbo.differentiable_loss(self.model, self.guide, *args, **kwargs)

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        with pyro.plate("cells", size=self.n_cells, subsample_size=x_ng.shape[0]):
            logits = self.fc(x_ng)
            pyro.sample("y_g", dist.Categorical(logits=logits), obs=y_n)

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        pass
