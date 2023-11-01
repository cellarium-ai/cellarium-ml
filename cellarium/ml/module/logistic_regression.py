# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .base_module import BaseModule


class LogisticRegression(BaseModule):
    def __init__(
        self,
        n_cells: int,
        g_genes: int,
        k_cell_types: int,
        W_prior_scale: float = 1.0,
        transform: torch.nn.Module | None = None,
        elbo: pyro.infer.ELBO | None = None,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.g_genes = g_genes
        self.k_cell_types = k_cell_types
        self.W_mean_gk = torch.nn.Parameter(torch.zeros(g_genes, k_cell_types))
        # TODO: use positive constraint
        self.W_sigma_gk = torch.nn.Parameter(torch.ones(g_genes, k_cell_types))
        self.W_prior_scale = W_prior_scale
        self.b_k = torch.nn.Parameter(torch.zeros(k_cell_types))
        self.transform = transform
        self.elbo = elbo or pyro.infer.Trace_ELBO()

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        y = tensor_dict["cell_type"]
        return (x, y), {}

    def forward(self, x_ng: torch.Tensor, cell_type_n: torch.Tensor) -> torch.Tensor:
        return self.elbo.differentiable_loss(self.model, self.guide, x_ng, cell_type_n)

    def model(self, x_ng: torch.Tensor, cell_type_n: torch.Tensor) -> None:
        W_prior_scale = torch.tensor(self.W_prior_scale, device=x_ng.device)
        W_gk = pyro.sample(
            "W_gk",
            dist.Laplace(0, W_prior_scale).expand([self.g_genes, self.k_cell_types]).to_event(2),
        )
        with pyro.plate("cells", size=self.n_cells, subsample_size=x_ng.shape[0]):
            logits = x_ng @ W_gk + self.b_k
            pyro.sample("cell_type", dist.Categorical(logits=logits), obs=cell_type_n)

    def guide(self, x_ng: torch.Tensor, cell_type_n: torch.Tensor) -> None:
        pyro.sample("W_gk", dist.Laplace(self.W_mean_gk, self.W_sigma_gk).to_event(2))
