# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence
import numpy as np
import pyro
import pyro.distributions as dist
import torch

from cellarium.ml.module.base_module import BaseModule
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
    assert_nonnegative,
    assert_positive,
)


class LogisticRegression(BaseModule):
    def __init__(
        self,
        n_cells: int,
        feature_schema: Sequence[str],
        c_categories: int,
        W_prior_scale: float = 1.0,
        transform: torch.nn.Module | None = None,
        elbo: pyro.infer.ELBO | None = None,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.feature_schema = np.array(feature_schema)
        self.g_genes = len(feature_schema)
        self.c_categories = c_categories
        self.W_mean_gc = torch.nn.Parameter(torch.zeros(self.g_genes, c_categories))
        # TODO: use positive constraint
        self.W_sigma_gc = torch.nn.Parameter(torch.ones(self.g_genes, c_categories))
        self.W_prior_scale = W_prior_scale
        self.b_k = torch.nn.Parameter(torch.zeros(c_categories))
        self.transform = transform
        self.elbo = elbo or pyro.infer.Trace_ELBO()

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        y = tensor_dict["cell_type"]
        return (x, y), {}

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray, y_n: torch.Tensor) -> torch.Tensor:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)
        return self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n)

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        W_prior_scale = torch.tensor(self.W_prior_scale, device=x_ng.device)
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, W_prior_scale).expand([self.g_genes, self.c_categories]).to_event(2),
        )
        with pyro.plate("cells", size=self.n_cells, subsample_size=x_ng.shape[0]):
            logits = x_ng @ W_gc + self.b_k
            pyro.sample("y", dist.Categorical(logits=logits), obs=y_n)

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        pyro.sample("W", dist.Laplace(self.W_mean_gc, self.W_sigma_gc).to_event(2))
