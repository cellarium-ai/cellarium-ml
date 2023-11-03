# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroParam
from torch.distributions import constraints

from cellarium.ml.module.base_module import BaseModule
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class LogisticRegression(BaseModule):
    """
    Logistic regression model.

    Args:
        n_obs:
            Number of observations.
        feature_schema:
            The variable names schema for the input data validation.
        c_categories:
            Number of categories.
        W_prior_scale:
            The scale of the Laplace prior for the weights.
    """

    def __init__(
        self,
        n_obs: int,
        feature_schema: Sequence[str],
        c_categories: int,
        W_prior_scale: float = 1.0,
    ) -> None:
        super().__init__()
        # data
        self.n_obs = n_obs
        self.feature_schema = np.array(feature_schema)
        self.g_features = len(feature_schema)
        self.c_categories = c_categories
        # parameters
        self.W_loc_gc = torch.nn.Parameter(torch.zeros(self.g_features, c_categories))
        self.W_scale_gc = PyroParam(
            torch.ones(self.g_features, c_categories),
            constraints=constraints.positive,
        )
        self.W_prior_scale = W_prior_scale
        self.b_c = torch.nn.Parameter(torch.zeros(c_categories))
        # loss
        self.elbo = pyro.infer.Trace_ELBO()

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x_ng = tensor_dict["X"]
        feature_g = tensor_dict["var_names"]
        y_n = tensor_dict["y_n"]
        return (x_ng, feature_g, y_n), {}

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray, y_n: torch.Tensor) -> torch.Tensor:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)
        return self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n)

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        W_prior_scale = torch.tensor(self.W_prior_scale, device=x_ng.device)
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, W_prior_scale).expand([self.g_features, self.c_categories]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0]):
            logits_nc = x_ng @ W_gc + self.b_c
            pyro.sample("y", dist.Categorical(logits=logits_nc), obs=y_n)

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        pyro.sample("W", dist.Normal(self.W_loc_gc, self.W_scale_gc).to_event(2))
