# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence

import lightning.pytorch as pl
import numpy as np
import pyro
import pyro.distributions as dist
import torch

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class LogisticRegression(CellariumModel):
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
        W_init_scale:
            Initialization scale for the ``W_gc`` parameter.
        seed:
            Random seed used to initialize parameters.
        log_metrics:
            Whether to log the histogram of the ``W_gc`` parameter.
    """

    def __init__(
        self,
        n_obs: int,
        feature_schema: Sequence[str],
        c_categories: int,
        W_prior_scale: float = 1.0,
        W_init_scale: float = 1.0,
        seed: int = 0,
        log_metrics: bool = True,
    ) -> None:
        super().__init__()

        # data
        self.n_obs = n_obs
        self.feature_schema = np.array(feature_schema)
        self.g_features = len(feature_schema)
        self.c_categories = c_categories

        rng = torch.Generator()
        rng.manual_seed(seed)
        # parameters
        self.W_prior_scale: torch.Tensor
        self.register_buffer("W_prior_scale", torch.tensor(W_prior_scale))
        self.W_gc = torch.nn.Parameter(W_init_scale * torch.randn((self.g_features, c_categories), generator=rng))
        self.b_c = torch.nn.Parameter(torch.zeros(c_categories))

        # loss
        self.elbo = pyro.infer.Trace_ELBO()

        self.log_metrics = log_metrics

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x_ng = tensor_dict["x_ng"]
        feature_g = tensor_dict["feature_g"]
        y_n = tensor_dict["y_n"]
        return (x_ng, feature_g, y_n), {}

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray, y_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng:
                The input data.
            feature_g:
                The variable names for the input data.
            y_n:
                The target data.

        Returns:
            The loss.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)
        return self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n)

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, self.W_prior_scale).expand([self.g_features, self.c_categories]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0]):
            logits_nc = x_ng @ W_gc + self.b_c
            pyro.sample("y", dist.Categorical(logits=logits_nc), obs=y_n)

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        pyro.sample("W", dist.Delta(self.W_gc).to_event(2))

    def on_batch_end(self, trainer: pl.Trainer) -> None:
        if trainer.global_rank != 0:
            return

        if not self.log_metrics:
            return

        if (trainer.global_step + 1) % trainer.log_every_n_steps != 0:  # type: ignore[attr-defined]
            return

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_histogram(
                    "W_gc",
                    self.W_gc,
                    global_step=trainer.global_step,
                )
