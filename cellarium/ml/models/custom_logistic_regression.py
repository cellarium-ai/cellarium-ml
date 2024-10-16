# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import time

import lightning.pytorch as pl
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional

from cellarium.ml.categorical_distribution import categorical_distribution
from cellarium.ml.data.fileio import read_pkl_from_gcs
from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class CustomLogisticRegression(CellariumModel, PredictMixin, ValidateMixin):
    """
    Logistic regression model.

    Args:
        n_obs:
            Number of observations.
        var_names_g:
            The variable names schema for the input data validation.
        y_categories:
            The categories for the target data.
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
        var_names_g: np.ndarray,
        y_categories: np.ndarray | None,
        W_prior_scale: float = 1.0,
        W_init_scale: float = 1.0,
        activation_fn: str = 'softmax',
        out_distribution: str = 'Categorical',
        seed: int = 0,
        probability_propagation_flag: bool = False,
        target_row_descendent_col_torch_tensor_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/target_row_descendent_col_torch_tensor.pkl',
        y_categories_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_filtered_sorted_unique_cells.pkl',
        log_metrics: bool = True,
    ) -> None:
        super().__init__()

        # data
        self.n_obs = n_obs
        self.var_names_g = var_names_g
        self.n_vars = len(var_names_g)
        #self.y_categories = y_categories
        self.y_categories = read_pkl_from_gcs(y_categories_path)
        self.target_row_descendent_col_torch_tensor = read_pkl_from_gcs(target_row_descendent_col_torch_tensor_path)
        self.n_categories = len(self.y_categories)
        self.activation_fn = getattr(torch.nn.functional, activation_fn)
        self.probability_propagation_flag = probability_propagation_flag
        self.out_distribution = getattr(categorical_distribution,'Pyro'+out_distribution)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = seed
        # parameters
        self._W_prior_scale = W_prior_scale
        self.W_init_scale = W_init_scale
        self.W_prior_scale: torch.Tensor
        self.register_buffer("W_prior_scale", torch.empty(()))
        self.W_gc = torch.nn.Parameter(torch.empty(self.n_vars, self.n_categories))
        self.b_c = torch.nn.Parameter(torch.empty(self.n_categories))
        self.reset_parameters()

        # loss
        self.elbo = pyro.infer.Trace_ELBO()

        self.log_metrics = log_metrics

    def reset_parameters(self) -> None:
        rng_device = self.W_gc.device.type if self.W_gc.device.type != "meta" else "cpu"
        rng = torch.Generator(device=rng_device)
        rng.manual_seed(self.seed)
        self.W_prior_scale.fill_(self._W_prior_scale)
        self.W_gc.data.normal_(0, self.W_init_scale, generator=rng)
        self.b_c.data.zero_()

    def forward(
        self, x_ng: torch.Tensor, var_names_g: np.ndarray, y_n: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                The input data.
            var_names_g:
                The variable names for the input data.
            y_n:
                The target data.
            y_categories:
                The categories for the input target data.

        Returns:
            A dictionary with the loss value.
        """
        #y_n = y_n.to(self.device)
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        #assert_arrays_equal("y_categories", y_categories, "self.y_categories", self.y_categories)
        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n)
        return {"loss": loss}

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, self.W_prior_scale).expand([self.n_vars, self.n_categories]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0]):
            logits_nc = x_ng @ W_gc + self.b_c
            activation_out = self.activation_fn(logits_nc.to(dtype=torch.float), dim=1)
            if (self.probability_propagation_flag==1):
                print("NIMISH ENTERED PP FLAG 1")
                activation_out = self.probability_propagation(activation_out_gpu=activation_out)
            if self.out_distribution == categorical_distribution.PyroCategorical:
                print(f"NIMISH OUT DIST IS CATEGORICAL AND Y_N TYPE IS {y_n.dtype}")
                print(f"NIMISH ACTIVATION OUT DTYPE BEFORE CATEGORICAL IS {activation_out.dtype}")
                pyro.sample("y", self.out_distribution(probs=activation_out), obs=y_n)
            elif self.out_distribution == dist.Bernoulli:
                pyro.sample("y", self.out_distribution(probs=activation_out).to_event(1), obs=y_n)

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        pyro.sample("W", dist.Delta(self.W_gc).to_event(2))

    def predict(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Predict the target logits.

        Args:
            x_ng:
                The input data.
            var_names_g:
                The variable names for the input data.

        Returns:
            A dictionary with the target logits.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        logits_nc = x_ng @ self.W_gc + self.b_c
        return {"y_logits_nc": logits_nc}

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
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

    def probability_propagation(self,activation_out_gpu:torch.tensor) -> torch.tensor:
        """
        for each column in activation_out_gpu_clone, col, we get a tensor of 
        activation_out_gpu[:,[children_indices[col]]],
        then we take the rowwise sume of these columns.
        then we add those values with all rows in 'col' column in activation_out_gpu_clone
        TRIAL CODE AVAILABLE IN TRIAL.IPYNB - PROBABILITY PROPAGATION CODE TRIAL
        """
        print(f"NIMISH ACTIVATION OUT BEFORE PP IS {activation_out_gpu[0:2,0:3]}")
        start_time = time.time()
        propagated_p = torch.einsum("nc,kc->nk", activation_out_gpu, self.target_row_descendent_col_torch_tensor.to(activation_out_gpu.device))
        end_time = time.time()
        print(f"NIMISH PP ELAPSED TIME IS {end_time - start_time} seconds")
        print(f"NIMISH ACTIVATION OUT AFTER PP IS {propagated_p[0:2,0:3]}")
        return propagated_p
