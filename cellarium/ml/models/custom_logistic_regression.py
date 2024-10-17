# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


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
        alpha: float = 0.5,
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
        self.alpha = alpha
        self.n_obs = n_obs
        self.var_names_g = var_names_g
        self.n_vars = len(var_names_g)
        #self.y_categories = y_categories
        self.y_categories = read_pkl_from_gcs(y_categories_path)
        self.target_row_descendent_col_torch_tensor = read_pkl_from_gcs(target_row_descendent_col_torch_tensor_path)
        self.n_categories = len(self.y_categories)
        self.activation_fn = getattr(torch.nn.functional, activation_fn)
        self.probability_propagation_flag = probability_propagation_flag
        if out_distribution == "Categorical":
            self.out_distribution = getattr(categorical_distribution,'Pyro'+out_distribution)
        else:
            self.out_distribution = getattr(dist,out_distribution)
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
        self, x_ng: torch.Tensor, var_names_g: np.ndarray, y_n: torch.Tensor, descendents_nc: torch.Tensor, **kwargs
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
        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n, descendents_nc)
        return {"loss": loss}

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor, descendents_nc: torch.Tensor) -> None:
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, self.W_prior_scale).expand([self.n_vars, self.n_categories]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0],dim=-2):
            logits_nc = x_ng @ W_gc + self.b_c
            activation_out = self.activation_fn(logits_nc.to(dtype=torch.float), dim=1)
            if (self.probability_propagation_flag==1):
                activation_out = self.probability_propagation(activation_out_gpu=activation_out)
            if self.out_distribution == categorical_distribution.PyroCategorical:
                #pyro.sample("y", self.out_distribution(probs=activation_out), obs=y_n)
                pyro.sample("y", dist.Categorical(probs=activation_out), obs=y_n)
            elif self.out_distribution == dist.Bernoulli:
                scale = self.get_scale(descendents_nc=descendents_nc) #n,c
                with pyro.poutine.scale(scale=scale):
                    with pyro.plate("categories", size=self.n_categories, dim=-1):
                        print(f"NIMISH XNG SHAPE 0 IS {x_ng.shape[0]}")
                        print(f"NIMISH OBS_N SHAPE IS {self.n_obs}")
                        print(f'NIMISH SCALE SHAPE IS {scale.shape}')
                        print(f"NIMISH N CATEGORIES ARE {self.n_categories}")
                        print(f"NIMISH Y_N SIZE IS {y_n.shape}")
                        print(f"NIMISH ACTIVATION OUT SHAPE IS {activation_out.shape}")
                        pyro.sample("y", self.out_distribution(probs=activation_out), obs=y_n)

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor, descendents_nc: torch.Tensor) -> None:
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
        propagated_p = torch.einsum("nc,kc->nk", activation_out_gpu, self.target_row_descendent_col_torch_tensor.to(device=activation_out_gpu.device))
        return torch.clamp(propagated_p, max=1.0)

    def get_scale(self, descendents_nc:torch.Tensor) -> torch.Tensor:
        """
        scale: torch.tensor(ones) same shape as y_n
        for each sample, 
            get indices of children of y_n_original (original single target),
            in scale, multiply these indices by alpha for that particular sample
        return scale
        - 1 way to get y_n_original is to use y_n and y_n_original from the extracts, and pass y_n as multilabel target and pass
        y_n_original as single label target
        - another way is: get column indices of 1s from y_n (ones_indices). then get the children column indices for all these
          one_indices.
        """
        return descendents_nc.masked_fill_(descendents_nc == 0, self.alpha)
