# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional

from cellarium.ml.distributions import PyroCategorical
from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


def compute_valid_mask(input_categories: list[str], output_categories: list[str]) -> list[int]:
    # Return indices of output_categories in the same order as input_categories
    index_map = {cat: i for i, cat in enumerate(output_categories)}
    return [index_map[cat] for cat in input_categories]


class SOCAM(CellariumModel, PredictMixin, ValidateMixin):
    """
    Logistic regression model for cell type ontology classification.

    Args:
        n_obs: Number of observations in the dataset (used to size the Pyro plate).
        var_names_g: The variable-name schema for the input data; used for validation.
        output_categories: Total number of target categories expected at prediction/validation time.
            Used when the trained model has fewer categories than the final output space.
        descendant_tensor: Binary (0/1) tensor of shape (n_categories, n_categories) defining
            the descendant relationships between categories. Row i contains ones
            for all categories considered descendants of category i (plus self). Used for
            probability-propagation.
        cl_names: Array of unique category identifiers for the training labels.
            Boolean or integer mask mapping the model's category indices to the
        probability_propagation_flag: If True, applies hierarchical probability propagation before sampling
            or predicting the output distribution.
        W_prior_scale: Scale (b) parameter of the Laplace prior on the weight matrix `W_gc`.
        W_init_scale: Standard deviation for initializing `W_gc`.
        seed: Random seed used to initialize parameters.
        log_metrics: If True, logs weight histograms (TensorBoard) during training.
            If True, logs weight histograms (TensorBoard) during training.
    """

    def __init__(
        self,
        n_obs: int,
        var_names_g: np.ndarray,
        descendant_tensor: torch.Tensor,
        cl_names: list[str],
        probability_propagation_flag: bool = True,
        W_prior_scale: float = 1.0,
        W_init_scale: float = 1.0,
        seed: int = 0,
        log_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.n_obs = n_obs
        self.var_names_g = var_names_g
        self.n_vars = len(var_names_g)
        self.cl_names = cl_names
        if descendant_tensor.shape[0] != descendant_tensor.shape[1]:
            raise ValueError("`descendant_tensor` should be a square matrix.")
        if descendant_tensor.trace() != descendant_tensor.shape[0]:
            raise ValueError(
                "`descendant_tensor` should have ones on the diagonal (each category is a descendant of itself)."
            )
        if len(cl_names) != descendant_tensor.shape[0]:
            raise ValueError("Length of `cl_names` should match the number of rows in `descendant_tensor`.")
        self._descendant_tensor = descendant_tensor
        self.register_buffer("descendant_tensor", descendant_tensor)
        self.n_categories = descendant_tensor.shape[0]
        self.probability_propagation_flag = probability_propagation_flag
        self.out_distribution = PyroCategorical
        self.seed = seed

        # parameters
        self._W_prior_scale = W_prior_scale
        self.W_init_scale = W_init_scale
        self.W_prior_scale: torch.Tensor
        self.register_buffer("W_prior_scale", torch.empty(()))
        self.W_gc = torch.nn.Parameter(torch.empty(self.n_vars, self.n_categories))
        self.b_c = torch.nn.Parameter(torch.empty(self.n_categories))
        self.elbo = pyro.infer.Trace_ELBO()
        self.log_metrics = log_metrics

        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng_device = self.W_gc.device.type if self.W_gc.device.type != "meta" else "cpu"
        rng = torch.Generator(device=rng_device)
        rng.manual_seed(self.seed)
        self.W_prior_scale.fill_(self._W_prior_scale)
        self.descendant_tensor.copy_(self._descendant_tensor)
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
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n)
        return {"loss": loss}

    def _compute_regression(self, x_ng: torch.Tensor, W_gc: torch.Tensor) -> torch.Tensor:
        logits_nc = x_ng @ W_gc + self.b_c
        return logits_nc

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, self.W_prior_scale).expand([self.n_vars, self.n_categories]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0], dim=-2):
            logits_nc = self._compute_regression(x_ng, W_gc)
            if self.probability_propagation_flag:
                logits_nc = self.propagated_logits(logits_nc=logits_nc)
            pyro.sample("y", self.out_distribution(logits=logits_nc), obs=y_n)

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
        logits_nc = self._compute_regression(x_ng, self.W_gc)
        probs_nc = torch.nn.functional.softmax(logits_nc, dim=1)
        if self.probability_propagation_flag:
            probs_nc = self.probability_propagation(probs_nc=probs_nc)
        return {"y_logits_nc": logits_nc, "cell_type_probs_nc": probs_nc}

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

    def probability_propagation(self, probs_nc: torch.Tensor) -> torch.Tensor:
        """
        Propagate probabilities up the hierarchy defined by `descendant_tensor` using matrix multiplication.
        This effectively sums the probabilities of all descendant categories for each category.
        The output is then clamped to a maximum of 1.0 to ensure valid probability values.

        Args:
            probs_nc: Tensor of shape (n, c) containing the probabilities for each category

        Returns:
            Tensor of shape (n, c) containing the propagated probabilities for each category
        """
        propagated_probs_nc = torch.einsum(
            "nc,kc->nk",
            probs_nc,
            self.descendant_tensor,
        )
        return torch.clamp(propagated_probs_nc, max=1.0)

    @torch.compile()
    def propagated_logits(self, logits_nc: torch.Tensor):
        """
        Perform probability propagation in logit space.

        Args:
            logits_nc: Tensor of shape (n, c) containing the logits for each category
        Returns:
            Tensor of shape (n, c) containing the logits for each category after propagation
        """
        propagated_logits_nc = self.logsumexp_propagated(logits_nc) - torch.logsumexp(logits_nc, dim=1, keepdim=True)
        return propagated_logits_nc

    def logsumexp_propagated(self, logits_nc):
        desc_matrix_cc = self.descendant_tensor
        temp = torch.where(desc_matrix_cc.T == 0, float("-inf"), logits_nc.unsqueeze(dim=-1) * desc_matrix_cc.T)
        return temp.logsumexp(dim=1)
