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
        self._subset_cache: dict[tuple[str, ...], tuple[list[str], list[int], torch.Tensor, dict[str, int]]] = {}
        self._full_label_lookup: dict[str, int] | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng_device = self.W_gc.device.type if self.W_gc.device.type != "meta" else "cpu"
        rng = torch.Generator(device=rng_device)
        rng.manual_seed(self.seed)
        self.W_prior_scale.fill_(self._W_prior_scale)
        self.descendant_tensor.copy_(self._descendant_tensor)
        self.W_gc.data.normal_(0, self.W_init_scale, generator=rng)
        self.b_c.data.zero_()
        self._subset_cache.clear()
        self._full_label_lookup = None

    def _get_subset_info(self, cl_name_subset: list[str]) -> tuple[list[str], list[int], torch.Tensor, dict[str, int]]:
        """
        Return the sorted category names, their indices into ``self.cl_names``, the
        corresponding submatrix of ``self.descendant_tensor``, and a pre-built label
        lookup dict ``{name: subset_index}`` for fast per-batch label conversion.
        Results are cached by the sorted tuple of names so that order does not matter
        and all outputs are only computed once per unique subset.

        Args:
            cl_name_subset:
                Category names to select. Must all be present in ``self.cl_names``.

        Returns:
            A 4-tuple ``(sorted_names, indices, descendant_tensor_subset_cc, label_lookup)``
            where ``sorted_names`` is the alphabetically sorted list of requested names,
            ``indices`` is the corresponding list of integer positions in ``self.cl_names``,
            ``descendant_tensor_subset_cc`` is the square submatrix of shape
            ``(len(indices), len(indices))``, and ``label_lookup`` maps each subset
            category name to its 0-based position within the sorted subset.
        """
        key = tuple(sorted(cl_name_subset))
        if key not in self._subset_cache:
            index_map = {cat: i for i, cat in enumerate(self.cl_names)}
            indices = [index_map[cat] for cat in key]
            ix = torch.tensor(indices, dtype=torch.long, device=self.descendant_tensor.device)
            descendant_tensor_subset_cc = self.descendant_tensor[ix][:, ix]
            label_lookup: dict[str, int] = {name: pos for pos, name in enumerate(key)}
            self._subset_cache[key] = (list(key), indices, descendant_tensor_subset_cc, label_lookup)
        return self._subset_cache[key]

    def _cl_names_to_indices(self, cl_names_n: np.ndarray, label_lookup: dict[str, int]) -> torch.Tensor:
        """
        Convert a per-cell array of string category names to a 1-D integer tensor of
        0-based indices into the active category list, using a pre-built lookup dict.

        Args:
            cl_names_n:
                Array of length n containing category name strings for each cell.
            label_lookup:
                Pre-built mapping from category name to 0-based integer index.
                Obtained from ``_get_subset_info`` or ``_full_label_lookup``.

        Returns:
            Long tensor of shape ``(n,)`` with integer category indices.

        Raises:
            ValueError: If any label in ``cl_names_n`` is not present in ``label_lookup``.
        """
        try:
            return torch.tensor([label_lookup[c] for c in cl_names_n], dtype=torch.long)
        except KeyError as exc:
            valid = sorted(label_lookup.keys())
            raise ValueError(f"Label {exc} is not in the active category list. Valid labels are: {valid}") from exc

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        cl_names_n: np.ndarray,
        cl_name_subset: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                The input data.
            var_names_g:
                The variable names for the input data.
            cl_names_n:
                Array of length n containing a category name string (from ``self.cl_names``) for
                each cell. When ``cl_name_subset`` is provided, every label must be a member of
                that subset.
            cl_name_subset:
                Optional list of category names (from ``self.cl_names``) to restrict
                training to. The list is sorted internally so order does not matter.
                When ``None``, all categories are used.

        Returns:
            A dictionary with the loss value.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        if cl_name_subset is not None:
            _, indices, descendant_tensor_subset_cc, label_lookup = self._get_subset_info(cl_name_subset)
        else:
            indices = None
            descendant_tensor_subset_cc = self.descendant_tensor
            if self._full_label_lookup is None:
                self._full_label_lookup = {name: i for i, name in enumerate(self.cl_names)}
            label_lookup = self._full_label_lookup
        y_n = self._cl_names_to_indices(cl_names_n, label_lookup)
        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n, indices, descendant_tensor_subset_cc)
        return {"loss": loss}

    def _compute_regression(self, x_ng: torch.Tensor, W_gc: torch.Tensor, b_c: torch.Tensor) -> torch.Tensor:
        return x_ng @ W_gc + b_c

    def model(
        self,
        x_ng: torch.Tensor,
        y_n: torch.Tensor,
        indices: list[int] | None = None,
        descendant_tensor_subset_cc: torch.Tensor | None = None,
    ) -> None:
        if indices is not None:
            n_cats = len(indices)
            b_c = self.b_c[indices]
            assert descendant_tensor_subset_cc is not None
            desc_cc = descendant_tensor_subset_cc
        else:
            n_cats = self.n_categories
            b_c = self.b_c
            desc_cc = descendant_tensor_subset_cc if descendant_tensor_subset_cc is not None else self.descendant_tensor
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, self.W_prior_scale).expand([self.n_vars, n_cats]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0], dim=-2):
            logits_nc = self._compute_regression(x_ng, W_gc, b_c)
            if self.probability_propagation_flag:
                logits_nc = self.propagate_logits(logits_nc, desc_cc)
            pyro.sample("y", self.out_distribution(logits=logits_nc), obs=y_n)

    def guide(
        self,
        x_ng: torch.Tensor,
        y_n: torch.Tensor,
        indices: list[int] | None = None,
        descendant_tensor_subset_cc: torch.Tensor | None = None,
    ) -> None:
        if indices is not None:
            pyro.sample("W", dist.Delta(self.W_gc[:, indices]).to_event(2))
        else:
            pyro.sample("W", dist.Delta(self.W_gc).to_event(2))

    @torch.compile()
    def propagate_probs(self, probs_nc: torch.Tensor, descendant_tensor_cc: torch.Tensor) -> torch.Tensor:
        """
        Propagate probabilities up the hierarchy defined by ``descendant_tensor_cc`` using matrix multiplication.
        This effectively sums the probabilities of all descendant categories for each category.
        The output is then clamped to a maximum of 1.0 to ensure valid probability values.

        Args:
            probs_nc: Tensor of shape (n, c) containing the probabilities for each category.
            descendant_tensor_cc: Binary tensor of shape (c, c) defining descendant relationships.

        Returns:
            Tensor of shape (n, c) containing the propagated probabilities for each category
        """
        propagated_probs_nc = torch.einsum(
            "nc,kc->nk",
            probs_nc,
            descendant_tensor_cc,
        )
        return torch.clamp(propagated_probs_nc, max=1.0)

    @torch.compile()
    def propagate_logits(self, logits_nc: torch.Tensor, descendant_tensor_cc: torch.Tensor) -> torch.Tensor:
        """
        Perform probability propagation in logit space.

        Args:
            logits_nc: Tensor of shape (n, c) containing the logits for each category.
            descendant_tensor_cc: Binary tensor of shape (c, c) defining descendant relationships.
        Returns:
            Tensor of shape (n, c) containing the logits for each category after propagation
        """
        propagated_logits_nc = self.logsumexp_propagated(logits_nc, descendant_tensor_cc) - torch.logsumexp(
            logits_nc, dim=1, keepdim=True
        )
        return propagated_logits_nc

    def logsumexp_propagated(self, logits_nc: torch.Tensor, desc_matrix_cc: torch.Tensor) -> torch.Tensor:
        temp = torch.where(desc_matrix_cc.T == 0, float("-inf"), logits_nc.unsqueeze(dim=-1) * desc_matrix_cc.T)
        return temp.logsumexp(dim=1)

    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        cl_name_subset: list[str] | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Predict the target logits.

        Args:
            x_ng:
                The input data.
            var_names_g:
                The variable names for the input data.
            cl_name_subset:
                Optional list of category names (from ``self.cl_names``) to restrict
                prediction to. The list is sorted internally so order does not matter.
                When ``None``, all categories are used. Output tensors will have shape
                ``(n, len(cl_name_subset))`` when provided.

        Returns:
            A dictionary with the target logits.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        if cl_name_subset is not None:
            _, indices, descendant_tensor_subset_cc, _ = self._get_subset_info(cl_name_subset)
            W_gc = self.W_gc[:, indices]
            b_c = self.b_c[indices]
        else:
            W_gc = self.W_gc
            b_c = self.b_c
            descendant_tensor_subset_cc = self.descendant_tensor
        logits_nc = self._compute_regression(x_ng, W_gc, b_c)
        probs_nc = torch.nn.functional.softmax(logits_nc, dim=1)
        if self.probability_propagation_flag:
            probs_nc = self.propagate_probs(probs_nc, descendant_tensor_subset_cc)
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
