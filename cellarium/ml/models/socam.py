# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypedDict

import lightning.pytorch as pl
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional

from cellarium.ml.distributions import UnconstrainedPyroCategorical
from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class NonleafInfo(TypedDict):
    nonleaf_desc_cc: torch.Tensor
    perm: torch.Tensor
    inv_perm: torch.Tensor


class SubsetInfo(TypedDict):
    names: list[str]
    indices: list[int]
    descendant_tensor_cc: torch.Tensor
    label_lookup: dict[str, int]
    nonleaf_desc_cc: torch.Tensor
    perm: torch.Tensor
    inv_perm: torch.Tensor


def _expand_with_ancestors(
    cl_name_subset: list[str],
    cl_names: list[str],
    descendant_tensor: torch.Tensor,
) -> list[str]:
    """Return ``cl_name_subset`` extended to include every ancestor of each node.

    ``descendant_tensor[k, j] == 1`` means j is a descendant of k, equivalently k
    is an ancestor of j.  So the ancestors of j are all k where column j is 1.

    Args:
        cl_name_subset: Category names requested by the user.
        cl_names: Full ordered list of category names (rows/cols of ``descendant_tensor``).
        descendant_tensor: Binary ``(C, C)`` tensor on any device.

    Returns:
        Sorted list containing every name in ``cl_name_subset`` plus all of their
        ancestors that are present in ``cl_names``.
    """
    index_map = {cat: i for i, cat in enumerate(cl_names)}
    expanded: set[int] = set()
    for name in cl_name_subset:
        j = index_map[name]
        # column j: entry k==1 means k is an ancestor of j (including j itself)
        ancestor_indices = (descendant_tensor[:, j] > 0).nonzero(as_tuple=True)[0].tolist()
        expanded.update(ancestor_indices)
    return sorted(cl_names[i] for i in expanded)


def _build_nonleaf_info(desc_matrix_cc: torch.Tensor) -> NonleafInfo:
    """Precompute the non-leaf descriptor tensors needed by ``propagate_logits``.

    Produces a permutation that places non-leaf categories first and leaves last,
    so the hot path can use ``torch.cat`` and simple slices instead of boolean-mask
    assignment.  Boolean-mask assignment internally calls ``aten.nonzero`` which
    ``torch.compile`` / inductor cannot fuse; integer-index gather and ``torch.cat``
    have no such restriction.

    Runs once per unique ``desc_matrix_cc`` (construction time, not the hot path).

    Args:
        desc_matrix_cc: Binary ``(c, c)`` descendant tensor.

    Returns:
        A :class:`NonleafInfo` dict with keys ``nonleaf_desc_cc``, ``perm``, and
        ``inv_perm``.  ``nonleaf_desc_cc`` is ``(c_nonleaf, c)`` with columns in
        ``perm`` order; ``perm`` places non-leaf indices first, leaf last;
        ``inv_perm`` restores the original column order.
    """
    nonleaf_mask = desc_matrix_cc.sum(dim=1) > 1
    nonleaf_indices = nonleaf_mask.nonzero(as_tuple=True)[0]  # (c_nonleaf,)
    leaf_indices = (~nonleaf_mask).nonzero(as_tuple=True)[0]  # (c_leaf,)
    perm = torch.cat([nonleaf_indices, leaf_indices])  # non-leaf first
    inv_perm = torch.argsort(perm)
    # Reorder desc columns to match the permuted input order used in propagate_logits
    nonleaf_desc_cc = desc_matrix_cc[nonleaf_indices][:, perm]  # (c_nonleaf, c)
    return {"nonleaf_desc_cc": nonleaf_desc_cc, "perm": perm, "inv_perm": inv_perm}


@torch.compile()
def propagate_probs(probs_nc: torch.Tensor, descendant_tensor_cc: torch.Tensor) -> torch.Tensor:
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


def _logsumexp_propagated(logits_nc: torch.Tensor, desc_matrix_cc: torch.Tensor) -> torch.Tensor:
    temp = torch.where(desc_matrix_cc.T == 0, float("-inf"), logits_nc.unsqueeze(dim=-1) * desc_matrix_cc.T)
    return temp.logsumexp(dim=1)


@torch.compile()
def propagate_logits(
    logits_nc: torch.Tensor,
    nonleaf_desc_cc: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
) -> torch.Tensor:
    """
    Perform probability propagation in logit space.

    Non-leaf output categories reduce over all their descendants via
    ``_logsumexp_propagated`` using a ``(c_nonleaf, c)`` submatrix, so the
    intermediate tensor is ``(n, c, c_nonleaf)`` rather than ``(n, c, c)``.
    Leaf output categories are the identity (logsumexp of a single element).

    ``perm`` / ``inv_perm`` sort columns so non-leaf outputs come first,
    allowing assembly via ``torch.cat`` and a single integer-index gather —
    avoiding ``aten.nonzero`` which breaks ``torch.compile``.

    Args:
        logits_nc: ``(n, c)`` raw logit tensor.
        nonleaf_desc_cc: ``(c_nonleaf, c)`` descendant rows for non-leaf outputs,
            with columns in ``perm`` order (from ``_build_nonleaf_info``).
        perm: ``(c,)`` permutation — non-leaf indices first, leaf last
            (from ``_build_nonleaf_info``).
        inv_perm: ``(c,)`` inverse of ``perm`` (from ``_build_nonleaf_info``).
    Returns:
        ``(n, c)`` propagated log-probability tensor in original column order.
    """
    c_nonleaf = nonleaf_desc_cc.shape[0]
    logits_reordered = logits_nc[:, perm]  # (n, c): non-leaf first
    nonleaf_part = _logsumexp_propagated(logits_reordered, nonleaf_desc_cc)  # (n, c_nonleaf)
    leaf_part = logits_reordered[:, c_nonleaf:]  # (n, c_leaf)
    out = torch.cat([nonleaf_part, leaf_part], dim=1)[:, inv_perm]  # (n, c) original order
    return out - torch.logsumexp(logits_nc, dim=1, keepdim=True)


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
        cl_name_subset: Optional list of category names (from ``cl_names``) to restrict
            training and prediction to. The list is sorted internally so order does not matter.
            When ``None``, all categories are used.
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
        cl_name_subset: list[str] | None = None,
        probability_propagation_flag: bool = True,
        W_prior_scale: float = 1e-2,
        W_init_scale: float = 1.0,
        seed: int = 0,
        log_metrics: bool = True,
        include_ancestors_of_cl_name_subset: bool = True,
    ) -> None:
        super().__init__()
        self.n_obs = n_obs
        self.var_names_g = var_names_g
        self.n_vars = len(var_names_g)
        self.cl_names = cl_names
        descendant_tensor = descendant_tensor.float()
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
        self.include_ancestors_of_cl_name_subset = include_ancestors_of_cl_name_subset
        if include_ancestors_of_cl_name_subset and cl_name_subset is not None:
            cl_name_subset = _expand_with_ancestors(cl_name_subset, cl_names, descendant_tensor)
        self.cl_name_subset = cl_name_subset
        self.probability_propagation_flag = probability_propagation_flag
        self.out_distribution = UnconstrainedPyroCategorical
        self.seed = seed

        # parameters
        self._W_prior_scale = W_prior_scale
        self.W_init_scale = W_init_scale
        self.W_prior_scale: torch.Tensor
        self.register_buffer("W_prior_scale", torch.empty(()))
        self.W_gc = torch.nn.Parameter(torch.empty(self.n_vars, self.n_categories, dtype=torch.float))
        self.b_c = torch.nn.Parameter(torch.empty(self.n_categories, dtype=torch.float))
        self.elbo = pyro.infer.Trace_ELBO()
        self.log_metrics = log_metrics
        self._subset_cache: dict[tuple[str, ...], SubsetInfo] = {}
        self._full_label_lookup: dict[str, int] | None = None
        self._full_nonleaf_info: NonleafInfo | None = None

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
        self._full_nonleaf_info = None

    def _get_subset_info(self, cl_name_subset: list[str]) -> SubsetInfo:
        """
        Return cached information for a category subset.

        Results are cached by the sorted tuple of names so that order does not
        matter and all outputs are computed only once per unique subset.

        Args:
            cl_name_subset:
                Category names to select. Must all be present in ``self.cl_names``.

        Returns:
            A :class:`SubsetInfo` dict with keys:

            * ``names``: sorted list of the requested category names.
            * ``indices``: their integer positions in ``self.cl_names``.
            * ``descendant_tensor_cc``: ``(c, c)`` submatrix of
              ``self.descendant_tensor``.
            * ``label_lookup``: ``{name: subset_index}`` for fast per-batch
              label conversion.
            * ``nonleaf_desc_cc``, ``perm``, ``inv_perm``: see
              :func:`_build_nonleaf_info`.
        """
        key = tuple(sorted(cl_name_subset))
        if key not in self._subset_cache:
            index_map = {cat: i for i, cat in enumerate(self.cl_names)}
            indices = [index_map[cat] for cat in key]
            ix = torch.tensor(indices, dtype=torch.long, device=self.descendant_tensor.device)
            descendant_tensor_cc = self.descendant_tensor[ix][:, ix]
            label_lookup: dict[str, int] = {name: pos for pos, name in enumerate(key)}
            nonleaf_info = _build_nonleaf_info(descendant_tensor_cc)
            self._subset_cache[key] = {
                "names": list(key),
                "indices": indices,
                "descendant_tensor_cc": descendant_tensor_cc,
                "label_lookup": label_lookup,
                "nonleaf_desc_cc": nonleaf_info["nonleaf_desc_cc"],
                "perm": nonleaf_info["perm"],
                "inv_perm": nonleaf_info["inv_perm"],
            }
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
                each cell. When ``self.cl_name_subset`` is set, every label must be a member of
                that subset.

        Returns:
            A dictionary with the loss value.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        if self.cl_name_subset is not None:
            info = self._get_subset_info(self.cl_name_subset)
            indices = info["indices"]
            label_lookup = info["label_lookup"]
            nonleaf_desc_cc = info["nonleaf_desc_cc"]
            perm = info["perm"]
            inv_perm = info["inv_perm"]
        else:
            indices = None
            if self._full_label_lookup is None:
                self._full_label_lookup = {name: i for i, name in enumerate(self.cl_names)}
            label_lookup = self._full_label_lookup
            if self._full_nonleaf_info is None:
                self._full_nonleaf_info = _build_nonleaf_info(self.descendant_tensor)
            nonleaf_desc_cc = self._full_nonleaf_info["nonleaf_desc_cc"]
            perm = self._full_nonleaf_info["perm"]
            inv_perm = self._full_nonleaf_info["inv_perm"]
        y_n = self._cl_names_to_indices(cl_names_n, label_lookup).to(x_ng.device)
        loss = self.elbo.differentiable_loss(
            self.model,
            self.guide,
            x_ng,
            y_n,
            indices,
            nonleaf_desc_cc,
            perm,
            inv_perm,
        )
        return {"loss": loss}

    @torch.compile()
    def _compute_regression(self, x_ng: torch.Tensor, W_gc: torch.Tensor, b_c: torch.Tensor) -> torch.Tensor:
        return x_ng @ W_gc + b_c

    def model(
        self,
        x_ng: torch.Tensor,
        y_n: torch.Tensor,
        indices: list[int] | None = None,
        nonleaf_desc_cc: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
        inv_perm: torch.Tensor | None = None,
    ) -> None:
        if indices is not None:
            n_cats = len(indices)
            b_c = self.b_c[indices]
        else:
            n_cats = self.n_categories
            b_c = self.b_c
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, self.W_prior_scale).expand([self.n_vars, n_cats]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0], dim=-2):
            logits_nc = self._compute_regression(x_ng, W_gc, b_c)
            if self.probability_propagation_flag:
                assert nonleaf_desc_cc is not None and perm is not None and inv_perm is not None
                logits_nc = propagate_logits(logits_nc, nonleaf_desc_cc, perm, inv_perm)
            pyro.sample("y", self.out_distribution(logits=logits_nc), obs=y_n)

    def guide(
        self,
        x_ng: torch.Tensor,
        y_n: torch.Tensor,
        indices: list[int] | None = None,
        nonleaf_desc_cc: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
        inv_perm: torch.Tensor | None = None,
    ) -> None:
        if indices is not None:
            pyro.sample("W", dist.Delta(self.W_gc[:, indices]).to_event(2))
        else:
            pyro.sample("W", dist.Delta(self.W_gc).to_event(2))

    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Predict the target logits.

        Args:
            x_ng:
                The input data.
            var_names_g:
                The variable names for the input data.

        Returns:
            A dictionary with the target logits. Output tensors have shape
            ``(n, len(self.cl_name_subset))`` when ``self.cl_name_subset`` is set,
            otherwise ``(n, n_categories)``.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        if self.cl_name_subset is not None:
            info = self._get_subset_info(self.cl_name_subset)
            indices = info["indices"]
            descendant_tensor_subset_cc = info["descendant_tensor_cc"]
            W_gc = self.W_gc[:, indices]
            b_c = self.b_c[indices]
        else:
            W_gc = self.W_gc
            b_c = self.b_c
            descendant_tensor_subset_cc = self.descendant_tensor
        logits_nc = self._compute_regression(x_ng, W_gc, b_c)
        probs_nc = torch.nn.functional.softmax(logits_nc, dim=1)
        if self.probability_propagation_flag:
            probs_nc = propagate_probs(probs_nc, descendant_tensor_subset_cc)
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
