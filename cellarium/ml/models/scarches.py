# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""scArches (single-cell Architectural Surgery) for reference mapping onto a trained scVI model."""

import logging
import warnings
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from typing_extensions import Self

from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.models.scanvi import SCANVI
from cellarium.ml.models.scvi import (
    LinearWithBatch,
    LinearWithBatchAndCovariates,
    SingleCellVariationalInference,
)
from cellarium.ml.utilities.core import call_func_with_batch

logger = logging.getLogger(__name__)

# the layers that take the batch, which scArches gives new-batch parameters
LINEAR_WITH_BATCH = (LinearWithBatch, LinearWithBatchAndCovariates)


class ScArchesLinearSurgeryMixin:
    """
    Mixin for :class:`~cellarium.ml.models.scvi.LinearWithBatch` and
    :class:`~cellarium.ml.models.scvi.LinearWithBatchAndCovariates` that extends them to support
    scArches new-batch surgery.

    The original ``bias_decoder`` handles old batches (frozen). A new ``new_batch_bias_weight`` parameter handles
    new batches via a weight-only linear projection of the new-batch one-hot. The two are summed. The new parameter
    is trainable, unless the layer is one that the surgery leaves frozen (see ``decoder_surgery`` of
    :class:`ScArches`), in which case it keeps its initial value.

    For a new-batch cell the frozen ``bias_decoder`` still sees a batch one-hot of zeros, so (if the decoder has
    hidden layers) it returns a constant, batch-independent offset. That offset is subtracted so that a new-batch
    cell receives only its own bias (plus, for
    :class:`~cellarium.ml.models.scvi.LinearWithBatchAndCovariates`, the frozen effect of its categorical covariates).
    Old-batch cells are unaffected.

    Each new batch starts as the *average reference batch*: its bias is the mean over the reference batches of their
    biases (without covariates). A zero bias would be that of no batch
    at all, which none of the reference cells had, so the layers that follow (e.g. BatchNorm with the statistics of
    the reference) would see inputs unlike any they were trained on.

    This mixin should always appear **before** the parent
    ``LinearWithBatch``/``LinearWithBatchAndCovariates`` in the MRO so that its
    ``compute_bias`` override takes effect.
    """

    # Declared here so mypy knows these attributes exist on any ScArchesLinearSurgeryMixin
    # instance; they are actually set by from_existing() and the concrete subclass MRO.
    n_batch_ref: int
    new_batch_bias_weight: nn.Parameter

    @classmethod
    def from_existing(
        cls,
        original: "LinearWithBatch | LinearWithBatchAndCovariates",
        n_batch_ref: int,
        n_new_batch: int,
        trainable: bool = True,
    ) -> "ScArchesLinearSurgeryMixin":
        """
        Construct an instance by adopting the frozen parameters of *original* and adding a
        new ``new_batch_bias_weight`` parameter for the new batches, initialized to the average reference batch
        (see :meth:`reset_new_batch_bias`).

        The parent ``__init__`` is bypassed so that no new parameter tensors are allocated for
        the existing weights; the original's parameters are shared (not copied).

        New parameters are explicitly created on the same device as the pretrained weights,
        overriding any ``torch.device("meta")`` default-device context that may be active
        during CLI instantiation.

        Args:
            original: the layer of the pretrained model.
            n_batch_ref: number of original (reference) batches.
            n_new_batch: number of new (query) batches being added.
            trainable: whether ``new_batch_bias_weight`` is trained, or frozen at its initial value.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)  # type: ignore[arg-type]
        # Scalar attributes from nn.Linear — obj is a LinearWithBatch subclass at runtime
        obj.in_features = original.in_features  # type: ignore[attr-defined]
        obj.out_features = original.out_features  # type: ignore[attr-defined]
        # Share frozen parameters and modules from the original
        obj.register_parameter("weight", original.weight)  # type: ignore[attr-defined]
        obj.register_parameter("bias", original.bias)  # type: ignore[attr-defined]
        obj.bias_decoder = original.bias_decoder  # type: ignore[attr-defined]
        # Surgery state — new weight lives on the same device as the pretrained model
        # (explicit device= overrides any torch.device("meta") context)
        pretrained_device = original.weight.device
        obj.n_batch_ref = n_batch_ref
        obj.new_batch_bias_weight = nn.Parameter(
            torch.zeros(original.out_features, n_new_batch, device=pretrained_device), requires_grad=trainable
        )
        obj.reset_new_batch_bias()
        return obj  # type: ignore[return-value]

    def average_reference_batch_bias(self) -> torch.Tensor:
        """
        The bias of the average reference batch, shape ``(out_features,)``: the mean over the reference batches of the
        bias of a cell of that batch with no covariates. :meth:`compute_bias` gives a new-batch cell its
        ``new_batch_bias_weight`` column in place of the bias of no batch, so with this as the column, a new-batch cell
        without covariates gets the mean of the biases of the reference batches (and with covariates, so does a
        cell when the bias is linear in the batch and covariates).
        """
        bias_decoder: nn.Module = self.bias_decoder  # type: ignore[attr-defined]
        first_linear = next(m for m in bias_decoder.modules() if isinstance(m, nn.Linear))
        n_covariates = first_linear.in_features - self.n_batch_ref
        device = first_linear.weight.device
        batch_bb = torch.eye(self.n_batch_ref, device=device)
        covariate_bp = torch.zeros(self.n_batch_ref, n_covariates, device=device) if n_covariates > 0 else None
        was_training = bias_decoder.training
        bias_decoder.eval()  # no dropout (and the running statistics of any BatchNorm)
        try:
            with torch.no_grad():
                bias_bo = super().compute_bias(  # type: ignore[misc]
                    batch_nb=batch_bb, categorical_covariate_np=covariate_bp
                )
        finally:
            bias_decoder.train(was_training)
        return bias_bo.mean(dim=0)

    def reset_new_batch_bias(self) -> None:
        """Set the bias of every new batch to that of the average reference batch."""
        if self.new_batch_bias_weight.device.type == "meta":
            return  # nothing to compute from: the values come from a checkpoint
        with torch.no_grad():
            self.new_batch_bias_weight.copy_(
                self.average_reference_batch_bias().unsqueeze(1).expand_as(self.new_batch_bias_weight)
            )

    def compute_bias(
        self,
        batch_nb: torch.Tensor,
        categorical_covariate_np: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Split ``batch_nb`` at ``n_batch_ref`` and route each portion to the appropriate decoder.

        Old-batch cells (indices < n_batch_ref) have their one-hot set in the first slice and
        zeros in the second, so the frozen ``bias_decoder`` produces their bias exactly as before
        surgery and the trainable ``new_batch_bias_weight`` contributes nothing.

        New-batch cells (indices >= n_batch_ref) have zeros in the first slice and their one-hot
        in the second. The frozen ``bias_decoder`` evaluated on an all-zero batch one-hot gives a
        constant offset (non-zero when the decoder has hidden layers), which is removed by
        subtracting the decoder's output for a zero batch one-hot and zero covariates. What remains is the
        frozen covariate effect plus the trainable ``new_batch_bias_weight`` term.
        """
        old_nb = batch_nb[..., : self.n_batch_ref]
        new_nb = batch_nb[..., self.n_batch_ref :]
        # super() dispatches to LinearWithBatch or LinearWithBatchAndCovariates
        old_bias = super().compute_bias(  # type: ignore[misc]
            batch_nb=old_nb, categorical_covariate_np=categorical_covariate_np
        )
        zero_covariate_1p = (
            None
            if categorical_covariate_np is None
            else categorical_covariate_np.new_zeros(1, categorical_covariate_np.shape[-1])
        )
        zero_bias_1o = super().compute_bias(  # type: ignore[misc]
            batch_nb=old_nb.new_zeros(1, self.n_batch_ref), categorical_covariate_np=zero_covariate_1p
        )
        is_new_batch_n1 = new_nb.sum(dim=-1, keepdim=True)
        new_bias = F.linear(new_nb, self.new_batch_bias_weight)
        return old_bias - is_new_batch_n1 * zero_bias_1o + new_bias


class ScArchesLinearWithBatch(ScArchesLinearSurgeryMixin, LinearWithBatch):
    """
    :class:`~cellarium.ml.models.scvi.LinearWithBatch` extended for scArches surgery.

    Created via :meth:`ScArchesLinearSurgeryMixin.from_existing`; do not call ``__init__``
    directly.
    """


class ScArchesLinearWithBatchAndCovariates(ScArchesLinearSurgeryMixin, LinearWithBatchAndCovariates):
    """
    :class:`~cellarium.ml.models.scvi.LinearWithBatchAndCovariates` extended for scArches surgery.

    Created via :meth:`ScArchesLinearSurgeryMixin.from_existing`; do not call ``__init__``
    directly.
    """


class ScArchesFinalAdditiveBiasLayer(nn.Module):
    """
    Replacement for :attr:`~cellarium.ml.models.scvi.DecoderSCVI.final_additive_bias_layer` that
    supports scArches new-batch surgery.

    The original layer is ``ReLU(Linear([batch_onehot, categorical_covariates]))``. Here the
    pre-activation of the frozen original (fed the old-batch one-hot and the covariates) is summed
    with a new trainable ``new_bias_weight`` projection of the new-batch one-hot, and the original
    ReLU is applied to the sum. Only the new-batch columns are trainable; the covariate weights stay
    frozen. For old-batch cells the new-batch one-hot is zero, so the output is identical to the original.

    ``new_bias_weight`` is initialised to a small positive constant (:attr:`NEW_BIAS_WEIGHT_INIT`), not to zero.
    The gradient of the ReLU at zero is zero, and for a new-batch cell without categorical covariates the
    pre-activation is exactly the new weight, so a zero initialisation would never receive a gradient.

    New parameters are explicitly created on the same device as the pretrained weights,
    overriding any ``torch.device("meta")`` default-device context.

    Args:
        original: the existing ``nn.Sequential(FullyConnectedLinear, ReLU)`` to freeze.
        n_batch_ref: number of original (reference) batches.
        n_new_batch: number of new (query) batches being added.
    """

    NEW_BIAS_WEIGHT_INIT = 1e-3

    def __init__(
        self,
        original: nn.Sequential,
        n_batch_ref: int,
        n_new_batch: int,
    ) -> None:
        super().__init__()
        self.frozen_layer = original
        self.n_batch_ref = n_batch_ref
        self.n_new_batch = n_new_batch
        out_features: int = original[0].out_features  # type: ignore[union-attr, assignment]
        pretrained_device = next(original.parameters()).device
        self.new_bias_weight = nn.Parameter(torch.empty(out_features, n_new_batch, device=pretrained_device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.new_bias_weight, self.NEW_BIAS_WEIGHT_INIT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``cat([extended_batch_nb, categorical_covariate_np], dim=-1)`` or just
               ``extended_batch_nb`` when there are no categorical covariates.  The batch
               portion has dimension ``n_batch_ref + n_new_batch``; any remaining dimensions
               are categorical.
        """
        n_total_batch = self.n_batch_ref + self.n_new_batch
        old_batch = x[..., : self.n_batch_ref]
        new_batch = x[..., self.n_batch_ref : n_total_batch]
        categorical = x[..., n_total_batch:]

        old_x = torch.cat([old_batch, categorical], dim=-1) if categorical.shape[-1] > 0 else old_batch
        linear, activation = self.frozen_layer[0], self.frozen_layer[1]
        return activation(linear(old_x) + F.linear(new_batch, self.new_bias_weight))


class ScArches(CellariumModel, PredictMixin, ValidateMixin):
    """
    scArches (single-cell Architectural Surgery) [1] adapter for a trained
    :class:`~cellarium.ml.models.SingleCellVariationalInference` model (or a subclass such as
    :class:`~cellarium.ml.models.SCANVI`).

    Surgery is performed **in-place** on *pretrained_scvi* during ``__init__``:

    * All existing parameters are frozen (``requires_grad=False``).
    * ``BatchNorm`` layers are kept in eval mode (also during training), so their running statistics
      are frozen too. Dropout is left as it is.
    * ``pretrained_scvi.n_batch`` is extended by *n_new_batch*.
    * The KL annealing schedule continues from where the pretrained model's training ended (its step and epoch
      are restored from its checkpoint): a fully annealed pretrained model is trained with the full KL weight.
    * For models with ``batch_embedded=False`` (one-hot batch representation):

      - Each :class:`~cellarium.ml.models.scvi.LinearWithBatch` and
        :class:`~cellarium.ml.models.scvi.LinearWithBatchAndCovariates` is replaced in-place by
        :class:`ScArchesLinearWithBatch` / :class:`ScArchesLinearWithBatchAndCovariates`.
        Each replacement adopts the original's frozen parameters and adds a new
        ``new_batch_bias_weight`` parameter (shape ``(out_features, n_new_batch)``,
        initialised to the bias of the average reference batch) for the new-batch bias.
      - Which of the new ``new_batch_bias_weight`` parameters are trained is set by ``decoder_surgery``. With
        ``"final_layer"`` (the default), those of the encoder and of the final layer of the decoder are trained, and
        those of the hidden layers of the decoder are frozen at the average reference batch. The final layer of the
        decoder gives a per-gene, per-batch offset of the logits of the normalized expression: a per-gene scaling
        of expression that is the same for every cell state, which calibrates the genes of a new batch without
        changing what a point of the latent space means. A batch bias in a hidden layer of the decoder, on the
        other hand, passes through nonlinearities, so its effect depends on ``z``, and training it would let the
        decoder map the latent space to expression differently for the new batches. With ``"all"``, every new
        parameter is trained.
      - If the decoder has a ``final_additive_bias_layer``, it is replaced by
        :class:`ScArchesFinalAdditiveBiasLayer` which adds a trainable projection of the
        new-batch one-hot (initialised to a small positive constant) to the frozen pre-activation. It is trained with
        either ``decoder_surgery``: it adds background counts per gene and batch, the same for every cell state.

    * For models with ``batch_embedded=True`` (learned batch embedding):

      - New embedding rows are added as separate trainable parameters
        (``batch_representation_mean_new_bd`` and ``batch_representation_std_unconstrained_new_bd``,
        initialised to the mean of the rows of the reference batches). The pretrained embedding
        tables are untouched and frozen.
      - A batch embedding feeds every layer that takes the batch, so the layers cannot be trained selectively, and
        ``decoder_surgery`` must be ``"all"``.

    Only the new-batch parameters are trainable. Cells from the original batches get exactly the
    outputs they got before surgery.

    **scANVI.** For a :class:`~cellarium.ml.models.SCANVI` model, the query cells are unlabeled, and
    ``unlabeled_prior`` sets the prior on ``z`` in their ELBO (see the ``unlabeled_prior`` argument of
    :class:`~cellarium.ml.models.SCANVI`). The default, ``"mixture"``, is the mixture of the class-conditional priors
    of the model weighted by ``p(c)``, without the (frozen) classifier, which otherwise gives the encoder a gradient
    that moves cells to where the classifier makes the KL terms small rather than to where they belong. ``p(c)`` (the
    ``y_prior`` of the model) is set to ``cell_type_proportions`` if given, and otherwise to the proportions of the
    cell types among the labeled cells that the model was trained on, if the model knows them (its
    ``class_counts``).

    **Batch indices.** By default (``shift_batch_index=True``) the ``batch_index_n`` passed to
    :meth:`forward`, :meth:`predict` and :meth:`validate` is interpreted as *query-local*: it must
    lie in ``[0, n_new_batch)`` and is shifted internally by ``n_batch_ref`` so that it addresses the new
    batches. This is what is wanted when the query dataset consists only of new batches. With
    ``shift_batch_index=False`` the indices are used as they are, so original batches
    (``0 .. n_batch_ref - 1``) and new batches (``n_batch_ref .. n_batch_ref + n_new_batch - 1``) can be
    mixed in the same pass.

    The methods :meth:`forward`, :meth:`predict` and :meth:`validate` accept any of the arguments of the
    corresponding methods of the wrapped model (e.g. ``cell_type_labels_n`` for
    :class:`~cellarium.ml.models.SCANVI`) and pass on those that it accepts.

    **CLI usage**: supply *pretrained_scvi* via a ``!CheckpointLoader`` YAML block pointing to
    a trained scVI checkpoint; *n_new_batch* is linked automatically from the data's
    ``batch_index_n`` batch key (assuming all batches in the new dataset are new batches).

    **References:**

    1. `Query to reference single-cell integration with transfer learning (Lotfollahi et al.)
       <https://www.nature.com/articles/s41587-021-01001-7>`_.

    Args:
        pretrained_scvi: a **trained** :class:`SingleCellVariationalInference` instance.
            Surgery is performed on this object; it should not be used independently after
            ``ScArches`` is constructed.
        n_new_batch: number of new (query) batches to add.
        shift_batch_index: if ``True`` (default), ``batch_index_n`` is query-local and is shifted by the
            number of original batches. If ``False``, ``batch_index_n`` is used as is.
        decoder_surgery: which layers of the decoder get trainable parameters for the new batches: ``"final_layer"``
            (default) for only its final layer (which must take the batch), or ``"all"`` for every layer that takes the
            batch. The layers of the encoder that take the batch are always trained.
        unlabeled_prior: (scANVI only) the prior on ``z`` for the unlabeled (query) cells: ``"mixture"`` (default),
            ``"scanvi"`` or ``"standard_normal"`` (see :class:`~cellarium.ml.models.SCANVI`). Ignored for scVI.
        cell_type_proportions: (scANVI only) the expected proportions (or numbers of cells) of the cell types in the
            query, by cell type name, which become ``p(c)``. Cell types that are not given get 0, and finer cell types
            than the classes of an ontology-aware model count towards their class. By default, the proportions of the
            labeled cells that the model was trained on (if it knows them), and otherwise the ``y_prior`` of the model
            as it is.
    """

    def __init__(
        self,
        pretrained_scvi: SingleCellVariationalInference,
        n_new_batch: int,
        shift_batch_index: bool = True,
        decoder_surgery: Literal["final_layer", "all"] = "final_layer",
        unlabeled_prior: Literal["mixture", "scanvi", "standard_normal"] = "mixture",
        cell_type_proportions: pd.Series | dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if pretrained_scvi.n_batch < 1:
            raise ValueError(
                f"pretrained_scvi has n_batch={pretrained_scvi.n_batch}. "
                "scArches requires a model trained with at least one batch."
            )
        if n_new_batch < 1:
            raise ValueError(f"n_new_batch must be >= 1, got {n_new_batch}.")
        if decoder_surgery not in ("final_layer", "all"):
            raise ValueError(f"decoder_surgery must be 'final_layer' or 'all', got {decoder_surgery!r}.")
        if decoder_surgery == "final_layer":
            if pretrained_scvi.batch_embedded:
                raise ValueError(
                    "decoder_surgery='final_layer' needs a one-hot batch representation (batch_embedded=False): a "
                    "batch embedding feeds every layer that takes the batch, so the hidden layers of the decoder "
                    "cannot be left frozen. Use decoder_surgery='all'."
                )
            if not isinstance(pretrained_scvi.decoder.normalized_count_decoder, LINEAR_WITH_BATCH):
                raise ValueError(
                    "decoder_surgery='final_layer' needs a final layer of the decoder that takes the batch "
                    "(LinearWithBatch or LinearWithBatchAndCovariates), so that the new batches can get their own "
                    f"per-gene offsets, but it is a {type(pretrained_scvi.decoder.normalized_count_decoder).__name__}. "
                    "Use a reference model whose decoder has such a final layer, or decoder_surgery='all'."
                )
        if not isinstance(pretrained_scvi, SCANVI) and cell_type_proportions is not None:
            raise ValueError("cell_type_proportions is only for scANVI models.")

        self.scvi = pretrained_scvi
        self.decoder_surgery = decoder_surgery
        # Continue the KL annealing schedule of the pretrained model instead of restarting it: the training position
        # of the pretrained model (restored from its checkpoint) is added to that of this training run.
        pretrained_scvi.step_offset = pretrained_scvi.step
        pretrained_scvi.epoch_offset = pretrained_scvi.epoch
        self.n_batch_ref: int = pretrained_scvi.n_batch
        self.n_new_batch: int = n_new_batch
        self.shift_batch_index = shift_batch_index

        # Freeze all existing parameters. This must happen before the surgery below creates
        # the new (trainable) parameters.
        pretrained_scvi.requires_grad_(False)

        # Extend the batch count so that batch_representation_from_batch_index generates
        # one-hots of the correct size for new-batch indices.
        pretrained_scvi.n_batch += n_new_batch

        if not pretrained_scvi.batch_embedded:
            # n_latent_batch == n_batch in the one-hot path.
            assert pretrained_scvi.n_latent_batch is not None
            pretrained_scvi.n_latent_batch += n_new_batch
            self._surgery_linear_layers()
            self._surgery_final_additive_bias()
        else:
            self._surgery_batch_embeddings()

        if isinstance(pretrained_scvi, SCANVI):
            pretrained_scvi.set_unlabeled_prior(unlabeled_prior)
            if unlabeled_prior != "standard_normal" or cell_type_proportions is not None:
                self._set_class_prior(pretrained_scvi, cell_type_proportions)

        # Put the BatchNorm layers into eval mode right away (see train()).
        self.train(self.training)

    # ------------------------------------------------------------------
    # Surgery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_class_prior(scanvi: SCANVI, cell_type_proportions: pd.Series | dict[str, float] | None) -> None:
        """Set ``p(c)`` of ``scanvi`` to ``cell_type_proportions``, or else to the proportions of its training data."""
        if cell_type_proportions is not None:
            scanvi.set_y_prior(scanvi.class_proportions(pd.Series(cell_type_proportions), space="partition"))
            return
        reference = scanvi.reference_class_proportions(space="partition")
        if reference is None:
            warnings.warn(
                "The scANVI model does not know how many labeled cells of each cell type it was trained on (it has no "
                "class_counts), so p(c) of the query cells stays the y_prior of the model. Give cell_type_proportions "
                "to set it.",
                UserWarning,
            )
            return
        scanvi.set_y_prior(reference)

    def _is_trained_layer(self, name: str) -> bool:
        """Whether the new-batch parameter of the layer ``name`` (in ``self.scvi``) is trained."""
        return self.decoder_surgery == "all" or not name.startswith("decoder.fully_connected.")

    def _surgery_linear_layers(self) -> None:
        """Replace LinearWithBatch/LinearWithBatchAndCovariates modules in self.scvi."""
        for name, module in list(self.scvi.named_modules()):
            replacement: nn.Module | None = None
            trainable = self._is_trained_layer(name)
            if isinstance(module, LinearWithBatchAndCovariates) and not isinstance(
                module, ScArchesLinearWithBatchAndCovariates
            ):
                replacement = ScArchesLinearWithBatchAndCovariates.from_existing(  # type: ignore[assignment]
                    module, self.n_batch_ref, self.n_new_batch, trainable=trainable
                )
            elif isinstance(module, LinearWithBatch) and not isinstance(module, ScArchesLinearWithBatch):
                replacement = ScArchesLinearWithBatch.from_existing(  # type: ignore[assignment]
                    module, self.n_batch_ref, self.n_new_batch, trainable=trainable
                )

            if replacement is not None:
                parent_name, _, child_name = name.rpartition(".")
                parent = self.scvi if not parent_name else self.scvi.get_submodule(parent_name)
                setattr(parent, child_name, replacement)

    def _surgery_final_additive_bias(self) -> None:
        """Replace decoder.final_additive_bias_layer with a surgery-aware wrapper."""
        decoder = self.scvi.decoder
        if decoder.final_additive_bias_layer is None:
            return
        decoder.final_additive_bias_layer = ScArchesFinalAdditiveBiasLayer(  # type: ignore[assignment]
            original=decoder.final_additive_bias_layer,
            n_batch_ref=self.n_batch_ref,
            n_new_batch=self.n_new_batch,
        )

    def _surgery_batch_embeddings(self) -> None:
        """
        Add new embedding rows as separate trainable parameters.

        The pretrained embedding tables stay as they are (frozen) and are found by
        :meth:`~cellarium.ml.models.SingleCellVariationalInference._full_batch_representation_table` together with
        the new rows. Keeping the old and new rows in different parameters means that the old rows are frozen by
        ``requires_grad=False`` alone, which every optimizer (including weight decay) respects and which survives
        copying and checkpointing.
        """
        scvi = self.scvi
        assert scvi.batch_representation_mean_bd is not None
        assert scvi.batch_representation_std_unconstrained_bd is not None
        n_latent_batch: int = scvi.batch_representation_mean_bd.shape[1]
        # Explicit device= so new rows are created on the same device as the pretrained
        # embeddings, not on whatever torch.device("meta") context may be active.
        device = scvi.batch_representation_mean_bd.device
        scvi.batch_representation_mean_new_bd = nn.Parameter(
            torch.zeros(self.n_new_batch, n_latent_batch, device=device)
        )
        scvi.batch_representation_std_unconstrained_new_bd = nn.Parameter(
            torch.zeros(self.n_new_batch, n_latent_batch, device=device)
        )
        self._reset_new_embedding_rows()

    def _reset_new_embedding_rows(self) -> None:
        """Start every new batch at the average reference batch: the mean of the rows of the reference batches."""
        scvi = self.scvi
        for name in ("batch_representation_mean", "batch_representation_std_unconstrained"):
            reference_bd: torch.Tensor = getattr(scvi, f"{name}_bd")
            new_bd: nn.Parameter = getattr(scvi, f"{name}_new_bd")
            if new_bd.device.type == "meta" or reference_bd.device.type == "meta":
                continue  # nothing to compute from: the values come from a checkpoint
            with torch.no_grad():
                new_bd.copy_(reference_bd.mean(dim=0, keepdim=True).expand_as(new_bd))

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> Self:
        """
        Set the training mode, but keep all ``BatchNorm`` layers of the wrapped model in eval mode so that their
        running statistics (and the statistics used for normalization) stay those of the pretrained model.
        Other layers, e.g. dropout, follow ``mode``.
        """
        super().train(mode)
        for module in self.scvi.modules():
            if isinstance(module, _BatchNorm):
                module.eval()
        return self

    # ------------------------------------------------------------------
    # CellariumModel interface
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise only the newly added scArches parameters; pretrained weights are unchanged."""
        for module in self.scvi.modules():
            if isinstance(module, ScArchesLinearSurgeryMixin):
                module.reset_new_batch_bias()
        # Widen to nn.Module | None so mypy knows ScArchesFinalAdditiveBiasLayer is reachable.
        final_bias: nn.Module | None = self.scvi.decoder.final_additive_bias_layer
        if isinstance(final_bias, ScArchesFinalAdditiveBiasLayer):
            final_bias.reset_parameters()
        if self.scvi.batch_embedded:
            self._reset_new_embedding_rows()

    @property
    def var_names_g(self) -> np.ndarray:
        return self.scvi.var_names_g

    # ------------------------------------------------------------------
    # Forward / predict / validate — delegate to self.scvi
    # ------------------------------------------------------------------

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert query-local ``batch_index_n`` to the index space of the extended model (if enabled)."""
        if not self.shift_batch_index or "batch_index_n" not in kwargs:
            return kwargs
        batch_index_n = kwargs["batch_index_n"]
        if ((batch_index_n < 0) | (batch_index_n >= self.n_new_batch)).any():
            raise ValueError(
                f"With shift_batch_index=True, batch_index_n must be query-local, i.e. in [0, {self.n_new_batch}). "
                "Use shift_batch_index=False to pass indices of the extended model (original batches first)."
            )
        return {**kwargs, "batch_index_n": batch_index_n + self.n_batch_ref}

    def forward(self, **kwargs: Any) -> dict:
        return call_func_with_batch(self.scvi.forward, self._prepare_kwargs(kwargs))

    def predict(self, **kwargs: Any) -> Any:
        return call_func_with_batch(self.scvi.predict, self._prepare_kwargs(kwargs))

    def validate(self, **kwargs: Any) -> None:  # type: ignore[override]
        call_func_with_batch(self.scvi.validate, self._prepare_kwargs(kwargs))

    def on_validation_epoch_start(self, trainer: pl.Trainer) -> None:
        self.scvi.on_validation_epoch_start(trainer)

    def on_validation_epoch_end(self, lightning_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        self.scvi.on_validation_epoch_end(lightning_module, trainer)

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        self.scvi.on_train_batch_end(trainer)

    # Note: ``on_load_checkpoint`` is deliberately not delegated to ``self.scvi``. It would overwrite the training
    # position of the pretrained model with that of this (shorter) run, which is already accounted for by the offsets.
