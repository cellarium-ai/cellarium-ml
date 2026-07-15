# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""scArches (single-cell Architectural Surgery) for reference mapping onto a trained scVI model."""

import logging

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.models.scvi import (
    LinearWithBatch,
    LinearWithBatchAndCovariates,
    SingleCellVariationalInference,
)

logger = logging.getLogger(__name__)


class ScArchesLinearSurgeryMixin:
    """
    Mixin for :class:`~cellarium.ml.models.scvi.LinearWithBatch` and
    :class:`~cellarium.ml.models.scvi.LinearWithBatchAndCovariates` that extends them to support
    scArches new-batch surgery.

    The original ``bias_decoder`` handles old batches (frozen). A new trainable
    ``new_batch_bias_weight`` parameter handles new batches via a weight-only linear projection.
    The two outputs are summed; because the input is a one-hot vector exactly one term is
    non-zero per cell.

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
    ) -> "ScArchesLinearSurgeryMixin":
        """
        Construct an instance by adopting the frozen parameters of *original* and adding a
        new trainable ``new_batch_bias_weight`` parameter for the new batches.

        The parent ``__init__`` is bypassed so that no new parameter tensors are allocated for
        the existing weights; the original's parameters are shared (not copied).

        New parameters are explicitly created on the same device as the pretrained weights,
        overriding any ``torch.device("meta")`` default-device context that may be active
        during CLI instantiation.
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
            torch.zeros(original.out_features, n_new_batch, device=pretrained_device)
        )
        return obj  # type: ignore[return-value]

    def compute_bias(
        self,
        batch_nb: torch.Tensor,
        categorical_covariate_np: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Split ``batch_nb`` at ``n_batch_ref`` and route each portion to the appropriate decoder.

        Old-batch cells (indices < n_batch_ref) have their one-hot set in the first slice and
        zeros in the second, so only the frozen ``bias_decoder`` produces a non-zero output.
        New-batch cells (indices >= n_batch_ref) have zeros in the first slice and their one-hot
        in the second, so only the trainable ``new_batch_bias_weight`` produces a non-zero
        output (because ``bias_decoder(zeros) = 0`` when no bias is present in its final layer).
        """
        old_nb = batch_nb[..., : self.n_batch_ref]
        new_nb = batch_nb[..., self.n_batch_ref :]
        # super() dispatches to LinearWithBatch or LinearWithBatchAndCovariates
        old_bias = super().compute_bias(  # type: ignore[misc]
            batch_nb=old_nb, categorical_covariate_np=categorical_covariate_np
        )
        new_bias = F.linear(new_nb, self.new_batch_bias_weight)
        return old_bias + new_bias


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

    The frozen original handles old batches; a new trainable ``new_bias_weight`` parameter
    (with a ReLU activation to mirror the original) handles new batches.  Both receive the same
    categorical covariate portion of the input (if present).

    New parameters are explicitly created on the same device as the pretrained weights,
    overriding any ``torch.device("meta")`` default-device context.

    Args:
        original: the existing ``nn.Sequential(FullyConnectedLinear, ReLU)`` to freeze.
        n_batch_ref: number of original (reference) batches.
        n_new_batch: number of new (query) batches being added.
        categorical_dims: total one-hot dimension of categorical covariates concatenated after
            the batch one-hot in the input tensor (0 when no categorical covariates are used).
    """

    def __init__(
        self,
        original: nn.Sequential,
        n_batch_ref: int,
        n_new_batch: int,
        categorical_dims: int,
    ) -> None:
        super().__init__()
        self.frozen_layer = original
        self.n_batch_ref = n_batch_ref
        self.n_new_batch = n_new_batch
        out_features: int = original[0].out_features  # type: ignore[union-attr, assignment]
        pretrained_device = next(original.parameters()).device
        self.new_bias_weight = nn.Parameter(
            torch.zeros(out_features, n_new_batch + categorical_dims, device=pretrained_device)
        )

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

        if categorical.shape[-1] > 0:
            old_x = torch.cat([old_batch, categorical], dim=-1)
            new_x = torch.cat([new_batch, categorical], dim=-1)
        else:
            old_x = old_batch
            new_x = new_batch

        return self.frozen_layer(old_x) + F.relu(F.linear(new_x, self.new_bias_weight))


def _make_freeze_old_rows_hook(n_new_batch: int):
    """Return a gradient hook that zeros rows 0..(n_batch_ref-1) of a 2-D parameter."""

    def hook(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[:-n_new_batch] = 0.0
        return g

    return hook


class ScArches(CellariumModel, PredictMixin, ValidateMixin):
    """
    scArches (single-cell Architectural Surgery) [1] adapter for a trained
    :class:`~cellarium.ml.models.SingleCellVariationalInference` model.

    Surgery is performed **in-place** on *pretrained_scvi* during ``__init__``:

    * All existing parameters are frozen (``requires_grad=False``).
    * ``pretrained_scvi.n_batch`` is extended by *n_new_batch*.
    * For models with ``batch_embedded=False`` (one-hot batch representation):

      - Each :class:`~cellarium.ml.models.scvi.LinearWithBatch` and
        :class:`~cellarium.ml.models.scvi.LinearWithBatchAndCovariates` is replaced in-place by
        :class:`ScArchesLinearWithBatch` / :class:`ScArchesLinearWithBatchAndCovariates`.
        Each replacement adopts the original's frozen parameters and adds a new trainable
        ``new_batch_bias_weight`` parameter (shape ``(out_features, n_new_batch)``,
        initialised to zero) for the new-batch bias.
      - If the decoder has a ``final_additive_bias_layer``, it is replaced by
        :class:`ScArchesFinalAdditiveBiasLayer` which routes old/new batch dimensions to a
        frozen and a trainable branch respectively.

    * For models with ``batch_embedded=True`` (learned batch embedding):

      - ``batch_representation_mean_bd`` and ``batch_representation_std_unconstrained_bd``
        are extended by *n_new_batch* new rows (mean initialised with small Gaussian noise,
        std initialised to zero).  Gradient hooks zero out gradients for the original rows so
        the pretrained embeddings remain unchanged.

    New batch indices must lie in ``[n_batch_ref, n_batch_ref + n_new_batch)``.  Old batch
    indices (0 to ``n_batch_ref - 1``) continue to work as before and can be mixed with new
    ones in the same forward pass.

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
    """

    def __init__(
        self,
        pretrained_scvi: SingleCellVariationalInference,
        n_new_batch: int,
    ) -> None:
        super().__init__()
        if pretrained_scvi.n_batch < 1:
            raise ValueError(
                f"pretrained_scvi has n_batch={pretrained_scvi.n_batch}. "
                "scArches requires a model trained with at least one batch."
            )
        if n_new_batch < 1:
            raise ValueError(f"n_new_batch must be >= 1, got {n_new_batch}.")

        self.scvi = pretrained_scvi
        self.n_batch_ref: int = pretrained_scvi.n_batch
        self.n_new_batch: int = n_new_batch

        # Freeze all existing parameters.
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

    # ------------------------------------------------------------------
    # Surgery helpers
    # ------------------------------------------------------------------

    def _surgery_linear_layers(self) -> None:
        """Replace LinearWithBatch/LinearWithBatchAndCovariates modules in self.scvi."""
        for name, module in list(self.scvi.named_modules()):
            replacement: nn.Module | None = None
            if isinstance(module, LinearWithBatchAndCovariates) and not isinstance(
                module, ScArchesLinearWithBatchAndCovariates
            ):
                replacement = ScArchesLinearWithBatchAndCovariates.from_existing(  # type: ignore[assignment]
                    module, self.n_batch_ref, self.n_new_batch
                )
            elif isinstance(module, LinearWithBatch) and not isinstance(module, ScArchesLinearWithBatch):
                replacement = ScArchesLinearWithBatch.from_existing(  # type: ignore[assignment]
                    module, self.n_batch_ref, self.n_new_batch
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
        categorical_dims = sum(self.scvi.n_cats_per_cov)
        decoder.final_additive_bias_layer = ScArchesFinalAdditiveBiasLayer(  # type: ignore[assignment]
            original=decoder.final_additive_bias_layer,
            n_batch_ref=self.n_batch_ref,
            n_new_batch=self.n_new_batch,
            categorical_dims=categorical_dims,
        )

    def _surgery_batch_embeddings(self) -> None:
        """Extend batch embedding tables and freeze old rows via gradient hooks."""
        scvi = self.scvi
        n_new = self.n_new_batch
        assert scvi.batch_representation_mean_bd is not None
        assert scvi.batch_representation_std_unconstrained_bd is not None

        n_latent_batch: int = scvi.batch_representation_mean_bd.shape[1]
        # Explicit device= so new rows are created on the same device as the pretrained
        # embeddings, not on whatever torch.device("meta") context may be active.
        device = scvi.batch_representation_mean_bd.device

        # Mean: extend with small-noise rows (like the identity-init used for old batches).
        old_mean = scvi.batch_representation_mean_bd.data
        new_mean_rows = torch.zeros(n_new, n_latent_batch, device=device)
        nn.init.normal_(new_mean_rows, mean=0.0, std=0.01)
        extended_mean = nn.Parameter(torch.cat([old_mean, new_mean_rows], dim=0))
        scvi.batch_representation_mean_bd = extended_mean
        extended_mean.register_hook(_make_freeze_old_rows_hook(n_new))

        # Std (unconstrained): extend with zeros (maps to std=1 after softplus/exp).
        old_std = scvi.batch_representation_std_unconstrained_bd.data
        new_std_rows = torch.zeros(n_new, n_latent_batch, device=device)
        extended_std = nn.Parameter(torch.cat([old_std, new_std_rows], dim=0))
        scvi.batch_representation_std_unconstrained_bd = extended_std
        extended_std.register_hook(_make_freeze_old_rows_hook(n_new))

    # ------------------------------------------------------------------
    # CellariumModel interface
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise only the newly added scArches parameters; pretrained weights are unchanged."""
        for module in self.scvi.modules():
            if isinstance(module, ScArchesLinearSurgeryMixin):
                nn.init.zeros_(module.new_batch_bias_weight)
        # Widen to nn.Module | None so mypy knows ScArchesFinalAdditiveBiasLayer is reachable.
        final_bias: nn.Module | None = self.scvi.decoder.final_additive_bias_layer
        if isinstance(final_bias, ScArchesFinalAdditiveBiasLayer):
            nn.init.zeros_(final_bias.new_bias_weight)
        if self.scvi.batch_embedded:
            n = self.n_new_batch
            if self.scvi.batch_representation_mean_bd is not None:
                nn.init.normal_(self.scvi.batch_representation_mean_bd.data[-n:], mean=0.0, std=0.01)
            if self.scvi.batch_representation_std_unconstrained_bd is not None:
                self.scvi.batch_representation_std_unconstrained_bd.data[-n:].zero_()

    @property
    def var_names_g(self) -> np.ndarray:
        return self.scvi.var_names_g

    # ------------------------------------------------------------------
    # Forward / predict / validate — delegate entirely to self.scvi
    # ------------------------------------------------------------------

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
        total_mrna_umis_n: torch.Tensor | None = None,
    ) -> dict:
        return self.scvi(
            x_ng=x_ng,
            var_names_g=var_names_g,
            batch_index_n=batch_index_n,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_index_nd=categorical_covariate_index_nd,
            total_mrna_umis_n=total_mrna_umis_n,
        )

    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
    ) -> dict:
        return self.scvi.predict(
            x_ng=x_ng,
            var_names_g=var_names_g,
            batch_index_n=batch_index_n,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_index_nd=categorical_covariate_index_nd,
        )

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
        total_mrna_umis_n: torch.Tensor | None = None,
        validation_cell_type_index_n: torch.Tensor | None = None,
    ) -> None:
        self.scvi.validate(
            trainer=trainer,
            pl_module=pl_module,
            batch_idx=batch_idx,
            x_ng=x_ng,
            var_names_g=var_names_g,
            batch_index_n=batch_index_n,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_index_nd=categorical_covariate_index_nd,
            total_mrna_umis_n=total_mrna_umis_n,
            validation_cell_type_index_n=validation_cell_type_index_n,
        )

    def on_validation_epoch_start(self, trainer: pl.Trainer) -> None:
        self.scvi.on_validation_epoch_start(trainer)

    def on_validation_epoch_end(self, lightning_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        self.scvi.on_validation_epoch_end(lightning_module, trainer)

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        self.scvi.on_train_batch_end(trainer)
