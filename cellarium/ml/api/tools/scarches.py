# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import warnings
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from cellarium.ml import CellariumModule
from cellarium.ml.api.cellariumdata import CellariumData, TrainedModule, _strip_ensembl_version
from cellarium.ml.api.tools.scanvi import _WARN_PARTITION_SIZE
from cellarium.ml.api.tools.scvi import _batch_categories, _train
from cellarium.ml.models import SCANVI, ScArches, SingleCellVariationalInference
from cellarium.ml.models.scvi import LinearWithBatch, LinearWithBatchAndCovariates
from cellarium.ml.transforms import Filter
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes

_WARN_MISSING_GENES_FRACTION = 0.1  # warn if more than this fraction of the genes of the reference is not in the data
_MAX_MISSING_GENES_FRACTION = 0.5  # and refuse if more than this fraction is not


def _single_batch_codes(obs_names: Any) -> np.ndarray:
    """The batch index 0 for every cell, for data that is all one batch (a module-level function, so it pickles)."""
    return np.zeros(len(obs_names), dtype=np.int32)


def _reference_module(cdata: CellariumData, reference: str | TrainedModule | CellariumModule) -> CellariumModule:
    """The configured module that ``reference`` names or is."""
    if isinstance(reference, str):
        if reference not in cdata.trained_modules:
            raise ValueError(
                f"No trained module '{reference}' in cdata.trained_modules (it has {list(cdata.trained_modules)}). "
                "Pass the module itself to use one trained on other data, e.g. reference=other.trained_modules['scvi']."
            )
        module = cdata.trained_modules[reference].module
    elif isinstance(reference, TrainedModule):
        module = reference.module
    elif isinstance(reference, CellariumModule):
        module = reference
    else:
        raise TypeError(
            "reference must be a key of cdata.trained_modules, a TrainedModule or a CellariumModule, "
            f"got {type(reference).__name__}."
        )
    module.configure_model()  # idempotent; makes its transforms and model accessible
    return module


def _encoder_takes_batch(model: SingleCellVariationalInference) -> bool:
    """Whether the batch of a cell enters its encoder, which is what makes its embedding depend on the batch."""
    return any(isinstance(m, (LinearWithBatch, LinearWithBatchAndCovariates)) for m in model.z_encoder.modules())


def _check_reference_model(model: nn.Module) -> SingleCellVariationalInference:
    """``model`` if scArches can map new batches onto it, and otherwise an error that says why it cannot."""
    if isinstance(model, ScArches):
        raise ValueError(
            "The reference module was itself made by scarches(). Map the cells onto the module that it was made from."
        )
    if not isinstance(model, SingleCellVariationalInference):
        raise TypeError(
            f"The reference must be an scVI or scANVI module (made by cml.tl.scvi or cml.tl.scanvi, or loaded from "
            f"a checkpoint of one), but its model is a {type(model).__name__}."
        )
    if not _encoder_takes_batch(model):
        raise ValueError(
            "The encoder of the reference model does not take the batch (it has no LinearWithBatch or "
            "LinearWithBatchAndCovariates layer), so neither its embedding of a cell nor (for scANVI) its cell type "
            "prediction depends on the batch: scArches would only train the decoder, and would change neither. The "
            "reference model can embed these cells as it is."
        )
    return model


def _query_transforms(reference: CellariumModule, var_names_g: np.ndarray) -> tuple[list[nn.Module], list[nn.Module]]:
    """
    The ``(cpu_transforms, transforms)`` of the reference module (copies), with every gene :class:`Filter` replaced by
    one to the genes of the model that takes the genes that the data does not have as zero counts. If the reference
    has no gene filter, one is put first, to put the genes of the data in the order of the model.
    """
    gene_filter = Filter(filter_list=list(var_names_g), ordering=True, allow_missing=True)

    def adapted(transforms: Iterable[nn.Module]) -> list[nn.Module]:
        return [copy.deepcopy(gene_filter if isinstance(t, Filter) else t) for t in transforms]

    cpu_transforms, transforms = adapted(reference.cpu_transforms), adapted(reference.transforms)
    if not any(isinstance(t, Filter) for t in [*cpu_transforms, *transforms]):
        cpu_transforms.insert(0, gene_filter)
    return cpu_transforms, transforms


def _ensembl_version_hint(data_var_names_g: np.ndarray, var_names_g: np.ndarray, n_missing: int) -> str:
    """
    A hint to pass ``var_names_strip_ensembl_version=True`` to :class:`CellariumData`, if the data's gene names
    ``data_var_names_g`` with the Ensembl versions stripped would have more of the genes ``var_names_g`` of the
    reference model than the ``n_missing`` that they lack; otherwise ``""``.
    """
    stripped = _strip_ensembl_version(data_var_names_g)
    n_missing_stripped = int((~np.isin(var_names_g, stripped)).sum())
    if n_missing_stripped >= n_missing:
        return ""
    example = data_var_names_g[stripped != data_var_names_g.astype(str)][0]  # one is stripped, as more are found
    return (
        f" The data's gene names have Ensembl versions (e.g. {str(example)!r}) and the reference model's "
        f"do not: with the versions stripped, {n_missing - n_missing_stripped:,} more of its genes would be found. "
        "Pass var_names_strip_ensembl_version=True to CellariumData to strip them."
    )


def _count_missing_genes(data_var_names_g: np.ndarray, var_names_g: np.ndarray) -> int:
    """
    The number of the genes ``var_names_g`` of the reference model that are not among the genes ``data_var_names_g``
    of the data, which are taken as zero counts. A warning if they are many, and an error if they are most of them,
    each with a hint to strip the Ensembl versions of the data's gene names if that would find more of them.
    """
    n_genes = len(var_names_g)
    n_missing = int((~np.isin(var_names_g, data_var_names_g)).sum())
    fraction = n_missing / n_genes
    if fraction > _MAX_MISSING_GENES_FRACTION:
        raise ValueError(
            f"{n_missing:,} of the {n_genes:,} genes of the reference model ({fraction:.1%}) are not in the data, more "
            f"than the {_MAX_MISSING_GENES_FRACTION:.0%} that can be taken as zero counts. Are the genes named the "
            "same way (e.g. Ensembl IDs in both)?" + _ensembl_version_hint(data_var_names_g, var_names_g, n_missing)
        )
    if fraction > _WARN_MISSING_GENES_FRACTION:
        warnings.warn(
            f"{n_missing:,} of the {n_genes:,} genes of the reference model ({fraction:.1%}) are not in the data, and "
            "are taken as zero counts, which the reference model takes as no expression."
            + _ensembl_version_hint(data_var_names_g, var_names_g, n_missing),
            UserWarning,
        )
    return n_missing


def scarches(
    cdata: CellariumData,
    reference: str | TrainedModule | CellariumModule,
    batch_key: str | None = None,
    max_epochs: int = 50,
    val_size: float = 0.1,
    early_stopping_patience: int = 3,
    optim_kwargs: dict | None = None,
    key_added: str | None = None,
    log_every_n_steps: int = 10,
    val_check_interval: int = 50,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> None:
    """
    Map the cells of ``cdata`` onto a trained scVI or scANVI model (the *reference*, which may have been trained on
    other data) with scArches [1]: the batches of ``cdata`` are added to the model as new batches, and only the
    parameters of the new batches are trained, on ``cdata``. Everything that the reference model learned stays as it
    is, so the cells are embedded (and, with scANVI, annotated) in the latent space of the reference.

    The result is a new module, stored as ``cdata.trained_modules[key_added]``, which the other tools use as any other,
    for example ``cml.tl.embed(cdata, key_added)`` and ``cml.tl.scanvi_annotate(cdata, key_added)``. The reference
    module is not changed.

    The batches of ``cdata`` are all new batches, even if some of them have the same names as batches of the
    reference. The genes are those of the reference model: genes of the data that the model does not have are left
    out, and genes of the model that the data does not have are taken as zero counts (with a warning if they are more
    than 10% of them, and an error if more than half).

    Example:
        >>> cml.tl.scarches(query, reference=reference_cdata.trained_modules["scanvi"], batch_key="donor_id")
        >>> cml.tl.embed(query, "scarches")
        >>> cml.tl.scanvi_annotate(query, "scarches")

    **References:**

    1. `Mapping single-cell data to reference atlases by transfer learning (Lotfollahi et al.)
       <https://www.nature.com/articles/s41587-021-01001-7>`_.

    Args:
        cdata: The cells to map (a :class:`CellariumData` or a view). The new module is stored in it.
        reference: The trained scVI or scANVI module to map onto: the key of a module in ``cdata.trained_modules``, or
            the module itself (a :class:`~cellarium.ml.api.cellariumdata.TrainedModule`, such as
            ``other.trained_modules["scanvi"]`` of the data it was trained on, or a
            :class:`~cellarium.ml.core.CellariumModule`). The batch of a cell must enter its encoder, as it does in
            the modules of :func:`~cellarium.ml.api.tools.scvi` and :func:`~cellarium.ml.api.tools.scanvi`, since
            otherwise the embedding does not depend on the batch and there is nothing for scArches to do.
        batch_key: The (categorical) ``obs`` column of the batches of ``cdata``, each of which is a new batch, or
            ``None`` if all the cells are of one batch.
        max_epochs: The most epochs of training (a fallback: training usually stops earlier, by early stopping on the
            validation loss).
        val_size: The fraction of the cells held out for validation, for early stopping.
        early_stopping_patience: The number of validation checks (see ``val_check_interval``) with no improvement in
            the validation loss before stopping. When training stops, the parameters of the new batches are put back
            to those of the best validation, so stopping a few checks late loses nothing.
        optim_kwargs: Keyword arguments of the optimizer (Adam, with no weight decay unless given here), by default
            ``{"lr": 1e-2}``. This is ten times the default of :func:`~cellarium.ml.api.tools.scvi`: only the few
            parameters of the new batches are trained, and Adam moves each parameter by about the learning rate per
            step, so a small one would take many steps to move the batch biases as far as they need to go.
        key_added: The key in ``cdata.trained_modules`` to store the new module under, replacing any module there. By
            default ``"<reference>_scarches"`` when ``reference`` is a key, and ``"scarches"`` otherwise.
        log_every_n_steps: How often (in training steps) per-step metrics such as ``train_loss`` are recorded in the
            training history.
        val_check_interval: Validate every this many training steps, or once per epoch if an epoch has fewer. Training
            converges in few steps since few parameters are trained, so the checks are frequent (with
            ``early_stopping_patience``, training stops 150 steps after the last improvement), and each one evaluates
            all of the ``val_size`` held-out cells: for a large dataset, consider a smaller ``val_size``.
        accelerator: The accelerator to train on, in ["cpu", "mps", "cuda", "auto"].

    Raises:
        ValueError: If ``batch_key`` is not an ``obs`` column, if the reference model is not one that scArches can map
            onto (see ``reference``), or if most of its genes are not in the data.
        TypeError: If the reference is not an scVI or scANVI module.

    Note:
        The KL warmup of the reference continues where its training left off, so a reference that finished its
        warmup is fine-tuned with the full KL weight. As for :func:`~cellarium.ml.api.tools.scvi`, a module whose
        training is interrupted is stored marked ``complete=False``, and ``cdata.datamodule`` is changed only while
        training.
    """
    if optim_kwargs is None:
        optim_kwargs = {"lr": 1e-2}
    if batch_key is not None and batch_key not in cdata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' is not a column of obs. The columns are: {list(cdata.obs.columns)}.")
    if key_added is None:
        key_added = f"{reference}_scarches" if isinstance(reference, str) else "scarches"

    reference_module = _reference_module(cdata, reference)
    reference_model = _check_reference_model(reference_module.model)
    n_missing_genes = _count_missing_genes(np.asarray(cdata.datamodule.var_names_g), reference_model.var_names_g)
    cpu_transforms, transforms = _query_transforms(reference_module, reference_model.var_names_g)

    if batch_key is None:
        batch_field = AnnDataField(attr="obs_names", convert_fn=_single_batch_codes)
        batch_categories: list[str] = []
        n_new_batch = 1
    else:
        batch_field = AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes)
        batch_categories = _batch_categories(cdata, batch_key)
        n_new_batch = len(batch_categories)
    batch_keys = {"batch_index_n": batch_field}

    # the surgery is done on a copy: the reference module stays as it is
    model = ScArches(copy.deepcopy(reference_model), n_new_batch=n_new_batch)
    print(
        f"scArches: mapping {n_new_batch:,} new batch{'es' if n_new_batch != 1 else ''} onto a "
        f"{type(reference_model).__name__} model of {model.n_batch_ref:,} batches; {n_missing_genes:,} of its "
        f"{len(reference_model.var_names_g):,} genes are not in the data."
    )
    if isinstance(reference_model, SCANVI) and reference_model.n_partition > _WARN_PARTITION_SIZE:
        warnings.warn(
            f"The cells have no labels, so each is marginalized over the {reference_model.n_partition} cell types of "
            "the scANVI model, which makes training slow and uses a lot of memory on the accelerator. Consider a "
            "smaller batch size.",
            UserWarning,
        )

    config: dict[str, Any] = {
        "reference": reference if isinstance(reference, str) else type(reference_model).__name__,
        "batch_key": batch_key,
        "n_new_batch": n_new_batch,
        "n_reference_batch": model.n_batch_ref,
        "n_missing_genes": n_missing_genes,
        "val_size": val_size,
    }
    if batch_categories:
        config["batch_categories"] = batch_categories
    _train(
        cdata,
        model,
        cpu_transforms,
        transforms=transforms,
        train_batch_keys=batch_keys,
        recorded_batch_keys=batch_keys,
        key_added=key_added,
        config=config,
        max_epochs=max_epochs,
        val_size=val_size,
        early_stopping_patience=early_stopping_patience,
        optim_kwargs=optim_kwargs,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        accelerator=accelerator,
        optim_fn=torch.optim.Adam,
        restore_best_weights=True,
    )
