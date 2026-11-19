# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from cellarium.ml.api.cellariumdata import CellariumData, _device_of, _unwrap_scarches
from cellarium.ml.api.tools.scvi import (
    _batch_categories,
    _model_kwargs,
    _select_genes,
    _train,
    _with_decoder_final_additive_bias,
)
from cellarium.ml.api.utils import release_memory_after, resolve_device
from cellarium.ml.models import SCANVI
from cellarium.ml.utilities.core import call_func_with_batch
from cellarium.ml.utilities.data import (
    AnnDataField,
    categories_to_codes,
    get_cl_descendant_tensor_from_owl,
    get_cl_names_from_owl,
)

# arguments of the model that scanvi() sets itself, from its own arguments and the labels
_RESERVED_KWARGS = {
    "classifier_type",
    "cell_type_categories",
    "descendant_tensor",
    "cl_names",
    "class_counts",
    "unlabeled_category",
}
_WARN_UNLABELED_FRACTION = 0.9  # warn if more than this fraction of the cells is unlabeled
_WARN_PARTITION_SIZE = 300  # warn if the unlabeled cells are marginalized over more classes (frontier nodes) than this


def _fill_unlabeled(labels: Any, unlabeled_category: str) -> np.ndarray:
    """The labels of some cells as an array of strings, with the missing ones (NaN) as ``unlabeled_category``."""
    values = pd.Series(labels).astype(object)
    return values.where(values.notna(), unlabeled_category).astype(str).to_numpy(dtype=object)


def _label_counts(cdata: CellariumData, labels_key: str, unlabeled_category: str) -> pd.Series:
    """
    The number of cells with each label in ``obs[labels_key]``, where cells with no label (NaN) count as
    ``unlabeled_category``. Reads only that one column of ``obs``.
    """
    column = cdata.obs[labels_key]
    counts = column.value_counts()  # without the missing values
    counts = counts[counts > 0]
    counts.index = counts.index.astype(str)
    counts = counts.astype(np.int64)
    n_missing = int(column.isna().sum())
    if n_missing:
        counts[unlabeled_category] = counts.get(unlabeled_category, 0) + n_missing
    return counts.sort_index()


def _ontology_inputs(ontology: str | tuple[torch.Tensor, list[str]]) -> tuple[torch.Tensor, list[str]]:
    """The ``(descendant_tensor, cl_names)`` of an ontology given as an OWL path or URL, or as that pair."""
    if isinstance(ontology, str):
        return get_cl_descendant_tensor_from_owl(ontology), get_cl_names_from_owl(ontology)
    if isinstance(ontology, tuple) and len(ontology) == 2:
        descendant_tensor, cl_names = ontology
        return torch.as_tensor(descendant_tensor), list(cl_names)
    raise TypeError("ontology must be the path or URL of an OWL file, or a tuple (descendant_tensor, cl_names).")


def _describe_labels(labels: dict[str, int], limit: int = 10) -> str:
    """``labels`` (with their numbers of cells) as text, for an error message."""
    shown = ", ".join(f"{label!r} ({n:,} cells)" for label, n in list(labels.items())[:limit])
    return shown + (f", and {len(labels) - limit} more" if len(labels) > limit else "")


def scanvi(
    cdata: CellariumData,
    batch_key: str,
    labels_key: str = "cell_type_ontology_term_id",
    n_latent: int = 10,
    ontology: str | tuple[torch.Tensor, list[str]] | None = None,
    decoder_final_additive_bias: bool = False,
    max_epochs: int = 200,
    val_size: float = 0.1,
    early_stopping_patience: int = 10,
    scanvi_kwargs: dict | None = None,
    optim_kwargs: dict | None = None,
    key_added: str = "scanvi",
    log_every_n_steps: int = 10,
    val_check_interval: int = 500,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
    unlabeled_category: str = "unknown",
) -> None:
    """
    Train an scANVI model on the data in the datamodule, using ``batch_key`` for batch correction and the cell type
    labels in ``labels_key`` for semi-supervised annotation. The labels may be missing for any number of cells.

    There are two kinds of classifier:

    * **flat** (``ontology=None``): the cell types are the values of ``labels_key`` (other than the unlabeled one),
      and the classifier chooses among them.
    * **ontology** (``ontology`` is given): the labels are Cell Ontology IDs (like ``"CL:0000540"``). The classifier
      is aware of the ontology, and the generative model is partitioned over a *frontier* of cell types that have
      enough cells. A label of a cell type that is more general than the frontier supervises the model through the
      frontier cell types below it, and one that is more specific is binned up to its frontier ancestor.

    In both, cells whose label is ``unlabeled_category`` (or missing) are unlabeled, and are never a class that the
    classifier predicts. Annotate cells with the trained model with :func:`scanvi_annotate`, and get an embedding of
    them with :func:`~cellarium.ml.api.tools.embed`.

    The model keeps the numbers of labeled cells of each cell type (its ``class_counts``), and its prior over the
    classes ``p(c)`` is their proportions (unless ``y_prior_probs`` is given in ``scanvi_kwargs``). These are also the
    default ``p(c)`` of the query cells of :func:`~cellarium.ml.api.tools.scarches`.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        batch_key: The ``obs`` column to use as the batch covariate for batch correction.
            Injected into ``cdata.datamodule.batch_keys`` as ``"batch_index_n"`` if not already present.
        labels_key: The ``obs`` column of the cell type labels (a missing value, or ``unlabeled_category``, means the
            cell is unlabeled). Injected into ``cdata.datamodule.batch_keys`` as ``"cell_type_labels_n"`` during
            training. Reading its values to count the cells of each label is not a pass over the counts.
        n_latent: Dimension of the scANVI latent space.
        ontology: ``None`` for a flat classifier. For an ontology-aware one, the path or URL of a Cell Ontology OWL
            file (which is parsed once, and needs ``owlready2``), or a tuple ``(descendant_tensor, cl_names)`` that
            has been prepared already, where ``descendant_tensor[i, j]`` is 1 if ``cl_names[j]`` is ``cl_names[i]``
            or one of its descendants (see :func:`~cellarium.ml.utilities.data.get_cl_descendant_tensor_from_owl`).
        decoder_final_additive_bias: If True, the decoder adds a per-gene, per-batch number of background counts to
            the mean counts of each cell (independent of its library size and state), like ambient RNA or the
            spillover of segmentation. It is in addition to the per-gene, per-batch scaling of expression of the final
            layer of the decoder, and :func:`~cellarium.ml.api.tools.scarches` trains both for new batches. It turns on
            ``final_additive_bias`` in the decoder of ``scanvi_kwargs`` if that has one, and otherwise in the default
            decoder. If False (default), the decoder is left as it is.
        max_epochs: Maximum number of training epochs (a fallback; training usually stops earlier
            via early stopping on the validation loss).
        val_size: Fraction of cells held out for validation (used for early stopping). Restored to
            its original value once training completes.
        early_stopping_patience: Number of validation checks (see ``val_check_interval``) with no improvement in
            validation loss before stopping early.
        scanvi_kwargs: Additional keyword arguments passed to :class:`~cellarium.ml.models.SCANVI`, overriding the
            defaults (e.g. ``classification_weight``, ``frontier_min_cells``, ``chunk_size``, ``encoder``,
            ``decoder``). The arguments that this function sets itself (``classifier_type``, ``cell_type_categories``,
            ``descendant_tensor``, ``cl_names``, ``class_counts`` and ``unlabeled_category``) cannot be among them.
        optim_kwargs: Additional keyword arguments passed to the optimizer (default: ``{"lr": 1e-3}``).
        key_added: The key under which the trained module and its history are stored in ``cdata.trained_modules``,
            replacing any module already there under that key.
        log_every_n_steps: How often (in training steps) per-step metrics such as ``train_loss`` are recorded in the
            training history. Metrics logged once per validation check, such as ``val_loss``, are recorded then.
        val_check_interval: Run validation every this many training steps, counted globally across epochs (so checks
            need not fall on epoch boundaries), or once per epoch if an epoch has fewer steps, whichever is more
            frequent. Each check evaluates all of the ``val_size`` held-out cells, so for very large datasets
            consider also lowering ``val_size``.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].
        unlabeled_category: The label that stands for an unlabeled cell, in addition to a missing value.

    Returns:
        None

    Raises:
        ValueError: If ``batch_key`` or ``labels_key`` is not an ``obs`` column, if no cell is labeled, or (with an
            ontology) if some label is not a cell type of the ontology.

    Note:
        A :class:`CellariumModule` instance containing the trained scANVI model is stored, along with the training
        history (``train_loss`` every ``log_every_n_steps`` steps, ``val_loss`` every validation check), as
        ``cdata.trained_modules[key_added]``. If training is interrupted (e.g. ``KeyboardInterrupt``), the partially
        trained module is stored the same way, marked ``complete=False``. Only the batch column is stored with it as
        the data that it needs to run, so it can embed and annotate cells that have no labels.

    Note:
        Before training, the numbers of labeled and unlabeled cells and the size of the partition that the unlabeled
        cells are marginalized over (the number of classes, or the size of the frontier) are printed, and a warning
        is given if most of the cells are unlabeled or the partition is large, as the cost of training grows with
        the unlabeled cells times the partition size. Training goes on regardless.

    Note:
        Temporarily mutates ``cdata.datamodule`` (injecting the ``"batch_index_n"`` and ``"cell_type_labels_n"``
        batch keys, restricting ``obs_columns_to_validate`` to their columns, and setting aside ``val_size``
        for validation) for the duration of training, and restores the original state afterward.
    """
    if not batch_key:
        raise ValueError("batch_key must be provided; scanvi() always performs batch correction.")
    reserved = sorted(_RESERVED_KWARGS & set(scanvi_kwargs or {}))
    if reserved:
        raise ValueError(f"scanvi_kwargs cannot contain {reserved}, which scanvi() sets from its own arguments.")
    if decoder_final_additive_bias:
        scanvi_kwargs = _with_decoder_final_additive_bias(scanvi_kwargs)
    columns = cdata.obs.columns
    for name, key in (("batch_key", batch_key), ("labels_key", labels_key)):
        if key not in columns:
            raise ValueError(f"{name} '{key}' is not a column of obs. The columns are: {list(columns)}.")

    counts = _label_counts(cdata, labels_key, unlabeled_category)
    n_unlabeled = int(counts.get(unlabeled_category, 0))
    n_labeled = int(counts.sum()) - n_unlabeled
    if n_labeled == 0:
        raise ValueError(f"No cell has a label in obs['{labels_key}'] (other than '{unlabeled_category}').")
    labeled_counts = counts.drop(unlabeled_category, errors="ignore")

    config: dict[str, Any] = {}
    if ontology is None:
        # the classes are the cell types in the data, and the unlabeled one is not among them
        classifier_kwargs: dict[str, Any] = {
            "cell_type_categories": sorted(labeled_counts.index),
            "class_counts": labeled_counts,
        }
    else:
        descendant_tensor, cl_names = _ontology_inputs(ontology)
        outside = labeled_counts[~labeled_counts.index.isin(cl_names)]
        if len(outside):
            raise ValueError(
                f"obs['{labels_key}'] has {len(outside)} label(s) that are not cell types of the ontology: "
                f"{_describe_labels(outside.to_dict())}."
            )
        classifier_kwargs = {
            "descendant_tensor": descendant_tensor,
            "cl_names": cl_names,
            "class_counts": labeled_counts,
            "classifier_type": "ontology",
        }
        if isinstance(ontology, str):
            config["ontology"] = ontology

    n_batch = cdata.datamodule.obs_key_nunique(batch_key)
    cpu_transforms, var_names_g = _select_genes(cdata)
    model = SCANVI(
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=n_latent,
        unlabeled_category=unlabeled_category,
        **classifier_kwargs,
        # p(c) is the proportions of the labeled cells, unless given
        **{"y_prior_probs": "class_counts", **_model_kwargs(scanvi_kwargs)},
    )

    # every label has to be one that the model can resolve, so that training does not fail on one of them midway
    if ontology is not None:
        unresolved = labeled_counts[~labeled_counts.index.isin(list(model._label_to_active_idx))]
        if len(unresolved):
            raise ValueError(
                f"The model cannot place these labels of obs['{labels_key}'] on its ontology: "
                f"{_describe_labels(unresolved.to_dict())}."
            )

    unlabeled_fraction = n_unlabeled / (n_labeled + n_unlabeled)
    if ontology is None:
        partition = f"{model.n_partition} cell types"
    else:
        partition = f"a frontier of {model.n_partition} cell types ({model.n_active} ontology nodes in all)"
    print(
        f"scANVI: {n_labeled:,} labeled and {n_unlabeled:,} unlabeled cells ({unlabeled_fraction:.1%} unlabeled), "
        f"with {partition}."
    )
    if unlabeled_fraction > _WARN_UNLABELED_FRACTION:
        warnings.warn(
            f"{unlabeled_fraction:.1%} of the cells are unlabeled, so the labels have little to supervise with.",
            UserWarning,
        )
    if n_unlabeled and model.n_partition > _WARN_PARTITION_SIZE:
        warnings.warn(
            f"The unlabeled cells are marginalized over {model.n_partition} cell types, which makes training slow "
            "and uses a lot of memory on the accelerator. Consider a smaller batch size, a smaller chunk_size, or "
            "(with an ontology) a larger frontier_min_cells in scanvi_kwargs.",
            UserWarning,
        )

    batch_keys = {"batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes)}
    label_field = AnnDataField(
        attr="obs",
        key=labels_key,
        convert_fn=functools.partial(_fill_unlabeled, unlabeled_category=unlabeled_category),
    )
    config.update(
        {
            "batch_key": batch_key,
            "batch_categories": _batch_categories(cdata, batch_key),
            "labels_key": labels_key,
            "n_latent": n_latent,
            "val_size": val_size,
            "decoder_final_additive_bias": model.decoder.final_additive_bias_layer is not None,
            "classifier_type": model.classifier_type,
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
        }
    )
    if ontology is not None:
        config["frontier_cl_names"] = list(model.frontier_cl_names)
    _train(
        cdata,
        model,
        cpu_transforms,
        # the labels are needed to train, but not to run the module, which then works on cells that have none
        train_batch_keys={**batch_keys, "cell_type_labels_n": label_field},
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
    )


def _node_depths(descendant_tensor: torch.Tensor) -> np.ndarray:
    """
    The length of the longest path from a root to each node of an ontology, from its ``descendant_tensor`` (with 1 at
    ``[i, j]`` if ``j`` is ``i`` or one of its descendants). A root is at depth 0.

    The longest chain of ancestors of a node is the longest path to it in the ontology, so the transitive closure is
    all that is needed. A proper ancestor of a node has fewer ancestors than it, so visiting the nodes by their
    number of ancestors visits the ancestors of each node first.
    """
    is_ancestor = np.asarray(descendant_tensor.cpu()) > 0  # [i, j]: i is an ancestor of j (or j itself)
    proper = is_ancestor.T.copy()  # [j, i]: i is a proper ancestor of j
    np.fill_diagonal(proper, False)
    depth = np.zeros(len(proper), dtype=np.int64)
    for j in np.argsort(proper.sum(axis=1), kind="stable"):
        if proper[j].any():
            depth[j] = depth[proper[j]].max() + 1
    return depth


def _select_nodes(
    probs_nc: torch.Tensor, depth_c: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each cell, the node of greatest depth among those whose probability is at least ``threshold`` (the one with
    the greatest probability among those of that depth).

    Returns:
        ``(index_n, confidence_n, assigned_n)``: the index of the node, its probability, and whether any node passed
        the threshold (for a cell where none did, the other two are meaningless and the confidence is 0).
    """
    passing = probs_nc >= threshold
    assigned = passing.any(dim=1)
    deepest = depth_c.unsqueeze(0).masked_fill(~passing, -1).max(dim=1, keepdim=True).values
    candidates = passing & (depth_c.unsqueeze(0) == deepest)
    index = probs_nc.masked_fill(~candidates, -1.0).argmax(dim=1)
    confidence = probs_nc.gather(1, index.unsqueeze(1)).squeeze(1) * assigned
    return index, confidence, assigned


@release_memory_after
@torch.no_grad()
def scanvi_annotate(
    cdata: CellariumData,
    module_key: str = "scanvi",
    key_added: str | None = None,
    threshold: float = 0.2,
    n_generative_samples: int = 8,
    classifier_target_proportions: pd.Series | dict[str, float] | Literal["reference"] | None = None,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
    batch_size: int | None = None,
) -> None:
    """
    Give each cell of ``cdata`` the cell type that a trained scANVI model predicts, in one pass over the cells, and
    store the labels, and the probability of each, in ``obs_computed``. There are two predictions of the cell type:
    that of the classifier ``q(c|z)``, and that of the generative model (see below).

    The classifier gives a probability to each cell type. With an ontology-aware classifier these are the
    probabilities that have been propagated up the ontology, so that the probability of a node includes that of
    all the nodes below it, and a cell may be confidently assigned to a general type while the specific type is
    uncertain. The label of a cell is the **deepest** node (the one with the longest path from the root of the
    ontology) whose probability is at least ``threshold``, and the one with the greatest probability among those at
    that depth. With a flat classifier, it is the cell type with the greatest probability, if that is at least
    ``threshold``. A cell for which no node passes the threshold is labeled with the unlabeled category of the model
    (``"unknown"``), with confidence 0.

    The threshold means something different for the two kinds of classifier. For an ontology-aware one it is the
    probability that the cell is of the node or any of its descendants, so a low value (such as the default) finds
    specific types, and a high one retreats to more general ones. For a flat one, it is the confidence in the one
    best cell type.

    **The generative prediction.** The class-conditional priors ``p(z|u,c)`` of the model also give a posterior over
    the cell types, by Bayes' rule with the prior ``p(c)`` of the model (see :meth:`~cellarium.ml.models.SCANVI.predict`
    and, for scArches, ``cell_type_proportions`` of :func:`~cellarium.ml.api.tools.scarches`). Its label is chosen by
    the same rule as that of the classifier. The two fail differently: the classifier confidently extends its decision
    regions to cells unlike any it was trained on, whereas no class prior explains such a cell well. How well the
    classes explain a cell is its *in-distribution score* (in nats, higher is better), which has no natural threshold:
    compare it with the scores of reference cells. Cells whose two labels differ, or with low scores, are worth a look.
    The generative prediction averages over ``n_generative_samples`` samples, each a pass over all the cell types, so
    it makes annotation slower; with ``n_generative_samples=0`` it is not made.

    **Classifier prior correction.** The classifier was trained with an implicit prior over the cell types: the
    proportions of the training labels, times the class-balancing weights in ontology mode. With
    ``classifier_target_proportions``, its probabilities are corrected to another prior (Saerens et al., 2002).
    ``"reference"`` is the proportions of the training labels, which changes nothing for a flat classifier and undoes
    the class balancing of an ontology-aware one (approximately). The model needs its ``class_counts`` for this.

    Any data the module needs to run (such as the batch column) is provided for you, see :meth:`CellariumData.using`.
    The module is the one in ``cdata.trained_modules[module_key]``, so for a view of cells (which starts with no
    trained modules) it is one that was trained on the view, or assigned to it. The cells need no labels.

    These are stored in ``obs_computed``, each a :class:`pandas.Series` indexed by :attr:`CellariumData.cell_ids`, for
    all the cells:

    * ``key_added``: the labels of the classifier (a categorical of the cell types that were assigned),
    * ``f"{key_added}_confidence"``: the probability of the label of the cell under the classifier,

    and with ``n_generative_samples > 0``:

    * ``f"{key_added}_generative"``: the labels of the generative model,
    * ``f"{key_added}_generative_confidence"``: the probability of that label under the generative model,
    * ``f"{key_added}_in_distribution_score"``: the in-distribution score of the cell.

    Args:
        cdata: The cells to annotate (a :class:`CellariumData` or a view). It is where the labels are stored.
        module_key: The key in ``cdata.trained_modules`` of the scANVI module (see :func:`scanvi`), or of a module
            that :func:`~cellarium.ml.api.tools.scarches` made from one.
        key_added: The key in ``obs_computed`` to store the labels under, replacing any entry there. By default
            ``"<module_key>_annotation"``.
        threshold: The least probability that a cell type must have to be the label, between 0 and 1.
        n_generative_samples: The number of samples to average over for the generative prediction, or 0 to not make
            it.
        classifier_target_proportions: The prior to correct the classifier to: ``"reference"``, or proportions (or
            numbers of cells) by cell type name (cell types not given get 0), or ``None`` (default) for no correction.
        accelerator: The accelerator for the pass, in ["cpu", "mps", "cuda", "auto"].
        batch_size: The number of cells per batch, by default that of the datamodule.

    Raises:
        TypeError: If the module is not an scANVI one.
        ValueError: If ``classifier_target_proportions`` has cell types that are not classes of the model, or is given
            for a model without ``class_counts``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")
    if n_generative_samples < 0:
        raise ValueError(f"n_generative_samples must be >= 0, got {n_generative_samples}.")
    key_added = f"{module_key}_annotation" if key_added is None else key_added
    device = resolve_device(accelerator)
    n = cdata.n_obs
    # the label (as a node index, -1 for none) and its probability, for the probabilities under each key of predict()
    keys = {"y_probs_nc": key_added}
    if n_generative_samples > 0:
        keys["generative_probs_nc"] = f"{key_added}_generative"
    node_index = {key: np.full(n, -1, dtype=np.int64) for key in keys}
    confidence = {key: np.zeros(n, dtype=np.float32) for key in keys}
    in_distribution_score = np.zeros(n, dtype=np.float32)

    with cdata.using(module_key) as trained:
        module = trained.module
        module.configure_model()  # idempotent; makes its transforms and model accessible
        model = module.model  # what runs (a ScArches passes the batches on to the scANVI model it wraps)
        scanvi_model = _unwrap_scarches(model)  # the scANVI model, which has the classes
        if not isinstance(scanvi_model, SCANVI):
            raise TypeError(f"The module '{module_key}' ({type(scanvi_model).__name__}) is not an scANVI module.")
        names = list(scanvi_model.active_cl_names)
        if scanvi_model.classifier_type == "ontology":
            depth = _node_depths(scanvi_model.active_descendant_tensor_cc)
        else:  # no node is more specific than another, so the label is the cell type of greatest probability
            depth = np.zeros(len(names), dtype=np.int64)
        depth_c = torch.as_tensor(depth, device=device)
        # the arguments of predict() that are not data
        predict_kwargs: dict[str, Any] = {"n_generative_samples": n_generative_samples}
        if classifier_target_proportions is not None:
            predict_kwargs["classifier_target_proportions_c"] = torch.as_tensor(
                _classifier_target_proportions(scanvi_model, classifier_target_proportions),
                dtype=torch.float32,
                device=device,
            )
        original_device = _device_of(model)
        model.to(device)
        try:
            for batch in cdata.predict_batches(
                module, device=device, batch_size=batch_size, desc=f"Annotating cells with '{module_key}'"
            ):
                prediction = call_func_with_batch(model.predict, {**batch, **predict_kwargs})
                positions = batch["row_indices_n"].numpy()
                for key in keys:
                    index, node_confidence, assigned = _select_nodes(prediction[key].float(), depth_c, threshold)
                    node_index[key][positions] = np.where(assigned.cpu().numpy(), index.cpu().numpy(), -1)
                    confidence[key][positions] = node_confidence.cpu().numpy()
                if n_generative_samples > 0:
                    in_distribution_score[positions] = prediction["in_distribution_score_n"].float().cpu().numpy()
        finally:
            if original_device is not None:
                model.to(original_device)

    unlabeled = scanvi_model.unlabeled_category
    categories = names if unlabeled in names else [*names, unlabeled]
    for key, label_key in keys.items():
        codes = np.where(node_index[key] >= 0, node_index[key], categories.index(unlabeled))
        labels = pd.Categorical.from_codes(codes, categories=categories).remove_unused_categories()
        cdata.obs_computed[label_key] = pd.Series(labels)
        cdata.obs_computed[f"{label_key}_confidence"] = pd.Series(confidence[key])
    if n_generative_samples > 0:
        cdata.obs_computed[f"{key_added}_in_distribution_score"] = pd.Series(in_distribution_score)

    for key, what in (("y_probs_nc", "the classifier"), ("generative_probs_nc", "the generative model")):
        if key not in keys:
            continue
        n_unassigned = int((node_index[key] < 0).sum())
        print(
            f"scanvi_annotate: by {what}, {n - n_unassigned:,} of {n:,} cells have a cell type with probability of at "
            f"least {threshold}; the other {n_unassigned:,} ({n_unassigned / max(n, 1):.1%}) are '{unlabeled}'."
        )
    if n_generative_samples > 0:
        n_differ = int((node_index["y_probs_nc"] != node_index["generative_probs_nc"]).sum())
        print(f"scanvi_annotate: the two labels differ for {n_differ:,} cells ({n_differ / max(n, 1):.1%}).")


def _classifier_target_proportions(
    model: SCANVI, proportions: pd.Series | dict[str, float] | Literal["reference"]
) -> np.ndarray:
    """The ``classifier_target_proportions`` of :func:`scanvi_annotate`, over the classifier outputs of ``model``."""
    if isinstance(proportions, str):
        if proportions != "reference":
            raise ValueError(
                f"classifier_target_proportions must be 'reference', proportions or None, not {proportions!r}."
            )
        reference = model.reference_class_proportions(space="classifier")
        if reference is None:
            raise ValueError(
                "classifier_target_proportions='reference' needs the numbers of labeled cells of each cell type that "
                "the model was trained on (class_counts), which this model does not have."
            )
        return reference
    return model.class_proportions(pd.Series(proportions), space="classifier")
