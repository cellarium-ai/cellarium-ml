# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import pandas as pd

from cellarium.ml import CellariumModule
from cellarium.ml.api._data_transforms import OnePassRecipe, SeuratV3Recipe, fit_or_reuse
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.models import HVGSeuratV3, OnePassMeanVarStd
from cellarium.ml.preprocessing import kotliar_compute_highly_variable_genes, seurat_compute_highly_variable_genes


def highly_variable_genes(
    cdata: CellariumData,
    n_top_genes: int = 4000,
    flavor: Literal["seurat_v3", "seurat", "kotliar"] = "seurat",
    batch_key: str | None = None,
    key_added: str | None = None,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> tuple[pd.DataFrame, CellariumModule]:
    """
    Compute highly variable genes using the specified flavor.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        n_top_genes: The number of top highly variable genes to select.
        flavor: The flavor of highly variable gene computation to use, in ["seurat_v3", "seurat", "kotliar"].
        batch_key: The ``obs`` column to use for batch-aware computation. If ``None``, batch effects are
            ignored. Supported when ``flavor`` is ``"seurat_v3"`` or ``"seurat"``; raises ``ValueError`` if given
            together with ``"kotliar"``. Injected into ``cdata.datamodule.batch_keys`` as ``"batch_index_n"``
            for the duration of training.
        key_added: The key under which the trained module is stored in ``cdata.trained_modules``, replacing any
            module already there under that key. Defaults to the kind of model trained: ``"hvg_seurat_v3"`` for
            ``flavor="seurat_v3"`` and ``"onepass"`` (a :class:`OnePassMeanVarStd`) for ``"seurat"`` and
            ``"kotliar"``.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].

    NOTE: sets the :attr:`hvg` property of the datamodule with the boolean mask of highly variable genes.

    Returns:
        A tuple containing:
            - A :class:`pandas.DataFrame` with the highly variable genes.
            - A :class:`CellariumModule` instance used for the computation, containing a trained model. It is also
              stored as ``cdata.trained_modules[key_added]`` (with an empty history, since these models log no
              metrics). If training is interrupted, the partially trained module is stored the same way, marked
              ``complete=False``.

    Note:
        The trained models are the statistics of the data, not of the flavor or ``n_top_genes``: a model already
        trained for the same statistics (for example by an earlier call with another ``n_top_genes`` or flavor) is
        reused instead of reading the data again. The ``"seurat"`` and ``"kotliar"`` flavors share a model.

        When ``batch_key`` is given, temporarily mutates ``cdata.datamodule`` (injecting a
        ``"batch_index_n"`` batch key and restricting ``obs_columns_to_validate`` to ``batch_key``)
        for the duration of training, and restores the original state afterward.
    """
    datamodule = cdata.datamodule

    if flavor not in ["seurat_v3", "seurat", "kotliar"]:
        raise ValueError("Unsupported flavor, choose from ['seurat_v3', 'seurat', 'kotliar']")

    if batch_key is not None and flavor == "kotliar":
        raise ValueError(
            f"batch_key is only supported for flavor='seurat_v3' and flavor='seurat'; "
            f"got flavor={flavor!r} with batch_key={batch_key!r}."
        )

    if flavor == "seurat_v3":
        trained = fit_or_reuse(
            cdata,
            SeuratV3Recipe(batch_key=batch_key),
            accelerator,
            key_added="hvg_seurat_v3" if key_added is None else key_added,
            n_top_genes=n_top_genes,
        )
        model = trained.module.model
        assert isinstance(model, HVGSeuratV3)
        hvg_df = model._compute_hvg_df(n_top_genes=n_top_genes)
    else:
        trained = fit_or_reuse(
            cdata,
            OnePassRecipe(batch_key=batch_key),
            accelerator,
            key_added="onepass" if key_added is None else key_added,
        )
        model = trained.module.model
        assert isinstance(model, OnePassMeanVarStd)
        if flavor == "seurat":
            # go by the batch_key asked for: a module fit with batches also serves a call without
            if batch_key is None:
                hvg_df = seurat_compute_highly_variable_genes(
                    var_names_g=datamodule.var_names_g,
                    mean_g=model.mean_g,
                    var_g=model.var_g,
                    n_top_genes=n_top_genes,
                )
            else:
                hvg_df = seurat_compute_highly_variable_genes(
                    var_names_g=datamodule.var_names_g,
                    mean_g=model.mean_g,
                    var_g=model.var_g,
                    n_top_genes=n_top_genes,
                    batch_mean_bg=model.batch_mean_bg,
                    batch_var_bg=model.batch_var_bg,
                    batch_ids=[str(category) for category in datamodule.dadc.obs_categories(batch_key)],
                )
        else:
            hvg_df = kotliar_compute_highly_variable_genes(
                var_names_g=datamodule.var_names_g,
                mean_g=model.mean_g,
                var_g=model.var_g,
                n_top_genes=n_top_genes,
            )

    # set the hvg property of CellariumData instance
    cdata.hvg = hvg_df["highly_variable"]

    return hvg_df, trained.module
