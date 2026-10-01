# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext
from typing import Literal

import lightning.pytorch as pl
import pandas as pd

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._datamodule_context import temporary_batch_keys
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.models import HVGSeuratV3, OnePassMeanVarStd
from cellarium.ml.preprocessing import kotliar_compute_highly_variable_genes, seurat_compute_highly_variable_genes
from cellarium.ml.transforms import Densify, Log1p, NormalizeTotal
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes


def highly_variable_genes(
    cdata: CellariumData,
    n_top_genes: int = 4000,
    flavor: Literal["seurat_v3", "seurat", "kotliar"] = "seurat",
    batch_key: str | None = None,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> tuple[pd.DataFrame, CellariumModule]:
    """
    Compute highly variable genes using the specified flavor.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        n_top_genes: The number of top highly variable genes to select.
        flavor: The flavor of highly variable gene computation to use, in ["seurat_v3", "seurat", "kotliar"].
        batch_key: The ``obs`` column to use for batch-aware computation. If ``None``, batch effects are
            ignored. Only supported when ``flavor="seurat_v3"``; raises ``ValueError`` if given together
            with any other flavor. Injected into ``cdata.datamodule.batch_keys`` as ``"batch_index_n"``
            for the duration of training.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].

    NOTE: sets the :attr:`hvg` property of the datamodule with the boolean mask of highly variable genes.

    Returns:
        A tuple containing:
            - A :class:`pandas.DataFrame` with the highly variable genes.
            - A :class:`CellariumModule` instance used for the computation, containing a trained model.

    Note:
        When ``batch_key`` is given, temporarily mutates ``cdata.datamodule`` (injecting a
        ``"batch_index_n"`` batch key and restricting ``obs_columns_to_validate`` to ``batch_key``)
        for the duration of training, and restores the original state afterward.
    """
    datamodule: CellariumAnnDataDataModule = cdata.datamodule

    if flavor not in ["seurat_v3", "seurat", "kotliar"]:
        raise ValueError("Unsupported flavor, choose from ['seurat_v3', 'seurat', 'kotliar']")

    if batch_key is not None and flavor != "seurat_v3":
        raise ValueError(
            f"batch_key is only supported for flavor='seurat_v3' (the only flavor that consumes batch "
            f"information); got flavor={flavor!r} with batch_key={batch_key!r}."
        )

    if flavor == "seurat_v3":
        n_batch = 1 if batch_key is None else datamodule.obs_key_nunique(batch_key)
        module = CellariumModule(
            transforms=[
                Densify(),
                NormalizeTotal(),
                Log1p(),
            ],
            model=HVGSeuratV3(
                var_names_g=datamodule.var_names_g,
                n_top_genes=n_top_genes,
                use_batch_key=batch_key is not None,
                n_batch=n_batch,
            ),
        )
    else:
        module = CellariumModule(
            transforms=[
                Densify(),
                NormalizeTotal(),
                Log1p(),
            ],
            model=OnePassMeanVarStd(
                var_names_g=datamodule.var_names_g,
                algorithm="shifted_data",
            ),
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    ctx = (
        temporary_batch_keys(
            datamodule, {"batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes)}
        )
        if batch_key is not None
        else nullcontext()
    )
    with ctx:
        trainer.fit(module, datamodule)

    if flavor == "seurat_v3":
        hvg_df = module.model._compute_hvg_df(n_top_genes=n_top_genes)
    elif flavor == "seurat":
        hvg_df = seurat_compute_highly_variable_genes(
            var_names_g=datamodule.var_names_g,
            mean_g=module.model.mean_g,
            var_g=module.model.var_g,
            n_top_genes=n_top_genes,
        )
    elif flavor == "kotliar":
        hvg_df = kotliar_compute_highly_variable_genes(
            var_names_g=datamodule.var_names_g,
            mean_g=module.model.mean_g,
            var_g=module.model.var_g,
            n_top_genes=n_top_genes,
        )

    # set the hvg property of CellariumData instance
    cdata.hvg = hvg_df["highly_variable"]

    return hvg_df, module
