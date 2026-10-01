# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import lightning.pytorch as pl

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.models import IncrementalPCA, OnePassMeanVarStd
from cellarium.ml.transforms import Densify, Filter, Log1p, NormalizeTotal, ZScore


def pca(
    cdata: CellariumData,
    n_components: int = 50,
    zscore: bool = True,
    onepass_module: CellariumModule | None = None,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> CellariumModule:
    """
    Train a PCA model on the data in the datamodule.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        n_components: The number of principal components to compute.
        zscore: Whether to apply z-score normalization using a trained onepass_module.
        onepass_module: A trained :class:`CellariumModule` used for z-score normalization.
            Required if ``zscore`` is True.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].

    Returns:
        A :class:`CellariumModule` instance containing the trained PCA model.
    """
    datamodule: CellariumAnnDataDataModule = cdata.datamodule
    if cdata.hvg is None:
        raise ValueError("cdata.hvg must be set before running PCA. Try running highly_variable_genes() first.")

    if zscore:
        if onepass_module is None:
            raise ValueError("trained onepass_module must be provided if zscore is True")

    assert isinstance(onepass_module, CellariumModule)
    onepass_model = onepass_module.model
    assert isinstance(onepass_model, OnePassMeanVarStd)

    filter = Filter(filter_list=cdata.hvg.index[cdata.hvg].tolist(), ordering=True)

    module = CellariumModule(
        cpu_transforms=[filter],
        transforms=[
            Densify(),
            NormalizeTotal(),
            Log1p(),
        ]
        + (
            []
            if not zscore
            else [
                ZScore(
                    var_names_g=onepass_model.var_names_g,
                    mean_g=onepass_model.mean_g,
                    std_g=onepass_model.std_g,
                    eps=1e-4,
                )
            ]
        ),
        model=IncrementalPCA(
            var_names_g=filter.filter_list,
            n_components=n_components,
            perform_mean_correction=not zscore,
        ),
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices="auto",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule)

    return module
