# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Literal

import lightning.pytorch as pl

from cellarium.ml import CellariumModule
from cellarium.ml.api._data_transforms import OnePassRecipe, fit_or_reuse
from cellarium.ml.api.cellariumdata import CellariumData, fit_and_register
from cellarium.ml.models import IncrementalPCA, OnePassMeanVarStd
from cellarium.ml.transforms import Filter, ZScore


def pca(
    cdata: CellariumData,
    n_components: int = 50,
    zscore: bool = True,
    key_added: str = "pca",
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> CellariumModule:
    """
    Train a PCA model on the data in the datamodule.

    The counts are normalized by the total count of each cell over all genes, log1p transformed, restricted to the
    highly variable genes (:attr:`CellariumData.hvg`), and then z-scored if ``zscore`` is True, as in scanpy.

    Args:
        cdata: :class:`CellariumData` instance containing the data. If ``cdata.hvg`` is not set (see
            :func:`~cellarium.ml.api.preprocessing.highly_variable_genes`), a warning is issued and all genes are used.
        n_components: The number of principal components to compute.
        zscore: Whether to z-score the genes. Their means and standard deviations are computed in a pass over the
            data, unless an earlier call has already computed them for the same preprocessing.
        key_added: The key under which the trained module is stored in ``cdata.trained_modules``,
            replacing any module already there under that key.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].

    Returns:
        A :class:`CellariumModule` instance containing the trained PCA model. It is also stored as
        ``cdata.trained_modules[key_added]`` (with an empty history, since PCA logs no metrics). If training is
        interrupted, the partially trained module is stored the same way, marked ``complete=False``.
    """
    # the same preprocessing defines the statistics for z-scoring and the transforms in front of the PCA
    recipe = OnePassRecipe(log1p=True)
    cpu_transforms = recipe.cpu_transforms()
    transforms = recipe.transforms()

    if cdata.hvg is None:
        warnings.warn("cdata.hvg is not set, so PCA uses all genes. See highly_variable_genes().", stacklevel=2)
        var_names_g = cdata.datamodule.var_names_g
    else:
        filter = Filter(filter_list=cdata.hvg.index[cdata.hvg].tolist(), ordering=True)
        cpu_transforms.append(filter)
        var_names_g = filter.filter_list

    config: dict[str, object] = {"n_components": n_components, "zscore": zscore}
    if zscore:
        onepass_model = fit_or_reuse(cdata, recipe, accelerator).module.model
        assert isinstance(onepass_model, OnePassMeanVarStd)
        # the statistics are those of all genes: ZScore picks the genes it is given
        transforms.append(
            ZScore(
                var_names_g=onepass_model.var_names_g,
                mean_g=onepass_model.mean_g,
                std_g=onepass_model.std_g,
                eps=1e-4,
            )
        )
        config["zscore_statistics"] = recipe.fingerprint()

    module = CellariumModule(
        cpu_transforms=cpu_transforms,
        transforms=transforms,
        model=IncrementalPCA(
            var_names_g=var_names_g,
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
    fit_and_register(cdata, trainer, module, key_added, config=config)

    return module
