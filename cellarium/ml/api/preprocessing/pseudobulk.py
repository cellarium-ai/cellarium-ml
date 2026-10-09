# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import anndata
import lightning.pytorch as pl
import pandas as pd

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._datamodule_context import temporary_batch_keys
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.models import OnePassMeanVarStd
from cellarium.ml.transforms import Densify, Log1p, NormalizeTotal, ZScore
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes


def _run_onepass(
    datamodule: CellariumAnnDataDataModule,
    n_batch: int,
    algorithm: Literal["naive", "shifted_data"],
    accelerator: Literal["cpu", "mps", "cuda", "auto"],
    extra_transforms: list,
) -> OnePassMeanVarStd:
    module = CellariumModule(
        transforms=[Densify(), *extra_transforms],
        model=OnePassMeanVarStd(
            var_names_g=datamodule.var_names_g,
            algorithm=algorithm,
            n_batch=n_batch,
            output_path=None,
        ),
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule)
    assert isinstance(module.model, OnePassMeanVarStd)
    return module.model


def pseudobulk(
    cdata: CellariumData,
    batch_key: str,
    normalize_total: bool = False,
    log1p: bool = False,
    zscore: bool = False,
    algorithm: Literal["naive", "shifted_data"] = "shifted_data",
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> anndata.AnnData:
    """
    Compute per-group pseudobulk mean and standard deviation using :class:`OnePassMeanVarStd`,
    grouping cells by ``batch_key``.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        batch_key: The ``obs`` column defining the pseudobulk groups. Injected into
            ``cdata.datamodule.batch_keys`` as ``"batch_index_n"`` if not already present.
        normalize_total: Whether to apply :class:`NormalizeTotal` before computing statistics.
        log1p: Whether to apply :class:`Log1p` (after ``normalize_total``, if both are set) before
            computing statistics.
        zscore: Whether to z-score genes (using overall, ungrouped mean/std computed in a
            preliminary pass, after ``normalize_total``/``log1p``) before computing per-group statistics.
        algorithm: ``"naive"`` or ``"shifted_data"`` (numerically stable), passed to :class:`OnePassMeanVarStd`.
        accelerator: The accelerator to use for training, in ["cpu", "mps", "cuda", "auto"].

    Returns:
        An :class:`~anndata.AnnData` with one row per ``batch_key`` group:
            - ``X`` and ``layers["pseudobulk_mean"]``: per-group mean expression.
            - ``layers["pseudobulk_std"]``: per-group (population) standard deviation.
            - ``obs["n_cells"]``: number of cells contributing to each group.
        Groups with zero cells will have ``NaN`` mean/std.

    Note:
        Temporarily mutates ``cdata.datamodule`` (injecting a ``"batch_index_n"`` batch key and
        restricting ``obs_columns_to_validate`` to ``batch_key``) for the duration of the run, and
        restores the original state afterward.
    """
    if not batch_key:
        raise ValueError("batch_key must be provided; pseudobulk() always groups by a batch_key.")

    datamodule: CellariumAnnDataDataModule = cdata.datamodule
    dadc = datamodule.dadc

    preprocessing_transforms: list = []
    if normalize_total:
        preprocessing_transforms.append(NormalizeTotal())
    if log1p:
        preprocessing_transforms.append(Log1p())

    if zscore:
        onepass_overall = _run_onepass(
            datamodule,
            n_batch=1,
            algorithm=algorithm,
            accelerator=accelerator,
            extra_transforms=preprocessing_transforms,
        )
        preprocessing_transforms = [
            *preprocessing_transforms,
            ZScore(
                var_names_g=onepass_overall.var_names_g,
                mean_g=onepass_overall.mean_g,
                std_g=onepass_overall.std_g,
                eps=1e-4,
            ),
        ]

    n_batch = datamodule.obs_key_nunique(batch_key)

    with temporary_batch_keys(
        datamodule, {"batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes)}
    ):
        group_labels = dadc.obs_categories(batch_key)

        model = _run_onepass(
            datamodule,
            n_batch=n_batch,
            algorithm=algorithm,
            accelerator=accelerator,
            extra_transforms=preprocessing_transforms,
        )

    mean_bg = model.batch_mean_bg.detach().cpu().numpy()
    std_bg = model.batch_var_bg.sqrt().detach().cpu().numpy()
    n_cells_b = model.x_size_b.detach().cpu().numpy().astype(int)

    var = dadc.var
    ad_field = datamodule.batch_keys["var_names_g"]
    assert isinstance(ad_field, AnnDataField)
    var_col = ad_field.key
    var = var.set_index(var_col).copy() if var_col is not None else var.copy()

    obs = pd.DataFrame({"n_cells": n_cells_b}, index=pd.Index(group_labels.astype(str), name=batch_key))

    return anndata.AnnData(
        X=mean_bg.copy(),
        obs=obs,
        var=var,
        layers={
            "pseudobulk_mean": mean_bg.copy(),
            "pseudobulk_std": std_bg,
        },
    )
