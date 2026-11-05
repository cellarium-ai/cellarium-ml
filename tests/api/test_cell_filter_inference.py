# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest

from cellarium.ml import CellariumModule
from cellarium.ml.api.cellariumdata import CellariumData, fit_and_register
from cellarium.ml.api.tools import embed, geometric_sketch
from cellarium.ml.models import IncrementalPCA
from cellarium.ml.transforms import CellFilter, Densify, Log1p, NormalizeTotal

# A module trained with a `CellFilter`, which drops cells in training, is used for inference: every cell of the data
# is embedded (and is in play for the sketch), whether the `CellFilter` is in the `transforms` or in the
# `cpu_transforms` (which the dataloader runs).
# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15


def _counts(cdata: CellariumData) -> np.ndarray:
    return np.asarray(cdata.datamodule.dadc[np.arange(cdata.n_obs)].X.todense()).sum(axis=1)


def _filtered_module(cdata: CellariumData, filter_on_cpu: bool) -> CellariumModule:
    counts = _counts(cdata)
    threshold = int(np.median(counts))
    assert 0 < (counts < threshold).sum() < len(counts)  # the filter drops some of the cells, and not all
    if filter_on_cpu:
        cpu_transforms = [Densify(), CellFilter(min_count_per_cell=threshold)]
        transforms = [NormalizeTotal(), Log1p()]
    else:
        cpu_transforms = []
        transforms = [Densify(), CellFilter(min_count_per_cell=threshold), NormalizeTotal(), Log1p()]
    return CellariumModule(
        cpu_transforms=cpu_transforms,
        transforms=transforms,
        model=IncrementalPCA(var_names_g=cdata.datamodule.var_names_g, n_components=3, perform_mean_correction=True),
    )


@pytest.fixture(params=[False, True], ids=["transforms", "cpu_transforms"])
def filter_on_cpu(request) -> bool:
    return request.param


@pytest.fixture
def trained_filtered_pca(cdata, filter_on_cpu):
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    fit_and_register(cdata, trainer, _filtered_module(cdata, filter_on_cpu), "pca")


def test_predict_batches_with_a_cell_filter_yield_a_row_for_every_cell(cdata, filter_on_cpu):
    module = _filtered_module(cdata, filter_on_cpu)

    positions = []
    for batch in cdata.predict_batches(module, batch_size=7):
        assert batch["x_ng"].shape[0] == len(batch["row_indices_n"])
        positions.append(batch["row_indices_n"].numpy())
    assert sorted(np.concatenate(positions)) == list(range(cdata.n_obs))


def test_predict_batches_with_workers_and_a_cell_filter_on_the_cpu(make_h5ad_files):
    cdata = CellariumData(
        h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30), datamodule_num_workers=2
    )
    module = _filtered_module(cdata, filter_on_cpu=True)

    n_cells = 0
    for batch in cdata.predict_batches(module, batch_size=7):
        assert batch["x_ng"].shape[0] == len(batch["row_indices_n"])
        n_cells += len(batch["row_indices_n"])
    assert n_cells == cdata.n_obs


def test_embed_embeds_the_cells_that_the_cell_filter_dropped_in_training(cdata, trained_filtered_pca):
    expected = cdata.to_anndata().obsm["X_pca"]  # computed in memory, with the pipeline's predict
    embed(cdata, "pca", accelerator="cpu")

    stored = cdata.obsm["X_pca"]
    assert stored.shape == (cdata.n_obs, 3)
    np.testing.assert_allclose(stored, expected, rtol=1e-4, atol=1e-4)


def test_the_sketch_stores_the_embedding_of_every_cell_and_has_every_cell_in_play(cdata, trained_filtered_pca):
    expected = cdata.to_anndata().obsm["X_pca"]
    # as many cells as there are: the sketch is all of them, including those that the filter drops in training
    result = geometric_sketch(cdata, target_n_cells=cdata.n_obs, embedding_module_key="pca", accelerator="cpu")

    in_sketch = result["obs_names_in_sketch"]
    assert isinstance(in_sketch, pd.Series)
    assert in_sketch.sum() == cdata.n_obs
    np.testing.assert_allclose(cdata.obsm["X_pca"], expected, rtol=1e-4, atol=1e-4)
