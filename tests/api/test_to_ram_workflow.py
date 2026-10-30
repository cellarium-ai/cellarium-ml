# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from cellarium.ml.api._rebatch import ShardChunkDataModule, ShardChunkDataset
from cellarium.ml.api._view_collections import RamCollection, unwrap_collection
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import pca

# 40 cells in two h5ad files of 20 (as h5ad files, or as a deltacells dataset in tiles of 15), as in the `cdata` fixture
# (tests/api/conftest.py), but read by dataloader workers, as the data of a real analysis is

SCATTERED = np.arange(0, 40, 3)


@pytest.fixture(params=["h5ad", "deltacells"])
def cdata_with_workers(request, make_h5ad_files, tmp_path, deltacells_kwargs) -> CellariumData:
    h5ad_paths = make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30)
    if request.param == "h5ad":
        cdata = CellariumData(h5ad_paths=h5ad_paths, datamodule_num_workers=2)
    else:
        pytest.importorskip("deltacells._core")
        from cellarium.ml.api import create_deltacells_dataset

        uri = create_deltacells_dataset(h5ad_paths, str(tmp_path / "deltacells"), tile_size=15, level=3, log=None)
        cdata = CellariumData.from_deltacells(uri, deltacells_kwargs=deltacells_kwargs, datamodule_num_workers=2)
    assert cdata.datamodule.num_workers == 2 and cdata.datamodule.prefetch_factor == 2
    return cdata


@pytest.mark.parametrize("persistent_workers", [False, True])
def test_the_view_in_memory_has_loader_settings_that_fit_no_workers(cdata_with_workers, persistent_workers):
    cdata_with_workers.datamodule.persistent_workers = persistent_workers
    ram = cdata_with_workers[SCATTERED].to_ram()

    datamodule = ram.datamodule
    assert isinstance(unwrap_collection(datamodule.dadc), RamCollection)
    assert datamodule.num_workers == 0 and datamodule.prefetch_factor is None and not datamodule.persistent_workers
    assert not isinstance(datamodule.train_dataset, ShardChunkDataset)  # the cells are in memory: no shards to read
    assert next(iter(datamodule.train_dataloader()))["x_ng"].shape[0] > 0


def test_the_analysis_of_a_view_in_memory_runs_on_data_that_has_workers(cdata_with_workers):
    ram = cdata_with_workers[SCATTERED].to_ram()

    highly_variable_genes(ram, n_top_genes=10, flavor="seurat_v3", accelerator="cpu")
    pca(ram, n_components=3, zscore=True, accelerator="cpu")

    assert ram.hvg is not None and int(ram.hvg.sum()) == 10
    assert "pca" in ram.trained_modules


def test_a_view_that_is_not_in_memory_still_reads_by_shard_with_workers(cdata_with_workers):
    view = cdata_with_workers[SCATTERED]
    datamodule = view.datamodule
    assert isinstance(datamodule, ShardChunkDataModule) and datamodule.reads_by_shard()
    assert datamodule.num_workers == 2 and datamodule.prefetch_factor == 2
    assert isinstance(datamodule.train_dataset, ShardChunkDataset)


def test_a_view_of_a_view_in_memory_is_in_memory_and_has_no_workers(cdata_with_workers):
    ram = cdata_with_workers[SCATTERED].to_ram()
    ram.datamodule.num_workers = 2  # not allowed for cells in memory, but a parent can have any setting
    sub = ram[[1, 3, 4]]

    datamodule = sub.datamodule
    assert datamodule.dadc.max_num_workers == 0
    assert not datamodule.reads_by_shard() and not isinstance(datamodule.train_dataset, ShardChunkDataset)
    assert datamodule.num_workers == 0 and datamodule.prefetch_factor is None
    batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch["x_ng"], torch.Tensor) and batch["x_ng"].shape[0] == 3
    with pytest.raises(ValueError, match="at most 0"):
        datamodule.num_workers = 2
        datamodule.train_dataloader()
