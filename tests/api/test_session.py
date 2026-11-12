# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from cellarium.ml.api import CellariumData, CellariumDataView
from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import embed, pca
from cellarium.ml.core import CellariumModule
from cellarium.ml.utilities.data import to_float_tensor

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`


@pytest.fixture
def analysed(cdata):
    """The data after the steps of an analysis."""
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(cdata, n_components=3, zscore=True, accelerator="cpu")
    embed(cdata, "pca", accelerator="cpu")
    cdata.obs_computed["a"] = pd.Series(pd.Categorical(["x", "y"] * 20), index=cdata.cell_ids).where(np.arange(40) < 30)
    cdata.obsm["X_extra"] = np.arange(80, dtype=np.float32).reshape(40, 2)
    return cdata


def test_a_saved_session_is_loaded_with_everything_in_it(analysed, tmp_path, deltacells_kwargs):
    analysed.save(str(tmp_path / "session"))
    kwargs = {"deltacells_kwargs": deltacells_kwargs} if analysed.deltacells_uri else {}
    loaded = CellariumData.load(str(tmp_path / "session"), **kwargs)

    assert loaded.n_obs == 40 and loaded.parent is None
    pd.testing.assert_series_equal(loaded.hvg, analysed.hvg)
    pd.testing.assert_series_equal(loaded.obs_computed["a"], analysed.obs_computed["a"], check_names=False)
    for key in ("X_pca", "X_extra"):
        assert isinstance(loaded.obsm[key], np.memmap)
        np.testing.assert_array_equal(loaded.obsm[key], analysed.obsm[key])

    old, new = analysed.trained_modules["pca"], loaded.trained_modules["pca"]
    assert type(new.module.model) is type(old.module.model)
    for name, value in old.module.state_dict().items():
        torch.testing.assert_close(new.module.state_dict()[name], value.cpu())
    assert (new.n_epochs, new.complete, new.config) == (old.n_epochs, old.complete, old.config)
    pd.testing.assert_frame_equal(new.history, old.history)

    # the embedding is still the one of the module in `trained_modules`, so it is used as it is
    assert loaded._cached_embedding_state("pca") == "current"
    np.testing.assert_allclose(loaded.to_anndata().obsm["X_pca"], analysed.to_anndata().obsm["X_pca"])


def test_a_saved_module_is_a_checkpoint_that_loads_on_its_own(analysed, tmp_path):
    analysed.save(str(tmp_path / "session"))
    index = list(analysed.trained_modules).index("pca")
    module = CellariumModule.load_from_checkpoint(
        tmp_path / "session" / "modules" / f"{index}.ckpt", map_location="cpu"
    )
    for name, value in analysed.trained_modules["pca"].module.state_dict().items():
        torch.testing.assert_close(module.state_dict()[name], value.cpu())


def test_a_view_is_saved_with_the_cells_that_it_has(analysed, tmp_path, deltacells_kwargs):
    view = analysed[np.arange(40) % 3 == 0]
    view.obs_computed["b"] = np.arange(view.n_obs, dtype=float)
    view.save(str(tmp_path / "session"))
    kwargs = {"deltacells_kwargs": deltacells_kwargs} if analysed.deltacells_uri else {}
    loaded = CellariumData.load(str(tmp_path / "session"), **kwargs)

    assert isinstance(loaded, CellariumDataView)
    np.testing.assert_array_equal(loaded.root_indices, view.root_indices)
    np.testing.assert_array_equal(loaded.cell_ids, view.cell_ids)
    np.testing.assert_array_equal(loaded.obs_computed["b"], view.obs_computed["b"])
    np.testing.assert_array_equal(loaded.obsm["X_pca"], view.obsm["X_pca"])
    assert loaded.root.n_obs == 40
    assert "pca" not in loaded.trained_modules  # a view starts without the modules of its parent


def test_the_data_can_be_loaded_from_another_location(h5ad_cdata, tmp_path):
    h5ad_cdata.obs_computed["a"] = np.arange(40.0)
    h5ad_cdata.save(str(tmp_path / "session"))
    moved = []
    for i, path in enumerate(h5ad_cdata.datamodule.dadc.filenames):
        moved.append(str(tmp_path / f"moved_{i}.h5ad"))
        shutil.copy(path, moved[-1])
        shutil.move(path, path + ".gone")

    with pytest.raises(ValueError, match="is not there"):
        CellariumData.load(str(tmp_path / "session"))
    loaded = CellariumData.load(str(tmp_path / "session"), h5ad_paths=moved, batch_size=7, max_gb=1)
    assert loaded.datamodule.batch_size == 7 and loaded.max_gb == 1
    np.testing.assert_array_equal(loaded.obs_computed["a"], np.arange(40.0))


def test_the_obs_columns_of_the_constructor_are_saved_and_can_be_replaced(make_h5ad_files, tmp_path):
    paths = make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30)
    columns = {"y_n": ("n_counts", to_float_tensor)}
    CellariumData(h5ad_paths=paths, obs_columns=columns).save(str(tmp_path / "session"))
    assert "y_n" in CellariumData.load(str(tmp_path / "session")).datamodule.batch_keys
    loaded = CellariumData.load(str(tmp_path / "session"), obs_columns={"z_n": ("n_counts", to_float_tensor)})
    assert "z_n" in loaded.datamodule.batch_keys and "y_n" not in loaded.datamodule.batch_keys

    with pytest.raises(ValueError, match="obs_columns cannot be pickled"):
        CellariumData(h5ad_paths=paths, obs_columns={"y_n": ("n_counts", lambda x: x)}).save(str(tmp_path / "other"))
    assert not (tmp_path / "other").exists()


def test_loading_onto_other_cells_is_refused(h5ad_cdata, make_h5ad_files, tmp_path):
    h5ad_cdata.save(str(tmp_path / "session"))

    renamed = []
    for i, path in enumerate(h5ad_cdata.datamodule.dadc.filenames):
        adata = ad.read_h5ad(path)
        adata.obs_names = [f"renamed_{i}_{j}" for j in range(adata.n_obs)]
        renamed.append(str(tmp_path / f"renamed_{i}.h5ad"))
        adata.write_h5ad(renamed[-1])
    with pytest.raises(ValueError, match="not the data that the session was saved with"):
        CellariumData.load(str(tmp_path / "session"), h5ad_paths=renamed)

    other = make_h5ad_files(n_files=2, cells_per_file=10, n_genes=30, seed=1)
    with pytest.raises(ValueError, match="has 20 cells, but the session was saved with 40"):
        CellariumData.load(str(tmp_path / "session"), h5ad_paths=other)


def test_saving_does_not_replace_what_it_should_not(analysed, tmp_path, deltacells_kwargs):
    path = str(tmp_path / "session")
    analysed.save(path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        analysed.save(path)
    analysed.save(path, overwrite=True)

    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "mine.txt").write_text("keep")
    with pytest.raises(ValueError, match="not a saved session"):
        analysed.save(str(tmp_path / "notes"), overwrite=True)
    assert (tmp_path / "notes" / "mine.txt").exists()

    # the obsm of the loaded data is read from the directory, which saving would delete
    kwargs = {"deltacells_kwargs": deltacells_kwargs} if analysed.deltacells_uri else {}
    loaded = CellariumData.load(path, **kwargs)
    with pytest.raises(ValueError, match="Save to a different directory"):
        loaded.save(path, overwrite=True)
    loaded.save(str(tmp_path / "session_2"))
    assert (tmp_path / "session" / "manifest.pkl").exists()


def test_obsm_that_is_not_an_array_is_not_saved(cdata, tmp_path):
    cdata.obsm["frame"] = pd.DataFrame(np.zeros((40, 2)))
    with pytest.raises(TypeError, match="only numeric arrays"):
        cdata.save(str(tmp_path / "session"))
    assert not (tmp_path / "session").exists()


def test_a_directory_without_a_manifest_is_not_loaded(tmp_path):
    (tmp_path / "unfinished").mkdir()
    with pytest.raises(ValueError, match="not a saved session"):
        CellariumData.load(str(tmp_path / "unfinished"))
