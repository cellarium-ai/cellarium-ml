# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch

from cellarium.ml.api import CellariumData
from cellarium.ml.api.tools import embed
from cellarium.ml.cli import main
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes

# Modules that the Lightning CLI trained, loaded as modules of a CellariumData. The data (see `make_h5ad_files`) has
# obs columns `cell_type` (3 categories) and `n_counts`, and 30 genes.

_DATA = """
data:
  dadc:
    class_path: cellarium.ml.data.DistributedAnnDataCollection
    init_args:
      filenames: {directory}/data_{{0..1}}.h5ad
      shard_size: 20
      max_cache_size: 2
      obs_columns_to_validate: []
  batch_keys:
    x_ng: {{attr: X, convert_fn: cellarium.ml.utilities.data.to_torch_sparse_csr}}
    var_names_g: {{attr: var_names}}
{extra_batch_keys}
  batch_size: 10
  num_workers: 0
trainer:
  accelerator: cpu
  devices: 1
  max_epochs: {max_epochs}
  default_root_dir: {directory}/{model_name}
"""

_SCVI = """
model:
  model:
    class_path: cellarium.ml.models.SingleCellVariationalInference
    init_args:
      n_batch: 3
      var_names_g: null
      n_latent: 4
      kl_warmup_steps: null
      kl_warmup_epochs: null
      encoder:
        hidden_layers: []
        final_layer: {class_path: torch.nn.Linear, init_args: {}}
      decoder:
        hidden_layers:
          - class_path: cellarium.ml.models.scvi.LinearWithBatch
            init_args: {out_features: 16, label_to_bias_hidden_layers: []}
        final_layer: {class_path: torch.nn.Linear, init_args: {}}
        final_additive_bias: true
  transforms:
    - cellarium.ml.transforms.Densify
  optim_fn: torch.optim.Adam
  optim_kwargs: {lr: 1e-3}
"""

_PCA = """
model:
  transforms:
    - cellarium.ml.transforms.Densify
    - class_path: cellarium.ml.transforms.NormalizeTotal
      init_args: {target_count: 10000}
    - cellarium.ml.transforms.Log1p
  model:
    class_path: cellarium.ml.models.IncrementalPCA
    init_args: {n_components: 3, perform_mean_correction: true}
"""


def _train_with_cli(directory: Path, model_name: str, model: str, extra_batch_keys: str, max_epochs: int = 1) -> str:
    config = model + _DATA.format(
        directory=directory, model_name=model_name, extra_batch_keys=extra_batch_keys, max_epochs=max_epochs
    )
    (directory / f"{model_name}.yaml").write_text(config)
    main([model_name, "fit", "--config", str(directory / f"{model_name}.yaml")])
    (checkpoint,) = (directory / model_name).rglob("*.ckpt")
    return str(checkpoint)


def _with_renamed_obs_column(h5ad_paths: list[str], directory: Path, old: str, new: str) -> list[str]:
    """Copies of the h5ad files in ``directory`` whose obs column ``old`` is named ``new``."""
    directory.mkdir()
    paths = []
    for i, path in enumerate(h5ad_paths):
        adata = ad.read_h5ad(path)
        adata.obs = adata.obs.rename(columns={old: new})
        paths.append(str(directory / f"data_{i}.h5ad"))
        adata.write_h5ad(paths[-1])
    return paths


@pytest.fixture
def h5ad_paths(make_h5ad_files) -> list[str]:
    return make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30)


@pytest.fixture
def scvi_checkpoint(h5ad_paths, tmp_path) -> str:
    """scVI trained for 3 epochs by the CLI, with the batch index taken from the obs column `cell_type`."""
    batch_index = (
        "    batch_index_n: {attr: obs, key: cell_type, convert_fn: cellarium.ml.utilities.data.categories_to_codes}"
    )
    return _train_with_cli(tmp_path, "scvi", _SCVI, batch_index, max_epochs=3)


@pytest.fixture
def pca_checkpoint(h5ad_paths, tmp_path) -> str:
    """PCA trained by the CLI, normalizing by the library size in the obs column `n_counts`."""
    return _train_with_cli(
        tmp_path, "incremental_pca", _PCA, "    total_mrna_umis_n: {attr: obs, key: n_counts}", max_epochs=1
    )


def test_load_module_from_ckpt_stores_the_module_with_what_the_checkpoint_says(h5ad_paths, scvi_checkpoint):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    cdata.load_module_from_ckpt(scvi_checkpoint, "scvi")
    trained = cdata.trained_modules["scvi"]
    assert trained.n_epochs == 3 and trained.complete
    assert trained.config == {"source": scvi_checkpoint} and trained.history.empty
    assert list(trained.batch_keys) == ["batch_index_n"]  # recovered from the checkpoint
    assert scvi_checkpoint in repr(trained)
    assert next(trained.module.parameters()).device.type in ("cpu", "meta")


def test_load_module_from_ckpt_counts_the_epochs_of_a_checkpoint_that_is_saved_when_training_ends(
    h5ad_paths, pca_checkpoint
):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    cdata.load_module_from_ckpt(pca_checkpoint, "pca")
    assert cdata.trained_modules["pca"].n_epochs == 1


def test_a_module_loaded_from_a_checkpoint_embeds_like_one_trained_by_the_api(h5ad_paths, scvi_checkpoint):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    cdata.load_module_from_ckpt(scvi_checkpoint, "scvi")
    embed(cdata, "scvi", accelerator="cpu")

    adata = cdata.to_anndata()
    module = cdata.trained_modules["scvi"].module
    module.configure_model()
    with torch.no_grad():
        prediction = module.model.eval().predict(
            x_ng=torch.as_tensor(adata.X.toarray()),
            var_names_g=np.array(adata.var_names),
            batch_index_n=torch.as_tensor(adata.obs["cell_type"].cat.codes.to_numpy().astype(np.int64))[:, None],
        )
    np.testing.assert_allclose(cdata.obsm["X_scvi"], prediction["x_ng"].numpy(), rtol=1e-4, atol=1e-4)


def test_only_the_fields_that_the_module_takes_are_recovered_from_the_checkpoint(h5ad_paths, tmp_path):
    # the checkpoint also records `unused_n`, which nothing in the module takes, from a column that this data lacks
    extra = (
        "    batch_index_n: {attr: obs, key: cell_type, convert_fn: cellarium.ml.utilities.data.categories_to_codes}\n"
        "    unused_n: {attr: obs, key: n_counts}"
    )
    checkpoint = _train_with_cli(tmp_path, "scvi", _SCVI, extra)
    cdata = CellariumData(h5ad_paths=_with_renamed_obs_column(h5ad_paths, tmp_path / "other", "n_counts", "umis"))
    cdata.load_module_from_ckpt(checkpoint, "scvi")
    assert list(cdata.trained_modules["scvi"].batch_keys) == ["batch_index_n"]


def test_a_module_that_needs_fields_the_data_does_not_have_says_how_to_load_it_again(h5ad_paths, scvi_checkpoint):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    cdata.load_module_from_ckpt(scvi_checkpoint, "scvi", batch_keys={})

    with pytest.raises(ValueError, match=r"needs the data fields \['batch_index_n'\]") as error:
        embed(cdata, "scvi", accelerator="cpu")
    assert f"load_module_from_ckpt('{scvi_checkpoint}', 'scvi', batch_keys=" in str(error.value)
    assert "batch_index_n" not in cdata.datamodule.batch_keys  # nothing is left behind


def test_batch_keys_of_the_user_replace_those_of_the_checkpoint(h5ad_paths, scvi_checkpoint):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    field = AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes)
    cdata.load_module_from_ckpt(scvi_checkpoint, "scvi", batch_keys={"batch_index_n": field})
    assert cdata.trained_modules["scvi"].batch_keys == {"batch_index_n": field}
    embed(cdata, "scvi", accelerator="cpu")
    assert cdata.obsm["X_scvi"].shape == (40, 4)


def test_a_recovered_field_that_reads_an_obs_column_the_data_lacks_raises_and_says_what_to_do(
    h5ad_paths, scvi_checkpoint, tmp_path
):
    paths = _with_renamed_obs_column(h5ad_paths, tmp_path / "other", "cell_type", "annotation")
    cdata = CellariumData(h5ad_paths=paths)

    with pytest.raises(ValueError, match=r"read from the obs column\(s\) \['cell_type'\]") as error:
        cdata.load_module_from_ckpt(scvi_checkpoint, "scvi")
    assert "batch_keys" in str(error.value)
    assert "scvi" not in cdata.trained_modules

    field = AnnDataField(attr="obs", key="annotation", convert_fn=categories_to_codes)
    cdata.load_module_from_ckpt(scvi_checkpoint, "scvi", batch_keys={"batch_index_n": field})
    embed(cdata, "scvi", accelerator="cpu")
    assert cdata.obsm["X_scvi"].shape == (40, 4)


def test_loading_a_module_again_under_the_same_key_makes_its_embedding_stale(h5ad_paths, scvi_checkpoint):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    cdata.load_module_from_ckpt(scvi_checkpoint, "scvi")
    embed(cdata, "scvi", accelerator="cpu")
    assert cdata._cached_embedding_state("scvi") == "current"

    cdata.load_module_from_ckpt(scvi_checkpoint, "scvi")
    assert cdata._cached_embedding_state("scvi") == "stale"


def test_a_module_loaded_from_a_checkpoint_works_on_a_view(h5ad_paths, pca_checkpoint):
    cdata = CellariumData(h5ad_paths=h5ad_paths)
    view = cdata[np.arange(5, 25)]
    view.load_module_from_ckpt(pca_checkpoint, "pca")
    embed(view, "pca", accelerator="cpu")
    assert view.obsm["X_pca"].shape == (20, 3)
    assert "pca" not in cdata.trained_modules
