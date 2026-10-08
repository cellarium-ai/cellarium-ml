# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from types import SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.api import CellariumData
from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import geometric_sketch, pca, scvi
from cellarium.ml.api.utils import PreciseProgressBar
from cellarium.ml.models import IncrementalPCA, SingleCellVariationalInference
from cellarium.ml.utilities.data import to_torch_sparse_csr


@pytest.fixture
def onepass_module(cdata):
    _, module = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    return module


# --- pca() -----------------------------------------------------------------------------------


def test_pca_without_hvg_warns_and_uses_all_genes(cdata):
    assert cdata.hvg is None
    with pytest.warns(UserWarning, match="all genes"):
        module = pca(cdata, n_components=3, zscore=True, accelerator="cpu")

    assert module.model.var_names_g.tolist() == cdata.datamodule.var_names_g.tolist()


def test_pca_with_hvg_does_not_warn(cdata, onepass_module):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*all genes")
        pca(cdata, n_components=3, zscore=True, accelerator="cpu")


def test_pca_returns_module_with_expected_shape(cdata, onepass_module):
    module = pca(cdata, n_components=3, zscore=True, accelerator="cpu")

    assert isinstance(module, CellariumModule)
    assert isinstance(module.model, IncrementalPCA)
    assert module.model.n_components == 3
    expected_genes = sorted(cdata.hvg.index[cdata.hvg].tolist())
    assert sorted(module.model.var_names_g.tolist()) == expected_genes


def test_pca_registers_trained_module(cdata, onepass_module):
    module = pca(cdata, n_components=3, zscore=True, accelerator="cpu")

    trained = cdata.trained_modules["pca"]
    assert trained.module is module
    assert trained.complete
    assert trained.history.empty
    assert "n_components=3" in repr(cdata)

    pca(cdata, n_components=3, accelerator="cpu", key_added="other_pca")
    assert {"onepass", "pca", "other_pca"} == set(cdata.trained_modules)


def test_pca_statistics_do_not_replace_the_hvg_onepass(cdata, onepass_module):
    onepass = cdata.trained_modules["onepass"]

    pca(cdata, n_components=3, zscore=True, accelerator="cpu")

    assert cdata.trained_modules["onepass"] is onepass
    assert onepass.config["log1p"] is False
    assert set(cdata.trained_modules) == {"onepass", "pca"}


def test_pca_reuses_the_zscore_statistics(cdata, onepass_module, fits):
    n_fits_by_hvg = len(fits)

    pca(cdata, n_components=3, accelerator="cpu")
    pca(cdata, n_components=2, accelerator="cpu", key_added="other_pca")  # another n_components: same statistics

    assert len(fits) == n_fits_by_hvg + 1
    assert fits[-1] == {"target_count": 10_000, "eps": 1e-6, "log1p": True, "batch_key": None}


def test_pca_without_zscore_fits_no_statistics(cdata, onepass_module, fits):
    n_fits_by_hvg = len(fits)

    pca(cdata, n_components=3, zscore=False, accelerator="cpu")

    assert len(fits) == n_fits_by_hvg


def test_pca_hvg_onepass_cannot_be_used_for_the_zscore(cdata, fits):
    """The HVG statistics are of the normalized counts; the z-score needs those of the log1p transformed ones."""
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")

    pca(cdata, n_components=3, accelerator="cpu")

    assert [config["log1p"] for config in fits] == [False, True]


# --- pca() against the preprocessing and the PCA computed by hand ----------------------------------------


N_CELLS_PER_FILE, N_GENES = 100, 24


@pytest.fixture
def counts_and_cdata(make_h5ad_files):
    """The dense counts of some larger synthetic h5ad files and a CellariumData over them."""
    h5ad_paths = make_h5ad_files(n_files=2, cells_per_file=N_CELLS_PER_FILE, n_genes=N_GENES, seed=3)
    adatas = [ad.read_h5ad(path) for path in h5ad_paths]
    counts_ng = np.vstack([adata.X.toarray() for adata in adatas]).astype(np.float64)
    assert (counts_ng.std(axis=0) > 0).all()  # no gene is constant, so no singular value is zero
    return counts_ng, CellariumData(h5ad_paths=h5ad_paths)


def preprocess_with_scanpy(counts_ng: np.ndarray, var_names_g: np.ndarray, keep_g: np.ndarray) -> np.ndarray:
    """Normalize over all genes and log1p transform, then keep the genes in `keep_g`, as scanpy does."""
    adata = ad.AnnData(X=counts_ng.copy(), var=pd.DataFrame(index=var_names_g))
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    return adata.X[:, keep_g]


def zscore_with_population_std(x_ng: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (x_ng - x_ng.mean(axis=0)) / (x_ng.std(axis=0) + eps)


@pytest.mark.parametrize("zscore", [True, False])
@pytest.mark.parametrize("use_hvg", [True, False])
def test_pca_matches_pca_computed_by_hand(counts_and_cdata, zscore, use_hvg):
    counts_ng, cdata = counts_and_cdata
    var_names_g = cdata.datamodule.var_names_g
    # the genes with the most counts, so that the preprocessing before the subsetting matters
    keep_g = np.sort(np.argsort(-counts_ng.sum(axis=0))[:10]) if use_hvg else np.arange(N_GENES)
    if use_hvg:
        cdata.hvg = var_names_g[keep_g]
    n_components = len(keep_g)  # all of them: the randomized svd is then exact

    x_ng = preprocess_with_scanpy(counts_ng, var_names_g, keep_g)
    x_ng = zscore_with_population_std(x_ng) if zscore else x_ng - x_ng.mean(axis=0)
    _, singular_values_k, components_kg = np.linalg.svd(x_ng, full_matrices=False)

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*all genes")
        module = pca(cdata, n_components=n_components, zscore=zscore, accelerator="cpu")

    model = module.model
    assert model.var_names_g.tolist() == var_names_g[keep_g].tolist()
    np.testing.assert_allclose(model.explained_variance_k.numpy(), singular_values_k**2 / len(x_ng), rtol=1e-3)
    # the components are those of the data, up to their signs
    cosines_k = np.sum(model.components_kg.numpy() * components_kg, axis=1)
    np.testing.assert_allclose(np.abs(cosines_k), 1.0, atol=1e-3)


def test_pca_embedding_matches_projection_computed_by_hand(counts_and_cdata):
    counts_ng, cdata = counts_and_cdata
    var_names_g = cdata.datamodule.var_names_g
    keep_g = np.sort(np.argsort(-counts_ng.sum(axis=0))[:10])
    cdata.hvg = var_names_g[keep_g]

    x_ng = zscore_with_population_std(preprocess_with_scanpy(counts_ng, var_names_g, keep_g))

    torch.manual_seed(0)
    module = pca(cdata, n_components=len(keep_g), zscore=True, accelerator="cpu")

    # the embedding of the cells of the first file, through the trained pipeline from the raw counts
    first_file = cdata.datamodule.dadc[:N_CELLS_PER_FILE]
    batch = {"x_ng": to_torch_sparse_csr(first_file.X), "var_names_g": np.array(first_file.var_names)}
    prediction = module(batch)["x_ng"]
    assert isinstance(prediction, torch.Tensor)
    embedding_nk = prediction.detach().numpy()

    expected_nk = x_ng[:N_CELLS_PER_FILE] @ module.model.components_kg.numpy().T
    np.testing.assert_allclose(embedding_nk, expected_nk, rtol=1e-3, atol=1e-3)


# --- geometric_sketch() -----------------------------------------------------------------------


def test_geometric_sketch_raises_without_obs_names_key(cdata):
    del cdata.datamodule.batch_keys["obs_names_n"]
    with pytest.raises(ValueError):
        geometric_sketch(cdata, target_n_cells=10)


def test_geometric_sketch_raises_n_pcs_without_embedding_module_key(cdata):
    with pytest.raises(ValueError):
        geometric_sketch(cdata, target_n_cells=10, n_pcs=2)


def test_geometric_sketch_raises_n_pcs_with_non_pca_embedding_module(cdata, onepass_module):
    # the onepass_module fixture trains through the api, so it is registered as "onepass"
    with pytest.raises(ValueError):
        geometric_sketch(cdata, target_n_cells=10, n_pcs=2, embedding_module_key="onepass")


def test_geometric_sketch_raises_for_unknown_embedding_module_key_and_lists_available(cdata, onepass_module):
    with pytest.raises(ValueError, match=r"No trained module 'scvi'.*Available: \['onepass'\]"):
        geometric_sketch(cdata, target_n_cells=10, embedding_module_key="scvi")


def test_geometric_sketch_with_pca_embedding_module_key(cdata, onepass_module):
    pca(cdata, n_components=3, zscore=True, accelerator="cpu")

    result = geometric_sketch(cdata, target_n_cells=10, embedding_module_key="pca", n_pcs=2)

    adata = result["adata"]
    assert isinstance(adata, ad.AnnData)
    assert adata.obsm["X_embedding"].shape[1] == 2


def test_geometric_sketch_with_scvi_embedding_module_key_provides_batch_keys_itself(cdata):
    scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    assert "batch_index_n" not in cdata.datamodule.batch_keys  # scvi() restored the datamodule

    # no `temporary_batch_keys` needed: the key brings what the scVI module needs along
    result = geometric_sketch(cdata, target_n_cells=10, embedding_module_key="scvi")

    adata = result["adata"]
    assert isinstance(adata, ad.AnnData)
    assert adata.obsm["X_embedding"].shape[1] == 3
    assert "batch_index_n" not in cdata.datamodule.batch_keys  # and put it back


def test_geometric_sketch_default_embedding_returns_expected_keys(cdata):
    n_obs = len(cdata.datamodule.dadc)
    result = geometric_sketch(cdata, target_n_cells=10, return_new_adata=True)

    assert set(result.keys()) == {"obs_names_in_sketch", "adata", "module"}

    sketch_mask = result["obs_names_in_sketch"]
    assert isinstance(sketch_mask, pd.Series)
    assert sketch_mask.dtype == bool
    assert len(sketch_mask) == n_obs
    assert 0 < sketch_mask.sum() <= n_obs

    adata = result["adata"]
    assert isinstance(adata, ad.AnnData)
    assert "X_embedding" in adata.obsm
    assert adata.n_obs == int(sketch_mask.sum())
    assert isinstance(result["module"], CellariumModule)

    # side effect: the sketch mask is also recorded in cdata.obs_computed
    pd.testing.assert_series_equal(cdata.obs_computed["in_sketch"], sketch_mask)


def test_geometric_sketch_with_sparse_coo_batches(make_h5ad_files, monkeypatch):
    # the mps accelerator gets sparse COO instead of CSR batches; force that layout on any platform
    monkeypatch.setattr(torch.mps, "is_available", lambda: True)
    cdata = CellariumData(h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30))
    assert next(iter(cdata.datamodule.train_dataloader()))["x_ng"].layout == torch.sparse_coo

    result = geometric_sketch(cdata, target_n_cells=10, return_new_adata=True)

    adata = result["adata"]
    assert isinstance(adata, ad.AnnData)
    sketch_mask = result["obs_names_in_sketch"]
    assert isinstance(sketch_mask, pd.Series)
    assert adata.n_obs == int(sketch_mask.sum())


def test_geometric_sketch_return_new_adata_false_gives_no_adata(cdata):
    result = geometric_sketch(cdata, target_n_cells=10, return_new_adata=False)
    assert result["adata"] is None
    assert isinstance(result["module"], CellariumModule)


# --- scvi() ------------------------------------------------------------------------------------


def test_scvi_raises_without_batch_key(cdata):
    with pytest.raises(ValueError):
        scvi(cdata, batch_key=None, max_epochs=1, accelerator="cpu")  # type: ignore[arg-type]


def test_scvi_returns_trained_module_and_restores_datamodule_state(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_n_train, original_n_val = datamodule.n_train, datamodule.n_val
    original_obs_columns_to_validate = getattr(datamodule.dadc, "obs_columns_to_validate", None)  # h5ad only

    module = scvi(
        cdata,
        batch_key="cell_type",
        n_latent=3,
        max_epochs=2,
        val_size=0.2,
        early_stopping_patience=1,
        accelerator="cpu",
    )

    assert isinstance(module, CellariumModule)
    assert isinstance(module.model, SingleCellVariationalInference)
    assert module.model.n_latent == 3
    assert module.model.n_batch == 3  # "cell_type" has 3 categories in the synthetic fixture

    # side-effect-free: datamodule state restored to what it was before scvi() ran
    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert datamodule.n_train == original_n_train
    assert datamodule.n_val == original_n_val
    assert getattr(datamodule.dadc, "obs_columns_to_validate", None) == original_obs_columns_to_validate


def test_scvi_registers_trained_module_with_history(cdata):
    module = scvi(
        cdata,
        batch_key="cell_type",
        n_latent=3,
        max_epochs=3,
        val_size=0.2,
        early_stopping_patience=100,
        log_every_n_steps=1,
        accelerator="cpu",
    )

    trained = cdata.trained_modules["scvi"]
    assert trained.module is module
    assert trained.complete
    assert trained.n_epochs == 3
    assert list(trained.history.columns) == ["step", "epoch", "metric", "value"]
    assert {"train_loss", "val_loss"} <= set(trained.history["metric"])
    # one validation per epoch, and the sanity check is not recorded
    assert len(trained.metric("val_loss")) == 3
    assert "scvi" in repr(cdata) and "n_latent=3" in repr(cdata)

    scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu", key_added="other")
    assert set(cdata.trained_modules) == {"scvi", "other"}


def test_using_provides_and_restores_the_batch_keys_a_trained_module_needs(cdata, onepass_module):
    scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    original_batch_keys = set(cdata.datamodule.batch_keys)

    with cdata.using("scvi") as trained:
        assert trained is cdata.trained_modules["scvi"]
        assert "batch_index_n" in cdata.datamodule.batch_keys
        assert "batch_index_n" in next(iter(cdata.datamodule.train_dataloader()))
    assert set(cdata.datamodule.batch_keys) == original_batch_keys

    # restored even if the block raises
    with pytest.raises(RuntimeError):
        with cdata.using("scvi"):
            raise RuntimeError
    assert set(cdata.datamodule.batch_keys) == original_batch_keys

    # a module that needs nothing extra leaves the datamodule alone
    with cdata.using("onepass"):
        assert set(cdata.datamodule.batch_keys) == original_batch_keys


def test_using_raises_for_unknown_key(cdata):
    with pytest.raises(ValueError, match="None have been trained yet"):
        with cdata.using("scvi"):
            pass


@pytest.mark.parametrize(
    "batch_size, val_check_interval, expected_val_steps",
    [
        # logged steps are 0-based (the index of the last batch before the check)
        (8, 500, [3, 7]),  # 4 batches per epoch, interval longer than an epoch: once per epoch (no error)
        (8, 2, [1, 3, 5, 7]),  # every 2 steps
        (5, 3, [2, 5, 8, 11]),  # 7 batches per epoch: every 3 steps, ignoring epoch boundaries
    ],
)
def test_scvi_val_check_interval_is_a_global_step_cadence_capped_at_once_per_epoch(
    cdata, batch_size, val_check_interval, expected_val_steps
):
    cdata.datamodule.batch_size = batch_size  # 32 training cells
    scvi(
        cdata,
        batch_key="cell_type",
        n_latent=3,
        max_epochs=2,
        val_size=0.2,
        early_stopping_patience=100,
        val_check_interval=val_check_interval,
        accelerator="cpu",
    )
    val_loss = cdata.trained_modules["scvi"].metric("val_loss")
    assert list(val_loss.index) == expected_val_steps


@pytest.mark.parametrize(
    "scvi_kwargs, expected_steps, expected_epochs",
    [
        (None, 1000, None),  # default: 1000 step warmup
        ({}, 1000, None),
        ({"kl_warmup_steps": 5}, 5, None),
        ({"kl_warmup_epochs": 7}, None, 7),  # the default step warmup must not also be applied
        ({"kl_warmup_steps": None, "kl_warmup_epochs": None}, None, None),  # no warmup
    ],
)
def test_scvi_kl_warmup_defaults_do_not_override_user_values(cdata, scvi_kwargs, expected_steps, expected_epochs):
    module = scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=1, scvi_kwargs=scvi_kwargs, accelerator="cpu")

    assert module.model.kl_warmup_steps == expected_steps
    assert module.model.kl_warmup_epochs == expected_epochs


def test_precise_progress_bar_formats_floats_with_more_significant_digits():
    # stand-ins for the parts of the Trainer and LightningModule that get_metrics reads
    trainer: Any = SimpleNamespace(
        loggers=[], progress_bar_metrics={"val_loss": 1523.4567891, "tiny": 0.00012345678, "n": 3}
    )
    pl_module: Any = None

    metrics = PreciseProgressBar(significant_digits=7).get_metrics(trainer, pl_module)

    # tqdm would show "1.52e+3"
    assert metrics == {"val_loss": "1523.457", "tiny": "0.0001234568", "n": 3}


def test_scvi_registers_partial_module_on_interrupt(cdata, monkeypatch):
    def interrupt_after_first_epoch(self, trainer):
        if trainer.current_epoch == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        SingleCellVariationalInference, "on_train_epoch_end", interrupt_after_first_epoch, raising=False
    )
    # Lightning turns the KeyboardInterrupt into a SystemExit after shutting down gracefully
    with pytest.raises(SystemExit):
        scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=5, log_every_n_steps=1, accelerator="cpu")

    trained = cdata.trained_modules["scvi"]
    assert not trained.complete
    assert len(trained.history) > 0
    assert "(interrupted)" in repr(cdata)
