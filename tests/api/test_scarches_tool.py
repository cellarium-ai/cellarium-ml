# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from cellarium.ml.api.cellariumdata import CellariumData, _data_fields_used
from cellarium.ml.api.tools import embed, scanvi, scanvi_annotate, scarches, scvi
from cellarium.ml.api.tools.scarches import _count_missing_genes
from cellarium.ml.models import SCANVI, ScArches
from cellarium.ml.models.scarches import ScArchesLinearSurgeryMixin

N_GENES = 30
FIT: dict[str, Any] = {"max_epochs": 2, "accelerator": "cpu"}


def _write_cdata(
    tmp_path, name: str, genes: list[str], obs_columns: dict[str, list[str]], seed: int, **cdata_kwargs: Any
) -> CellariumData:
    """
    40 cells in two h5ad files, with the genes ``genes`` and a categorical obs column for each of ``obs_columns``, as a
    CellariumData with ``cdata_kwargs``.
    """
    rng = np.random.default_rng(seed)
    j = np.arange(20)
    paths = []
    for i in range(2):
        obs = pd.DataFrame(
            {
                column: pd.Categorical(np.array(categories)[j % len(categories)], categories=categories)
                for column, categories in obs_columns.items()
            },
            index=[f"{name}{i}_cell{k}" for k in j],
        )
        x = sp.csr_matrix(rng.poisson(lam=3.0, size=(20, len(genes))).astype(np.float32))
        path = tmp_path / f"{name}_{i}.h5ad"
        ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path)
        paths.append(str(path))
    return CellariumData(h5ad_paths=paths, **cdata_kwargs)


@pytest.fixture
def reference(tmp_path) -> CellariumData:
    """Data with a batch and cell types, of the genes gene0 ... gene29."""
    genes = [f"gene{k}" for k in range(N_GENES)]
    return _write_cdata(tmp_path, "ref", genes, {"batch": ["b0", "b1"], "cell_type": ["T", "B", "NK"]}, seed=0)


@pytest.fixture
def query(tmp_path) -> CellariumData:
    """Other data, without the batch column of the reference, with three donors, a few of the genes of the reference
    missing, two others, and the genes in another order."""
    genes = [f"gene{k}" for k in range(3, N_GENES)][::-1] + ["other0", "other1"]
    return _write_cdata(tmp_path, "query", genes, {"donor": ["d0", "d1", "d2"]}, seed=1)


def _trained_scanvi(reference: CellariumData):
    scanvi(reference, batch_key="batch", labels_key="cell_type", n_latent=3, **FIT)
    return reference.trained_modules["scanvi"]


def test_scarches_maps_new_data_onto_a_scanvi_module_trained_on_other_data(reference, query):
    trained_reference = _trained_scanvi(reference)
    reference_model = trained_reference.module.model
    reference_state = {name: value.clone() for name, value in reference_model.state_dict().items()}

    scarches(query, reference=trained_reference, batch_key="donor", **FIT)

    trained = query.trained_modules["scarches"]
    model = trained.module.model
    assert isinstance(model, ScArches) and isinstance(model.scvi, SCANVI)
    assert (model.n_batch_ref, model.n_new_batch) == (2, 3)
    assert trained.config["batch_categories"] == ["d0", "d1", "d2"]
    assert trained.config["n_missing_genes"] == 3
    assert trained.batch_keys["batch_index_n"].key == "donor"
    assert "batch_index_n" in _data_fields_used(trained.module)[1]  # required by the model that the ScArches wraps

    # the reference module is untouched: same parameters, same batches, no surgery
    assert reference_model.n_batch == 2
    assert not any(isinstance(m, ScArchesLinearSurgeryMixin) for m in reference_model.modules())
    for name, value in reference_model.state_dict().items():
        torch.testing.assert_close(value, reference_state[name])

    # only the parameters of the new batches were trained
    mapped_state = model.scvi.state_dict()
    for name, value in reference_state.items():
        torch.testing.assert_close(mapped_state[name].cpu(), value, msg=f"{name} changed")
    new_weights = [m.new_batch_bias_weight for m in model.scvi.modules() if isinstance(m, ScArchesLinearSurgeryMixin)]
    assert new_weights and any(weight.abs().sum() > 0 for weight in new_weights)

    # and the other tools use the module as any other
    embed(query, "scarches", accelerator="cpu")
    assert query.obsm["X_scarches"].shape == (40, 3)
    scanvi_annotate(query, "scarches", threshold=0.0, accelerator="cpu")
    assert set(query.obs_computed["scarches_annotation"].cat.categories) <= {"B", "NK", "T"}


def test_the_reference_cannot_run_on_data_without_its_batch_column_and_says_to_use_scarches(reference, query):
    query.trained_modules["scanvi"] = _trained_scanvi(reference)
    with pytest.raises(ValueError, match=r"obs columns that this data does not have.*cml\.tl\.scarches"):
        embed(query, "scanvi", accelerator="cpu")


def test_one_new_batch_survives_saving_and_loading_the_session(reference, query, tmp_path):
    scvi(reference, batch_key="batch", n_latent=3, **FIT)
    query.trained_modules["reference"] = reference.trained_modules["scvi"]
    scarches(query, reference="reference", **FIT)  # batch_key=None: the cells are all one batch
    trained = query.trained_modules["reference_scarches"]
    assert trained.module.model.n_new_batch == 1 and "batch_categories" not in trained.config
    # the KL warmup continued from where the training of the reference left off
    assert trained.module.model.scvi.step_offset == reference.trained_modules["scvi"].module.model.step > 0
    embed(query, "reference_scarches", accelerator="cpu")

    query.save(str(tmp_path / "session"))
    loaded = CellariumData.load(str(tmp_path / "session"))
    embed(loaded, "reference_scarches", key_added="again", accelerator="cpu")
    np.testing.assert_allclose(loaded.obsm["again"], query.obsm["X_reference_scarches"], rtol=1e-5, atol=1e-6)


def test_a_reference_whose_encoder_does_not_take_the_batch_is_an_error(reference, query):
    encoder = {
        "hidden_layers": [{"class_path": "torch.nn.Linear", "init_args": {"out_features": 16}}],
        "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
    }
    scvi(reference, batch_key="batch", n_latent=3, scvi_kwargs={"encoder": encoder}, **FIT)
    with pytest.raises(ValueError, match="encoder of the reference model does not take the batch"):
        scarches(query, reference=reference.trained_modules["scvi"], **FIT)
    assert "scarches" not in query.trained_modules


def test_data_without_most_of_the_genes_of_the_reference_is_an_error(reference, tmp_path):
    scvi(reference, batch_key="batch", n_latent=3, **FIT)
    other = _write_cdata(tmp_path, "other", [f"ENSG{k}" for k in range(N_GENES)], {"donor": ["d0"]}, seed=2)
    with pytest.raises(
        ValueError, match=r"30 of the 30 genes of the reference model \(100.0%\) are not in the data"
    ) as error:
        scarches(other, reference=reference.trained_modules["scvi"], **FIT)
    assert "var_names_strip_ensembl_version" not in str(error.value)  # stripping would not find them


def test_data_with_ensembl_versions_that_the_reference_lacks_is_told_to_strip_them(tmp_path):
    ensembl_ids = [f"ENSG{k:011d}" for k in range(N_GENES)]
    reference = _write_cdata(tmp_path, "ref", ensembl_ids, {"batch": ["b0", "b1"]}, seed=0)
    scvi(reference, batch_key="batch", n_latent=3, **FIT)
    versioned_ids = [f"{name}.{k % 3 + 1}" for k, name in enumerate(ensembl_ids)]

    query = _write_cdata(tmp_path, "query", versioned_ids, {"donor": ["d0"]}, seed=1)
    with pytest.raises(ValueError, match="Pass var_names_strip_ensembl_version=True") as error:
        scarches(query, reference=reference.trained_modules["scvi"], **FIT)
    assert "30 more of its genes would be found" in str(error.value)

    query = _write_cdata(
        tmp_path, "query", versioned_ids, {"donor": ["d0"]}, seed=1, var_names_strip_ensembl_version=True
    )
    scarches(query, reference=reference.trained_modules["scvi"], **FIT)
    assert query.trained_modules["scarches"].config["n_missing_genes"] == 0


def test_the_warning_for_missing_genes_also_tells_to_strip_ensembl_versions():
    reference_genes = np.array([f"ENSG{k:011d}" for k in range(N_GENES)])
    # a fifth of the genes have versions
    data_genes = np.array([f"{name}.1" if k < 6 else name for k, name in enumerate(reference_genes)])
    with pytest.warns(UserWarning, match="6 more of its genes would be found"):
        _count_missing_genes(_FakeCData(data_genes), reference_genes)


class _FakeCData:
    """Just the ``datamodule.var_names_g`` of a CellariumData."""

    def __init__(self, var_names_g: np.ndarray):
        self.datamodule = type("FakeDataModule", (), {"var_names_g": var_names_g})()
