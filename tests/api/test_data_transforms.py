# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from cellarium.ml.api._data_transforms import OnePassRecipe, SeuratV3Recipe, fit_or_reuse
from cellarium.ml.api.cellariumdata import FitCache, TrainedModule
from cellarium.ml.models import HVGSeuratV3, OnePassMeanVarStd
from cellarium.ml.transforms import Densify, Log1p, NormalizeTotal


def test_onepass_recipe_defaults_are_those_of_normalize_total():
    normalize_total = NormalizeTotal()
    recipe = OnePassRecipe()

    assert (recipe.target_count, recipe.eps) == (normalize_total.target_count, normalize_total.eps)


@pytest.mark.parametrize("log1p", [False, True])
def test_onepass_recipe_builds_the_module_without_writing_csv(cdata, log1p):
    recipe = OnePassRecipe(target_count=500, eps=0.1, log1p=log1p)

    module = recipe.build_module(cdata)
    module.configure_model()

    assert isinstance(module.model, OnePassMeanVarStd)
    assert module.model.output_path is None
    assert module.model.var_names_g.tolist() == cdata.datamodule.var_names_g.tolist()
    assert [type(t) for t in recipe.cpu_transforms()] == [NormalizeTotal]
    assert [type(t) for t in recipe.transforms()] == [Densify, *([Log1p] if log1p else [])]
    (normalize_total,) = recipe.cpu_transforms()
    assert (normalize_total.target_count, normalize_total.eps) == (500, 0.1)


def test_onepass_recipe_with_batch_key_builds_the_batches(cdata):
    recipe = OnePassRecipe(batch_key="cell_type")

    module = recipe.build_module(cdata)
    module.configure_model()

    assert module.model.n_batch == 3
    assert set(recipe.batch_keys()) == {"batch_index_n"}
    assert OnePassRecipe().batch_keys() == {}


def test_seurat_v3_recipe_builds_the_module_without_writing_csv(cdata):
    module = SeuratV3Recipe(batch_key="cell_type").build_module(cdata, n_top_genes=5)
    module.configure_model()

    assert isinstance(module.model, HVGSeuratV3)
    assert module.model.output_path is None
    assert module.model.n_batch == 3
    assert module.model.use_batch_key


@pytest.mark.parametrize(
    "requested, cached, served",
    [
        (OnePassRecipe(), OnePassRecipe(), True),
        (OnePassRecipe(), OnePassRecipe(batch_key="a"), True),  # no batches asked for: any batches will do
        (OnePassRecipe(batch_key="a"), OnePassRecipe(batch_key="a"), True),
        (OnePassRecipe(batch_key="a"), OnePassRecipe(), False),
        (OnePassRecipe(batch_key="a"), OnePassRecipe(batch_key="b"), False),
        (OnePassRecipe(log1p=True), OnePassRecipe(), False),
        (OnePassRecipe(), OnePassRecipe(log1p=True, batch_key="a"), False),
        (OnePassRecipe(target_count=500), OnePassRecipe(), False),
        (OnePassRecipe(eps=0.1), OnePassRecipe(), False),
        (OnePassRecipe(), SeuratV3Recipe(), False),
        (SeuratV3Recipe(), SeuratV3Recipe(), True),
        (SeuratV3Recipe(), SeuratV3Recipe(batch_key="a"), False),  # the batches change what the model computes
        (SeuratV3Recipe(batch_key="a"), SeuratV3Recipe(), False),
        (SeuratV3Recipe(), OnePassRecipe(), False),
    ],
)
def test_recipe_is_served_by(requested, cached, served):
    assert requested.is_served_by(cached) == served


def test_fit_cache_prefers_the_exact_recipe():
    unbatched, batched = TrainedModule(module=None), TrainedModule(module=None)  # type: ignore[arg-type]
    cache = FitCache()
    cache[OnePassRecipe(batch_key="a")] = batched
    assert cache.find(OnePassRecipe()) is batched

    cache[OnePassRecipe()] = unbatched
    assert cache.find(OnePassRecipe()) is unbatched
    assert cache.find(OnePassRecipe(batch_key="a")) is batched
    assert cache.find(OnePassRecipe(batch_key="b")) is None
    assert cache.find(OnePassRecipe(log1p=True)) is None


def test_fingerprint_describes_what_is_computed():
    assert OnePassRecipe().fingerprint() == "normalize_total(target_count=10000, eps=1e-06)"
    assert OnePassRecipe(log1p=True).fingerprint() == "normalize_total(target_count=10000, eps=1e-06) > log1p"
    assert OnePassRecipe(batch_key="a").fingerprint().endswith(" by a")
    assert SeuratV3Recipe().fingerprint() == "hvg_seurat_v3"


def test_fit_or_reuse_without_key_leaves_trained_modules_alone(cdata, fits):
    trained = fit_or_reuse(cdata, OnePassRecipe(log1p=True), accelerator="cpu")

    assert dict(cdata.trained_modules) == {}
    assert trained.complete
    assert fit_or_reuse(cdata, OnePassRecipe(log1p=True), accelerator="cpu") is trained
    assert len(fits) == 1


def test_fit_or_reuse_with_key_stores_the_cached_module(cdata, fits):
    trained = fit_or_reuse(cdata, OnePassRecipe(), accelerator="cpu")
    again = fit_or_reuse(cdata, OnePassRecipe(), accelerator="cpu", key_added="mine")

    assert again is trained
    assert cdata.trained_modules["mine"] is trained
    assert len(fits) == 1


def test_fit_or_reuse_restores_the_datamodule_after_batched_fit(cdata):
    original_batch_keys = set(cdata.datamodule.batch_keys)

    trained = fit_or_reuse(cdata, OnePassRecipe(batch_key="cell_type"), accelerator="cpu")

    assert set(cdata.datamodule.batch_keys) == original_batch_keys
    assert set(trained.batch_keys) == {"batch_index_n"}  # what the module needs in order to run
