# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Recipes for the statistics models that api functions fit over all the genes of ``cdata.datamodule``: the
:class:`~cellarium.ml.models.OnePassMeanVarStd` that the highly variable gene flavors ``"seurat"`` and ``"kotliar"``
and the z-scoring of :func:`~cellarium.ml.api.tools.pca` read, and the :class:`~cellarium.ml.models.HVGSeuratV3`.

A recipe says everything that determines what such a model computes. It builds the module to fit, and it is the key
under which the fitted module is kept in the private cache of the :class:`~cellarium.ml.api.CellariumData`, so that
asking for the same statistics again (a different ``n_top_genes``, another HVG flavor, another PCA) does not read the
data again. The models do not know what is later done with their statistics (the HVG flavor and ``n_top_genes``, for
example), so those are not part of a recipe.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal

import lightning.pytorch as pl
from torch import nn

from cellarium.ml import CellariumModule
from cellarium.ml.api._datamodule_context import temporary_batch_keys
from cellarium.ml.api.cellariumdata import CellariumData, TrainedModule, fit_and_register
from cellarium.ml.models import HVGSeuratV3, OnePassMeanVarStd
from cellarium.ml.transforms import Densify, Log1p, NormalizeTotal
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes


def _batch_keys(batch_key: str | None) -> dict[str, AnnDataField]:
    if batch_key is None:
        return {}
    return {"batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes)}


@dataclass(frozen=True)
class OnePassRecipe:
    """
    A :class:`~cellarium.ml.models.OnePassMeanVarStd` over the library-size normalized (and optionally log1p
    transformed) counts of all genes, with per-``batch_key`` statistics if ``batch_key`` is given. The pooled
    ``mean_g`` and ``std_g`` do not depend on the batches.

    The defaults of ``target_count`` and ``eps`` are those of :class:`~cellarium.ml.transforms.NormalizeTotal`.
    """

    target_count: int = 10_000
    eps: float = 1e-6
    log1p: bool = False
    batch_key: str | None = None

    def cpu_transforms(self) -> list[nn.Module]:
        """The transforms to run on the cpu, in front of any gene :class:`~cellarium.ml.transforms.Filter`."""
        return [NormalizeTotal(target_count=self.target_count, eps=self.eps)]

    def transforms(self) -> list[nn.Module]:
        """The transforms to run after those of :meth:`cpu_transforms`."""
        return [Densify(), *([Log1p()] if self.log1p else [])]

    def batch_keys(self) -> dict[str, AnnDataField]:
        """The data fields beyond ``cdata.datamodule.batch_keys`` that the module needs in order to run."""
        return _batch_keys(self.batch_key)

    def build_module(self, cdata: CellariumData) -> CellariumModule:
        n_batch = 1 if self.batch_key is None else cdata.datamodule.obs_key_nunique(self.batch_key)
        return CellariumModule(
            cpu_transforms=self.cpu_transforms(),
            transforms=self.transforms(),
            model=OnePassMeanVarStd(
                var_names_g=cdata.datamodule.var_names_g,
                algorithm="shifted_data",
                n_batch=n_batch,
                output_path=None,
            ),
        )

    def is_served_by(self, other: Any) -> bool:
        """
        Whether a module fitted for the recipe ``other`` can stand in for this one: every field but ``batch_key`` is
        the same, and either this recipe has no ``batch_key`` or it is the same.
        """
        return (
            isinstance(other, OnePassRecipe)
            and (other.target_count, other.eps, other.log1p) == (self.target_count, self.eps, self.log1p)
            and (self.batch_key is None or other.batch_key == self.batch_key)
        )

    def fingerprint(self) -> str:
        """A short description of the statistics, for the record of what was computed."""
        steps = [f"normalize_total(target_count={self.target_count}, eps={self.eps})"]
        if self.log1p:
            steps.append("log1p")
        batches = "" if self.batch_key is None else f" by {self.batch_key}"
        return " > ".join(steps) + batches


@dataclass(frozen=True)
class SeuratV3Recipe:
    """A :class:`~cellarium.ml.models.HVGSeuratV3` over the counts of all genes, per ``batch_key`` if it is given."""

    batch_key: str | None = None

    def batch_keys(self) -> dict[str, AnnDataField]:
        """The data fields beyond ``cdata.datamodule.batch_keys`` that the module needs in order to run."""
        return _batch_keys(self.batch_key)

    def build_module(self, cdata: CellariumData, n_top_genes: int) -> CellariumModule:
        """
        ``n_top_genes`` is what the model precomputes when it is fit; the fitted statistics do not depend on it, so
        the result can be read for any number of genes.
        """
        n_batch = 1 if self.batch_key is None else cdata.datamodule.obs_key_nunique(self.batch_key)
        return CellariumModule(
            transforms=[Densify()],
            model=HVGSeuratV3(
                var_names_g=cdata.datamodule.var_names_g,
                n_top_genes=n_top_genes,
                use_batch_key=self.batch_key is not None,
                n_batch=n_batch,
                output_path=None,
            ),
        )

    def is_served_by(self, other: Any) -> bool:
        """The batch structure changes what the model computes, so only the same recipe serves."""
        return self == other

    def fingerprint(self) -> str:
        """A short description of the statistics, for the record of what was computed."""
        return "hvg_seurat_v3" + ("" if self.batch_key is None else f" by {self.batch_key}")


def fit_or_reuse(
    cdata: CellariumData,
    recipe: OnePassRecipe | SeuratV3Recipe,
    accelerator: Literal["cpu", "mps", "cuda", "auto"],
    key_added: str | None = None,
    **build_kwargs: Any,
) -> TrainedModule:
    """
    The trained module for ``recipe``: one already fit for the recipe (or one that stands in for it, see
    ``recipe.is_served_by``) if ``cdata`` has it, else a new one fit over ``cdata.datamodule``. New fits are kept
    for later calls.

    If ``key_added`` is given, the trained module is also stored as ``cdata.trained_modules[key_added]``. Without
    it the module is only in the private cache, so it does not replace anything the user can see.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        recipe: The statistics to fit.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].
        key_added: The key under which to store the trained module in ``cdata.trained_modules``, replacing any
            module already there under that key.
        **build_kwargs: Passed to ``recipe.build_module`` when a new module is fit.
    """
    trained = cdata._fit_cache.find(recipe)
    if trained is None:
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        batch_keys = recipe.batch_keys()
        ctx = temporary_batch_keys(cdata.datamodule, batch_keys) if batch_keys else nullcontext()
        with ctx:
            trained = fit_and_register(
                cdata,
                trainer,
                recipe.build_module(cdata, **build_kwargs),  # type: ignore[arg-type]
                key_added,
                config=asdict(recipe),
                batch_keys=batch_keys,
            )
        cdata._fit_cache[recipe] = trained
    elif key_added is not None:
        cdata.trained_modules[key_added] = trained
    return trained
