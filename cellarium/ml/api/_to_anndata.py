# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import TYPE_CHECKING, Any, Literal, Sequence

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from cellarium.ml import CellariumPipeline
from cellarium.ml.api._view_collections import RamCollection, load_into_ram, unwrap_collection
from cellarium.ml.api.utils import release_unused_memory
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.models import IncrementalPCA, SingleCellVariationalInference
from cellarium.ml.utilities.data import collate_fn

if TYPE_CHECKING:
    from cellarium.ml.api.cellariumdata import CellariumData

# the trained models whose prediction is an embedding of the cells
_EMBEDDING_MODELS = (IncrementalPCA, SingleCellVariationalInference)

# arguments of `geometric_sketch` that `to_anndata` sets itself
_RESERVED_SKETCH_KWARGS = ("target_n_cells", "seed", "return_new_adata")


def _check_selection_arguments(
    downsample_method: str | None, target_n_cells: int | None, geometric_sketch_kwargs: dict[str, Any] | None
) -> None:
    if downsample_method not in (None, "random", "geometric_sketch"):
        raise ValueError(f"downsample_method must be None, 'random' or 'geometric_sketch', got {downsample_method!r}.")
    if downsample_method is None:
        if target_n_cells is not None:
            raise ValueError("target_n_cells needs a downsample_method ('random' or 'geometric_sketch').")
    else:
        if target_n_cells is None:
            raise ValueError(f"downsample_method={downsample_method!r} needs target_n_cells.")
        if isinstance(target_n_cells, bool) or not isinstance(target_n_cells, (int, np.integer)) or target_n_cells < 1:
            raise ValueError(f"target_n_cells must be a positive integer, got {target_n_cells!r}.")
    if geometric_sketch_kwargs:
        if downsample_method != "geometric_sketch":
            raise ValueError("geometric_sketch_kwargs only apply to downsample_method='geometric_sketch'.")
        reserved = [name for name in _RESERVED_SKETCH_KWARGS if name in geometric_sketch_kwargs]
        if reserved:
            raise ValueError(
                f"{reserved} cannot be given in geometric_sketch_kwargs: to_anndata sets them "
                "(from its own target_n_cells and seed)."
            )


def _embedding_keys(cdata: "CellariumData", collect_embeddings: bool | str | Sequence[str]) -> tuple[list[str], bool]:
    """The keys of ``cdata.trained_modules`` to embed the cells with, and whether they were asked for one by one."""
    trained_modules = cdata.trained_modules
    if collect_embeddings is True:
        keys = [key for key, trained in trained_modules.items() if isinstance(trained.module.model, _EMBEDDING_MODELS)]
        return keys, False
    if collect_embeddings is False:
        return [], True
    keys = [collect_embeddings] if isinstance(collect_embeddings, str) else list(collect_embeddings)
    for key in keys:
        if not isinstance(key, str):
            raise TypeError(f"collect_embeddings must be a bool or the keys of trained modules, got {key!r}.")
        if key not in trained_modules:
            available = list(trained_modules)
            raise ValueError(
                f"collect_embeddings: no trained module '{key}' in cdata.trained_modules. "
                + (f"Available: {available}." if available else "None have been trained yet.")
            )
        if not isinstance(trained_modules[key].module.model, _EMBEDDING_MODELS):
            raise ValueError(
                f"collect_embeddings: the module '{key}' is a {type(trained_modules[key].module.model).__name__}, "
                f"which is not one of the embedding models ({[m.__name__ for m in _EMBEDDING_MODELS]})."
            )
    return keys, True


def _default_embedding_module_key(cdata: "CellariumData") -> str | None:
    """The key of the PCA to sketch with (``"pca"``, if there is one), or ``None`` after a warning if there is none."""
    pcas = [key for key, trained in cdata.trained_modules.items() if isinstance(trained.module.model, IncrementalPCA)]
    if pcas:
        return "pca" if "pca" in pcas else pcas[0]
    warnings.warn(
        "No PCA has been trained on these cells, so the geometric sketch is based on a random projection of the "
        "normalized, log-transformed counts. A PCA embedding may give a better sketch: compute one first with "
        "cml.tl.pca(cdata, ...), or pass geometric_sketch_kwargs={'embedding_module_key': ...}."
    )
    return None


def _with_schema_categories(obs: pd.DataFrame, collection: Any) -> pd.DataFrame:
    """
    ``obs`` with the categories of the categorical columns of h5ad files in the order of their schema, which is the
    order that the batches of the datamodule have, so that the codes of a column are those that modules were trained
    with. (Other collections already keep the categories of their columns.)
    """
    root = unwrap_collection(collection)
    if not isinstance(root, DistributedAnnDataCollection):
        return obs
    for column, dtype in root.schema.attr_values["obs"].dtypes.items():
        if column not in obs or not isinstance(dtype, pd.CategoricalDtype):
            continue
        if isinstance(obs[column].dtype, pd.CategoricalDtype):
            obs[column] = obs[column].cat.set_categories(dtype.categories, ordered=dtype.ordered)
        else:
            obs[column] = pd.Categorical(obs[column], dtype=dtype)
    return obs


def _embed(cdata: "CellariumData", key: str, ram: RamCollection, chunk_size: int = 10_000) -> np.ndarray:
    """The embedding of the cells in ``ram`` by the trained module ``key``, run over them in chunks on the CPU."""
    # `using` makes the data fields that the module needs (such as scVI's batch column) part of the batch keys
    with cdata.using(key) as trained:
        batch_keys = dict(cdata.datamodule.batch_keys)
        module = trained.module
        module.configure_model()  # idempotent; makes its transforms and model accessible
        pipeline = CellariumPipeline([*(module.cpu_transforms or []), *module.transforms, module.model])
        was_training = module.training
        module.eval()  # inference only: BatchNorm uses (and does not update) its running statistics
        try:
            parts = []
            with torch.no_grad():
                for start in tqdm(range(0, ram.n_obs, chunk_size), desc=f"Embedding cells with '{key}'", unit="chunk"):
                    cells = ram[start : start + chunk_size]
                    batch = tree_map(lambda field: field(cells), batch_keys)
                    parts.append(torch.as_tensor(pipeline.predict(collate_fn([batch]))["x_ng"]).cpu().numpy())
        finally:
            module.train(was_training)
    return np.concatenate(parts)


def _obsm_value(value: Any, positions: np.ndarray, names: pd.Index) -> Any:
    """The rows ``positions`` of an ``obsm`` entry of ``cdata``, as something that AnnData takes."""
    from cellarium.ml.api.cellariumdata import _subset_rows

    subset = _subset_rows(value, positions)
    if isinstance(subset, pd.DataFrame):
        subset.index = names
    elif isinstance(subset, pd.Series):
        subset = subset.to_numpy()
    elif isinstance(subset, torch.Tensor):
        subset = subset.numpy()
    return subset


def to_anndata(
    cdata: "CellariumData",
    downsample_method: Literal["random", "geometric_sketch"] | None,
    target_n_cells: int | None,
    geometric_sketch_kwargs: dict[str, Any] | None,
    collect_embeddings: bool | str | Sequence[str],
    seed: int,
    max_gb: float | None,
) -> anndata.AnnData:
    """The implementation of :meth:`CellariumData.to_anndata`, which documents the arguments."""
    from cellarium.ml.api.cellariumdata import _per_cell_series
    from cellarium.ml.api.tools.geometric_sketch import geometric_sketch

    _check_selection_arguments(downsample_method, target_n_cells, geometric_sketch_kwargs)
    embedding_keys, explicit = _embedding_keys(cdata, collect_embeddings)
    kept_keys = []
    for key in embedding_keys:
        if f"X_{key}" in cdata.obsm:
            message = f"obsm['X_{key}'] already exists, so the embedding of the module '{key}' cannot be added to it"
            if explicit:
                raise ValueError(f"{message}: rename it or leave '{key}' out of collect_embeddings.")
            warnings.warn(f"{message}; the embedding is left out.")
        else:
            kept_keys.append(key)
    embedding_keys = kept_keys

    n_obs = cdata.n_obs
    all_columns = list(cdata.obs.columns)
    obs_computed = dict(cdata.obs_computed)  # as they are before the sketch adds its own
    clashing = [f"computed_{key}" for key in obs_computed if f"computed_{key}" in all_columns]
    if clashing:
        raise ValueError(f"obs already has the columns {clashing}, which obs_computed would be added to.")
    max_bytes = cdata._max_bytes(max_gb)

    if downsample_method is not None and target_n_cells >= n_obs:  # type: ignore[operator]
        downsample_method = None  # every cell is kept, so there is nothing to sample or sketch

    # which cells, and (for a sketch, which has them already) their counts
    x: sp.csr_matrix | None = None
    sketch_names: pd.Index | None = None
    if downsample_method is None:
        positions = np.arange(n_obs, dtype=np.int64)
    elif downsample_method == "random":
        assert target_n_cells is not None
        positions = np.sort(np.random.default_rng(seed).choice(n_obs, size=target_n_cells, replace=False))
    else:
        assert target_n_cells is not None
        sketch_kwargs = dict(geometric_sketch_kwargs or {})
        if "embedding_module_key" not in sketch_kwargs:
            sketch_kwargs["embedding_module_key"] = _default_embedding_module_key(cdata)
        sketch = geometric_sketch(
            cdata, target_n_cells=target_n_cells, return_new_adata=True, seed=seed, **sketch_kwargs
        )
        sketch_adata = sketch["adata"]
        assert isinstance(sketch_adata, anndata.AnnData)
        # the rows of the sketch are in the order that the cells streamed by: put them in the order of the data
        found = cdata.obs.index.get_indexer(sketch_adata.obs_names)
        if (found < 0).any():
            raise RuntimeError("The geometric sketch has cells that are not in the data.")
        order = np.argsort(found, kind="stable")
        positions = found[order].astype(np.int64)
        x = sp.csr_matrix(sketch_adata.X[order], dtype=np.float32)
        sketch_names = sketch_adata.obs_names[order]
        # the counts are in x now: do not hold the sketch's copy of them (nor the sketch module) for the rest of this
        del sketch, sketch_adata

    selected = cdata if len(positions) == n_obs else cdata[positions]
    if x is None:
        x = load_into_ram(selected.datamodule.dadc, max_bytes).x
    if x.shape[1] != len(cdata.var):
        raise RuntimeError(f"The counts have {x.shape[1]} genes but var has {len(cdata.var)}.")

    obs = _with_schema_categories(selected.obs[list(selected.obs.columns)], selected.datamodule.dadc)
    obs.index = pd.Index(obs.index.astype(str), name=None)
    if sketch_names is not None and not obs.index.equals(sketch_names.astype(str)):
        raise RuntimeError("The cells of the geometric sketch do not match the rows of obs.")
    names = obs.index
    for key, value in obs_computed.items():
        series = _per_cell_series(value, n_obs, f"obs_computed['{key}']")
        obs[f"computed_{key}"] = series.iloc[positions].set_axis(names)

    var = cdata.var.copy()
    var["highly_variable"] = (
        False if cdata.hvg is None else cdata.hvg.reindex(var.index).fillna(False).astype(bool).to_numpy()
    )

    obsm = {key: _obsm_value(value, positions, names) for key, value in cdata.obsm.items()}
    if embedding_keys:
        dadc = selected.datamodule.dadc
        ram = RamCollection(x, obs, dadc.var, dadc.var_names)
        for key in embedding_keys:
            try:
                obsm[f"X_{key}"] = _embed(cdata, key, ram)
            except Exception as error:
                if explicit:
                    raise
                warnings.warn(f"The embedding of the module '{key}' could not be computed and is left out: {error}")
    adata = anndata.AnnData(X=x, obs=obs, var=var, obsm=obsm)
    # the passes over the data above leave a lot of freed memory in the heap: give it back before the caller goes on
    release_unused_memory()
    return adata
