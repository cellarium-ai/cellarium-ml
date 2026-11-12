# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Saving a :class:`~cellarium.ml.api.CellariumData` to a directory, and opening it again: :func:`save_session` and
:func:`load_session`, which :meth:`CellariumData.save` and :meth:`CellariumData.load` call.

The cell counts are not copied. The directory holds the arguments that opened the data, so that loading opens it again
(from the same place, or from another that has the same cells), and what was computed from it:

* ``manifest.pkl``: the arguments, what identifies the cells, and the keys of the rest. It is written last, so a
  directory without it is an unfinished save;
* ``obs_computed.parquet``, as :meth:`CellariumData.save_labels` writes it;
* ``obsm/<i>.npy``: one array each, which loading memory maps rather than reads;
* ``hvg.parquet``;
* ``modules/<i>.ckpt`` and ``modules/<i>.meta.pt``: each trained module as a Lightning checkpoint (read with
  :meth:`CellariumModule.load_from_checkpoint`, with no trainer involved), and the record of its training.

A view is saved as the data it descends from (its ``root``) and the positions of its cells in it.
"""

import pickle
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.api._obsm_storage import RowSubset
from cellarium.ml.api.cellariumdata import CellariumData, CellariumDataView, TrainedModule

FORMAT_VERSION = 1
_MANIFEST = "manifest.pkl"
_CHUNK_BYTES = 64 * 2**20  # the rows of an obsm array are copied to its file this much at a time


def _backing_file(value: Any) -> Path | None:
    """The file that an on-disk array (a memory map, or a :class:`RowSubset` of one) reads from."""
    base = value._base if isinstance(value, RowSubset) else value
    filename = getattr(base, "filename", None)
    return None if filename is None else Path(filename).resolve()


def _edge_positions(root: CellariumData) -> np.ndarray:
    """The first and last cell of each shard (file, or tile) of the data."""
    limits = np.asarray(root.datamodule.dadc.limits, dtype=np.int64)
    return np.unique(np.concatenate([[0], limits[:-1], limits - 1]))


def _fingerprint(root: CellariumData) -> dict[str, Any]:
    """What identifies the cells of the data: how many there are, the names of the cells at the shard edges, and the
    cell IDs (``None`` if they are the positions, which they are unless the data was made by ``to_deltacells``)."""
    ids = root.cell_ids
    positions = _edge_positions(root)
    return {
        "n_obs": root.n_obs,
        "edge_positions": positions,
        "edge_names": root._obs._names_at(positions).astype(str).tolist(),
        "cell_ids": None if isinstance(ids, pd.RangeIndex) else ids.to_numpy(),
    }


def _check_fingerprint(root: CellariumData, saved: dict[str, Any], where: str) -> None:
    """Raise a ``ValueError`` if the cells of ``root`` are not those that the fingerprint ``saved`` is of."""
    if root.n_obs != saved["n_obs"]:
        raise ValueError(
            f"The data at {where} has {root.n_obs} cells, but the session was saved with {saved['n_obs']}. "
            "Pass h5ad_paths= or deltacells_uri= to load the session onto the data that it was saved with."
        )
    positions = saved["edge_positions"]
    names = root._obs._names_at(positions).astype(str).tolist()
    for position, expected, found in zip(positions, saved["edge_names"], names):
        if expected != found:
            raise ValueError(
                f"The data at {where} is not the data that the session was saved with: cell {position} is "
                f"'{found}' there, and was '{expected}'. Pass h5ad_paths= or deltacells_uri= to load the session "
                "onto the data that it was saved with."
            )
    expected_ids = np.arange(root.n_obs) if saved["cell_ids"] is None else saved["cell_ids"]
    if not np.array_equal(root.cell_ids.to_numpy(), expected_ids):
        raise ValueError(
            f"The cell IDs of the data at {where} are not those that the session was saved with (for example, "
            "the data is a deltacells dataset made by to_deltacells, and was saved as the data it was made from, "
            "or the other way around)."
        )


def _check_picklable(value: Any, what: str, hint: str) -> None:
    try:
        pickle.dumps(value)
    except Exception as error:
        raise ValueError(f"{what} cannot be pickled, so it cannot be saved ({error}). {hint}") from error


def _save_array(value: Any, path: Path) -> None:
    """Write ``value`` (an array, or an on-disk array) to the ``.npy`` file ``path``, a chunk of rows at a time."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype, shape=tuple(value.shape))
    step = max(1, _CHUNK_BYTES // max(1, value.nbytes // max(1, len(value))))
    for start in range(0, len(value), step):
        out[start : start + step] = value[start : start + step]
    out.flush()
    del out


def _save_module(trained: TrainedModule, directory: Path, i: int) -> None:
    module = trained.module
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": pl.__version__,
    }
    # the hook of the model keeps its own state in the checkpoint (that of CellariumModule needs a trainer: not called)
    on_save_checkpoint = getattr(module.model, "on_save_checkpoint", None)
    if callable(on_save_checkpoint):
        on_save_checkpoint(checkpoint)
    torch.save(checkpoint, directory / f"{i}.ckpt")
    meta = {
        "history": trained.history,
        "config": trained.config,
        "n_epochs": trained.n_epochs,
        "complete": trained.complete,
        "batch_keys": trained.batch_keys,
    }
    torch.save(meta, directory / f"{i}.meta.pt")


def save_session(cdata: CellariumData, path: str, overwrite: bool) -> None:
    """The implementation of :meth:`CellariumData.save`, which documents the arguments."""
    target = Path(path).expanduser().resolve()
    root = cdata.root

    # everything that can be wrong is found before anything is written or removed
    for key, value in cdata.obsm.items():
        file = _backing_file(value)
        if file is not None and target in file.parents:
            raise ValueError(
                f"obsm['{key}'] is read from {target}, which this would overwrite. Save to a different directory "
                f"(for example cdata.save('{target}_2')), and delete {target} afterwards if you no longer need it."
            )
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists. Pass overwrite=True to replace it.")
        if not target.is_dir() or (any(target.iterdir()) and not (target / _MANIFEST).exists()):
            raise ValueError(f"{target} exists and is not a saved session, so it is not replaced.")
    for key, value in cdata.obsm.items():
        if not isinstance(value, (np.ndarray, RowSubset)) or value.ndim < 1 or value.dtype.hasobject:
            raise TypeError(
                f"obsm['{key}'] is a {type(value).__name__}, but only numeric arrays can be saved. "
                "Delete it from obsm, or convert it to an array."
            )
    init_kwargs = {**root._init_kwargs, "max_gb": cdata.max_gb}
    _check_picklable(
        init_kwargs["obs_columns"],
        "obs_columns",
        "Define its functions at the top level of a module (not as lambdas or inside functions), or pass "
        "obs_columns= to load instead.",
    )
    for key, trained in cdata.trained_modules.items():
        _check_picklable(
            (trained.config, trained.batch_keys),
            f"The configuration of trained_modules['{key}']",
            "Its batch_keys have functions that cannot be pickled.",
        )

    obsm_keys = list(cdata.obsm)
    module_keys = list(cdata.trained_modules)
    manifest = {
        "format_version": FORMAT_VERSION,
        "init_kwargs": init_kwargs,
        "fingerprint": _fingerprint(root),
        "root_indices": cdata._root_indices,
        "obsm": obsm_keys,
        "trained_modules": module_keys,
        "current_embeddings": [key for key in module_keys if cdata._cached_embedding_state(key) == "current"],
    }

    if target.exists() and overwrite:
        shutil.rmtree(target)
    (target / "obsm").mkdir(parents=True)
    (target / "modules").mkdir()
    if cdata.obs_computed:
        cdata.save_labels(list(cdata.obs_computed), str(target / "obs_computed.parquet"))
    for i, key in enumerate(obsm_keys):
        _save_array(cdata.obsm[key], target / "obsm" / f"{i}.npy")
    if cdata.hvg is not None:
        cdata.hvg.rename("highly_variable").to_frame().to_parquet(target / "hvg.parquet")
    for i, key in enumerate(module_keys):
        _save_module(cdata.trained_modules[key], target / "modules", i)
    with open(target / _MANIFEST, "wb") as manifest_file:
        pickle.dump(manifest, manifest_file)


def _open_root(kwargs: dict[str, Any]) -> CellariumData:
    where = kwargs["deltacells_uri"] if kwargs["h5ad_paths"] is None else kwargs["h5ad_paths"]
    locations = [where] if isinstance(where, str) else list(where)
    missing = [location for location in locations if "://" not in location and not Path(location).exists()]
    if missing:
        raise ValueError(
            f"The data that the session was saved with is not there: {missing[:5]}"
            f"{' (and more)' if len(missing) > 5 else ''} does not exist. "
            "Pass h5ad_paths= or deltacells_uri= to load the session onto the data from another location."
        )
    try:
        return CellariumData(**kwargs)
    except Exception as error:
        raise ValueError(
            f"Could not open the data at {where}: {error} "
            "Pass h5ad_paths= or deltacells_uri= to load the session onto the data from another location."
        ) from error


def load_session(
    path: str,
    h5ad_paths: list[str] | None,
    deltacells_uri: str | None,
    deltacells_kwargs: dict[str, Any] | None,
    obs_parquet_path: str | None,
    obs_columns: dict[str, tuple[str, Callable]] | None,
    batch_size: int | None,
    datamodule_num_workers: int | None,
    max_gb: float | None,
) -> CellariumData:
    """The implementation of :meth:`CellariumData.load`, which documents the arguments."""
    directory = Path(path).expanduser()
    if not (directory / _MANIFEST).exists():
        raise ValueError(f"{directory} is not a saved session (it has no {_MANIFEST}), or its saving did not finish.")
    with open(directory / _MANIFEST, "rb") as manifest_file:
        manifest = pickle.load(manifest_file)
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"{directory} has format version {manifest['format_version']}, but this version reads {FORMAT_VERSION}."
        )

    kwargs = dict(manifest["init_kwargs"])
    if h5ad_paths is not None or deltacells_uri is not None:  # (passing both is for the constructor to refuse)
        kwargs.update(h5ad_paths=h5ad_paths, deltacells_uri=deltacells_uri)
        if h5ad_paths is not None and deltacells_kwargs is None:
            kwargs["deltacells_kwargs"] = None
    overrides = dict(
        deltacells_kwargs=deltacells_kwargs,
        obs_parquet_path=obs_parquet_path,
        obs_columns=obs_columns,
        batch_size=batch_size,
        datamodule_num_workers=datamodule_num_workers,
        max_gb=max_gb,
    )
    kwargs.update({name: value for name, value in overrides.items() if value is not None})

    root = _open_root(kwargs)
    where = kwargs["deltacells_uri"] if kwargs["h5ad_paths"] is None else kwargs["h5ad_paths"]
    _check_fingerprint(root, manifest["fingerprint"], str(where))
    cdata = root if manifest["root_indices"] is None else CellariumDataView(root, manifest["root_indices"])

    if (directory / "obs_computed.parquet").exists():
        cdata.load_labels(str(directory / "obs_computed.parquet"))
    for i, key in enumerate(manifest["obsm"]):
        cdata.obsm[key] = np.load(directory / "obsm" / f"{i}.npy", mmap_mode="r")
    if (directory / "hvg.parquet").exists():
        cdata.hvg = pd.read_parquet(directory / "hvg.parquet")["highly_variable"]
    for i, key in enumerate(manifest["trained_modules"]):
        module = CellariumModule.load_from_checkpoint(directory / "modules" / f"{i}.ckpt", map_location="cpu")
        meta = torch.load(directory / "modules" / f"{i}.meta.pt", map_location="cpu", weights_only=False)
        cdata.trained_modules[key] = TrainedModule(module=module, **meta)
    for key in manifest["current_embeddings"]:
        cdata._embedding_sources[f"X_{key}"] = (cdata.trained_modules[key], cdata.obsm[f"X_{key}"])
    return cdata
