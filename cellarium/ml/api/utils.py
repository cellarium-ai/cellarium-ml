# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

import ctypes
import difflib
import functools
import gc
import multiprocessing as mp
import shutil
import sys
import tempfile
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

import anndata
import h5py
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import Logger
from torch.utils._pytree import tree_leaves
from torch.utils.data import DataLoader
from tqdm import tqdm

T = TypeVar("T")

# Create a global placeholder for the process workers so they only authenticate once.
_GCS_FS = None


def get_gcs_fs():
    """Ensure each worker process only authenticates and opens a session ONCE."""
    global _GCS_FS
    if _GCS_FS is None:
        import gcsfs

        _GCS_FS = gcsfs.GCSFileSystem()
    return _GCS_FS


def resolve_device(accelerator: Literal["cpu", "mps", "cuda", "auto"]) -> torch.device:
    """The device for ``accelerator``: the best available of cuda, mps and cpu for ``"auto"``."""
    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(accelerator)


def release_unused_memory() -> None:
    """
    Give memory that this process has freed back to the operating system, as far as the platform allows.

    After passes over large data, Python and numpy leave freed memory in the allocator's heap, where it still counts
    against the process (and the machine) even though nothing refers to it. This collects garbage, then asks the C
    allocator to return what it can (``malloc_trim`` with glibc on Linux, ``malloc_zone_pressure_relief`` on macOS),
    and empties the cache of the GPU allocator (CUDA or MPS). It is best effort: where a step is not available (other
    platforms, or a C library without these functions) that step is skipped, and it never raises.
    """
    gc.collect()
    try:
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        elif sys.platform == "darwin":
            libsystem = ctypes.CDLL("libSystem.dylib")
            libsystem.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libsystem.malloc_zone_pressure_relief(None, 0)  # all zones, as much as possible
    except (OSError, AttributeError):
        pass  # not glibc (for example musl) or the function is missing: nothing to release this way
    try:
        if torch.cuda.is_initialized():
            # the workspaces that cuBLAS keeps for matrix products stay reserved otherwise (private function: it is
            # absent in some versions of torch)
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass  # a failure to empty a cache must never affect the caller


def _shutdown_loader_workers(loader: DataLoader) -> None:
    """
    Stop the worker processes that ``loader`` keeps alive between epochs (``persistent_workers``), and forget them, so
    that the loader starts new ones if it is iterated again. A loader whose dataset wraps another loader (the loaders
    of :class:`~cellarium.ml.api._rebatch.RebatchedDataset`) is followed to the one that has the workers.
    """
    inner = getattr(loader.dataset, "loader", None)
    if isinstance(inner, DataLoader):
        _shutdown_loader_workers(inner)
    iterator = getattr(loader, "_iterator", None)  # a private attribute of the DataLoader, set with persistent workers
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        loader._iterator = None  # type: ignore[attr-defined]


def shutdown_trainer_workers(trainer: pl.Trainer) -> None:
    """
    Stop the dataloader worker processes that ``trainer`` has left alive, as a fit that is over has no use for them.

    With ``persistent_workers`` the workers of a dataloader stay alive after its epochs, for as long as something refers
    to the dataloader, and the trainer does until it is garbage collected (which can be long after the fit, for
    example when the function that made it goes on to make another pass): each worker holds the tiles it read and
    what it allocated, and a later pass starts workers of its own. This is best effort, and never raises: the
    dataloaders are those that the trainer exposes, and stopping a persistent worker uses a private function of torch,
    so it does nothing where either is not available.
    """
    for name in ("train_dataloader", "val_dataloaders", "test_dataloaders", "predict_dataloaders"):
        try:
            loaders = tree_leaves(getattr(trainer, name))
        except Exception:
            continue
        for loader in loaders:
            if isinstance(loader, DataLoader):
                try:
                    _shutdown_loader_workers(loader)
                except Exception:
                    pass


def release_memory_after(function: Callable) -> Callable:
    """
    Decorate a function that makes a pass on an accelerator outside of a Trainer (a Trainer releases its memory when
    its fit ends): once the function has returned, or raised, and its locals (such as the last batch) are gone, call
    :func:`release_unused_memory`, so that the memory that the pass used is not left reserved.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        finally:
            release_unused_memory()

    return wrapper


class SeekableHTTPFile:
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self._cache = {}

    def read(self, size=-1):
        if size == -1:
            # Read from current position to end
            response = requests.get(self.url, headers={"Range": f"bytes={self.pos}-"})
            data = response.content
            self.pos += len(data)
            return data
        else:
            # Read specific number of bytes
            end_pos = self.pos + size - 1
            cache_key = (self.pos, end_pos)

            if cache_key not in self._cache:
                response = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end_pos}"})
                self._cache[cache_key] = response.content

            data = self._cache[cache_key]
            self.pos += len(data)
            return data

    def seek(self, pos, whence=0):
        if whence == 0:  # SEEK_SET
            self.pos = pos
        elif whence == 1:  # SEEK_CUR
            self.pos += pos
        elif whence == 2:  # SEEK_END
            # Get file size first
            response = requests.head(self.url)
            size = int(response.headers.get("content-length", 0))
            self.pos = size + pos
        return self.pos

    def tell(self):
        return self.pos

    def close(self):
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _h5py_read_n_obs(h5handle: h5py.File) -> int:
    idx_col = h5handle["obs"].attrs["_index"]
    try:
        n_obs = h5handle[f"obs/{idx_col}"].shape[0]
    except AttributeError:
        # can happen if somehow the obs index is saved as a categorical (not supposed to be allowed)
        n_obs = h5handle[f"obs/{idx_col}/codes"].shape[0]
    return n_obs


def _h5py_read_var_names(h5handle: h5py.File) -> np.ndarray:
    idx_col = h5handle["var"].attrs["_index"]
    try:
        var_names = h5handle[f"var/{idx_col}"][:]
    except AttributeError:
        # can happen if somehow the var index is saved as a categorical (not supposed to be allowed)
        var_names = h5handle[f"var/{idx_col}/categories"][:]
    return var_names


def _h5py_read_obs(h5handle: h5py.File) -> pd.DataFrame:
    return anndata.io.read_elem(h5handle["obs"])


def get_h5ad_file_n_cells(h5ad_path: str) -> int:
    """
    Get the number of cells in each h5ad file in a list of paths.
    """
    n_cells = _h5ad_file_read_elem(h5ad_path, fun=_h5py_read_n_obs)
    assert isinstance(n_cells, int), "Expected int from _h5py_read_n_obs"
    return n_cells


def get_h5ad_file_obs(h5ad_path: str) -> pd.DataFrame:
    """
    Read only the `obs` dataframe from a single h5ad file, loading as little of the file as possible.
    """
    obs = _h5ad_file_read_elem(h5ad_path, fun=_h5py_read_obs)
    assert isinstance(obs, pd.DataFrame), "Expected pd.DataFrame from _h5py_read_obs"
    return obs


def _h5ad_source(h5ad_paths: list[str]) -> str:
    """Where the h5ad files live, for progress bar labels: GCS, URLs, local files or mixed locations."""
    sources = {
        "GCS" if path.startswith("gs://") else "URLs" if path.startswith(("http://", "https://")) else "local files"
        for path in h5ad_paths
    }
    return sources.pop() if len(sources) == 1 else "mixed locations"


def _map_h5ad_files(fun: Callable[[str], T], h5ad_paths: Sequence[str], what: str) -> list[T]:
    """
    ``fun`` applied to each of ``h5ad_paths``, in order, using a process pool that terminates instantly when there
    are many files. ``fun`` must be picklable (a module-level function).
    """
    desc = f"Reading {what} from {_h5ad_source(list(h5ad_paths))}"
    # opt for a quick for loop if the number of files is small
    if len(h5ad_paths) < 40:
        return [fun(h5ad_path) for h5ad_path in tqdm(h5ad_paths, desc=desc, unit="file")]

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=12)

    try:
        # pool.imap preserves the order of the results, just like executor.map
        return list(tqdm(pool.imap(fun, h5ad_paths), total=len(h5ad_paths), desc=desc, unit="file"))
    finally:
        # as soon as we have our list, instantly nuke the worker processes.
        pool.terminate()
        pool.join()  # Wait for the OS to acknowledge they are dead (takes milliseconds)


def get_h5ad_files_n_cells(h5ad_paths: list[str]) -> list[int]:
    """
    Get the number of cells using a process pool that terminates instantly.
    """
    return _map_h5ad_files(get_h5ad_file_n_cells, h5ad_paths, "n_obs")


def _h5ad_paths_to_read(h5ad_paths: Sequence[str], nexus_extract_uniform_sizes: bool) -> list[str]:
    """The files whose sizes are read: all of them, or only the first and last of uniformly sized Nexus extracts."""
    return [h5ad_paths[0], h5ad_paths[-1]] if nexus_extract_uniform_sizes else list(h5ad_paths)


def _limits_from_n_cells(n_files: int, n_cells_read: Sequence[int], nexus_extract_uniform_sizes: bool) -> np.ndarray:
    """The limits of ``n_files`` files from the sizes of the files that :func:`_h5ad_paths_to_read` gives."""
    if nexus_extract_uniform_sizes:
        # every file but the last has the size of the first
        shard_sizes = [n_cells_read[0]] * (n_files - 1) + [n_cells_read[-1]]
    else:
        shard_sizes = list(n_cells_read)
    return np.cumsum(shard_sizes)


def get_h5ad_files_limits(h5ad_paths: list[str], nexus_extract_uniform_sizes: bool = False) -> np.ndarray:
    """
    Return the `limits` to be used in constructing a :class:`~cellarium.ml.data.DistributedAnnDataCollection`
    based on sizes of the provided h5ad files.
    """
    n_cells = get_h5ad_files_n_cells(_h5ad_paths_to_read(h5ad_paths, nexus_extract_uniform_sizes))
    return _limits_from_n_cells(len(h5ad_paths), n_cells, nexus_extract_uniform_sizes)


@dataclass(frozen=True)
class ObsColumnSchema:
    """
    How an h5ad file stores one ``obs`` column, as read from its HDF5 metadata (no values of the column are read).

    Attributes:
        encoding: The anndata encoding type: ``"array"``, ``"string-array"``, ``"categorical"``,
            ``"nullable-integer"``, ``"nullable-boolean"`` or ``"nullable-string-array"``.
        dtype: The type of the values (of the categories, for a categorical column): a numpy dtype name, or ``"str"``.
        categories: The categories of a categorical column, in their stored order (``None`` for other columns).
        ordered: Whether a categorical column is ordered.
    """

    encoding: str
    dtype: str
    categories: tuple[Hashable, ...] | None = None
    ordered: bool = False

    @property
    def is_categorical(self) -> bool:
        return self.encoding == "categorical"

    @property
    def is_numeric(self) -> bool:
        """Whether the column holds (possibly nullable) integers or floats."""
        if self.encoding not in ("array", "nullable-integer") or self.dtype == "str":
            return False
        return np.dtype(self.dtype).kind in "iuf"

    @property
    def type_key(self) -> tuple[str, str, bool]:
        """What must agree across files for the column to have the same type in each (categories aside)."""
        return (self.encoding, self.dtype, self.ordered)

    @property
    def categories_key(self) -> frozenset | tuple | None:
        """What must agree across files for a categorical column to have the same categories in each: their set, or
        their sequence for an ordered column (whose order is meaningful)."""
        if self.categories is None:
            return None
        return self.categories if self.ordered else frozenset(self.categories)

    def describe_type(self) -> str:
        if self.is_categorical:
            return f"{'ordered ' if self.ordered else ''}categorical of {self.dtype}"
        return self.dtype if self.encoding in ("array", "string-array") else f"{self.encoding} {self.dtype}"


@dataclass(frozen=True)
class H5adObsInfo:
    """The number of cells of an h5ad file, and how it stores each of its ``obs`` columns (in their stored order)."""

    n_obs: int
    columns: dict[str, ObsColumnSchema]


def _h5py_value_dtype(dataset: h5py.Dataset) -> str:
    return "str" if h5py.check_string_dtype(dataset.dtype) is not None else dataset.dtype.name


def _h5py_read_obs_column_schema(elem: h5py.Group | h5py.Dataset) -> ObsColumnSchema:
    encoding = str(elem.attrs.get("encoding-type", "array"))
    if encoding == "categorical":
        assert isinstance(elem, h5py.Group)
        return ObsColumnSchema(
            encoding,
            _h5py_value_dtype(elem["categories"]),
            categories=tuple(anndata.io.read_elem(elem["categories"]).tolist()),
            ordered=bool(elem.attrs.get("ordered", False)),
        )
    # the nullable encodings are groups of "values" and "mask"
    values = elem.get("values") if isinstance(elem, h5py.Group) else elem
    return ObsColumnSchema(encoding, _h5py_value_dtype(values) if isinstance(values, h5py.Dataset) else "")


def _h5py_read_obs_info(h5handle: h5py.File) -> H5adObsInfo:
    obs = h5handle["obs"]
    names = [str(name) for name in obs.attrs.get("column-order", [])]
    return H5adObsInfo(
        n_obs=_h5py_read_n_obs(h5handle),
        columns={name: _h5py_read_obs_column_schema(obs[name]) for name in names},
    )


def get_h5ad_file_obs_info(h5ad_path: str) -> H5adObsInfo:
    """
    The number of cells of an h5ad file and the schema of its ``obs`` columns, read from the HDF5 metadata (and the
    categories of the categorical columns) without reading the values of any column.
    """
    return _h5ad_file_read_elem(h5ad_path, fun=_h5py_read_obs_info)


def get_h5ad_files_obs_info(
    h5ad_paths: list[str], nexus_extract_uniform_sizes: bool = False
) -> tuple[np.ndarray, dict[str, H5adObsInfo]]:
    """
    The ``limits`` of the files (as :func:`get_h5ad_files_limits` gives them) and the
    :class:`H5adObsInfo` of each file that was read, by path, in one pass that opens each file once.

    With ``nexus_extract_uniform_sizes``, only the first and last files are read (the files of a Nexus extract all
    have the same size and schema), so only they are in the returned dict.
    """
    paths = _h5ad_paths_to_read(h5ad_paths, nexus_extract_uniform_sizes)
    infos = _map_h5ad_files(get_h5ad_file_obs_info, paths, "obs schemas")
    limits = _limits_from_n_cells(len(h5ad_paths), [info.n_obs for info in infos], nexus_extract_uniform_sizes)
    return limits, dict(zip(paths, infos))


def _list_preview(values: Sequence[Any], n: int = 10) -> str:
    """``values`` as a list for an error message, with at most ``n`` of them shown."""
    shown = ", ".join(repr(value) for value in values[:n])
    return f"[{shown}{f', ... ({len(values) - n} more)' if len(values) > n else ''}]"


def _files_preview(paths: Sequence[str], n: int = 3) -> str:
    """How many ``paths`` there are, and the names of the first ``n`` of them, for an error message."""
    names = ", ".join(os.path.basename(path) for path in paths[:n])
    count = f"{len(paths)} file{'s' if len(paths) != 1 else ''}"
    return f"{count} ({names}{', ...' if len(paths) > n else ''})"


def _group_paths(paths: Sequence[str], key: Callable[[str], Hashable]) -> dict[Hashable, list[str]]:
    """``paths`` grouped by ``key``, the groups and the paths within each in the order of ``paths``."""
    groups: dict[Hashable, list[str]] = {}
    for path in paths:
        groups.setdefault(key(path), []).append(path)
    return groups


def _obs_columns_problem(obs_infos: Mapping[str, H5adObsInfo]) -> str | None:
    """Why the files do not have the same obs columns (whatever their order), or ``None`` if they do."""
    paths = list(obs_infos)
    groups = _group_paths(paths, lambda path: frozenset(obs_infos[path].columns))
    if len(groups) == 1:
        return None
    reference = list(obs_infos[paths[0]].columns)
    lines = [
        f"The obs columns differ across the {len(paths)} h5ad files, but every file must have the same obs columns "
        "(their order does not matter)."
    ]
    for group_paths in groups.values():
        columns = obs_infos[group_paths[0]].columns
        missing = [column for column in reference if column not in columns]
        extra = [column for column in columns if column not in reference]
        if not missing and not extra:
            lines.append(
                f"  {_files_preview(group_paths)}: the {len(reference)} obs columns of the first file, "
                f"{_list_preview(reference)}"
            )
            continue
        differences = [
            *([f"without {len(missing)} of them, {_list_preview(missing)}"] if missing else []),
            *([f"with {len(extra)} more, {_list_preview(extra)}"] if extra else []),
        ]
        lines.append(f"  {_files_preview(group_paths)}: {'; '.join(differences)}")
    lines.append("Check that h5ad_paths lists only the files that you meant to load.")
    return "\n".join(lines)


def _obs_type_problems(obs_infos: Mapping[str, H5adObsInfo]) -> list[str]:
    """The obs columns whose type (the encoding and dtype) is not the same in every file, described."""
    paths = list(obs_infos)
    problems = []
    for column in obs_infos[paths[0]].columns:
        groups = _group_paths(paths, lambda path: obs_infos[path].columns[column].type_key)
        if len(groups) > 1:
            types = "; ".join(
                f"{obs_infos[group_paths[0]].columns[column].describe_type()} in {_files_preview(group_paths)}"
                for group_paths in groups.values()
            )
            problems.append(f"obs column {column!r} has different types: {types}")
    return problems


def _obs_category_problems(obs_infos: Mapping[str, H5adObsInfo]) -> tuple[list[str], list[str]]:
    """
    The categorical obs columns (of the same type in every file) whose categories are not the same in every file,
    described, and those of them that :func:`harmonize_obs_categories` can harmonize (the unordered ones).
    """
    paths = list(obs_infos)
    first = obs_infos[paths[0]]
    problems = []
    harmonizable = []
    for column, schema in first.columns.items():
        if not schema.is_categorical or any(obs_infos[p].columns[column].type_key != schema.type_key for p in paths):
            continue
        differing = [p for p in paths if obs_infos[p].columns[column].categories_key != schema.categories_key]
        if not differing:
            continue
        assert schema.categories is not None
        example = obs_infos[differing[0]].columns[column]
        assert example.categories is not None
        missing = [c for c in schema.categories if c not in example.categories]
        extra = [c for c in example.categories if c not in schema.categories]
        difference = (
            f"lacks {_list_preview(missing)} and has {_list_preview(extra)}"
            if missing or extra
            else "has them in a different order (the column is ordered, so the order matters)"
        )
        problems.append(
            f"obs column {column!r} has different categories in {_files_preview(differing)}: the first file "
            f"({os.path.basename(paths[0])}) has the {len(schema.categories)} categories "
            f"{_list_preview(list(schema.categories))}, and {os.path.basename(differing[0])} {difference}"
        )
        if not schema.ordered:
            harmonizable.append(column)
    return problems, harmonizable


def validate_h5ad_obs_schemas(obs_infos: Mapping[str, H5adObsInfo]) -> None:
    """
    Check that h5ad files have the same ``obs`` schema, as the files of a
    :class:`~cellarium.ml.data.DistributedAnnDataCollection` must: the same obs columns (in any order), each stored
    with the same type, and each categorical column with the same categories (in any order, unless the column is
    ordered).

    Args:
        obs_infos: The :class:`H5adObsInfo` of each file, by path (see :func:`get_h5ad_files_obs_info`).

    Raises:
        ValueError: If the files do not have the same obs schema. The message tells which columns and files differ, and
            how to harmonize the categories of categorical columns with :func:`harmonize_obs_categories`.
    """
    columns_problem = _obs_columns_problem(obs_infos)
    if columns_problem is not None:
        raise ValueError(columns_problem)
    category_problems, harmonizable = _obs_category_problems(obs_infos)
    problems = _obs_type_problems(obs_infos) + category_problems
    if not problems:
        return
    lines = [
        "The obs of the h5ad files do not agree: every file must store each obs column with the same type, and each "
        "categorical obs column with the same categories (their order does not matter, unless the column is ordered).",
        *(f"  - {problem}" for problem in problems),
    ]
    if harmonizable:
        lines.append(
            "To give the categorical columns the same categories in every file (the union of their categories, the "
            "files being rewritten in place), run:\n"
            f"    cellarium.ml.api.harmonize_obs_categories(h5ad_paths, columns={harmonizable!r})"
        )
    raise ValueError("\n".join(lines))


def check_obs_columns_exist(requested: Mapping[str, str], available: Collection[str]) -> None:
    """
    Check that the obs columns that arguments name are among the ``available`` obs columns.

    Args:
        requested: The obs column that each argument names, by a description of the argument (e.g.
            ``{"total_mrna_umis_column": "count"}``).
        available: The obs columns of the data.

    Raises:
        ValueError: If some of the columns do not exist, with the closest existing names as suggestions. (Not a
            ``KeyError``, which shows its message as a quoted repr, so that the lines of the message would not break.)
    """
    problems = []
    for argument, column in requested.items():
        if column in available:
            continue
        close = difflib.get_close_matches(column, list(available), n=3)
        suggestion = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        problems.append(f"{argument}={column!r} is not an obs column.{suggestion}")
    if problems:
        problems.append(f"The obs columns are: {_list_preview(sorted(available), n=50)}")
        raise ValueError("\n".join(problems))


def harmonize_obs_categories(h5ad_paths: Sequence[str], columns: Sequence[str] | None = None) -> dict[str, list]:
    """
    Give categorical ``obs`` columns the same categories in every h5ad file, rewriting the files in place, as
    :class:`~cellarium.ml.api.CellariumData` requires of its files (see :func:`validate_h5ad_obs_schemas`).

    The categories of a column become the union of its categories in all of the files, in the order in which they
    first appear (file by file, in the order of ``h5ad_paths``). The codes of each file are remapped, so every cell
    keeps its value (and a missing value stays missing). Only the files whose categories differ from the union are
    rewritten, and in them only the ``obs`` columns being harmonized.

    HDF5 does not reclaim the space of the data that is replaced, so a rewritten file grows by the size of the old
    columns (``h5repack`` reclaims it).

    Example::

        >>> harmonize_obs_categories(h5ad_paths, columns=["sample"])
        {'sample': ['sample_1', 'sample_2', 'sample_3']}

    Args:
        h5ad_paths: The local h5ad files.
        columns: The categorical obs columns to harmonize. If ``None``, all of the unordered categorical columns whose
            categories are not the same in every file.

    Returns:
        The categories that each harmonized column has now, by column.

    Raises:
        ValueError: If some files are not local, or a column is not categorical with the same type in every file, or
            is ordered with categories that are not the same in every file (the order of the union would be arbitrary).
    """
    remote = [path for path in h5ad_paths if path.startswith(("gs://", "http://", "https://"))]
    if remote:
        raise ValueError(f"Only local h5ad files can be rewritten, but {_files_preview(remote)} are not local.")
    paths = list(h5ad_paths)
    obs_infos = dict(zip(paths, _map_h5ad_files(get_h5ad_file_obs_info, paths, "obs schemas")))

    if columns is None:
        _, columns = _obs_category_problems(obs_infos)
    for column in columns:
        schemas = [obs_infos[path].columns.get(column) for path in paths]
        if any(schema is None or not schema.is_categorical for schema in schemas):
            raise ValueError(f"obs column {column!r} is not categorical in every file.")
        type_keys = {schema.type_key for schema in schemas if schema is not None}
        if len(type_keys) > 1:
            raise ValueError(f"obs column {column!r} does not have the same type in every file.")
        if schemas[0] is not None and schemas[0].ordered:
            if len({schema.categories for schema in schemas if schema is not None}) > 1:
                raise ValueError(
                    f"obs column {column!r} is ordered, and its categories differ across files, so they cannot be "
                    "harmonized automatically: rewrite the column with the categories in the order you want."
                )

    def categories(path: str, column: str) -> list:
        return list(obs_infos[path].columns[column].categories or ())

    unions = {column: list(dict.fromkeys(c for path in paths for c in categories(path, column))) for column in columns}
    for path in tqdm(paths, desc="Harmonizing obs categories", unit="file"):
        to_rewrite = [column for column in columns if categories(path, column) != unions[column]]
        if not to_rewrite:
            continue
        with h5py.File(path, "r+") as h5handle:
            obs = h5handle["obs"]
            for column in to_rewrite:
                values = anndata.io.read_elem(obs[column])
                assert isinstance(values, pd.Categorical)
                harmonized = values.set_categories(pd.Index(unions[column], dtype=values.categories.dtype))
                # written under a temporary name first, so that the column is never missing if this is interrupted
                temporary = f"__harmonizing__{column}"
                if temporary in obs:
                    del obs[temporary]
                anndata.io.write_elem(obs, temporary, harmonized)
                del obs[column]
                obs.move(temporary, column)
    return unions


def _write_obs_shard(args: tuple[str, str]) -> None:
    """Read `obs` from one h5ad file and write it to its own parquet shard file."""
    h5ad_path, shard_path = args
    obs = get_h5ad_file_obs(h5ad_path)
    table = pa.Table.from_pandas(obs, preserve_index=True)
    pq.write_table(table, shard_path)


def _unified_field(name: str, fields: list[pa.Field]) -> pa.Field:
    """
    Reconcile one column's field across shards. If every shard agrees, keep that type as-is
    (preserving a categorical/dictionary-encoded column's dtype in the common case where h5ad
    files were preprocessed to share identical categories). If shards disagree only in how many
    categories they saw -- e.g. two shards' pandas Categoricals differ in cardinality enough to
    cross an int8/int16/int32 dictionary-index-width boundary -- fall back to the shared
    underlying value type so the files can still merge. A true type disagreement (not explained by
    dictionary-encoding) is a real schema mismatch and is raised as an error.
    """
    types = {f.type for f in fields}
    if len(types) == 1:
        return fields[0]
    value_types = {t.value_type if pa.types.is_dictionary(t) else t for t in types}
    if len(value_types) == 1:
        return pa.field(name, next(iter(value_types)))
    raise ValueError(f"obs column {name!r} has incompatible types across h5ad files: {types}")


def _unified_schema(shard_schemas: list[pa.Schema], h5ad_paths: list[str]) -> pa.Schema:
    """
    The schema of the merged shards: their columns, in the order of the first shard (the shards may have them in any
    order), each of the type that :func:`_unified_field` reconciles.
    """
    first = shard_schemas[0]
    for schema, h5ad_path in zip(shard_schemas, h5ad_paths):
        if set(schema.names) != set(first.names):
            missing = [name for name in first.names if name not in schema.names]
            extra = [name for name in schema.names if name not in first.names]
            raise ValueError(
                f"obs columns are not identical across h5ad files: {h5ad_path!r} lacks {_list_preview(missing)} and "
                f"has {_list_preview(extra)}, compared with {h5ad_paths[0]!r}"
            )
    fields = [_unified_field(name, [schema.field(name) for schema in shard_schemas]) for name in first.names]
    return pa.schema(fields, metadata=first.metadata)


def _merge_parquet_shards(h5ad_paths: list[str], shard_paths: list[str], output_path: str) -> None:
    """
    Merge per-file parquet shards into a single parquet file, one shard at a time, so the full
    `obs` dataframe is never held in memory at once. Shards are merged in `shard_paths` order,
    which callers should keep aligned with the desired final row order.
    """
    # a cheap, metadata-only pass (no data read) to reconcile shard schemas before writing anything
    target_schema = _unified_schema([pq.ParquetFile(p).schema_arrow for p in shard_paths], h5ad_paths)

    writer = pq.ParquetWriter(output_path, target_schema)
    try:
        for h5ad_path, shard_path in tqdm(
            list(zip(h5ad_paths, shard_paths)), desc="Merging obs parquet shards", unit="file"
        ):
            # the columns in the order of the target schema, whatever their order in this file
            table = pq.read_table(shard_path).select(target_schema.names)
            try:
                table = table.cast(target_schema)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                raise ValueError(f"obs schema of {h5ad_path!r} is incompatible with the unified obs schema") from e
            writer.write_table(table)
    finally:
        writer.close()


def write_obs_parquet(h5ad_paths: list[str], output_path: str, processes: int = 12) -> None:
    """
    Read `obs` from each h5ad file in `h5ad_paths` and write it to a single local parquet file at
    `output_path`, using a process pool of workers that each read one file and write their own
    parquet shard, followed by a sequential, one-shard-at-a-time merge into the final file.

    Row order in the output matches the concatenation order of `h5ad_paths` (each file's rows in
    their original order), the same convention used by :func:`get_h5ad_files_limits`.

    All files are assumed to have the same `obs` columns (in any order: the output has those of the
    first file, in its order); a `ValueError` is raised if a mismatch is found while merging shards.
    """
    if not h5ad_paths:
        raise ValueError("h5ad_paths must be a non-empty list")

    desc = f"Reading obs from {_h5ad_source(h5ad_paths)}"
    tmpdir = tempfile.mkdtemp(prefix="cellarium_obs_parquet_")
    try:
        shard_paths = [os.path.join(tmpdir, f"{i:08d}.parquet") for i in range(len(h5ad_paths))]
        args = list(zip(h5ad_paths, shard_paths))

        # opt for a quick for loop if the number of files is small
        if len(h5ad_paths) < 40:
            for a in tqdm(args, desc=desc, unit="file"):
                _write_obs_shard(a)
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=processes)
            try:
                # imap_unordered: each worker writes directly to its own pre-assigned shard path,
                # so completion order doesn't matter and we avoid waiting on stragglers.
                list(
                    tqdm(
                        pool.imap_unordered(_write_obs_shard, args),
                        total=len(args),
                        desc=desc,
                        unit="file",
                    )
                )
            finally:
                # as soon as all shards are written, instantly nuke the worker processes.
                pool.terminate()
                pool.join()  # Wait for the OS to acknowledge they are dead (takes milliseconds)

        _merge_parquet_shards(h5ad_paths, shard_paths, output_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_h5ad_file_var_names_g(h5ad_path: str) -> np.ndarray:
    """
    Get var_names_g from an h5ad file.
    """
    var_names_g = _h5ad_file_read_elem(h5ad_path, fun=_h5py_read_var_names)
    assert isinstance(var_names_g, np.ndarray), "Expected numpy array from _h5py_read_var_names"
    return var_names_g.astype(str)


def _h5ad_file_read_elem(h5ad_path: str, fun: Callable[[h5py.File], T]) -> T:
    """
    Read info from an h5ad file, loading as little of it as possible.
    """

    def _gcloud_version(h5ad_path: str) -> T:
        fs = get_gcs_fs()
        # with fs.open(h5ad_path, "rb") as f:
        with fs.open(h5ad_path, "rb", block_size=65536, cache_type="blockcache") as f:
            with h5py.File(f) as h5handle:
                return fun(h5handle)

    def _local_version(h5ad_path: str) -> T:
        with h5py.File(h5ad_path, "r") as h5handle:
            return fun(h5handle)

    def _url_version(h5ad_path: str) -> T:
        """Optimized version that streams only the needed parts of the file"""
        with SeekableHTTPFile(h5ad_path) as f:
            with h5py.File(f, "r") as h5handle:
                return fun(h5handle)

    if h5ad_path.startswith("gs://"):
        out = _gcloud_version(h5ad_path)
    elif h5ad_path.startswith("http://") or h5ad_path.startswith("https://"):
        out = _url_version(h5ad_path)
    else:
        out = _local_version(h5ad_path)

    return out


def h5ad_paths_from_google_bucket(gs_bucket_path: str) -> list[str]:
    """
    Helper function to get h5ad file paths from a Google Cloud Storage bucket, like a Cellarium Nexus curriculum
    """
    if not gs_bucket_path.startswith("gs://"):
        raise ValueError("Invalid Google Cloud Storage bucket path -- must start with 'gs://'")
    fs = get_gcs_fs()
    paths = fs.ls(gs_bucket_path[5:])
    return [f"gs://{path}" for path in paths if path.endswith(".h5ad")]


class PreciseProgressBar(TQDMProgressBar):
    """
    A :class:`~lightning.pytorch.callbacks.TQDMProgressBar` that shows float metrics with ``significant_digits``
    significant digits. tqdm shows numbers with 3 (e.g. ``1.52e+3``), which hides small changes in large values
    such as a loss in the thousands.
    """

    def __init__(self, *args: Any, significant_digits: int = 5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.significant_digits = significant_digits

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, int | str | float | dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        # tqdm leaves strings as they are
        return {k: f"{v:.{self.significant_digits}g}" if isinstance(v, float) else v for k, v in metrics.items()}


class LossHistory(Logger):
    """
    An in-memory Lightning logger that records every metric the module logs, at the cadence it is logged, with no
    files written. Pass it as ``logger=`` to :class:`lightning.pytorch.Trainer` (and set ``log_every_n_steps``, which
    controls how often per-step metrics such as ``train_loss`` are recorded; epoch-level metrics such as ``val_loss``
    are recorded once per epoch). Lightning skips the pre-training validation sanity check, so it is not recorded.

    The record is :attr:`history`, a long-format table with columns ``step``, ``epoch``, ``metric`` and ``value``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._steps: list[int | None] = []
        self._epochs: list[int | None] = []
        self._metrics: list[str] = []
        self._values: list[float] = []

    @property
    def name(self) -> str:
        return "loss_history"

    @property
    def version(self) -> str:
        return ""

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        metrics = dict(metrics)
        epoch = metrics.pop("epoch", None)
        for metric, value in metrics.items():
            self._steps.append(step)
            self._epochs.append(None if epoch is None else int(epoch))
            self._metrics.append(metric)
            self._values.append(float(value))

    @property
    def history(self) -> pd.DataFrame:
        """The logged metrics as a long-format table with columns ``step``, ``epoch``, ``metric``, ``value``."""
        return pd.DataFrame(
            {"step": self._steps, "epoch": self._epochs, "metric": self._metrics, "value": self._values}
        )
