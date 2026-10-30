# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

import ctypes
import functools
import gc
import multiprocessing as mp
import shutil
import sys
import tempfile
from collections.abc import Mapping
from typing import Any, Callable, Literal

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
from tqdm import tqdm

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


def get_h5ad_files_n_cells(h5ad_paths: list[str]) -> list[int]:
    """
    Get the number of cells using a process pool that terminates instantly.
    """
    desc = f"Reading n_obs from {_h5ad_source(h5ad_paths)}"
    # opt for a quick for loop if the number of files is small
    if len(h5ad_paths) < 40:
        return [get_h5ad_file_n_cells(h5ad_path) for h5ad_path in tqdm(h5ad_paths, desc=desc, unit="file")]

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=12)

    try:
        # pool.imap preserves the order of the results, just like executor.map
        results = list(
            tqdm(
                pool.imap(get_h5ad_file_n_cells, h5ad_paths),
                total=len(h5ad_paths),
                desc=desc,
                unit="file",
            )
        )
        return results
    finally:
        # as soon as we have our list, instantly nuke the worker processes.
        pool.terminate()
        pool.join()  # Wait for the OS to acknowledge they are dead (takes milliseconds)


def get_h5ad_files_limits(h5ad_paths: list[str], nexus_extract_uniform_sizes: bool = False) -> np.ndarray:
    """
    Return the `limits` to be used in constructing a :class:`~cellarium.ml.data.DistributedAnnDataCollection`
    based on sizes of the provided h5ad files.
    """
    if nexus_extract_uniform_sizes:
        # we can look at just the first and last file
        first_and_last_shard_size = get_h5ad_files_n_cells([h5ad_paths[0], h5ad_paths[-1]])
        shard_sizes = [first_and_last_shard_size[0]] * (len(h5ad_paths) - 1) + [first_and_last_shard_size[1]]
    else:
        shard_sizes = get_h5ad_files_n_cells(h5ad_paths)
    limits = np.cumsum(shard_sizes)
    return limits


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


def _unified_schema(shard_schemas: list[pa.Schema]) -> pa.Schema:
    first = shard_schemas[0]
    if any(schema.names != first.names for schema in shard_schemas):
        raise ValueError("obs columns are not identical across h5ad files")
    fields = [_unified_field(name, [schema.field(name) for schema in shard_schemas]) for name in first.names]
    return pa.schema(fields, metadata=first.metadata)


def _merge_parquet_shards(h5ad_paths: list[str], shard_paths: list[str], output_path: str) -> None:
    """
    Merge per-file parquet shards into a single parquet file, one shard at a time, so the full
    `obs` dataframe is never held in memory at once. Shards are merged in `shard_paths` order,
    which callers should keep aligned with the desired final row order.
    """
    # a cheap, metadata-only pass (no data read) to reconcile shard schemas before writing anything
    target_schema = _unified_schema([pq.ParquetFile(p).schema_arrow for p in shard_paths])

    writer = pq.ParquetWriter(output_path, target_schema)
    try:
        for h5ad_path, shard_path in tqdm(
            list(zip(h5ad_paths, shard_paths)), desc="Merging obs parquet shards", unit="file"
        ):
            table = pq.read_table(shard_path)
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

    All files are assumed to share an identical `obs` schema; a `ValueError` is raised if a
    mismatch is found while merging shards.
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


def _h5ad_file_read_elem(
    h5ad_path: str, fun: Callable[[h5py.File], int | np.ndarray | pd.DataFrame]
) -> np.ndarray | int | pd.DataFrame:
    """
    Read info from an h5ad file, loading as little of it as possible.
    """

    def _gcloud_version(h5ad_path: str) -> int | np.ndarray | pd.DataFrame:
        fs = get_gcs_fs()
        # with fs.open(h5ad_path, "rb") as f:
        with fs.open(h5ad_path, "rb", block_size=65536, cache_type="blockcache") as f:
            with h5py.File(f) as h5handle:
                return fun(h5handle)

    def _local_version(h5ad_path: str) -> int | np.ndarray | pd.DataFrame:
        with h5py.File(h5ad_path, "r") as h5handle:
            return fun(h5handle)

    def _url_version(h5ad_path: str) -> int | np.ndarray | pd.DataFrame:
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
