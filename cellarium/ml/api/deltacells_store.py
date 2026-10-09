# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from cellarium.ml.api._view_collections import _obs_frame
from cellarium.ml.api.utils import get_gcs_fs
from cellarium.ml.data.distributed_deltacells import _import_deltacells

_GCS_PREFIX = "gs://"
_MAX_COUNT = 65535  # the largest count that deltacells can store


def _check_local_h5ad_paths(h5ad_paths: Sequence[str]) -> list[str]:
    if isinstance(h5ad_paths, str) or len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths must be a non-empty list of paths")
    for path in h5ad_paths:
        if "://" in path:
            raise ValueError(
                f"{path!r} is not a local file: create_deltacells_dataset reads local h5ad files only. "
                "Download them first (for example with `gcloud storage cp`)."
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return list(h5ad_paths)


def _upload_directory(local_dir: str, remote_dir: str, fs: Any, max_workers: int = 8) -> None:
    """Copy ``local_dir`` to ``remote_dir`` (without the ``gs://`` prefix) with the manifest last, so that a dataset
    that is only partly uploaded has no manifest and is recognised as incomplete."""
    from deltacells.manifest import MANIFEST_NAME as manifest_name

    names = sorted(
        os.path.relpath(os.path.join(root, f), local_dir).replace(os.sep, "/")
        for root, _, files in os.walk(local_dir)
        for f in files
    )
    if manifest_name not in names:
        raise RuntimeError(f"The conversion did not write {manifest_name} in {local_dir}")

    def put(name: str) -> None:
        fs.put_file(os.path.join(local_dir, name), f"{remote_dir}/{name}")

    others = [n for n in names if n != manifest_name]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(tqdm(pool.map(put, others), total=len(others), desc="Uploading", unit="file"))
    put(manifest_name)


def create_deltacells_dataset(
    h5ad_paths: Sequence[str],
    output: str,
    tile_size: int = 10_000,
    obs_exclude: Sequence[str] = (),
    sort_genes: bool = True,
    overwrite: bool = False,
    threads: int | None = None,
    workers: int | None = None,
    sort_genes_max_files: int | None = 10,
    staging_dir: str | None = None,
    log: Callable[[str], None] | None = print,
    filesystem: Any = None,
    **convert_kwargs: Any,
) -> str:
    """
    Create a `deltacells <https://github.com/cellarium-ai/cellarium-ml/tree/main/deltacells>`_ dataset from h5ad files,
    for fast training with :meth:`~cellarium.ml.api.CellariumData.from_deltacells`.

    The cells are stored in the order of ``h5ad_paths`` (and the order within each file), so
    ``CellariumData(h5ad_paths)`` and ``CellariumData.from_deltacells(output)`` see the same cells in the same order.
    All files must have the same ``var_names`` and ``X`` must hold integer counts up to 65535. The genes are stored in a
    different order (most expressed first, which compresses much better); the ``obs`` columns, ``obs_names`` and ``var``
    are stored too (``CellariumData`` needs the ``obs_names``, so ``obs`` and ``obs_names`` cannot be turned off).

    The files can be larger than memory (a single file of tens of GB is fine): ``X`` is read from disk a tile at a time,
    in parallel worker processes, so it should be stored as CSR (dense ``X`` works but is much slower and heavier).
    The cells are **not shuffled**: a tile holds consecutive cells, so if the files are sorted by something (donor,
    tissue, ...) shuffle the cells across the files first, for example with a backed shuffling tool such as
    ``annslicer``.

    The h5ad files must be local files. ``output`` is a local directory or a ``gs://bucket/prefix``; for the latter the
    dataset is written to a local staging directory first and then uploaded (the manifest last), which needs ``gcsfs``.

    Args:
        h5ad_paths: Local h5ad files, in cell order.
        output: Local directory or ``gs://bucket/prefix`` to create. It must be empty or not exist unless ``overwrite``.
        tile_size: Cells per tile (the unit of reading and of shuffling); ``10_000`` is a good default.
        obs_exclude: ``obs`` columns not to store.
        sort_genes: Store the genes by decreasing total counts. The counts are summed over up to
            ``sort_genes_max_files`` files.
        overwrite: Replace an existing dataset at ``output``.
        threads: Compression threads in each worker (default: the cores divided by ``workers``).
        workers: Worker processes that read, encode and write tiles (default: the number of cores, limited so that the
            tiles being written fit in about 70% of the available memory; a worker needs roughly 50 bytes per nonzero
            of a tile).
        sort_genes_max_files: Files, spread evenly over ``h5ad_paths``, that the gene order is computed from (all of
            them if there are no more, or if ``None``). One pass over ``X`` of the files used, so with thousands of
            (shuffled) files a few are enough.
        staging_dir: Where to write the dataset before uploading it to ``gs://`` (default: the system temp directory).
            The dataset is a fraction of the size of the h5ad files but can still be large.
        log: Called with progress messages (``None`` to be silent).
        filesystem: An fsspec filesystem to upload with instead of ``gcsfs`` (for testing).
        **convert_kwargs: Further arguments of :func:`deltacells.convert.convert_h5ad`, for example ``level``,
            ``max_categories`` or ``var=False``.

    Returns:
        ``output``.
    """
    _import_deltacells()  # a clear error if deltacells is missing
    from deltacells.convert import convert_h5ad

    files = _check_local_h5ad_paths(h5ad_paths)
    convert = dict(
        obs=True,  # the api reads the obs_names of every batch
        obs_names=True,
        tile_size=tile_size,
        obs_exclude=obs_exclude,
        sort_genes=sort_genes,
        threads=threads,
        workers=workers,
        sort_genes_max_files=sort_genes_max_files,
        log=log,
        **convert_kwargs,
    )
    # a path is a literal here, not a glob pattern; the order of the list is the order of the cells
    patterns = [glob.escape(f) for f in files]

    def write(directory: str, overwrite: bool) -> None:
        convert_h5ad(patterns, directory, sort_files=False, overwrite=overwrite, **convert)

    write_dataset(output, write, overwrite=overwrite, staging_dir=staging_dir, filesystem=filesystem)
    return output


def write_dataset(
    output: str,
    write: Callable[[str, bool], None],
    overwrite: bool = False,
    staging_dir: str | None = None,
    filesystem: Any = None,
) -> None:
    """
    Create a dataset at ``output``, a local directory or a ``gs://bucket/prefix``, by calling ``write(directory,
    overwrite)``, which must write the dataset to the local ``directory`` (replacing one that is there if
    ``overwrite``). For ``gs://`` the dataset is written to a local staging directory first and then uploaded, the
    manifest last, replacing what is at ``output`` if ``overwrite``.
    """
    if not output.startswith(_GCS_PREFIX):
        write(output, overwrite)
        return

    fs = filesystem if filesystem is not None else get_gcs_fs()
    remote_dir = output[len(_GCS_PREFIX) :].rstrip("/")
    if fs.exists(remote_dir) and fs.ls(remote_dir) and not overwrite:
        raise FileExistsError(f"{output} is not empty; pass overwrite=True to replace it")
    staging = tempfile.mkdtemp(prefix="cellarium_deltacells_", dir=staging_dir)
    try:
        local = os.path.join(staging, "dataset")
        write(local, False)
        if fs.exists(remote_dir):
            fs.rm(remote_dir, recursive=True)
        _upload_directory(local, remote_dir, fs)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def write_collection_to_deltacells(
    collection: Any,
    output: str,
    tile_size: int = 10_000,
    sort_genes: bool = True,
    sort_genes_max_tiles: int = 10,
    overwrite: bool = False,
    staging_dir: str | None = None,
    filesystem: Any = None,
    **writer_kwargs: Any,
) -> str:
    """
    Write all the cells of ``collection`` (a :class:`~cellarium.ml.data.DistributedCollection`), in order, as a
    deltacells dataset at ``output`` (see :func:`write_dataset`), in tiles of ``tile_size`` cells with their ``obs``
    (every column, and the ``obs_names``) and ``var``. This is one pass over the cells, plus one over up to
    ``sort_genes_max_tiles`` tiles spread over them, to sort the genes by decreasing total counts (better
    compression). ``X`` must hold integer counts from 0 to 65535. If it does not, or the writing fails otherwise,
    nothing is left at a local ``output``. ``writer_kwargs`` are passed to :class:`deltacells.DatasetWriter`.

    Returns:
        ``output``.
    """
    _import_deltacells()  # a clear error if deltacells is missing
    from deltacells.obs import ObsSchema
    from deltacells.writer import DatasetWriter

    n = len(collection)
    tile_starts = list(range(0, n, tile_size))

    def read(start: int) -> tuple[sp.csr_matrix, pd.DataFrame]:
        batch = collection.read(np.arange(start, min(start + tile_size, n)))
        x = sp.csr_matrix(batch.X)
        if x.nnz and (x.data.min() < 0 or x.data.max() > _MAX_COUNT or np.any(x.data != np.round(x.data))):
            raise ValueError(
                f"The tile of cells {start} to {start + x.shape[0]} does not hold integer counts from 0 to "
                f"{_MAX_COUNT}, which deltacells requires."
            )
        obs = _obs_frame(batch)
        obs["obs_names"] = np.asarray(obs.index, dtype=str)
        return x, obs.reset_index(drop=True)

    _, first_obs = read(0)
    schema = ObsSchema.infer([first_obs])
    gene_order = None
    if sort_genes:
        totals = np.zeros(collection.n_vars)
        for start in tile_starts[:: max(1, len(tile_starts) // sort_genes_max_tiles)][:sort_genes_max_tiles]:
            totals += np.asarray(read(start)[0].sum(axis=0)).ravel()
        gene_order = np.argsort(-totals, kind="stable")

    def write(directory: str, overwrite: bool) -> None:
        # refuses (FileExistsError) to write into a directory that has something in it unless `overwrite`, before
        # touching it, so that a failure here leaves what was there alone
        writer = DatasetWriter(
            directory,
            n_genes=collection.n_vars,
            tile_size=tile_size,
            gene_order=gene_order,
            var_names=[str(name) for name in collection.var_names],
            var=collection.var,
            obs_schema=schema,
            overwrite=overwrite,
            **writer_kwargs,
        )
        try:
            with writer:
                for start in tqdm(tile_starts, desc="Writing tiles", unit="tile"):
                    writer.add_tile(*read(start))
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)  # a directory without a manifest is not a dataset
            raise

    write_dataset(output, write, overwrite=overwrite, staging_dir=staging_dir, filesystem=filesystem)
    return output
