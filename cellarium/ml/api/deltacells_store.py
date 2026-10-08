# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from cellarium.ml.api.utils import get_gcs_fs
from cellarium.ml.data.distributed_deltacells import _import_deltacells

_GCS_PREFIX = "gs://"


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
    remote = output.startswith(_GCS_PREFIX)
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

    if not remote:
        convert_h5ad(patterns, output, sort_files=False, overwrite=overwrite, **convert)
        return output

    fs = filesystem if filesystem is not None else get_gcs_fs()
    remote_dir = output[len(_GCS_PREFIX) :].rstrip("/")
    if fs.exists(remote_dir) and fs.ls(remote_dir) and not overwrite:
        raise FileExistsError(f"{output} is not empty; pass overwrite=True to replace it")
    staging = tempfile.mkdtemp(prefix="cellarium_deltacells_", dir=staging_dir)
    try:
        local = os.path.join(staging, "dataset")
        convert_h5ad(patterns, local, sort_files=False, **convert)
        if fs.exists(remote_dir):
            fs.rm(remote_dir, recursive=True)
        _upload_directory(local, remote_dir, fs)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output
