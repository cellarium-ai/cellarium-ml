# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from deltacells.convert import convert_h5ad
from deltacells.obs import ObsSchema
from deltacells.writer import DatasetWriter
from tqdm import tqdm

from cellarium.ml.api._view_collections import _obs_frame

_GCS_PREFIX = "gs://"
_MAX_COUNT = 2**31 - 1  # the largest count that deltacells can store


def _check_h5ad_paths(h5ad_paths: Sequence[str]) -> list[str]:
    """Check that ``h5ad_paths`` are local files that exist or ``gs://`` paths (those are checked when they are read, in
    parallel: checking thousands of them one by one here would be slow)."""
    if isinstance(h5ad_paths, str) or len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths must be a non-empty list of paths")
    for path in h5ad_paths:
        if "://" in path:
            if not path.startswith(_GCS_PREFIX):
                raise ValueError(f"{path!r} is not supported: h5ad files must be local paths or gs:// paths")
        elif not os.path.isfile(path):
            raise FileNotFoundError(path)
    return list(h5ad_paths)


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
    All files must have the same ``var_names`` and ``X`` must hold non-negative integer counts (up to 2**31 - 1; counts
    above 65535 are stored exactly, but a tile in which more than about 0.5% of the nonzeros are that large is an error:
    it is not raw counts). The genes are stored in a
    different order (most expressed first, which compresses much better); the ``obs`` columns, ``obs_names`` and ``var``
    are stored too (``CellariumData`` needs the ``obs_names``, so ``obs`` and ``obs_names`` cannot be turned off).

    The files can be larger than memory (a single file of tens of GB is fine): ``X`` is read from disk a tile at a time,
    in parallel worker processes, so it should be stored as CSR (dense ``X`` works but is much slower and heavier).
    The cells are **not shuffled**: a tile holds consecutive cells, so if the files are sorted by something (donor,
    tissue, ...) shuffle the cells across the files first, for example with a backed shuffling tool such as
    ``annslicer``.

    The h5ad files and ``output`` can be local or in Google Cloud Storage (``gs://...``; needs ``gcsfs``). The
    credentials are the Google *Application Default Credentials*, not the login of ``gcloud`` / ``gsutil``: on a
    Compute Engine VM run ``gcloud auth application-default login`` if the VM's service account cannot write to
    ``output`` (a permission problem is reported before the files are read).

    The dataset never has to fit on the local disk. Each ``gs://`` h5ad file is downloaded to
    ``staging_dir`` just before it is needed and deleted when it is done, and each tile of a ``gs://`` output is
    uploaded as soon as it is written (the manifest last) and then deleted. So ``staging_dir`` needs room for the h5ad
    files that are in use at the same time, roughly ``5 * workers`` of them, plus a few tiles. List the ``gs://`` files
    one by one (they are not expanded). The files in GCS are only ever read, never changed or deleted.

    ``overwrite`` deletes the dataset at ``output`` first (also an incomplete one left by a failed run); if ``output``
    has anything else in it, or an input file is inside it, this raises instead and nothing is deleted. If the
    conversion fails part way, what was uploaded stays in a ``gs://`` ``output`` (without a manifest, so it is not a
    usable dataset): run again with ``overwrite=True``, or delete it yourself.

    Args:
        h5ad_paths: h5ad files (local paths or ``gs://`` paths), in cell order.
        output: Local directory or ``gs://bucket/prefix`` to create. It must be empty or not exist unless ``overwrite``.
        tile_size: Cells per tile (the unit of reading and of shuffling); ``10_000`` is a good default.
        obs_exclude: ``obs`` columns not to store.
        sort_genes: Store the genes by decreasing total counts. The counts are summed over up to
            ``sort_genes_max_files`` files.
        overwrite: Replace an existing dataset at ``output`` (see above).
        threads: Compression threads in each worker (default: the cores divided by ``workers``).
        workers: Worker processes that read, encode and write tiles (default: the number of cores, limited so that the
            tiles being written fit in about 70% of the available memory; a worker needs roughly 50 bytes per nonzero
            of a tile).
        sort_genes_max_files: Files, spread evenly over ``h5ad_paths``, that the gene order is computed from (all of
            them if there are no more, or if ``None``). One pass over ``X`` of the files used, so with thousands of
            (shuffled) files a few are enough.
        staging_dir: Scratch directory for the ``gs://`` h5ad files being read and the tiles waiting to be uploaded
            (default: the system temp directory). It is not used if everything is local.
        log: Called with progress messages (``None`` to be silent).
        filesystem: An fsspec filesystem to use for ``gs://`` paths instead of ``gcsfs`` (for testing).
        **convert_kwargs: Further arguments of :func:`deltacells.convert.convert_h5ad`, for example ``level`` (the zstd
            level, ``19`` by default: the smallest files, but slow to write; ``12`` is about four times faster and 6%
            larger), ``max_categories`` or ``var=False``, ``progress_interval`` (seconds between the progress lines
            with the speed and the time left, 60 by default) and ``scan_workers`` (the processes that read the
            metadata of the files first; they are started with ``spawn``, so a script that calls this needs an
            ``if __name__ == "__main__":`` guard).

    Returns:
        ``output``.
    """
    files = _check_h5ad_paths(h5ad_paths)
    convert = dict(
        obs=True,  # the api reads the obs_names of every batch
        obs_names=True,
        tile_size=tile_size,
        obs_exclude=obs_exclude,
        sort_genes=sort_genes,
        threads=threads,
        workers=workers,
        sort_genes_max_files=sort_genes_max_files,
        overwrite=overwrite,
        staging_dir=staging_dir,
        filesystem=filesystem,
        log=log,
        **convert_kwargs,
    )
    # a local path is a literal here, not a glob pattern (a gs:// path never is); the order of the list is the order of
    # the cells
    patterns = [f if f.startswith(_GCS_PREFIX) else glob.escape(f) for f in files]
    convert_h5ad(patterns, output, sort_files=False, **convert)
    return output


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
    deltacells dataset at ``output`` (a local directory or ``gs://bucket/prefix``), in tiles of ``tile_size`` cells with
    their ``obs`` (every column, and the ``obs_names``) and ``var``. This is one pass over the cells, plus one over up
    to ``sort_genes_max_tiles`` tiles spread over them, to sort the genes by decreasing total counts (better
    compression). ``X`` must hold integer counts from 0 to 2**31 - 1. If it does not, or the writing fails otherwise,
    nothing is left at a local ``output``, and a ``gs://`` ``output`` has no manifest. A ``gs://`` ``output`` is
    uploaded tile by tile as it is written (the manifest last), through a small scratch directory in ``staging_dir``;
    ``overwrite`` deletes only the files of a deltacells dataset that are at ``output`` (and raises if there is
    anything else). ``writer_kwargs`` are passed to :class:`deltacells.DatasetWriter`.

    Returns:
        ``output``.
    """
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

    # refuses (FileExistsError) to write into an output that has something in it unless `overwrite`, and deletes only
    # the files of a deltacells dataset, so that a failure before this leaves what was there alone
    with DatasetWriter(
        output,
        n_genes=collection.n_vars,
        tile_size=tile_size,
        gene_order=gene_order,
        var_names=[str(name) for name in collection.var_names],
        var=collection.var,
        obs_schema=schema,
        overwrite=overwrite,
        scratch_dir=staging_dir,
        filesystem=filesystem,
        **writer_kwargs,
    ) as writer:  # on an error this leaves nothing at a local output (and no manifest at a remote one)
        for start in tqdm(tile_starts, desc="Writing tiles", unit="tile"):
            writer.add_tile(*read(start))
    return output
