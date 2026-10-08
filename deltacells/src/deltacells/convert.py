# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert h5ad shards (any cell counts per file) into a deltacells dataset of exact-size tiles."""

from __future__ import annotations

import glob
import multiprocessing
import os
import re
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import accumulate

import numpy as np
import scipy.sparse as sp

from deltacells.manifest import DEFAULT_OBS_TILES_PER_FILE, Manifest
from deltacells.writer import DEFAULT_CHUNKS, DEFAULT_LEVEL, DatasetWriter, write_tile_file

try:  # optional: needed only to store obs / var
    from deltacells.obs import DEFAULT_MAX_CATEGORIES, ObsSchema
except ImportError:  # pragma: no cover
    DEFAULT_MAX_CATEGORIES, ObsSchema = 20_000, None  # type: ignore[assignment,misc]


def natural_sort_key(path: str) -> list[int | str]:
    """Sort ``shard_2.h5ad`` before ``shard_10.h5ad``."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", path)]


def resolve_files(patterns: Sequence[str] | str, sort: bool = True) -> list[str]:
    """Expand glob patterns / paths into a de-duplicated list of files.

    With ``sort`` the files are naturally sorted. Without it the order of ``patterns`` is kept (the files matched by one
    pattern are still sorted), so an explicit list of paths gives the cells in the order of the list. A pattern that
    matches nothing raises ``FileNotFoundError``.
    """
    files: dict[str, None] = {}
    for pattern in [patterns] if isinstance(patterns, str) else patterns:
        matched = sorted(glob.glob(pattern, recursive=True), key=natural_sort_key)
        if not matched:
            raise FileNotFoundError(f"no files matched {pattern!r}")
        files.update(dict.fromkeys(matched))
    return sorted(files, key=natural_sort_key) if sort else list(files)


# A worker's peak memory is about this many bytes per nonzero of its tile plus a fixed base (see ``_default_workers``).
_BYTES_PER_NNZ = 48
_WORKER_BASE_BYTES = 400 * 2**20
_MEMORY_FRACTION = 0.7  # of the available memory that the workers may use


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return os.cpu_count() or 1


def _available_memory() -> int | None:
    """Bytes of memory new processes can use: ``MemAvailable`` limited by the cgroup limit, if any. ``None`` if unknown."""
    available: int | None = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        pass
    if available is None:
        try:
            available = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):  # pragma: no cover
            return None
    for limit_file in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(limit_file) as f:
                available = min(available, int(f.read()))
        except (OSError, ValueError):  # no such file, or "max" (no limit)
            continue
    return available


def _default_workers(n_jobs: int, job_bytes: float) -> int:
    """The number of worker processes: the cores, limited by how many jobs of ``job_bytes`` fit in memory."""
    workers = min(_cpu_count(), n_jobs)
    available = _available_memory()
    if available is not None:
        workers = min(workers, int(_MEMORY_FRACTION * available // job_bytes))
    return max(1, workers)


def _mp_context():
    # fork is cheap and does not re-import (or re-run) the calling script; the workers only use h5py, numpy and scipy
    return multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else None)


def _evenly_spaced(n: int, k: int | None) -> list[int]:
    """``k`` indices spread evenly over ``range(n)`` (all of them if ``k`` is ``None`` or at least ``n``)."""
    if k is None or k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _x_work_bytes(path: str) -> int:
    """Estimated working memory for processing all the rows of ``X`` in ``path`` at once (to size the workers)."""
    import h5py

    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):  # dense: the block itself and its sparse copy (assuming 30% nonzeros)
            n, g = x.shape
            return int(n * g * (x.dtype.itemsize + 0.3 * _BYTES_PER_NNZ))
        return int(x["indptr"][-1]) * _BYTES_PER_NNZ


def _read_rows(path: str, start: int, end: int) -> sp.csr_matrix:
    """Rows ``start:end`` of ``X`` in an h5ad file as CSR, reading only those rows from disk (``X`` may be CSR or dense)."""
    import h5py

    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            return sp.csr_matrix(x[start:end])
        fmt = str(x.attrs.get("encoding-type", x.attrs.get("h5sparse_format")))
        if fmt not in ("csr_matrix", "csr"):
            raise ValueError(f"X of {path} is stored as {fmt}; only CSR and dense X are supported")
        shape = x.attrs.get("shape", x.attrs.get("h5sparse_shape"))
        indptr = x["indptr"][start : end + 1].astype(np.int64)
        lo, hi = int(indptr[0]), int(indptr[-1])
        return sp.csr_matrix((x["data"][lo:hi], x["indices"][lo:hi], indptr - lo), shape=(end - start, int(shape[1])))


def _tile_pieces(i: int, tile_size: int, offsets: Sequence[int], files: Sequence[str]) -> list[tuple[str, int, int]]:
    """The ``(file, first row, last row + 1)`` pieces that tile ``i`` is made of; ``offsets`` are the first cells of the files."""
    lo, hi = i * tile_size, min((i + 1) * tile_size, offsets[-1])
    k = bisect_right(offsets, lo) - 1
    pieces = []
    while lo < hi:
        end = min(hi, offsets[k + 1])
        if end > lo:
            pieces.append((files[k], lo - offsets[k], end - offsets[k]))
        lo, k = end, k + 1
    return pieces


def _column_sums_job(job: tuple[str, int, int, int]) -> np.ndarray:
    path, start, end, n_genes = job
    x = _read_rows(path, start, end)
    return np.bincount(x.indices, weights=x.data, minlength=n_genes)


def _gene_totals(files: Sequence[str], n_obs: Sequence[int], n_genes: int, step: int, workers: int) -> np.ndarray:
    """Total counts of every gene over ``files``, reading ``step`` rows at a time (in ``workers`` processes)."""
    jobs = [(f, s, min(s + step, n), n_genes) for f, n in zip(files, n_obs) for s in range(0, n, step)]
    totals = np.zeros(n_genes, dtype=np.float64)
    if workers == 1 or len(jobs) == 1:
        for job in jobs:
            totals += _column_sums_job(job)
    else:
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=_mp_context()) as pool:
            for part in pool.map(_column_sums_job, jobs):
                totals += part
    return totals


_TILE_JOB: dict = {}  # the settings of a tile-writing worker, set by ``_init_tile_worker``


def _init_tile_worker(root: str, gene_order: np.ndarray | None, n_chunks: int, level: int, threads: int) -> None:
    _TILE_JOB.update(root=root, gene_order=gene_order, n_chunks=n_chunks, level=level, threads=threads)


def _write_tile_job(job: tuple[int, list[tuple[str, int, int]]]) -> tuple[int, int, int]:
    """Read the pieces of one tile, encode it and write its file; returns ``(n_cells, nnz, n_bytes)``."""
    index, pieces = job
    parts = [_read_rows(*piece) for piece in pieces]
    x = parts[0] if len(parts) == 1 else sp.vstack(parts, format="csr")
    del parts
    return write_tile_file(
        _TILE_JOB["root"],
        index,
        x,
        gene_order=_TILE_JOB["gene_order"],
        n_chunks=_TILE_JOB["n_chunks"],
        level=_TILE_JOB["level"],
        threads=_TILE_JOB["threads"],
    )


def convert_h5ad(
    paths: Sequence[str] | str,
    output: str,
    *,
    tile_size: int = 10_000,
    sort_genes: bool = True,
    sort_files: bool = True,
    n_chunks: int = DEFAULT_CHUNKS,
    level: int = DEFAULT_LEVEL,
    threads: int | None = None,
    workers: int | None = None,
    sort_genes_max_files: int | None = 10,
    overwrite: bool = False,
    obs: bool = True,
    obs_names: bool = True,
    obs_exclude: Sequence[str] = (),
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    obs_tiles_per_file: int = DEFAULT_OBS_TILES_PER_FILE,
    var: bool = True,
    log: Callable[[str], None] | None = print,
) -> Manifest:
    """Write ``output`` from h5ad files (glob patterns or paths), preserving cell order.

    The files may hold any number of cells (a single file can be much larger than memory): ``X`` is read from disk a tile at a
    time, in ``workers`` processes that each read, encode and write whole tiles, so the result depends only on the cell order
    and ``tile_size``. ``X`` can be CSR (the efficient case) or dense. All files must have the same ``var_names`` and ``X`` must
    hold integer counts up to 65535. The files are naturally sorted unless ``sort_files=False``, which keeps the order of
    ``paths``. The cells are not shuffled: each tile holds consecutive cells, so files that are sorted by something
    (donor, tissue, ...) give tiles that are as homogeneous; shuffle the cells across the files first if that matters.

    With ``sort_genes`` the genes are reordered by decreasing total counts (which compresses much better); the original column
    of each output gene is stored as ``gene_order``. The totals come from a pass over ``X`` of up to ``sort_genes_max_files``
    files spread evenly over ``paths`` (all the files if there are no more than that, or if it is ``None``), which is a proxy
    for the whole dataset when the files are shuffled across each other.

    ``workers`` is the number of processes that write tiles (default: the number of cores, limited so that the tiles in
    flight fit in about 70% of the available memory) and ``threads`` the compression threads of each one (default: the
    cores divided by ``workers``, so about one in total per core).

    With ``obs`` (needs pyarrow and pandas) the cell metadata is stored too, as tile-aligned parquet shards (see
    :mod:`deltacells.obs`): one pass over all files builds a global vocabulary for each categorical column (columns with more than
    ``max_categories`` distinct values are stored as strings), ``obs_names`` adds the cell names as a string column, and
    ``obs_exclude`` drops columns. With ``var`` the gene table is stored as ``var.parquet``.
    """
    import anndata

    say = log or (lambda _: None)
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    if sort_genes_max_files is not None and sort_genes_max_files < 1:
        raise ValueError("sort_genes_max_files must be at least 1 (or None for all the files)")
    if os.path.exists(output) and os.listdir(output) and not overwrite:
        raise FileExistsError(f"{output} is not empty; pass overwrite=True to replace it")
    files = resolve_files(paths, sort=sort_files)
    say(f"{len(files)} files: {files[0]} ... {files[-1]}")
    if obs or var:
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "storing obs / var needs pandas and pyarrow (pip install deltacells[obs]); or pass obs=False, var=False"
            ) from e

    state: dict = {"var_names": None, "var": None, "n_obs": [], "x_bytes": 0}

    def scan() -> Iterator:
        """One pass over the files (not reading ``X``): checks var_names, counts cells and yields each file's obs frame (or None)."""
        for path in files:
            a = anndata.read_h5ad(path, backed="r")
            try:
                names = np.asarray(a.var_names, dtype=str)
                if state["var_names"] is None:
                    state["var_names"], state["var"] = names, a.var.copy()
                elif not np.array_equal(state["var_names"], names):
                    raise ValueError(f"var_names of {path} differ from those of {files[0]}")
                state["n_obs"].append(a.n_obs)
                state["x_bytes"] += _x_work_bytes(path)
                yield _obs_frame(a, obs_names, obs_exclude) if obs else None
            finally:
                a.file.close()

    schema = None
    if obs:
        schema = ObsSchema.infer(scan(), max_categories=max_categories)
        as_strings = [c.name for c in schema.columns if c.kind == "string"]
        say(
            f"obs: {len(schema.columns)} columns ({sum(c.kind == 'category' for c in schema.columns)} categorical, strings: {as_strings})"
        )
    else:
        for _ in scan():
            pass
    n_obs = state["n_obs"]
    var_names = state["var_names"]
    n_genes = len(var_names)
    say(f"{sum(n_obs)} cells x {n_genes} genes (cells per file: min {min(n_obs)}, max {max(n_obs)})")

    n_cells = sum(n_obs)
    if n_cells == 0:
        raise ValueError("the h5ad files hold no cells")
    n_tiles = -(-n_cells // tile_size)
    if workers is None:
        tile_bytes = _WORKER_BASE_BYTES + 1.5 * state["x_bytes"] / n_cells * min(tile_size, n_cells)
        workers = _default_workers(n_tiles, tile_bytes)
    workers = min(workers, n_tiles)
    if threads is None:
        threads = min(max(1, _cpu_count() // workers), 2 * n_chunks)
    say(f"{n_tiles} tiles, written by {workers} worker process(es) with {threads} compression thread(s) each")

    if sort_genes:
        used = _evenly_spaced(len(files), sort_genes_max_files)
        say(
            f"gene order: summing the counts of {len(used)} of {len(files)} files ({sum(n_obs[k] for k in used)} cells)"
        )
        totals = _gene_totals([files[k] for k in used], [n_obs[k] for k in used], n_genes, tile_size, workers)
        gene_order = np.argsort(-totals, kind="stable")
    else:
        used, gene_order = [], None

    metadata = {
        "source": "h5ad",
        "n_source_files": len(files),
        "first_file": files[0],
        "last_file": files[-1],
        "sort_genes": sort_genes,
        "sort_genes_n_files": len(used),
    }
    writer = DatasetWriter(
        output, n_genes=n_genes, tile_size=tile_size, gene_order=gene_order, var_names=list(var_names),
        var=state["var"] if var else None, obs_schema=schema, obs_tiles_per_file=obs_tiles_per_file,
        n_chunks=n_chunks, level=level, threads=threads, metadata=metadata, overwrite=overwrite,
    )  # fmt: skip

    def obs_tiles() -> Iterator:
        """The obs of each tile, in order: one Arrow table encoded with the global schema (so shards with different category
        sets combine) per tile. Runs in this process, one file at a time, while the workers write the tiles."""
        import pyarrow as pa

        buf = None
        for path in files:
            a = anndata.read_h5ad(path, backed="r")
            try:
                table = schema.encode(_obs_frame(a, obs_names, obs_exclude))
            finally:
                a.file.close()
            buf = table if buf is None else pa.concat_tables([buf, table])
            while buf.num_rows >= tile_size:
                yield buf.slice(0, tile_size)
                buf = buf.slice(tile_size)
        if buf is not None and buf.num_rows > 0:
            yield buf

    obs_iter = obs_tiles() if schema is not None else None
    offsets = [0, *accumulate(n_obs)]

    def commit(i: int, get_result: Callable[[], tuple[int, int, int]]) -> None:
        obs_table = None if obs_iter is None else next(obs_iter)  # while the workers are busy with the tiles
        n, nnz, n_bytes = get_result()
        writer.commit_tile(i, n, nnz, n_bytes, obs=obs_table)
        say(f"[tile {i + 1}/{n_tiles}] {n} cells, {n_bytes / 1e6:.1f} MB")

    init_args = (output, gene_order, n_chunks, level, threads)
    if workers == 1:
        _init_tile_worker(*init_args)
        for i in range(n_tiles):
            commit(i, lambda i=i: _write_tile_job((i, _tile_pieces(i, tile_size, offsets, files))))
    else:
        pool = ProcessPoolExecutor(workers, mp_context=_mp_context(), initializer=_init_tile_worker, initargs=init_args)
        pending: dict[int, Future] = {}
        submitted = 0
        try:
            for i in range(n_tiles):
                while (
                    submitted < n_tiles and len(pending) < 4 * workers
                ):  # keep the workers busy, but not run ahead forever
                    job = (submitted, _tile_pieces(submitted, tile_size, offsets, files))
                    pending[submitted] = pool.submit(_write_tile_job, job)
                    submitted += 1
                commit(i, pending.pop(i).result)
        except BrokenProcessPool as e:
            raise RuntimeError(
                "a worker process died, probably because it ran out of memory; pass a smaller `workers`"
            ) from e
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    manifest = writer.close()
    total = sum(manifest.tile_bytes)
    say(f"done: {manifest.n_tiles} tiles, {total / 1e6:.1f} MB = {total / manifest.n_cells / 1e3:.2f} kB per cell")
    if manifest.has_obs:
        say(f"obs: {sum(manifest.obs_shard_bytes) / 1e6:.1f} MB in {manifest.n_obs_shards} parquet shard(s)")
    return manifest


def _obs_frame(adata, include_names: bool, exclude: Sequence[str]):
    """The obs frame of an AnnData object, with the cell names as an ``obs_names`` column and ``exclude`` dropped."""
    df = adata.obs.drop(columns=[c for c in exclude if c in adata.obs.columns]).copy()
    if include_names:
        if "obs_names" in df.columns:
            raise ValueError("the obs already has an 'obs_names' column; pass obs_names=False or rename it")
        df["obs_names"] = np.asarray(adata.obs_names, dtype=str)
    return df
