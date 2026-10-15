# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert h5ad shards (any cell counts per file) into a deltacells dataset of exact-size tiles."""

from __future__ import annotations

import glob
import multiprocessing
import os
import re
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import accumulate

import numpy as np
import scipy.sparse as sp

from deltacells.manifest import DEFAULT_OBS_TILES_PER_FILE, Manifest
from deltacells.progress import DEFAULT_INTERVAL, Progress, format_duration, say_flushed
from deltacells.sink import is_remote, open_sink
from deltacells.staging import BYTES_PER_NNZ, FileMeta, Stager, ordered_map, read_h5ad_meta
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
    matches nothing raises ``FileNotFoundError``. A remote file (``gs://bucket/file.h5ad``) is taken as it is, not as a
    pattern: remote files must be listed one by one (and are checked when they are read).
    """
    files: dict[str, None] = {}
    for pattern in [patterns] if isinstance(patterns, str) else patterns:
        if is_remote(pattern):
            files[pattern] = None
            continue
        matched = sorted(glob.glob(pattern, recursive=True), key=natural_sort_key)
        if not matched:
            raise FileNotFoundError(f"no files matched {pattern!r}")
        files.update(dict.fromkeys(matched))
    return sorted(files, key=natural_sort_key) if sort else list(files)


# A worker's peak memory is about this many bytes per nonzero of its tile plus a fixed base (see ``_default_workers``).
_BYTES_PER_NNZ = BYTES_PER_NNZ
_WORKER_BASE_BYTES = 400 * 2**20
_MEMORY_FRACTION = 0.7  # of the available memory that the workers may use
_SCAN_THREADS = 16  # files whose metadata is read at the same time in threads (see ``_scan_processes``)
_SCAN_MIN_FILES = 32  # fewer files than this are scanned in this process: starting the processes takes seconds
_SCAN_PROCESSES_MAX = 24  # the scan waits for the network, so a bit more than one process per core is useful


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return os.cpu_count() or 1


def _scan_job(job: tuple[str, bool, bool]) -> FileMeta:
    """Read the metadata of one file, in a scan process (which makes its own filesystem for a remote file)."""
    path, want_obs, want_var = job
    return read_h5ad_meta(path, None, obs=want_obs, var=want_var)


def _scan_processes(n_files: int, filesystem, scan_workers: int | None) -> int:
    """How many processes scan the metadata (0: threads in this process).

    h5py serialises the reads it makes through a Python file object (that is how it reads a remote file), so scanning remote
    files with threads is no faster than with one thread (about 1.3 files/s from GCS); the processes are what makes the scan
    parallel. They need a filesystem that every process can make for itself, so a ``filesystem`` that is passed in (for
    testing) is used from threads.
    """
    if scan_workers is not None and scan_workers < 1:
        raise ValueError("scan_workers must be at least 1")
    if scan_workers == 1:
        return 0
    if scan_workers is None:
        if filesystem is not None or n_files < _SCAN_MIN_FILES:
            return 0
        scan_workers = min(_SCAN_PROCESSES_MAX, max(2, 3 * _cpu_count() // 2))
    elif filesystem is not None:
        raise ValueError(
            "scan_workers > 1 cannot be used with a filesystem that is passed in (processes make their own)"
        )
    return min(scan_workers, n_files)


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
    # fork is cheap and does not re-import (or re-run) the calling script; the workers only use h5py, numpy and scipy (they
    # never touch the network: remote files are downloaded by the main process)
    return multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else None)


def _evenly_spaced(n: int, k: int | None) -> list[int]:
    """``k`` indices spread evenly over ``range(n)`` (all of them if ``k`` is ``None`` or at least ``n``)."""
    if k is None or k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _read_rows(path: str, start: int, end: int) -> sp.csr_matrix:
    """Rows ``start:end`` of ``X`` in a local h5ad file as CSR, reading only those rows from disk (``X`` may be CSR or dense)."""
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


# --- the gene order ------------------------------------------------------------------------------------------------------


def _column_sums_job(job: tuple[str, int, int, int]) -> np.ndarray:
    path, start, end, n_genes = job
    x = _read_rows(path, start, end)
    return np.bincount(x.indices, weights=x.data, minlength=n_genes)


def _gene_totals(
    stager: Stager,
    n_obs: Sequence[int],
    n_genes: int,
    step: int,
    workers: int,
    on_file: Callable[[], None] | None = None,
) -> np.ndarray:
    """Total counts of every gene over the files of ``stager`` (each released after its use), reading ``step`` rows at a time
    (in ``workers`` processes). ``on_file`` is called when a file is done."""
    totals = np.zeros(n_genes, dtype=np.float64)
    in_flight = max(2, workers)  # files that are being read at the same time
    pool = None if workers == 1 else ProcessPoolExecutor(workers, mp_context=_mp_context())

    def submit(job) -> Future:
        if pool is not None:
            return pool.submit(_column_sums_job, job)
        future: Future = Future()
        future.set_result(_column_sums_job(job))
        return future

    def collect(k: int, futures: list[Future]) -> None:
        for future in futures:
            np.add(totals, future.result(), out=totals)
        stager.release(k)
        if on_file is not None:
            on_file()

    pending: deque = deque()
    try:
        for k, n in enumerate(n_obs):
            stager.prefetch_through(k + in_flight)
            path = stager.local(k)
            pending.append((k, [submit((path, s, min(s + step, n), n_genes)) for s in range(0, n, step)]))
            while len(pending) > in_flight:
                collect(*pending.popleft())
        while pending:
            collect(*pending.popleft())
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return totals


# --- writing the tiles ---------------------------------------------------------------------------------------------------

_TILE_JOB: dict = {}  # the settings of a tile-writing worker, set by ``_init_tile_worker``


def _init_tile_worker(root: str, gene_order: np.ndarray | None, n_chunks: int, level: int, threads: int) -> None:
    _TILE_JOB.update(root=root, gene_order=gene_order, n_chunks=n_chunks, level=level, threads=threads)


def _write_tile_job(job: tuple[int, list[tuple[str, int, int]], list[str]]) -> tuple[int, int, int]:
    """Read the pieces of one tile (of local files), encode it and write its file. Returns ``(n_cells, nnz, n_bytes)``.

    ``job`` is ``(index, pieces, sources)`` where ``sources`` names the original files, for messages.
    """
    index, pieces, sources = job
    parts = [_read_rows(*piece) for piece in pieces]
    x = parts[0] if len(parts) == 1 else sp.vstack(parts, format="csr")
    del parts
    try:
        return write_tile_file(
            _TILE_JOB["root"],
            index,
            x,
            gene_order=_TILE_JOB["gene_order"],
            n_chunks=_TILE_JOB["n_chunks"],
            level=_TILE_JOB["level"],
            threads=_TILE_JOB["threads"],
        )
    except ValueError as e:
        raise ValueError(f"tile {index} (from {', '.join(sources)}): {e}") from e


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
    staging_dir: str | None = None,
    filesystem=None,
    scan_workers: int | None = None,
    progress_interval: float = DEFAULT_INTERVAL,
    log: Callable[[str], None] | None = say_flushed,
) -> Manifest:
    """Write ``output`` from h5ad files (glob patterns or paths), preserving cell order.

    The files may hold any number of cells (a single file can be much larger than memory): ``X`` is read from disk a tile at a
    time, in ``workers`` processes that each read, encode and write whole tiles, so the result depends only on the cell order
    and ``tile_size``. ``X`` can be CSR (the efficient case) or dense. All files must have the same ``var_names`` and ``X`` must
    hold non-negative integer counts (up to 2**31 - 1; counts above 65535 are stored exactly, as rare "exceptions", and a tile
    with more than about 0.5% of them is an error: it is not raw counts). The files are naturally sorted unless
    ``sort_files=False``, which keeps the order of ``paths``. The cells are not shuffled: each tile holds consecutive cells, so files that are sorted by
    something (donor, tissue, ...) give tiles that are as homogeneous; shuffle the cells across the files first if that matters.

    The h5ad files, and ``output``, can be in Google Cloud Storage (``gs://bucket/...``, with the default credentials; needs
    ``gcsfs``, see ``deltacells[gcs]``). Remote files must be listed one by one (they are not globbed). Their metadata is read
    with a few ranged reads each; for the tiles each file is downloaded in full to a scratch directory (``staging_dir``) just
    before it is needed and deleted after, so the scratch space needed is that of the files in use at the same time (roughly
    ``5 * workers`` files), not of all of them. A remote ``output`` is written the same way: each tile is uploaded when it is
    done and then deleted, and the manifest is uploaded last. Nothing is ever deleted from the input files, and ``overwrite``
    only deletes the files of a deltacells dataset that are at ``output`` (it raises if there is anything else, or if an input
    file is inside ``output``). If the conversion fails what was uploaded stays at ``output``, with no manifest: re-run with
    ``overwrite=True`` or delete it yourself.

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

    That first pass reads the metadata of every file, in ``scan_workers`` processes (default: about 1.5 per core, up to 24,
    when there are at least 32 files and no ``filesystem`` is passed in; ``1`` scans in this process, with threads). The
    processes are started with ``spawn``, so a script that calls this must be guarded by ``if __name__ == "__main__":``.

    ``log`` receives the progress messages (``None``: none). A line about the phase that is running (scan, gene order, tiles,
    finishing) is written every ``progress_interval`` seconds, the first one after ten seconds, with how far it is, its speed
    over the last minutes and the time left; ``0`` leaves only the line that starts and the line that ends each phase.
    """
    progress = Progress(log, progress_interval)
    try:
        return _convert_h5ad(
            progress, time.monotonic(), paths, output,
            tile_size=tile_size, sort_genes=sort_genes, sort_files=sort_files, n_chunks=n_chunks, level=level,
            threads=threads, workers=workers, sort_genes_max_files=sort_genes_max_files, overwrite=overwrite,
            obs=obs, obs_names=obs_names, obs_exclude=obs_exclude, max_categories=max_categories,
            obs_tiles_per_file=obs_tiles_per_file, var=var, staging_dir=staging_dir, filesystem=filesystem,
            scan_workers=scan_workers, log=log,
        )  # fmt: skip
    finally:
        progress.close()  # the thread that prints the status lines must not outlive the conversion, whatever happened


def _convert_h5ad(
    progress: Progress,
    t_start: float,
    paths: Sequence[str] | str,
    output: str,
    *,
    tile_size: int,
    sort_genes: bool,
    sort_files: bool,
    n_chunks: int,
    level: int,
    threads: int | None,
    workers: int | None,
    sort_genes_max_files: int | None,
    overwrite: bool,
    obs: bool,
    obs_names: bool,
    obs_exclude: Sequence[str],
    max_categories: int,
    obs_tiles_per_file: int,
    var: bool,
    staging_dir: str | None,
    filesystem,
    scan_workers: int | None,
    log: Callable[[str], None] | None,
) -> Manifest:
    """The conversion itself (see :func:`convert_h5ad`); ``progress`` reports on its phases."""
    say = log or (lambda _: None)
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    if sort_genes_max_files is not None and sort_genes_max_files < 1:
        raise ValueError("sort_genes_max_files must be at least 1 (or None for all the files)")
    files = resolve_files(paths, sort=sort_files)
    say(f"{len(files):,} files: {files[0]} ... {files[-1]}")
    sink = open_sink(output, filesystem)
    replaced = sink.check(overwrite, files)  # now, so that a mistake does not cost a slow scan
    say(
        f"the output {output} has a dataset of {len(replaced)} files, which will be deleted when the scan is done"
        if replaced
        else f"the output {output} is empty"
    )
    sink.probe()  # for the same reason: a missing permission to write is found now, not after the scan
    if obs or var:
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "storing obs / var needs pandas and pyarrow (pip install deltacells[obs]); or pass obs=False, var=False"
            ) from e

    state: dict = {"var_names": None, "var": None, "n_obs": [], "x_bytes": 0, "sizes": []}
    scan_processes = _scan_processes(len(files), filesystem, scan_workers)
    scan_threads = _SCAN_THREADS if any(is_remote(f) for f in files) else 1

    def scan() -> Iterator:
        """One pass over the files (not reading ``X``): checks var_names, counts cells and yields each file's obs frame (or None)."""
        if scan_processes:
            pool: Executor = ProcessPoolExecutor(scan_processes, mp_context=multiprocessing.get_context("spawn"))
            jobs = [(files[k], obs, k == 0) for k in range(len(files))]  # the gene table comes from the first file
            fn, items, window = _scan_job, jobs, 2 * scan_processes
        else:
            pool = ThreadPoolExecutor(scan_threads)
            fn = lambda k: read_h5ad_meta(files[k], filesystem, obs=obs, var=k == 0)  # noqa: E731
            items, window = range(len(files)), 2 * scan_threads
        try:
            for k, meta in enumerate(ordered_map(pool, fn, items, window)):
                if state["var_names"] is None:
                    state["var_names"], state["var"] = meta.var_names, meta.var
                elif not np.array_equal(state["var_names"], meta.var_names):
                    raise ValueError(f"var_names of {files[k]} differ from those of {files[0]}")
                state["n_obs"].append(meta.n_obs)
                state["x_bytes"] += meta.x_bytes
                state["sizes"].append(meta.size)
                progress.advance()
                yield _obs_frame(meta.obs, obs_names, obs_exclude) if obs else None
        except BrokenProcessPool as e:
            raise RuntimeError(
                "a process that reads the metadata of the files died; pass scan_workers=1 to read them in this process"
            ) from e
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    how = (
        f"{scan_processes} processes"
        if scan_processes
        else f"{scan_threads} threads"
        if scan_threads > 1
        else "this process"
    )
    progress.start("scan", len(files), "files", header=f"reading the metadata of {len(files):,} files ({how})")
    schema = None
    if obs:
        schema = ObsSchema.infer(scan(), max_categories=max_categories)
    else:
        for _ in scan():
            pass
    progress.finish()
    if schema is not None:
        as_strings = [c.name for c in schema.columns if c.kind == "string"]
        say(
            f"obs: {len(schema.columns)} columns ({sum(c.kind == 'category' for c in schema.columns)} categorical, strings: {as_strings})"
        )
    n_obs = state["n_obs"]
    var_names = state["var_names"]
    n_genes = len(var_names)
    known_sizes = [s for s in state["sizes"] if s]
    total_gb = f", {sum(known_sizes) / 1e9:,.0f} GB" if len(known_sizes) == len(files) else ""
    say(
        f"{sum(n_obs):,} cells x {n_genes:,} genes in {len(files):,} files{total_gb} "
        f"(cells per file: min {min(n_obs):,}, max {max(n_obs):,})"
    )

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
    say(f"{n_tiles:,} tiles, written by {workers} worker process(es) with {threads} compression thread(s) each")

    if sort_genes:
        used = _evenly_spaced(len(files), sort_genes_max_files)
        progress.start(
            "gene order", len(used), "files",
            header=f"gene order: summing the counts of {len(used)} of {len(files):,} files ({sum(n_obs[k] for k in used):,} cells)",
        )  # fmt: skip
        with Stager(
            [files[k] for k in used],
            [1] * len(used),
            scratch_dir=staging_dir,
            sizes=[state["sizes"][k] for k in used],
            filesystem=filesystem,
        ) as sampled:
            totals = _gene_totals(
                sampled, [n_obs[k] for k in used], n_genes, tile_size, workers, on_file=progress.advance
            )
        gene_order = np.argsort(-totals, kind="stable")
        progress.finish()
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
    # the first change to the output: what is there is deleted now (``overwrite``), after everything above had its chance to fail
    if replaced:
        say(f"deleting the {len(replaced):,} files of the dataset that is in {output} ...")
    writer = DatasetWriter(
        output, n_genes=n_genes, tile_size=tile_size, gene_order=gene_order, var_names=list(var_names),
        var=state["var"] if var else None, obs_schema=schema, obs_tiles_per_file=obs_tiles_per_file,
        n_chunks=n_chunks, level=level, threads=threads, metadata=metadata, overwrite=overwrite,
        inputs=files, scratch_dir=staging_dir, filesystem=filesystem,
    )  # fmt: skip

    offsets = [0, *accumulate(n_obs)]
    index_of = {f: k for k, f in enumerate(files)}
    tile_pieces = [_tile_pieces(i, tile_size, offsets, files) for i in range(n_tiles)]
    tile_files = [[index_of[f] for f, _, _ in pieces] for pieces in tile_pieces]  # the files of each tile, in order
    uses = [0] * len(files)
    for ks in tile_files:
        for k in ks:
            uses[k] += 1
    ahead = max(2, workers)  # files that are downloaded before they are needed
    stager = Stager(files, uses, scratch_dir=staging_dir, sizes=state["sizes"], filesystem=filesystem)
    if stager.scratch is not None:
        sizes = [s for s in state["sizes"] if s]
        if sizes:
            n_staged = min(len(files), 4 * workers + ahead + 2)
            say(
                f"remote files are staged in {stager.scratch}: up to about {n_staged} at a time "
                f"({n_staged * np.mean(sizes) / 1e9:.1f} GB)"
            )

    def make_job(i: int):
        """The job for tile ``i``, with its files downloaded (and the next few requested)."""
        stager.prefetch_through(max(tile_files[i]) + ahead)
        pieces = [(stager.local(k), lo, hi) for k, (_, lo, hi) in zip(tile_files[i], tile_pieces[i])]
        return i, pieces, [files[k] for k in tile_files[i]]

    def obs_tiles() -> Iterator:
        """The obs of each tile, in order: one Arrow table encoded with the global schema (so shards with different category
        sets combine) per tile. Runs in this process, one file at a time, while the workers write the tiles. The files are
        read from the staged copies, which are still there: a tile's files are released only after the tile is committed."""
        import pyarrow as pa

        buf = None
        for k in range(len(files)):
            if n_obs[k] == 0:
                continue
            frame = _obs_frame(read_h5ad_meta(stager.local(k)).obs, obs_names, obs_exclude)
            table = schema.encode(frame)
            buf = table if buf is None else pa.concat_tables([buf, table])
            while buf.num_rows >= tile_size:
                yield buf.slice(0, tile_size)
                buf = buf.slice(tile_size)
        if buf is not None and buf.num_rows > 0:
            yield buf

    obs_iter = obs_tiles() if schema is not None else None

    def commit(i: int, get_result: Callable[[], tuple]) -> None:
        obs_table = None if obs_iter is None else next(obs_iter)  # while the workers are busy with the tiles
        n, nnz, n_bytes = get_result()
        writer.commit_tile(i, n, nnz, n_bytes, obs=obs_table)
        for k in tile_files[i]:
            stager.release(k)
        progress.advance(1, n_bytes)

    def tile_detail() -> str:
        parts = []
        if stager.scratch is not None:
            parts.append(f"{stager.n_staged} files staged ({stager.staged_bytes / 1e9:.1f} GB)")
        if writer.is_remote:
            parts.append(f"{writer.pending_uploads} uploads pending")
        return ", ".join(parts)

    init_args = (writer.root, gene_order, n_chunks, level, threads)
    progress.start(
        "tiles", n_tiles, "tiles", bytes_label="written", detail=tile_detail, header=f"writing {n_tiles:,} tiles"
    )
    try:
        if workers == 1:
            _init_tile_worker(*init_args)
            for i in range(n_tiles):
                job = make_job(i)
                commit(i, lambda: _write_tile_job(job))  # run at once, so the loop variable has not moved on
        else:
            pool = ProcessPoolExecutor(
                workers, mp_context=_mp_context(), initializer=_init_tile_worker, initargs=init_args
            )
            pending: dict[int, Future] = {}
            submitted = 0
            try:
                for i in range(n_tiles):
                    while (
                        submitted < n_tiles and len(pending) < 4 * workers
                    ):  # keep the workers busy, but not run ahead forever
                        pending[submitted] = pool.submit(_write_tile_job, make_job(submitted))
                        submitted += 1
                    commit(i, pending.pop(i).result)
            except BrokenProcessPool as e:
                raise RuntimeError(
                    "a worker process died, probably because it ran out of memory; pass a smaller `workers`"
                ) from e
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        progress.finish()
        progress.start(
            "finishing",
            header="finishing: the obs shards, the last uploads and the manifest",
            detail=lambda: f"{writer.pending_uploads} uploads pending" if writer.is_remote else "",
        )
        manifest = writer.close()
        progress.finish()
    except BaseException as e:
        writer.abort()
        if writer.is_remote and writer.n_published:
            hint = (
                f"The conversion failed, and what it had uploaded to {output} is an incomplete dataset (it has no manifest). "
                "Re-run with overwrite=True to replace it, or delete it yourself."
            )
            if hasattr(e, "add_note"):
                e.add_note(hint)
            else:
                say(hint)
        raise
    finally:
        stager.close()
    total = sum(manifest.tile_bytes)
    say(
        f"done in {format_duration(time.monotonic() - t_start)}: {manifest.n_tiles:,} tiles, {total / 1e6:,.1f} MB = "
        f"{total / manifest.n_cells / 1e3:.2f} kB per cell"
    )
    if manifest.has_obs:
        say(f"obs: {sum(manifest.obs_shard_bytes) / 1e6:.1f} MB in {manifest.n_obs_shards} parquet shard(s)")
    return manifest


def _obs_frame(obs_df, include_names: bool, exclude: Sequence[str]):
    """The obs frame of a file, with the cell names as an ``obs_names`` column and ``exclude`` dropped."""
    df = obs_df.drop(columns=[c for c in exclude if c in obs_df.columns]).copy()
    if include_names:
        if "obs_names" in df.columns:
            raise ValueError("the obs already has an 'obs_names' column; pass obs_names=False or rename it")
        df["obs_names"] = np.asarray(obs_df.index, dtype=str)
    return df
