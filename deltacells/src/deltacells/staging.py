# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Reading h5ad files that may be remote (``gs://...``).

* :func:`read_h5ad_meta` reads what the conversion needs to know about a file (``obs``, ``var``, the shape of ``X``) without
  reading ``X``: from a remote file this takes a handful of large ranged reads.
* :class:`Stager` gives the conversion local paths of the files, downloading the remote ones to a scratch directory just
  before they are needed and deleting them when they are not. ``X`` is read from the local copies, a tile at a time.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import shutil
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from deltacells.sink import get_filesystem, is_remote, retry, split_url

# Block size of the cache of a remote file: the metadata of an h5ad file is read with a few requests of this size (with the
# 64 KB that fsspec uses by default it takes about a hundred requests per file, which is 10 times slower).
BLOCK_SIZE = 8 << 20


def ordered_map(pool: ThreadPoolExecutor, fn: Callable[[Any], Any], items: Iterable[Any], window: int) -> Iterator[Any]:
    """``map`` over a thread pool that yields results in order, with at most ``window`` tasks in flight."""
    it = iter(items)
    pending: deque = deque(pool.submit(fn, x) for x in itertools.islice(it, window))
    while pending:
        result = pending.popleft().result()
        for x in itertools.islice(it, 1):
            pending.append(pool.submit(fn, x))
        yield result


def _read_elem(group: Any) -> Any:
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(group)


@contextlib.contextmanager
def open_h5(path: str, fs: Any = None):
    """An open ``h5py.File`` for a local path, or for a remote URL through its (fsspec) filesystem ``fs``."""
    import h5py

    if not is_remote(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            yield f
        return
    fs = get_filesystem(path, fs)
    with fs.open(split_url(path)[1], "rb", block_size=BLOCK_SIZE, cache_type="blockcache") as fileobj:
        with h5py.File(fileobj, "r") as f:
            yield f


# A worker's memory use is about this many bytes per nonzero of its tile (see ``convert._default_workers``).
BYTES_PER_NNZ = 48


def x_work_bytes(x: Any) -> int:
    """Estimated working memory for processing all the rows of ``X`` (an open h5py group or dataset) at once."""
    import h5py

    if isinstance(x, h5py.Dataset):  # dense: the block itself and its sparse copy (assuming 30% nonzeros)
        n, g = x.shape
        return int(n * g * (x.dtype.itemsize + 0.3 * BYTES_PER_NNZ))
    return int(x["indptr"][-1]) * BYTES_PER_NNZ


def x_shape(path: str, x: Any) -> tuple[int, int]:
    """The shape of ``X`` (a dense dataset or a CSR group); raises for any other storage of ``X``."""
    import h5py

    if isinstance(x, h5py.Dataset):
        return int(x.shape[0]), int(x.shape[1])
    fmt = str(x.attrs.get("encoding-type", x.attrs.get("h5sparse_format")))
    if fmt not in ("csr_matrix", "csr"):
        raise ValueError(f"X of {path} is stored as {fmt}; only CSR and dense X are supported")
    shape = x.attrs.get("shape", x.attrs.get("h5sparse_shape"))
    return int(shape[0]), int(shape[1])


def _var_names(var_group: Any) -> np.ndarray:
    index = var_group.attrs.get("_index")
    if isinstance(index, bytes):
        index = index.decode()
    if index is not None and index in var_group:  # just the index column, not the whole table
        names = var_group[index][()]
    else:
        names = _read_elem(var_group).index.to_numpy()
    return np.asarray([n.decode() if isinstance(n, bytes) else n for n in names], dtype=str)


@dataclass
class FileMeta:
    """What the conversion needs to know about one h5ad file, without ``X``."""

    var_names: np.ndarray
    n_obs: int
    x_bytes: int
    size: int | None  # in bytes, for a remote file
    obs: Any = None  # pandas.DataFrame, indexed by the cell names
    var: Any = None  # pandas.DataFrame


def read_h5ad_meta(path: str, fs: Any = None, *, obs: bool = True, var: bool = False) -> FileMeta:
    """Read the ``var_names``, the number of cells and (if asked) the ``obs`` and the whole ``var`` of an h5ad file, which
    can be local or remote, without reading ``X``."""
    size = None
    if is_remote(path):
        fs = get_filesystem(path, fs)
        size = int(fs.info(split_url(path)[1])["size"])
    with open_h5(path, fs) as f:
        if "X" not in f:
            raise ValueError(f"{path} has no X")
        n_obs, n_vars = x_shape(path, f["X"])
        names = _var_names(f["var"])
        if len(names) != n_vars:
            raise ValueError(f"{path}: X has {n_vars} columns but var has {len(names)} rows")
        obs_df = _read_elem(f["obs"]) if obs else None
        if obs_df is not None and len(obs_df) != n_obs:
            raise ValueError(f"{path}: X has {n_obs} rows but obs has {len(obs_df)}")
        return FileMeta(
            var_names=names,
            n_obs=n_obs,
            x_bytes=x_work_bytes(f["X"]),
            size=size,
            obs=obs_df,
            var=_read_elem(f["var"]) if var else None,
        )


class Stager:
    """Local paths for h5ad files, some of which may be remote.

    A local file is used where it is. A remote file is downloaded, by :meth:`prefetch_through` (in background threads) or
    when :meth:`local` first needs it, to a scratch directory made inside ``scratch_dir``, and deleted by :meth:`release`
    once it has been released as many times as it is used (``uses[k]``). So the scratch space needed is that of the files
    that are in use or prefetched at the same moment, not of all of them.

    A download is written to a temporary name, checked against the size of the remote file and retried if it fails.

    Args:
        files: The h5ad files, local paths or URLs.
        uses: For each file, how many times it will be released; a file that is never used is never downloaded.
        scratch_dir: Where to make the scratch directory (default: the system temp directory).
        sizes: The sizes of the remote files, to check the downloads (``None`` or missing: not checked).
        filesystem: An fsspec filesystem to use instead of the default for the protocol (for testing).
        threads: Downloads at the same time.
    """

    def __init__(
        self,
        files: Sequence[str],
        uses: Sequence[int],
        *,
        scratch_dir: str | None = None,
        sizes: Sequence[int | None] | None = None,
        filesystem: Any = None,
        threads: int = 8,
    ) -> None:
        self.files = list(files)
        self._uses = list(uses)
        self._sizes = list(sizes) if sizes is not None else [None] * len(self.files)
        self._filesystem = filesystem
        self._remote = [is_remote(f) for f in self.files]
        self._futures: dict[int, Future] = {}
        self._released: set[int] = set()
        self._next = 0  # the files before this one have been requested
        self._lock = threading.Lock()
        self._scratch: str | None = None
        self._pool: ThreadPoolExecutor | None = None
        if any(self._remote):
            self._scratch = tempfile.mkdtemp(prefix="deltacells_inputs_", dir=scratch_dir)
            self._pool = ThreadPoolExecutor(threads)

    @property
    def scratch(self) -> str | None:
        return self._scratch

    def _download(self, k: int) -> str:
        assert self._scratch is not None
        url, dest = self.files[k], os.path.join(self._scratch, f"{k:07d}.h5ad")
        fs = get_filesystem(url, self._filesystem)

        def fetch() -> None:
            tmp = dest + ".part"
            try:
                fs.get_file(split_url(url)[1], tmp)
                expected = self._sizes[k]
                if expected is not None and os.path.getsize(tmp) != expected:
                    raise OSError(f"{url} downloaded with {os.path.getsize(tmp)} bytes, expected {expected}")
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        retry(fetch, f"downloading {url}")
        return dest

    def _request(self, k: int) -> None:
        with self._lock:
            if k in self._released:
                raise RuntimeError(f"{self.files[k]} was released and is needed again (a bug)")
            if self._remote[k] and k not in self._futures and self._uses[k] > 0:
                assert self._pool is not None
                self._futures[k] = self._pool.submit(self._download, k)

    def prefetch_through(self, last: int) -> None:
        """Start downloading, in order, every remote file up to index ``last`` that has not been requested yet."""
        while self._next <= min(last, len(self.files) - 1):
            self._request(self._next)
            self._next += 1

    def local(self, k: int) -> str:
        """The local path of file ``k``, waiting for its download if it is remote."""
        if not self._remote[k]:
            return self.files[k]
        self._request(k)
        return self._futures[k].result()

    def release(self, k: int) -> None:
        """File ``k`` is no longer needed by one more of its uses; delete its local copy after the last one."""
        with self._lock:
            self._uses[k] -= 1
            if self._uses[k] > 0 or not self._remote[k]:
                return
            self._released.add(k)
            future = self._futures.pop(k, None)
        if future is not None:
            with contextlib.suppress(Exception):
                os.remove(future.result())

    def close(self) -> None:
        """Stop downloading and delete everything in the scratch directory."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def __enter__(self) -> Stager:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
