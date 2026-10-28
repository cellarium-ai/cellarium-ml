# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Per-cell arrays (entries of ``cdata.obsm``) that are too big for memory.

An api function that computes a value for every cell (for example the loadings of the programs of
:class:`~cellarium.ml.api.NMFOutput`) allocates its result with an :class:`ObsmWriter`: in memory if it fits in the
budget (``max_gb`` of :class:`~cellarium.ml.api.CellariumData`), and otherwise in a memory-mapped ``.npy`` file in the
temporary directory, which is deleted when the array is no longer referenced (or when the interpreter exits). The
finished on-disk array is a read-only :class:`numpy.memmap`, so it works wherever an ndarray does, without being read
into memory.

A view of the cells (see :class:`~cellarium.ml.api.CellariumDataView`) takes the rows of its cells from such an
array with :func:`subset_rows`: into memory if they fit in the budget, and otherwise as a :class:`RowSubset`, a lazy
read-only view of the rows of the parent's file that copies nothing.
"""

import shutil
import tempfile
import weakref
from collections.abc import Iterator
from typing import Any

import numpy as np


class RowSubset:
    """
    A lazy, read-only view of some rows (in the given order) of a 2-dimensional array that lives on disk, such as a
    :class:`numpy.memmap`. Nothing is copied: rows are read from the file when they are asked for. It holds on to the
    array, so the file lives as long as any :class:`RowSubset` of it.

    It supports ``len``, ``shape``, ``ndim``, ``dtype`` and ``nbytes``, and indexing by integers, slices, lists,
    integer arrays and boolean masks of the rows, optionally followed by an index of the columns (``x[rows, cols]``).
    These return ndarrays that hold only what was asked for. ``np.asarray(x)`` (also :meth:`to_numpy`) reads all of the
    rows into memory, which is how everything else that expects an ndarray (or something array-like, such as a
    :class:`pandas.DataFrame`) gets them. :meth:`take` makes a :class:`RowSubset` of a subset of the rows.

    Args:
        base: The array whose rows to view.
        positions: The rows of ``base`` that this holds, in order, as integers.
    """

    def __init__(self, base: np.ndarray, positions: np.ndarray):
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1:
            raise ValueError(f"positions must be 1-dimensional, got shape {positions.shape}.")
        if len(positions) > 0 and (positions.min() < 0 or positions.max() >= len(base)):
            raise IndexError(f"positions must lie in [0, {len(base)}).")
        self._base = base
        self._positions = positions

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._positions), *self._base.shape[1:])

    @property
    def ndim(self) -> int:
        return self._base.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._base.dtype

    @property
    def nbytes(self) -> int:
        """The size in bytes of the rows, as an ndarray in memory."""
        return int(np.prod(self.shape, dtype=np.int64)) * self._base.dtype.itemsize

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, key: Any) -> Any:
        parts = key if isinstance(key, tuple) else (key,)
        if any(part is Ellipsis or part is None for part in parts):
            raise IndexError("`...` and `None` are not supported; index np.asarray(x) instead.")
        rows = self._positions[parts[0]]
        result = self._base[(rows,) + parts[1:]]
        # a plain ndarray in memory, even where numpy returns a view of the file (for a single row)
        return np.array(result) if isinstance(result, np.ndarray) else result

    def __iter__(self) -> Iterator[np.ndarray]:
        # makes this list-like for pandas, which then converts it with __array__
        for i in range(len(self)):
            yield self[i]

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("This array is read-only.")

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        if copy is False:
            raise ValueError("The rows are read from disk, which needs a copy.")
        rows = np.asarray(self._base[self._positions])
        return rows if dtype is None else rows.astype(dtype, copy=False)

    def to_numpy(self) -> np.ndarray:
        """All of the rows, read into memory."""
        return np.asarray(self)

    def take(self, positions: Any) -> "RowSubset":
        """The :class:`RowSubset` of the rows ``positions`` of this one (still of the same file)."""
        return RowSubset(self._base, self._positions[np.asarray(positions)])

    def __repr__(self) -> str:
        return f"RowSubset(shape={self.shape}, dtype={self.dtype})"


def subset_rows(value: "np.memmap | RowSubset", positions: np.ndarray, max_bytes: int | None) -> Any:
    """
    The rows ``positions`` of an array that lives on disk (a :class:`numpy.memmap` or :class:`RowSubset`): as an ndarray
    in memory if that takes at most ``max_bytes`` (always, if ``max_bytes`` is ``None``), and otherwise as a lazy
    :class:`RowSubset` that reads them from the file when asked.
    """
    lazy = value.take(positions) if isinstance(value, RowSubset) else RowSubset(value, positions)
    if max_bytes is None or lazy.nbytes <= max_bytes:
        return np.asarray(lazy)
    return lazy


class ObsmWriter:
    """
    A context manager that allocates a per-cell array to be filled in (by assigning to :attr:`array`), in memory if it
    takes at most ``max_bytes`` and otherwise in a memory-mapped file in the temporary directory.

    On leaving the ``with`` block normally, :attr:`result` is the finished array: the ndarray itself, or for a file a
    read-only :class:`numpy.memmap` of it, which deletes the file when it is no longer referenced. If the block raises,
    the file is deleted straight away.

    Args:
        n_rows: The number of cells.
        n_cols: The number of columns.
        max_bytes: The most memory to use for the array. ``None`` for no limit.
        dtype: The dtype of the array.
    """

    def __init__(self, n_rows: int, n_cols: int, max_bytes: int | None, dtype: Any = np.float32):
        self.shape = (n_rows, n_cols)
        self.dtype = np.dtype(dtype)
        nbytes = n_rows * n_cols * self.dtype.itemsize
        #: whether the array is allocated in a file (because it does not fit in ``max_bytes``)
        self.on_disk = max_bytes is not None and nbytes > max_bytes
        self._directory: str | None = None
        self._array: np.ndarray | None = None
        self._result: np.ndarray | None = None

    def __enter__(self) -> "ObsmWriter":
        if self.on_disk:
            self._directory = tempfile.mkdtemp(prefix="cellarium_obsm_")
            path = f"{self._directory}/obsm.npy"
            self._array = np.lib.format.open_memmap(path, mode="w+", dtype=self.dtype, shape=self.shape)
        else:
            self._array = np.zeros(self.shape, dtype=self.dtype)
        return self

    @property
    def array(self) -> np.ndarray:
        """The array to fill in."""
        if self._array is None:
            raise RuntimeError("The array is only available inside the `with` block.")
        return self._array

    @property
    def result(self) -> np.ndarray:
        """The finished array, available after the ``with`` block."""
        if self._result is None:
            raise RuntimeError("The result is only available after the `with` block has completed.")
        return self._result

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        array, self._array = self._array, None
        if not self.on_disk:
            self._result = array
            return
        assert isinstance(array, np.memmap) and self._directory is not None
        if exc_type is not None:
            del array
            shutil.rmtree(self._directory, ignore_errors=True)
            return
        array.flush()
        del array
        result = np.load(f"{self._directory}/obsm.npy", mmap_mode="r")
        # the file goes when nothing refers to the array any more (slices and RowSubsets refer to it), or at exit
        weakref.finalize(result, shutil.rmtree, self._directory, True)
        self._result = result
