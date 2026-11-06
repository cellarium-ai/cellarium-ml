# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Collecting the rows of a large sparse matrix as they arrive in pieces, and putting them together in a different order.
"""

import os
import shutil
import tempfile
import weakref
from typing import BinaryIO

import numpy as np
import scipy.sparse as sp

# The entries of the collected rows are read back, when the rows are put together, in windows of this many entries
# (40 MB of float32, and of int32 column indices), which are looked at one after the other.
_SLAB_ENTRIES = 10_000_000


class SparseRowCollector:
    """
    Collects the rows of a sparse matrix that arrive in pieces (each a :class:`scipy.sparse.csr_matrix` of some of the
    rows, with the position that each row has in the final matrix), and :meth:`finish` assembles them into one CSR
    matrix with the rows in the order of their positions.

    This is what ``scipy.sparse.vstack`` of the pieces and then a reordering of the rows of the result do, but the
    pieces are kept in files in the temporary directory, not in memory: they are in the operating system's cache, which
    gives the memory back when it is needed, so only the matrix that is assembled takes memory (the pieces and the
    matrix are not in memory together, which would double what is needed). :meth:`finish` makes the matrix (which has
    the size that it needs, now that all the rows are known) from the files, which it deletes. The files are deleted
    also if the collector is dropped, or the interpreter exits, without :meth:`finish`.
    """

    def __init__(self) -> None:
        self._directory: str | None = None
        self._files: tuple[BinaryIO, BinaryIO] | None = None  # the values and the column indices, appended to
        self._finalizer: weakref.finalize | None = None
        self._n_entries = 0
        self._row_nnz: list[np.ndarray] = []
        self._positions: list[np.ndarray] = []
        self._n_cols: int | None = None

    def __len__(self) -> int:
        """The number of rows collected."""
        return sum(len(positions) for positions in self._positions)

    def _paths(self) -> tuple[str, str]:
        assert self._directory is not None
        return os.path.join(self._directory, "values.f32"), os.path.join(self._directory, "columns.i32")

    def _open_files(self) -> tuple[BinaryIO, BinaryIO]:
        if self._files is None:
            self._directory = tempfile.mkdtemp(prefix="cellarium_rows_")
            self._finalizer = weakref.finalize(self, shutil.rmtree, self._directory, True)
            values_path, columns_path = self._paths()
            self._files = (open(values_path, "wb"), open(columns_path, "wb"))
        return self._files

    def close(self) -> None:
        """Forget the rows collected, and delete the files that hold them."""
        if self._files is not None:
            for file in self._files:
                file.close()
        if self._finalizer is not None:
            self._finalizer()  # deletes the directory
        self._directory, self._files, self._finalizer = None, None, None
        self._n_entries = 0
        self._row_nnz, self._positions, self._n_cols = [], [], None

    def append(self, rows: sp.csr_matrix, positions: np.ndarray) -> None:
        """Add the rows of ``rows`` (a CSR matrix), which go to ``positions`` (one for each row) of the result."""
        if rows.shape[0] != len(positions):
            raise ValueError(f"{rows.shape[0]} rows but {len(positions)} positions.")
        if self._n_cols is None:
            self._n_cols = rows.shape[1]
        elif rows.shape[1] != self._n_cols:
            raise ValueError(f"The rows have {rows.shape[1]} columns, but earlier ones had {self._n_cols}.")
        values_file, columns_file = self._open_files()
        np.ascontiguousarray(rows.data, dtype=np.float32).tofile(values_file)
        np.ascontiguousarray(rows.indices, dtype=np.int32).tofile(columns_file)
        self._n_entries += len(rows.data)
        self._row_nnz.append(np.diff(rows.indptr).astype(np.int64))
        self._positions.append(np.asarray(positions, dtype=np.int64))

    def finish(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """
        The CSR matrix of all the rows collected, in the order of their positions, and the order itself: ``order[i]`` is
        the number (in the order that the rows were appended in) of the row that is row ``i`` of the matrix, for putting
        anything else that goes with the rows (such as their names) in the same order. The collector is empty after this
        (its files are deleted).

        The matrix has float32 values and int32 column indices, or int64 ones if it has more than 2**31 - 1 entries (a
        scipy CSR matrix has one dtype for its indices, and its row pointers need it).
        """
        if self._n_cols is None:
            raise ValueError("No rows were collected.")
        n_cols, n_entries = self._n_cols, self._n_entries
        positions = np.concatenate(self._positions)
        row_nnz = np.concatenate(self._row_nnz)
        n_rows = len(positions)
        order = np.argsort(positions, kind="stable")
        self._positions, self._row_nnz = [], []

        index_dtype = np.int32 if n_entries <= np.iinfo(np.int32).max else np.int64
        # where each row starts in the files (the rows as they came) and in the result
        arrived = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(row_nnz, out=arrived[1:])
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(row_nnz[order], out=indptr[1:])
        rank = np.empty(n_rows, dtype=np.int64)
        rank[order] = np.arange(n_rows)
        start_in_result = indptr[rank]
        del rank, row_nnz

        values = np.empty(n_entries, dtype=np.float32)
        columns = np.empty(n_entries, dtype=index_dtype)
        if n_entries > 0:
            assert self._files is not None
            for file in self._files:
                file.close()
            self._files = None
            values_path, columns_path = self._paths()
            source_values = np.memmap(values_path, dtype=np.float32, mode="r", shape=(n_entries,))
            source_columns = np.memmap(columns_path, dtype=np.int32, mode="r", shape=(n_entries,))
            for window_start in range(0, n_entries, _SLAB_ENTRIES):
                window_stop = min(window_start + _SLAB_ENTRIES, n_entries)
                # the rows that have entries in this window (one that starts in an earlier window, or ends in a later
                # one, is copied in parts)
                first = int(np.searchsorted(arrived, window_start, side="right")) - 1
                last = int(np.searchsorted(arrived, window_stop, side="left"))
                row_starts = arrived[first : last + 1].tolist()
                result_starts = start_in_result[first:last].tolist()
                for i in range(last - first):
                    begin, end = max(row_starts[i], window_start), min(row_starts[i + 1], window_stop)
                    if end <= begin:  # an empty row
                        continue
                    target = result_starts[i] + (begin - row_starts[i])
                    values[target : target + end - begin] = source_values[begin:end]
                    columns[target : target + end - begin] = source_columns[begin:end]
            del source_values, source_columns
        self.close()

        result = sp.csr_matrix((values, columns, indptr.astype(index_dtype)), shape=(n_rows, n_cols))
        return result, order
