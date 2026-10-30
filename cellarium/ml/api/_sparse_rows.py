# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Collecting the rows of a large sparse matrix as they arrive in pieces, and putting them together in a different order.
"""

import numpy as np
import scipy.sparse as sp

# The values of the rows are kept in slabs of this many entries (40 MB of float32, and of int32 column indices). That is
# more than the largest size that the C allocator serves from its heap (32 MB), so a slab is its own memory mapping, and
# the memory goes back to the operating system as soon as the slab is freed. A numpy array for each piece (a few MB)
# would be served from the heap instead, where freed memory in the middle is not returned, and the matrix being
# assembled would be on top of the pieces that it is made of.
_SLAB_ENTRIES = 10_000_000


class SparseRowCollector:
    """
    Collects the rows of a sparse matrix that arrive in pieces (each a :class:`scipy.sparse.csr_matrix` of some of the
    rows, with the position that each row has in the final matrix), and :meth:`finish` assembles them into one CSR
    matrix with the rows in the order of their positions.

    This is what ``scipy.sparse.vstack`` of the pieces and then a reordering of the rows of the result do, with about
    one copy of the values in memory at a time instead of two or three: the values of the pieces are copied into slabs
    as they come, and :meth:`finish` fills in the matrix (which has the size that it needs, now that all the rows are
    known) from the slabs, which it frees one at a time as it goes.
    """

    def __init__(self) -> None:
        self._slabs: list[tuple[np.ndarray, np.ndarray] | None] = []  # (values, column indices)
        self._fill = _SLAB_ENTRIES  # the entries used in the last slab (initially "full", so the first piece adds one)
        self._row_nnz: list[np.ndarray] = []
        self._positions: list[np.ndarray] = []
        self._n_cols: int | None = None

    def __len__(self) -> int:
        """The number of rows collected."""
        return sum(len(positions) for positions in self._positions)

    def append(self, rows: sp.csr_matrix, positions: np.ndarray) -> None:
        """Add the rows of ``rows`` (a CSR matrix), which go to ``positions`` (one for each row) of the result."""
        if rows.shape[0] != len(positions):
            raise ValueError(f"{rows.shape[0]} rows but {len(positions)} positions.")
        if self._n_cols is None:
            self._n_cols = rows.shape[1]
        elif rows.shape[1] != self._n_cols:
            raise ValueError(f"The rows have {rows.shape[1]} columns, but earlier ones had {self._n_cols}.")
        self._row_nnz.append(np.diff(rows.indptr).astype(np.int64))
        self._positions.append(np.asarray(positions, dtype=np.int64))
        values = rows.data.astype(np.float32, copy=False)
        columns = rows.indices.astype(np.int32, copy=False)
        n_entries, written = len(values), 0
        while written < n_entries:
            if self._fill == _SLAB_ENTRIES:
                self._slabs.append((np.empty(_SLAB_ENTRIES, np.float32), np.empty(_SLAB_ENTRIES, np.int32)))
                self._fill = 0
            slab_values, slab_columns = self._slabs[-1]  # type: ignore[misc]
            n = min(_SLAB_ENTRIES - self._fill, n_entries - written)
            slab_values[self._fill : self._fill + n] = values[written : written + n]
            slab_columns[self._fill : self._fill + n] = columns[written : written + n]
            self._fill += n
            written += n

    def finish(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """
        The CSR matrix of all the rows collected, in the order of their positions, and the order itself: ``order[i]`` is
        the number (in the order that the rows were appended in) of the row that is row ``i`` of the matrix, for putting
        anything else that goes with the rows (such as their names) in the same order. The collector is empty after this
        (its slabs are freed as they are used).

        The matrix has float32 values and int32 column indices, or int64 ones if it has more than 2**31 - 1 entries (a
        scipy CSR matrix has one dtype for its indices, and its row pointers need it).
        """
        if self._n_cols is None:
            raise ValueError("No rows were collected.")
        positions = np.concatenate(self._positions)
        row_nnz = np.concatenate(self._row_nnz)
        n_rows, n_cols = len(positions), self._n_cols
        order = np.argsort(positions, kind="stable")
        self._positions, self._row_nnz = [], []

        n_entries = int(row_nnz.sum())
        index_dtype = np.int32 if n_entries <= np.iinfo(np.int32).max else np.int64
        # where each row starts in the pieces as they came (the slabs, one after the other) and in the result
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
        for k in range(len(self._slabs)):
            slab = self._slabs[k]
            assert slab is not None
            slab_start = k * _SLAB_ENTRIES
            slab_stop = min(slab_start + _SLAB_ENTRIES, n_entries)
            # the rows that have entries in this slab (one that starts in an earlier slab, or ends in a later one, is
            # copied in parts)
            first = int(np.searchsorted(arrived, slab_start, side="right")) - 1
            last = int(np.searchsorted(arrived, slab_stop, side="left"))
            row_starts = arrived[first : last + 1].tolist()
            result_starts = start_in_result[first:last].tolist()
            for i in range(last - first):
                begin, end = max(row_starts[i], slab_start), min(row_starts[i + 1], slab_stop)
                if end <= begin:  # an empty row
                    continue
                target = result_starts[i] + (begin - row_starts[i])
                values[target : target + end - begin] = slab[0][begin - slab_start : end - slab_start]
                columns[target : target + end - begin] = slab[1][begin - slab_start : end - slab_start]
            self._slabs[k] = None  # free it now: the result is being made from the rest
        self._slabs, self._fill = [], _SLAB_ENTRIES

        result = sp.csr_matrix((values, columns, indptr.astype(index_dtype)), shape=(n_rows, n_cols))
        return result, order
