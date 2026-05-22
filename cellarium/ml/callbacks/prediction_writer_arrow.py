# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import threading
import warnings
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pyarrow as pa
import torch


class _BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with a bounded submission queue.

    Blocks on :meth:`submit` when *max_queue_size* tasks are already in flight,
    preventing the caller from queuing an unbounded number of batches in memory.
    """

    def __init__(self, max_workers: int, max_queue_size: int) -> None:
        self._bound_queue: Queue = Queue(max_queue_size)
        super().__init__(max_workers=max_workers)

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        self._bound_queue.put(None)  # blocks when all slots are occupied
        future = super().submit(fn, *args, **kwargs)

        def _done(_: Future) -> None:
            self._bound_queue.get()

        future.add_done_callback(_done)
        return future


class PredictionWriterArrow(pl.callbacks.BasePredictionWriter):
    r"""
    Write prediction batches to Arrow IPC (Feather v2) files in parallel.

    Intended to be paired with :class:`~cellarium.ml.models.DataPreformatter` (or
    any other :class:`~cellarium.ml.models.PredictMixin` model) in a ``predict``
    CLI run.  After all ``cpu_transforms`` and ``transforms`` in the pipeline have
    been applied, the accumulated post-transform batch is received via
    ``write_on_batch_end`` and written asynchronously to disk.

    **Field classification** (same conventions as the old ``DataPreformatter``):

    * Keys ending in ``_g`` (but **not** ``_ng``) or ``_categories`` → Arrow
      **schema metadata** (written once per shard, not per row).
    * All other array keys → Arrow **columns** (one value per cell).

    **Per-cell column encoding:**

    * 2-D numeric arrays (e.g. ``x_ng``) → ``FixedSizeBinary(n_cols × itemsize)``,
      each row packed as the dtype given by *matrix_dtype*.
    * String / object arrays → ``large_utf8``.
    * Integer arrays → ``int32``.
    * All other numerics → ``float32``.

    **Output file naming:**

    ``<output_dir>/rank<rank:02d>_shard<counter:06d>.arrow``

    .. note::

        Set ``return_predictions: false`` in the top-level trainer config to
        prevent Lightning from accumulating all predictions in memory::

            return_predictions: false

    .. note::

        When running with multiple distributed ranks each rank writes its own
        series of files starting from counter 0.

    Args:
        output_dir:
            Directory in which Arrow IPC files will be written. Created
            automatically if it does not exist.
        compression:
            Compression codec for Arrow IPC files. ``"zstd"`` (default) gives
            the best ratio for sparse integer count data; ``"lz4"`` is faster
            to decompress; ``None`` disables compression.
        num_write_workers:
            Number of threads encoding and writing Arrow shards in parallel.
            Submission blocks when all workers are busy, bounding peak extra
            memory to roughly
            ``num_write_workers × shard_size × n_genes × itemsize`` bytes.
        matrix_dtype:
            NumPy dtype name for storing 2-D matrix fields (e.g. ``x_ng``).
            ``"float32"`` (default) preserves full single-precision range.
            ``"float16"`` halves file size but has a maximum representable
            value of 65504 — values outside ``[−65504, 65504]`` are silently
            **clamped** to ``±65504`` and a :class:`RuntimeWarning` with
            per-gene overflow counts is emitted by :meth:`flush` at the end
            of prediction.
    """

    def __init__(
        self,
        output_dir: str,
        compression: str | None = "zstd",
        num_write_workers: int = 8,
        matrix_dtype: str = "float32",
    ) -> None:
        super().__init__(write_interval="batch")
        _allowed_dtypes = {"float32", "float16"}
        if matrix_dtype not in _allowed_dtypes:
            raise ValueError(f"matrix_dtype must be one of {sorted(_allowed_dtypes)!r}, got {matrix_dtype!r}")
        self.output_dir = output_dir
        self.compression = compression
        self.num_write_workers = num_write_workers
        self.matrix_dtype = matrix_dtype
        self._matrix_np_dtype = np.dtype(matrix_dtype)
        self._shard_counter: int = 0
        self._executor: _BoundedThreadPoolExecutor = _BoundedThreadPoolExecutor(
            max_workers=num_write_workers, max_queue_size=num_write_workers
        )
        self._futures: list[Future] = []
        # Overflow tracking — only populated when matrix_dtype="float16"
        self._overflow_lock = threading.Lock()
        self._overflow_counts: dict[str, np.ndarray] = {}  # field → per-gene int64 overflow count
        self._var_names_g: np.ndarray | None = None  # captured for overflow reporting

    def __del__(self) -> None:
        """Ensure the executor shuts down on object deletion."""
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(val: Any) -> np.ndarray | None:
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        if isinstance(val, np.ndarray):
            return val
        return None

    @staticmethod
    def _get_rank() -> int:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    def _write_arrow(self, batch: dict[str, Any], filename: str | None = None) -> None:
        """Classify batch fields and write one Arrow IPC shard.

        When *filename* is ``None`` (synchronous / direct-call path) the filename
        is computed here and ``_shard_counter`` is incremented. When *filename*
        is provided (async path from :meth:`_submit_write`) the caller has already
        assigned the name and incremented the counter in the main thread to
        guarantee sequential ordering.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Convert tensors → numpy; skip non-array scalars
        np_batch: dict[str, np.ndarray] = {}
        for key, val in batch.items():
            arr = self._to_numpy(val)
            if arr is not None:
                np_batch[key] = arr

        if not np_batch:
            return

        # Infer n_obs from the first array with ndim >= 1
        n_obs: int | None = None
        for val in np_batch.values():
            if val.ndim >= 1:
                n_obs = val.shape[0]
                break
        if n_obs is None:
            return

        # Classify: schema metadata vs per-row columns
        # Rule: keys ending in `_g` (but not `_ng`) or `_categories` → metadata
        per_row: dict[str, np.ndarray] = {}
        metadata_arrays: dict[str, np.ndarray] = {}
        for key, val in np_batch.items():
            if key.endswith("_categories") or (key.endswith("_g") and not key.endswith("_ng")):
                metadata_arrays[key] = val
            else:
                per_row[key] = val

        # Build Arrow columns for per-row fields
        arrow_columns: list[pa.Array] = []
        arrow_fields: list[pa.Field] = []

        for key, val in per_row.items():
            if val.ndim == 2:
                n, d = val.shape
                if self._matrix_np_dtype == np.dtype("float16"):
                    with np.errstate(over="ignore"):
                        val_cast = val.astype(np.float16)
                    overflow_mask = np.isinf(val_cast)  # True where overflow occurred
                    if np.any(overflow_mask):
                        f16_max = np.finfo(np.float16).max
                        val_cast = val_cast.copy()
                        val_cast[val_cast == np.float16(np.inf)] = np.float16(f16_max)
                        val_cast[val_cast == np.float16(-np.inf)] = np.float16(-f16_max)
                        per_gene_counts = overflow_mask.sum(axis=0).astype(np.int64)
                        with self._overflow_lock:
                            if key not in self._overflow_counts:
                                self._overflow_counts[key] = np.zeros(d, dtype=np.int64)
                            self._overflow_counts[key] += per_gene_counts
                            if self._var_names_g is None and "var_names_g" in metadata_arrays:
                                self._var_names_g = metadata_arrays["var_names_g"].copy()
                else:
                    val_cast = val.astype(self._matrix_np_dtype)
                byte_width = d * self._matrix_np_dtype.itemsize
                col = pa.array(
                    [val_cast[i].tobytes() for i in range(n)],
                    type=pa.binary(byte_width),
                )
                arrow_fields.append(pa.field(key, pa.binary(byte_width)))
            elif val.dtype.kind in ("U", "O", "S"):
                col = pa.array(val.tolist(), type=pa.large_utf8())
                arrow_fields.append(pa.field(key, pa.large_utf8()))
            elif val.dtype.kind in ("i", "u"):
                col = pa.array(val.astype(np.int32), type=pa.int32())
                arrow_fields.append(pa.field(key, pa.int32()))
            else:
                col = pa.array(val.astype(np.float32), type=pa.float32())
                arrow_fields.append(pa.field(key, pa.float32()))
            arrow_columns.append(col)

        # Build schema metadata
        schema_meta: dict[bytes, bytes] = {
            b"cellarium_arrow_version": b"1",
            b"n_obs": str(n_obs).encode(),
            b"matrix_dtype": self._matrix_np_dtype.name.encode(),
        }
        for key, val in metadata_arrays.items():
            if key == "var_names_g" or (key.endswith("_g") and val.dtype.kind in ("U", "O")):
                schema_meta[key.encode()] = "\n".join(val.tolist()).encode()
                if key == "var_names_g":
                    schema_meta[b"n_genes"] = str(len(val)).encode()
            else:
                schema_meta[key.encode()] = json.dumps(val.tolist()).encode()

        schema = pa.schema(arrow_fields, metadata=schema_meta)
        record_batch = pa.record_batch(arrow_columns, schema=schema)

        if filename is None:
            # Synchronous path: compute filename and advance counter here.
            rank = self._get_rank()
            filename = os.path.join(self.output_dir, f"rank{rank:02d}_shard{self._shard_counter:06d}.arrow")
            self._shard_counter += 1

        options = pa.ipc.IpcWriteOptions(compression=self.compression)
        with pa.OSFile(filename, "wb") as sink:
            with pa.ipc.new_file(sink, schema, options=options) as writer:
                writer.write_batch(record_batch)

    def _submit_write(self, batch: dict[str, Any]) -> None:
        """Assign filename in the main thread then submit encoding + write to the thread pool."""
        rank = self._get_rank()
        os.makedirs(self.output_dir, exist_ok=True)
        filename = os.path.join(self.output_dir, f"rank{rank:02d}_shard{self._shard_counter:06d}.arrow")
        self._shard_counter += 1
        future: Future = self._executor.submit(self._write_arrow, batch, filename)
        self._futures.append(future)
        # Opportunistically prune completed futures and surface any write errors.
        live: list[Future] = []
        for f in self._futures:
            if f.done():
                f.result()  # re-raises if the thread threw an exception
            else:
                live.append(f)
        self._futures = live

    def flush(self) -> None:
        """Drain all pending write threads, report float16 overflow warnings, and re-raise write errors."""
        self._executor.shutdown(wait=True)
        for f in self._futures:
            f.result()
        self._futures.clear()

        # Collect overflow state and reset under lock so worker threads see a clean slate next run.
        with self._overflow_lock:
            overflow_snapshot = {k: v.copy() for k, v in self._overflow_counts.items()}
            var_names = self._var_names_g.copy() if self._var_names_g is not None else None
            self._overflow_counts.clear()
            self._var_names_g = None

        for field, counts in overflow_snapshot.items():
            total = int(counts.sum())
            if total == 0:
                continue
            affected_indices = np.where(counts > 0)[0]
            n_affected = len(affected_indices)
            top_idx = affected_indices[np.argsort(counts[affected_indices])[::-1][:10]]
            if var_names is not None and len(var_names) == len(counts):
                gene_str = ", ".join(f"{var_names[i]} ({counts[i]})" for i in top_idx)
            else:
                gene_str = ", ".join(f"idx={i} ({counts[i]})" for i in top_idx)
            warnings.warn(
                f"PredictionWriterArrow: {total} value(s) in field '{field}' overflowed float16 "
                f"and were clamped to \u00b1{np.finfo(np.float16).max}. "
                f"Affected genes: {n_affected}. Top genes (up to 10): {gene_str}. "
                f"Consider using matrix_dtype='float32' to avoid precision loss.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Recreate the executor so the callback can be reused across multiple predict runs.
        self._executor = _BoundedThreadPoolExecutor(
            max_workers=self.num_write_workers, max_queue_size=self.num_write_workers
        )

    # ------------------------------------------------------------------
    # BasePredictionWriter interface
    # ------------------------------------------------------------------

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: dict[str, Any],
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Submit the post-transform batch to the async write pool."""
        self._submit_write(prediction)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Drain the write pool and re-raise any write exceptions."""
        self.flush()
