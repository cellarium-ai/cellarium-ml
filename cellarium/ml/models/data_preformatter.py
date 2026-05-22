# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from typing import Any

import numpy as np
import pyarrow as pa
import torch

from cellarium.ml.models.model import CellariumModel, PredictMixin


class DataPreformatter(CellariumModel, PredictMixin):
    r"""
    Writes processed batches to Arrow IPC (Feather v2) files.

    Intended to be used as the model in a ``predict`` CLI run, converting h5ad
    shards (with all ``cpu_transforms`` and ``transforms`` already applied) into
    fast Arrow shards that can be loaded without any AnnData / HDF5 overhead.

    **Naming conventions used when classifying batch fields:**

    * Fields whose key ends in ``_g`` (e.g. ``var_names_g``, ``mean_g``) but
      **not** ``_ng`` are per-gene constants → stored in Arrow **schema metadata**.
    * Fields whose key ends in ``_categories`` are global category arrays →
      stored in Arrow **schema metadata** as JSON-encoded string lists.
    * All other fields are per-cell → stored as Arrow **columns**.

    **Per-cell column encoding:**

    * 2-D numeric arrays (e.g. ``x_ng``) → ``FixedSizeBinary(n_cols × 2)``, each
      row packed as ``float16``.
    * String / object arrays → ``large_utf8``.
    * Integer arrays → ``int32``.
    * All other numerics → ``float32``.

    **Output file naming:**

    ``<output_dir>/rank<rank:02d>_shard<counter:06d>.arrow``

    where *rank* is the PyTorch distributed rank (0 for single-process runs).

    .. note::

        When running with multiple distributed ranks each rank writes its own
        series of files starting from counter 0.  After preformatting you can
        renumber the files into a single flat sequence if desired.

    Args:
        output_dir:
            Directory in which Arrow IPC files will be written.  Created
            automatically if it does not exist.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self._shard_counter: int = 0

    def reset_parameters(self) -> None:
        # No parameters or buffers to reset.
        pass

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

    def _write_arrow(self, batch: dict[str, Any]) -> None:
        """Classify batch fields and write one Arrow IPC shard."""
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
                # Matrix field (e.g., x_ng): pack each row as float16 bytes
                val_f16 = val.astype(np.float16)
                n, d = val_f16.shape
                byte_width = d * 2
                col = pa.array(
                    [val_f16[i].tobytes() for i in range(n)],
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

        rank = self._get_rank()
        filename = os.path.join(self.output_dir, f"rank{rank:02d}_shard{self._shard_counter:06d}.arrow")
        with pa.OSFile(filename, "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                writer.write_batch(record_batch)

        self._shard_counter += 1

    # ------------------------------------------------------------------
    # CellariumModel interface
    # ------------------------------------------------------------------

    def forward(self, **kwargs: Any) -> dict:
        """
        Write one Arrow IPC shard from the batch and return an empty dict
        (no loss is produced).

        The ``**kwargs`` annotation causes :func:`~cellarium.ml.utilities.core.call_func_with_batch`
        to pass all batch keys to this method.
        """
        self._write_arrow(kwargs)
        return {}

    # ------------------------------------------------------------------
    # PredictMixin interface
    # ------------------------------------------------------------------

    def predict(self, **kwargs: Any) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Write one Arrow IPC shard from the batch during predict mode.

        The ``**kwargs`` annotation causes :func:`~cellarium.ml.utilities.core.call_func_with_batch`
        to pass all batch keys to this method.
        """
        self._write_arrow(kwargs)
        return {}
