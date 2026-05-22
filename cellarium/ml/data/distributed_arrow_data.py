# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
from boltons.cacheutils import LRU
from braceexpand import braceexpand


class DistributedArrowDataCollection:
    r"""
    Distributed collection of pre-formatted Arrow IPC shards.

    Each shard is an Arrow IPC file (Feather v2 layout) written by
    :class:`~cellarium.ml.models.DataPreformatter`.  Per-cell fields are stored as
    Arrow columns; per-gene and global category fields are stored in the Arrow
    schema metadata so they are read once per shard rather than once per cell.

    The :meth:`__getitem__` method returns a ``dict[str, numpy.ndarray]`` directly
    (no :class:`~cellarium.ml.utilities.data.AnnDataField` mapping is required),
    so ``batch_keys`` must be set to ``None`` in the corresponding
    :class:`~cellarium.ml.core.CellariumDataModule`.

    Schema metadata keys written by :class:`~cellarium.ml.models.DataPreformatter`:

    * ``b"cellarium_arrow_version"`` – format version string (``"1"``)
    * ``b"n_obs"`` – number of cells in the shard
    * ``b"n_genes"`` – number of genes / features
    * ``b"var_names_g"`` – newline-joined gene name string
    * ``b"<key>_categories"`` – JSON-encoded list of category strings for each
      categorical field

    Example::

        >>> dadc = DistributedArrowDataCollection(
        ...     "output/rank00_shard{000000..000999}.arrow",
        ...     shard_size=1800,
        ...     max_cache_size=2,
        ... )
        >>> batch = dadc[0:1800]   # returns dict[str, np.ndarray]

    Args:
        filenames:
            Names of Arrow IPC files.  May be a brace-expanded glob string or an
            explicit sequence of paths.
        limits:
            Cumulative cell counts (upper exclusive bound) for each shard.
            If ``None``, inferred from ``shard_size`` and ``last_shard_size``.
        shard_size:
            Number of cells in each shard.  Required when ``limits`` is ``None``.
        last_shard_size:
            Cell count of the final shard when it differs from ``shard_size``.
            Only meaningful when ``shard_size`` is also set.
        max_cache_size:
            Maximum number of Arrow record batches to keep in the LRU cache.
        cache_size_strictly_enforced:
            If ``True``, raise an error when a single ``__getitem__`` call would
            require more shards than ``max_cache_size``.
    """

    def __init__(
        self,
        filenames: Sequence[str] | str,
        limits: list[int] | None = None,
        shard_size: int | None = None,
        last_shard_size: int | None = None,
        max_cache_size: int = 1,
        cache_size_strictly_enforced: bool = True,
    ) -> None:
        self.filenames = list(braceexpand(filenames) if isinstance(filenames, str) else filenames)

        if shard_size is None and last_shard_size is not None:
            raise ValueError("If `last_shard_size` is specified then `shard_size` must also be specified.")
        if limits is None:
            if shard_size is None:
                raise ValueError("If `limits` is `None` then `shard_size` must be specified.")
            limits = [shard_size * (i + 1) for i in range(len(self.filenames))]
            if last_shard_size is not None:
                limits[-1] = limits[-1] - shard_size + last_shard_size
        else:
            limits = list(limits)

        if len(limits) != len(self.filenames):
            raise ValueError(
                f"The number of limits ({len(limits)}) must match the number of filenames ({len(self.filenames)})."
            )

        self.limits = limits
        self.max_cache_size = max_cache_size
        self.cache_size_strictly_enforced = cache_size_strictly_enforced
        self.cache: LRU = LRU(max_cache_size)

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def n_obs(self) -> int:
        """Total number of observations (cells) across all shards."""
        return self.limits[-1]

    @property
    def n_vars(self) -> int:
        """Number of variables (genes/features)."""
        meta = self._read_schema_metadata(self.filenames[0])
        return int(meta.get(b"n_genes", b"0"))

    def __len__(self) -> int:
        return self.n_obs

    # ------------------------------------------------------------------
    # Schema metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_schema_metadata(filename: str) -> dict[bytes, bytes]:
        """Read Arrow schema metadata from a file without loading cell data."""
        with pa.memory_map(filename, "r") as source:
            # open_file reads only the footer (schema + batch offsets) up front
            meta = dict(pa.ipc.open_file(source).schema.metadata or {})
        return meta

    def get_schema_metadata(self) -> dict[str, Any]:
        """
        Return schema-level metadata without reading any cell data.

        Reads the Arrow schema footer from the first shard only.

        Returns:
            A dict containing:

            * ``"n_obs"`` – total observation count (:class:`int`)
            * ``"n_vars"`` – number of genes (:class:`int`)
            * ``"var_names_g"`` – :class:`numpy.ndarray` of gene names
            * ``"<key>_categories"`` – :class:`numpy.ndarray` of category strings for
              every categorical batch field stored in the Arrow metadata
        """
        meta = self._read_schema_metadata(self.filenames[0])

        result: dict[str, Any] = {
            "n_obs": self.n_obs,
            "n_vars": int(meta.get(b"n_genes", b"0")),
        }

        if b"var_names_g" in meta:
            raw = meta[b"var_names_g"].decode()
            result["var_names_g"] = np.array(raw.split("\n")) if raw else np.array([], dtype=str)

        for k, v in meta.items():
            k_str = k.decode()
            if k_str.endswith("_categories"):
                result[k_str] = np.array(json.loads(v.decode()))

        return result

    # ------------------------------------------------------------------
    # Shard loading
    # ------------------------------------------------------------------

    def _get_shard(self, shard_idx: int) -> pa.RecordBatch:
        """Load a shard into the LRU cache and return it."""
        filename = self.filenames[shard_idx]
        if filename in self.cache:
            return self.cache[filename]

        with pa.memory_map(filename, "r") as source:
            batch = pa.ipc.open_file(source).get_batch(0)

        if len(self.cache) < self.cache.max_size:
            self.cache[filename] = batch

        return batch

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int | list[int] | slice) -> dict[str, np.ndarray]:
        """
        Return a dict of numpy arrays for the requested cell indices.

        Per-row columns are concatenated across shards if the request spans multiple
        shards.  Schema-level fields (``var_names_g``, ``*_categories``) are read
        from the first touched shard and added to the returned dict.

        Args:
            idx: A single cell index, a list of cell indices, or a slice.

        Returns:
            ``dict[str, numpy.ndarray]`` with one entry per Arrow column plus one
            entry per schema-metadata field.
        """
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(self.n_obs)))

        idx_arr = np.asarray(idx, dtype=np.intp)
        limits_arr = np.asarray(self.limits, dtype=np.intp)
        # shard_indices[i] = which shard cell idx_arr[i] belongs to
        shard_indices = np.searchsorted(limits_arr, idx_arr, side="right")

        unique_shards = np.unique(shard_indices)
        if self.cache_size_strictly_enforced and len(unique_shards) > self.max_cache_size:
            raise ValueError(
                f"Expected the number of Arrow shards ({len(unique_shards)}) to be "
                f"no more than the max cache size ({self.max_cache_size})."
            )

        parts: list[dict[str, np.ndarray]] = []
        metadata_dict: dict[str, np.ndarray] | None = None
        # perm[flat_i] = original request position of the i-th cell in shard-sorted order.
        # Used to restore the caller's requested ordering after concatenating parts.
        perm = np.empty(len(idx_arr), dtype=np.intp)
        flat_offset = 0

        for shard_idx in unique_shards:
            orig_positions = np.where(shard_indices == shard_idx)[0]
            perm[flat_offset : flat_offset + len(orig_positions)] = orig_positions
            flat_offset += len(orig_positions)

            global_idx = idx_arr[orig_positions]

            batch = self._get_shard(int(shard_idx))
            shard_start = int(self.limits[shard_idx - 1]) if shard_idx > 0 else 0
            local_idx = (global_idx - shard_start).tolist()

            part: dict[str, np.ndarray] = {}
            for col_name in batch.schema.names:
                col_taken = batch.column(col_name).take(local_idx)
                col_type = col_taken.type

                if pa.types.is_fixed_size_binary(col_type):
                    # x_ng: packed float16 – one fixed-size binary row per cell
                    byte_width: int = col_type.byte_width
                    n_genes = byte_width // 2
                    offset = col_taken.offset
                    values_buf = col_taken.buffers()[1]
                    arr_bytes = values_buf[offset * byte_width : (offset + len(col_taken)) * byte_width]
                    part[col_name] = np.frombuffer(arr_bytes, dtype=np.float16).reshape(len(local_idx), n_genes).copy()
                elif pa.types.is_dictionary(col_type):
                    part[col_name] = np.array(col_taken.to_pylist())
                elif pa.types.is_large_string(col_type) or pa.types.is_string(col_type):
                    part[col_name] = np.array(col_taken.to_pylist(), dtype=object)
                else:
                    part[col_name] = col_taken.to_numpy(zero_copy_only=False)

            # Read schema metadata once from the first shard we touch
            if metadata_dict is None:
                metadata_dict = {}
                meta = batch.schema.metadata or {}
                if b"var_names_g" in meta:
                    raw = meta[b"var_names_g"].decode()
                    metadata_dict["var_names_g"] = np.array(raw.split("\n")) if raw else np.array([], dtype=str)
                for k, v in meta.items():
                    k_str = k.decode()
                    if k_str.endswith("_categories"):
                        metadata_dict[k_str] = np.array(json.loads(v.decode()))

            parts.append(part)

        # Concatenate per-row fields across shards (result is in shard-sorted order)
        if len(parts) == 1:
            per_row: dict[str, np.ndarray] = parts[0]
        else:
            per_row = {}
            for key in parts[0]:
                per_row[key] = np.concatenate([p[key] for p in parts], axis=0)

        # Restore the caller's original request order via inverse permutation
        inv_perm = np.argsort(perm, kind="stable")
        result = {key: val[inv_perm] for key, val in per_row.items()}

        # Merge schema-level fields (constant, not per-row)
        if metadata_dict:
            result.update(metadata_dict)

        return result

    # ------------------------------------------------------------------
    # Pickling (worker handoff)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["cache"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Rebuild an EMPTY cache – no file I/O on worker init
        self.cache = LRU(self.max_cache_size)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DistributedArrowDataCollection with n_obs={self.n_obs}, n_shards={len(self.filenames)}"
