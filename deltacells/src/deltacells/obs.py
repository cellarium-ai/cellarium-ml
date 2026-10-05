# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Per-cell metadata ("obs") for deltacells datasets. Requires ``pyarrow`` and ``pandas`` (``pip install deltacells[obs]``).

*Authoring* (remote layout): the obs table is stored as parquet **shards** (``obs/obs_0000.parquet`` ...), each holding up to
``obs_tiles_per_file`` tiles with exactly **one row group per tile**, so that obs row group ``i`` always describes the cells of
tile ``i``. Categorical columns are stored as small integer codes (``-1`` = missing) with one **global vocabulary** per column in
``obs/schema.json``; columns with more than ``max_categories`` distinct values (default 20,000) are stored as plain strings.
Parquet statistics are disabled to keep footers small.

*Reading*: obs is never fetched together with tiles. The first time columns are requested they are **localized**: only those
columns' bytes are read from the shards (ranged reads, parallel across row groups) and written to a node-local cache as one
memory-mapped Arrow IPC file per column (one record batch per tile). Columns can be added later without touching the ones already
there; the cache is shared by every process of a node and guarded by a file lock. Lookups for a batch of cells then cost
microseconds and no network.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import logging
import os
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from deltacells.backends import Backend, RangeFile
from deltacells.manifest import OBS_SCHEMA_NAME, VAR_TABLE_NAME, Manifest

try:  # POSIX file locking (not available on Windows: localization is then not protected against concurrent processes)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger("deltacells.obs")

DEFAULT_MAX_CATEGORIES = 20_000
SCHEMA_FORMAT = "deltacells-obs"
SCHEMA_VERSION = 1
KINDS = ("category", "numeric", "bool", "string")


# ---------------------------------------------------------------------------------------------- schema


@dataclass
class ColumnSpec:
    """One stored obs column.

    ``kind`` is ``"category"`` (stored as integer codes into ``categories``, ``-1`` = missing), ``"numeric"``, ``"bool"`` or
    ``"string"``. ``arrow_type`` is the stored Arrow type (``"int16"`` for the codes of a categorical column).
    """

    name: str
    kind: str
    arrow_type: str
    categories: list[str] | None = None

    def pa_type(self) -> pa.DataType:
        return pa.type_for_alias(self.arrow_type)


def _classify(name: str, s: pd.Series) -> str:
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_numeric_dtype(dt):
        return "numeric"
    if pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt):
        return "string"
    raise ValueError(
        f"obs column {name!r} has unsupported dtype {dt}; exclude it (supported: categorical, numeric, bool, string)"
    )


def _codes_type(n_categories: int) -> str:
    return "int8" if n_categories <= 127 else "int16" if n_categories <= 32_767 else "int32"


def _to_str_object(s: pd.Series) -> pd.Series:
    """Values as Python strings, keeping missing values as ``None`` (so that NaN does not become the string 'nan')."""
    out = s.astype(object)
    mask = out.notna().to_numpy()
    out = out.where(mask, None)
    if mask.any():
        out[mask] = out[mask].map(str)
    return out


def _merge_numeric(name: str, a: str, b: str) -> str:
    """Common numeric Arrow type of ``a`` and ``b`` (type names as in ``str(pa.DataType)``, e.g. 'float' is float32)."""
    if a == b:
        return a
    try:
        merged = np.promote_types(pa.type_for_alias(a).to_pandas_dtype(), pa.type_for_alias(b).to_pandas_dtype())
        return str(pa.from_numpy_dtype(merged))
    except (TypeError, ValueError, pa.ArrowNotImplementedError) as e:  # pragma: no cover
        raise ValueError(f"obs column {name!r} has incompatible types {a} and {b} across the data") from e


class ObsSchema:
    """The stored layout of the obs table: column names, kinds, Arrow types and the global vocabulary of each categorical column."""

    def __init__(self, columns: Sequence[ColumnSpec]) -> None:
        names = [c.name for c in columns]
        if len(set(names)) != len(names):
            raise ValueError("duplicate obs column names")
        for c in columns:
            if c.kind not in KINDS:
                raise ValueError(f"unknown column kind {c.kind!r}")
            if (c.kind == "category") != (c.categories is not None):
                raise ValueError(f"column {c.name!r}: categories are required for (and only for) kind 'category'")
        self.columns = list(columns)
        self._by_name = {c.name: c for c in columns}

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> ColumnSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown obs column {name!r}; available: {self.names}") from None

    def arrow_schema(self) -> pa.Schema:
        return pa.schema([pa.field(c.name, c.pa_type()) for c in self.columns])

    def to_json(self) -> str:
        cols = [
            {"name": c.name, "kind": c.kind, "arrow_type": c.arrow_type, "categories": c.categories}
            for c in self.columns
        ]
        return json.dumps({"format": SCHEMA_FORMAT, "version": SCHEMA_VERSION, "columns": cols})

    @classmethod
    def from_json(cls, text: str | bytes) -> ObsSchema:
        d = json.loads(text)
        if d.get("format") != SCHEMA_FORMAT or d.get("version") != SCHEMA_VERSION:
            raise ValueError("not a deltacells obs schema (or an unsupported version)")
        return cls([ColumnSpec(**c) for c in d["columns"]])

    @classmethod
    def infer(
        cls,
        frames: Iterable[pd.DataFrame],
        *,
        max_categories: int = DEFAULT_MAX_CATEGORIES,
        columns: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
    ) -> ObsSchema:
        """Infer the schema from every obs frame of the dataset (so that the vocabularies are global).

        Categorical columns become ``"category"`` with the union of all frames' categories (in order of first appearance), unless
        that union exceeds ``max_categories``, in which case the column is stored as strings. Object / string columns become
        ``"string"``, booleans ``"bool"``, everything else numeric. A column must keep its kind across frames.
        """
        kinds: dict[str, str] = {}
        types: dict[str, str] = {}
        vocabs: dict[str, dict[str, None]] = {}
        order: list[str] = []
        n_frames = 0
        for df in frames:
            n_frames += 1
            names = [c for c in (columns if columns is not None else df.columns) if c not in set(exclude)]
            missing = [c for c in names if c not in df.columns]
            if missing:
                raise ValueError(f"obs frame {n_frames - 1} lacks columns {missing}")
            if n_frames > 1 and set(names) != set(order):
                raise ValueError(f"obs frame {n_frames - 1} has different columns than the first one")
            for name in names:
                s = df[name]
                kind = _classify(str(name), s)
                if name not in kinds:
                    order.append(name)
                    kinds[name] = kind
                elif kinds[name] != kind:
                    raise ValueError(f"obs column {name!r} is {kinds[name]!r} in some frames and {kind!r} in others")
                if kind == "category":
                    v = vocabs.setdefault(name, {})
                    if len(v) <= max_categories:  # stop growing once it is going to be a string column anyway
                        for c in s.cat.categories:
                            v.setdefault(str(c), None)
                elif kind in ("numeric", "bool"):
                    t = str(pa.Array.from_pandas(s).type)
                    if name in types and kind == "numeric":
                        t = _merge_numeric(str(name), types[name], t)
                    types[name] = t
        if n_frames == 0:
            raise ValueError("no obs frames given")
        specs = []
        for name in order:
            kind = kinds[name]
            if kind == "category":
                vocab = list(vocabs[name])
                if len(vocab) > max_categories:
                    specs.append(ColumnSpec(str(name), "string", "string"))
                else:
                    specs.append(ColumnSpec(str(name), "category", _codes_type(len(vocab)), vocab))
            elif kind == "string":
                specs.append(ColumnSpec(str(name), "string", "string"))
            elif kind == "bool":
                specs.append(ColumnSpec(str(name), "bool", "bool"))
            else:
                specs.append(ColumnSpec(str(name), "numeric", types[name]))
        return cls(specs)

    def _encode_column(self, spec: ColumnSpec, s: pd.Series) -> pa.Array:
        if spec.kind == "category":
            vocab = pd.Index(spec.categories)
            if isinstance(s.dtype, pd.CategoricalDtype):
                mapping = vocab.get_indexer(s.cat.categories.astype(str))
                codes = s.cat.codes.to_numpy()
                used = codes >= 0
                out = np.full(len(s), -1, dtype=np.int64)
                out[used] = mapping[codes[used]]
                unknown = used & (out == -1)
            else:
                values = _to_str_object(s)
                out = vocab.get_indexer(values.to_numpy())
                unknown = values.notna().to_numpy() & (out == -1)
            if unknown.any():
                bad = s[unknown].iloc[0]
                raise ValueError(f"obs column {spec.name!r}: value {bad!r} is not in the dataset's vocabulary")
            return pa.array(out.astype(spec.arrow_type))
        if spec.kind == "string":
            return pa.array(_to_str_object(s), type=pa.string(), from_pandas=True)
        return pa.Array.from_pandas(s, type=spec.pa_type())

    def encode(self, df: pd.DataFrame) -> pa.Table:
        """Convert a pandas obs frame into the stored representation (an Arrow table with this schema)."""
        names = set(df.columns)
        missing, extra = [c for c in self.names if c not in names], [c for c in df.columns if c not in self._by_name]
        if missing or extra:
            raise ValueError(f"obs columns do not match the schema (missing: {missing}, unexpected: {extra})")
        return pa.Table.from_arrays(
            [self._encode_column(c, df[c.name]) for c in self.columns], schema=self.arrow_schema()
        )


# ---------------------------------------------------------------------------------------------- writing


def _sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def obs_fingerprint(schema_json: str, tile_cells: Sequence[int], shard_sha256: Sequence[str]) -> str:
    """Identifies the obs content of a dataset (used to key the local cache so that a changed dataset never reuses stale files)."""
    payload = json.dumps(
        {"schema": schema_json, "tile_cells": list(tile_cells), "shards": list(shard_sha256)}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class ObsWriter:
    """Writes one encoded obs table per tile, in tile order, into parquet shards of ``tiles_per_file`` row groups."""

    def __init__(self, root: str, schema: ObsSchema, *, tiles_per_file: int, compression_level: int = 9) -> None:
        if tiles_per_file < 1:
            raise ValueError("obs_tiles_per_file must be at least 1")
        self.root, self.schema, self.tiles_per_file, self.level = root, schema, tiles_per_file, compression_level
        os.makedirs(os.path.join(root, "obs"), exist_ok=True)
        self._arrow_schema = schema.arrow_schema()
        self._writer: pq.ParquetWriter | None = None
        self._tmp: str | None = None
        self._n_tiles = 0
        self.shard_bytes: list[int] = []
        self.shard_sha256: list[str] = []

    def _shard_path(self, shard: int) -> str:
        return os.path.join(self.root, Manifest.obs_shard_name(shard))

    def _finish_shard(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        final = self._shard_path(len(self.shard_bytes))
        os.replace(self._tmp, final)
        self.shard_bytes.append(os.path.getsize(final))
        self.shard_sha256.append(_sha256_file(final))
        self._writer = self._tmp = None

    def add_tile(self, table: pa.Table) -> None:
        if not table.schema.equals(self._arrow_schema):
            raise ValueError("obs table does not match the schema")
        if self._n_tiles % self.tiles_per_file == 0:
            self._finish_shard()
            self._tmp = self._shard_path(self._n_tiles // self.tiles_per_file) + f".tmp{os.getpid()}"
            self._writer = pq.ParquetWriter(
                self._tmp,
                self._arrow_schema,
                compression="zstd",
                compression_level=self.level,
                use_dictionary=True,
                write_statistics=False,
            )
        assert self._writer is not None
        self._writer.write_table(table, row_group_size=max(1, table.num_rows))  # one row group per tile
        self._n_tiles += 1

    def close(self, tile_cells: Sequence[int]) -> tuple[str, str]:
        """Finish the last shard and write ``obs/schema.json``; returns ``(schema_json, fingerprint)``."""
        self._finish_shard()
        schema_json = self.schema.to_json()
        path = os.path.join(self.root, OBS_SCHEMA_NAME)
        with open(path + ".tmp", "w") as f:
            f.write(schema_json)
        os.replace(path + ".tmp", path)
        return schema_json, obs_fingerprint(schema_json, tile_cells, self.shard_sha256)


def write_var_table(root: str, var: pd.DataFrame) -> None:
    """Write the gene table (rows = output columns, in order; the index is kept) as ``var.parquet``."""
    path = os.path.join(root, VAR_TABLE_NAME)
    pq.write_table(pa.Table.from_pandas(var, preserve_index=True), path + ".tmp", compression="zstd")
    os.replace(path + ".tmp", path)


def read_var_table(backend: Backend) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(bytes(backend.read(VAR_TABLE_NAME)))).to_pandas()


# ---------------------------------------------------------------------------------------------- reading


def resolve_cache_dir(cache_dir: str | os.PathLike | None = None) -> Path:
    """``cache_dir`` if given, else ``$DELTACELLS_CACHE``, else ``$XDG_CACHE_HOME/deltacells`` / ``~/.cache/deltacells``."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    if os.environ.get("DELTACELLS_CACHE"):
        return Path(os.environ["DELTACELLS_CACHE"]).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "deltacells"


def _ordered_map(
    pool: ThreadPoolExecutor, fn: Callable[[Any], Any], items: Iterable[Any], window: int
) -> Iterator[Any]:
    """``map`` over a thread pool that yields results in order while keeping at most ``window`` tasks in flight."""
    it = iter(items)
    pending: deque = deque(pool.submit(fn, x) for x in itertools.islice(it, window))
    while pending:
        result = pending.popleft().result()
        for x in itertools.islice(it, 1):
            pending.append(pool.submit(fn, x))
        yield result


class _CorruptCache(RuntimeError):
    pass


class ObsStore:
    """Access to the obs table of a dataset through a local, additive, per-column cache. Get it from ``DeltaCellsDataset.obs``.

    Args:
        backend: Where the dataset lives.
        manifest: Its manifest.
        cache_dir: Cache root (see :func:`resolve_cache_dir`); files live in ``<root>/<fingerprint>/obs/``.
        threads: Concurrent row-group reads while localizing.
    """

    def __init__(
        self, backend: Backend, manifest: Manifest, *, cache_dir: str | os.PathLike | None = None, threads: int = 8
    ) -> None:
        if not manifest.has_obs:
            raise ValueError("this dataset has no obs")
        self.backend, self.manifest, self.threads = backend, manifest, threads
        self.cache_root = resolve_cache_dir(cache_dir)
        self._init_runtime()

    def _init_runtime(self) -> None:
        self._schema: ObsSchema | None = None
        self._dtypes: dict[str, pd.CategoricalDtype] = {}
        self._readers: dict[str, tuple[Any, Any]] = {}  # column -> (memory map, IPC reader)
        self._limits = self.manifest.limits
        self._starts = np.concatenate([[0], self._limits[:-1]]).astype(np.int64)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "manifest": self.manifest,
            "threads": self.threads,
            "cache_root": self.cache_root,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_runtime()

    def close(self) -> None:
        """Release the memory maps (they are reopened on demand)."""
        for mm, _ in self._readers.values():
            with contextlib.suppress(Exception):
                mm.close()
        self._readers.clear()

    # -- metadata

    @property
    def schema(self) -> ObsSchema:
        if self._schema is None:
            self._schema = ObsSchema.from_json(bytes(self.backend.read(OBS_SCHEMA_NAME)))
        return self._schema

    @property
    def columns(self) -> list[str]:
        return self.schema.names

    def kind(self, column: str) -> str:
        return self.schema[column].kind

    def categories(self, column: str) -> list[str]:
        """The global vocabulary of a categorical column: code ``i`` means ``categories[i]``, code ``-1`` means missing."""
        spec = self.schema[column]
        if spec.categories is None:
            raise ValueError(f"column {column!r} is not categorical (kind {spec.kind!r})")
        return list(spec.categories)

    def _check_columns(self, columns: Sequence[str]) -> list[str]:
        cols = [columns] if isinstance(columns, str) else list(columns)
        unknown = [c for c in cols if c not in self.schema]
        if unknown:
            raise KeyError(f"unknown obs columns {unknown}; available: {self.columns}")
        return cols

    # -- cache

    @property
    def cache_dir(self) -> Path:
        return self.cache_root / self.manifest.obs_fingerprint / "obs"

    def _path(self, column: str) -> Path:
        return self.cache_dir / f"{quote(column, safe='')}.arrow"

    def localized_columns(self) -> list[str]:
        """Columns already present in the local cache."""
        return [c for c in self.columns if self._path(c).exists()]

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir.parent / ".lock", "a+") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def localize(self, columns: Sequence[str] | str | None = None, *, threads: int | None = None) -> list[str]:
        """Make sure ``columns`` (default: all) are in the local cache, reading only their bytes from the dataset.

        Idempotent and cheap when everything is already local; safe to call from several processes at once (one of them does the
        work, the others wait). Returns the columns that had to be fetched.
        """
        cols = self.columns if columns is None else self._check_columns(columns)
        if all(self._path(c).exists() for c in cols):
            return []
        with self._locked():
            missing = [c for c in cols if not self._path(c).exists()]  # another process may have fetched them meanwhile
            if missing:
                self._build(missing, threads or self.threads)
        return missing

    def _shard_metadata(self) -> list[tuple[str, int, Any]]:
        out = []
        for s in range(self.manifest.n_obs_shards):
            name, tiles = self.manifest.obs_shard_name(s), self.manifest.obs_shard_tiles(s)
            size = self.backend.size(name)
            md = pq.read_metadata(RangeFile(self.backend, name, size))
            rows = [md.row_group(g).num_rows for g in range(md.num_row_groups)]
            if rows != [self.manifest.tile_cells[t] for t in tiles]:
                raise ValueError(f"{name}: row groups do not match the tiles of the dataset")
            out.append((name, size, md))
        return out

    def _build(self, columns: list[str], threads: int) -> None:
        shards = self._shard_metadata()
        tmps = {c: Path(f"{self._path(c)}.tmp{os.getpid()}") for c in columns}
        writers: dict[str, Any] = {}
        sinks: dict[str, Any] = {}
        n_tiles = self.manifest.n_tiles
        log.info("localizing obs columns %s (%d tiles) to %s", columns, n_tiles, self.cache_dir)

        def read_tile(t: int) -> pa.Table:
            s, g = divmod(t, self.manifest.obs_tiles_per_file)
            name, size, md = shards[s]
            pf = pq.ParquetFile(RangeFile(self.backend, name, size), metadata=md, pre_buffer=True)
            return pf.read_row_group(g, columns=columns)

        try:
            with ThreadPoolExecutor(threads) as pool:
                for t, table in enumerate(_ordered_map(pool, read_tile, range(n_tiles), window=max(2, 2 * threads))):
                    if table.num_rows != self.manifest.tile_cells[t]:
                        raise ValueError(
                            f"obs row group {t} has {table.num_rows} rows, expected {self.manifest.tile_cells[t]}"
                        )
                    for c in columns:
                        arr = table.column(c).combine_chunks()
                        if c not in writers:
                            sinks[c] = pa.OSFile(str(tmps[c]), "wb")
                            writers[c] = pa.ipc.new_file(sinks[c], pa.schema([pa.field(c, arr.type)]))
                        writers[c].write_batch(pa.record_batch([arr], names=[c]))
                    if n_tiles >= 100 and (t + 1) % max(1, n_tiles // 10) == 0:
                        log.info("localized %d / %d tiles", t + 1, n_tiles)
            for c in columns:
                writers[c].close()
                sinks[c].close()
                os.replace(tmps[c], self._path(c))
        except BaseException:
            for c in columns:
                with contextlib.suppress(Exception):
                    writers[c].close()
                    sinks[c].close()
                tmps[c].unlink(missing_ok=True)
            raise

    def _reader(self, column: str):
        hit = self._readers.get(column)
        if hit is not None:
            return hit[1]
        path = self._path(column)
        mm = pa.memory_map(str(path), "r")
        try:
            reader = pa.ipc.open_file(mm)  # a truncated file has no footer and fails here
            n = self.manifest.n_tiles
            ok = reader.num_record_batches == n and all(
                reader.get_batch(t).num_rows == self.manifest.tile_cells[t] for t in {0, n // 2, n - 1}
            )
        except pa.ArrowInvalid:
            ok = False
        if not ok:
            mm.close()
            path.unlink(missing_ok=True)
            raise _CorruptCache(f"the cached obs column {column!r} was incomplete or corrupt; it has been deleted")
        self._readers[column] = (mm, reader)
        return reader

    # -- lookups

    @staticmethod
    def _take_from(arr: pa.Array, local: np.ndarray) -> np.ndarray:
        if (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)) and arr.null_count == 0:
            return arr.to_numpy(zero_copy_only=True)[local]  # a zero-copy view of the memory map, then a numpy gather
        return arr.take(pa.array(local)).to_numpy(zero_copy_only=False)

    def take(
        self, indices: Sequence[int] | np.ndarray, columns: Sequence[str] | str, *, warn: bool = True
    ) -> dict[str, np.ndarray]:
        """Values of ``columns`` for the cells with the given global indices, in that order, as numpy arrays.

        Categorical columns come back as integer codes (``-1`` = missing; see :meth:`categories`), strings as object arrays,
        numeric columns with missing values as float64 with NaN. Columns that are not in the local cache yet are fetched first
        (with a warning unless ``warn=False``: list the columns you need up front with :meth:`localize` to avoid stalling a
        training run).
        """
        cols = self._check_columns(columns)
        idx = np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).ravel()
        if len(idx) and (idx.min() < 0 or idx.max() >= self.manifest.n_cells):
            raise IndexError(f"cell index out of range [0, {self.manifest.n_cells})")
        missing = [c for c in cols if not self._path(c).exists()]
        if missing:
            if warn:
                warnings.warn(
                    f"obs columns {missing} are not in the local cache ({self.cache_dir}); fetching them now. "
                    "Call dataset.obs.localize([...]) (or pass obs_columns=...) up front to avoid this stall.",
                    stacklevel=2,
                )
            self.localize(missing)
        tile_ids = np.searchsorted(self._limits, idx, side="right")
        local = idx - self._starts[tile_ids]
        tiles = list(dict.fromkeys(tile_ids.tolist()))
        sels = {t: np.nonzero(tile_ids == t)[0] for t in tiles}
        out: dict[str, np.ndarray] = {}
        for c in cols:
            for attempt in (0, 1):
                try:
                    reader = self._reader(c)
                    break
                except _CorruptCache:
                    if attempt:
                        raise
                    self.localize([c])
            parts = [self._take_from(reader.get_batch(t).column(0), local[sels[t]]) for t in tiles]
            if not parts:
                out[c] = np.empty(0, dtype=self.schema[c].pa_type().to_pandas_dtype())
                continue
            dtype = np.result_type(*[p.dtype for p in parts]) if parts[0].dtype != object else object
            arr = np.empty(len(idx), dtype=dtype)
            for t, p in zip(tiles, parts):
                arr[sels[t]] = p
            out[c] = arr
        return out

    def _pandas_dtype(self, column: str) -> pd.CategoricalDtype:
        """The (cached) pandas dtype of a categorical column; building it validates the vocabulary, which is costly if large."""
        dt = self._dtypes.get(column)
        if dt is None:
            dt = self._dtypes[column] = pd.CategoricalDtype(self.schema[column].categories, ordered=False)
        return dt

    def take_pandas(
        self, indices: Sequence[int] | np.ndarray, columns: Sequence[str] | str, *, warn: bool = True
    ) -> pd.DataFrame:
        """Like :meth:`take` but as a DataFrame (``RangeIndex``): categorical columns are pandas categoricals carrying the
        dataset's *global* vocabulary (so ``.cat.categories`` and ``.cat.codes`` are the same in every batch)."""
        cols = self._check_columns(columns)
        taken = self.take(indices, cols, warn=warn)
        data: dict[str, Any] = {}
        for c in cols:
            if self.schema[c].kind == "category":
                data[c] = pd.Categorical.from_codes(taken[c], dtype=self._pandas_dtype(c))
            else:
                data[c] = taken[c]
        return pd.DataFrame(data, columns=cols)

    def warmup(self) -> None:
        """Pay the one-off cost (about 0.2 s) of Arrow's first compute call now, e.g. when a DataLoader worker starts."""
        pa.array([0, 1]).take(pa.array([1]))

    def to_pandas(self, columns: Sequence[str] | str | None = None, *, categorical: bool = True) -> pd.DataFrame:
        """The whole obs table (or some columns) as a DataFrame with a ``RangeIndex`` over all cells, fetching missing columns.

        Categorical columns become pandas categoricals (``categorical=False`` gives the vocabulary strings instead). Meant for
        analysis and plotting, not for training batches.
        """
        cols = self.columns if columns is None else self._check_columns(columns)
        self.localize(cols)
        data: dict[str, Any] = {}
        for c in cols:
            chunked = self._reader(c).read_all().column(0)
            spec = self.schema[c]
            if spec.kind == "category":
                codes = chunked.to_numpy()
                cat = pd.Categorical.from_codes(codes, dtype=self._pandas_dtype(c))
                data[c] = cat if categorical else np.asarray(cat.astype(object))
            else:
                data[c] = chunked.to_pandas()
        return pd.DataFrame(data)

    # -- diagnostics

    def column_nbytes(self) -> dict[str, int]:
        """Compressed bytes per column in the dataset's parquet shards (what localizing the column reads)."""
        total = dict.fromkeys(self.columns, 0)
        for _, _, md in self._shard_metadata():
            for g in range(md.num_row_groups):
                rg = md.row_group(g)
                for i in range(rg.num_columns):
                    total[rg.column(i).path_in_schema] += rg.column(i).total_compressed_size
        return total

    def verify(self) -> None:
        """Re-read every obs shard in full and check its checksum, structure and schema; raises ``ValueError`` on any problem."""
        m = self.manifest
        for s in range(m.n_obs_shards):
            name = m.obs_shard_name(s)
            data = bytes(self.backend.read(name))
            if len(data) != m.obs_shard_bytes[s] or hashlib.sha256(data).hexdigest() != m.obs_shard_sha256[s]:
                raise ValueError(f"{name}: checksum mismatch (the file changed or is corrupt)")
            md = pq.read_metadata(io.BytesIO(data))
            if md.schema.to_arrow_schema().names != self.schema.names:
                raise ValueError(f"{name}: columns differ from obs/schema.json")
            rows = [md.row_group(g).num_rows for g in range(md.num_row_groups)]
            if rows != [m.tile_cells[t] for t in m.obs_shard_tiles(s)]:
                raise ValueError(f"{name}: row groups do not match the tiles of the dataset")
        if obs_fingerprint(self.schema.to_json(), m.tile_cells, m.obs_shard_sha256) != m.obs_fingerprint:
            raise ValueError("obs fingerprint does not match the schema and shards")
