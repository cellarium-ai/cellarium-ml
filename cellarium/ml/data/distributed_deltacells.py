# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from torch.utils._pytree import tree_leaves

from cellarium.ml.data.distributed_collection import DistributedCollection
from cellarium.ml.utilities.data import AnnDataField

logger = logging.getLogger(__name__)

#: Attributes of an AnnData that a :class:`DeltaCellsBatch` provides (and so :class:`AnnDataField` can read).
SUPPORTED_ATTRS = ("X", "obs", "obs_names", "var", "var_names")


def _import_deltacells() -> Any:
    try:
        import deltacells
    except ImportError as e:  # pragma: no cover
        raise ImportError("DistributedDeltaCellsCollection requires the `deltacells` package.") from e
    if not hasattr(deltacells, "open_dataset"):  # e.g. an unrelated directory named `deltacells` on sys.path
        raise ImportError(
            "Found a module named `deltacells` that is not the deltacells package; install it (`pip install -e deltacells`)."
        )
    return deltacells


def _obs_columns_needed(batch_keys: Any, available: Sequence[str]) -> list[str]:
    """The obs columns that reading the :class:`AnnDataField` leaves of ``batch_keys`` needs, in order of first use."""
    columns: dict[str, None] = {}
    for field in tree_leaves(batch_keys):
        if not isinstance(field, AnnDataField):
            continue
        attr = field.attr.split(".")[0]
        if attr not in SUPPORTED_ATTRS:
            raise ValueError(
                f"The deltacells collection cannot provide {field.attr!r}; supported AnnData attributes are {SUPPORTED_ATTRS}."
            )
        if attr == "obs":
            keys = list(available) if field.key is None else [field.key] if isinstance(field.key, str) else field.key
            for key in keys:
                columns[key] = None
        elif attr == "obs_names":
            columns["obs_names"] = None
    return list(columns)


class DeltaCellsObs:
    """Lazy ``obs`` of a :class:`DeltaCellsBatch`: indexing it with a column name (or a list of names) reads just those columns."""

    def __init__(self, collection: "DistributedDeltaCellsCollection", indices: np.ndarray) -> None:
        self._collection = collection
        self._indices = indices

    @property
    def columns(self) -> list[str]:
        return self._collection.obs_columns_available

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, key: str | Sequence[str]) -> pd.Series | pd.DataFrame:
        keys = [key] if isinstance(key, str) else list(key)
        df = self._collection.take_obs(self._indices, keys)
        return df[key] if isinstance(key, str) else df


class DeltaCellsBatch:
    """
    A batch of cells that behaves like an :class:`~anndata.AnnData` for the attributes
    :class:`~cellarium.ml.utilities.data.AnnDataField` reads: ``X`` (float32 CSR), ``obs`` (pandas, with categorical
    columns carrying the dataset's global categories), ``obs_names``, ``var``, ``var_names``.

    Everything is read lazily, only when the attribute is accessed, and ``X`` once.
    """

    def __init__(self, collection: "DistributedDeltaCellsCollection", indices: np.ndarray) -> None:
        self._collection = collection
        self._indices = indices
        self._X: scipy.sparse.csr_matrix | None = None

    @property
    def n_obs(self) -> int:
        return len(self._indices)

    @property
    def n_vars(self) -> int:
        return self._collection.n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_obs, self.n_vars

    def __len__(self) -> int:
        return self.n_obs

    @property
    def X(self) -> scipy.sparse.csr_matrix:
        if self._X is None:
            batch = self._collection.dataset.get_batch(self._indices)
            self._X = scipy.sparse.csr_matrix((batch.values, batch.indices, batch.indptr), shape=self.shape)
        return self._X

    @property
    def obs(self) -> DeltaCellsObs:
        return DeltaCellsObs(self._collection, self._indices)

    @property
    def obs_names(self) -> pd.Index:
        return pd.Index(self._collection.take_obs(self._indices, ["obs_names"])["obs_names"])

    @property
    def var(self) -> pd.DataFrame:
        return self._collection.var

    @property
    def var_names(self) -> np.ndarray:
        return self._collection.var_names

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"{type(self).__name__} provides only the AnnData attributes {SUPPORTED_ATTRS}, not {name!r}"
        )


class DistributedDeltaCellsCollection(DistributedCollection):
    r"""
    A collection of cells stored as a `deltacells <https://github.com/cellarium-ai/cellarium-ml/tree/main/deltacells>`_
    dataset: one object per *tile* of consecutive cells (the shards of this collection), compressed so that a tile is
    read and decoded in about a tenth of a second, with the cell metadata (obs) kept beside the tiles and copied to a
    node-local cache, one column at a time, only when a field needs it.

    Indexing returns a :class:`DeltaCellsBatch`, an AnnData-like batch, so :class:`~cellarium.ml.utilities.data.AnnDataField`
    and all of its ``convert_fn`` s work as they do with h5ad files. Differences from h5ad data to be aware of:

    * ``X`` holds integer counts (at most 65535) as float32 and the genes are in the dataset's stored order, which
      differs from the original order; models identify genes by ``var_names_g``.
    * The categories of categorical obs columns are global (the same in every batch and every shard) and are strings.
    * Only the attributes ``X``, ``obs``, ``obs_names``, ``var`` and ``var_names`` exist.

    Example::

        >>> from cellarium.ml import CellariumAnnDataDataModule
        >>> from cellarium.ml.data import DistributedDeltaCellsCollection
        >>> from cellarium.ml.utilities.data import AnnDataField, categories_to_codes, to_torch_sparse_csr

        >>> dadc = DistributedDeltaCellsCollection("gs://bucket-name/dataset")
        >>> dm = CellariumAnnDataDataModule(
        ...     dadc,
        ...     batch_keys={
        ...         "x_ng": AnnDataField(attr="X", convert_fn=to_torch_sparse_csr),
        ...         "var_names_g": AnnDataField(attr="var_names"),
        ...         "y_n": AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes),
        ...     },
        ...     batch_size=5000,
        ...     shuffle=True,
        ...     num_workers=4,
        ... )

    The obs columns used by ``batch_keys`` are copied to the local cache by the datamodule's ``prepare_data``.

    Args:
        uri:
            Dataset location: a local path or ``gs://bucket/prefix``.
        max_cached_tiles:
            Decoded tiles kept in memory per process (about 8 bytes per nonzero each).
        max_prefetch_tiles:
            Compressed tiles kept (or being fetched) per process.
        io_threads:
            Concurrent tile fetches per process.
        decode_threads:
            Threads used to decode one tile (``1`` is right when each worker has about one core).
        cache_dir:
            Where obs columns are cached (default: ``$DELTACELLS_CACHE`` or ``~/.cache/deltacells``).
        verify:
            Check the CRC32 of every tile that is read.
        obs_columns:
            Additional obs columns to make local in :meth:`prepare` (for example for plotting later); the columns that
            ``batch_keys`` use are always made local.
    """

    supports_prefetch = True

    def __init__(
        self,
        uri: str,
        max_cached_tiles: int = 1,
        max_prefetch_tiles: int = 8,
        io_threads: int = 4,
        decode_threads: int = 1,
        cache_dir: str | None = None,
        verify: bool = False,
        obs_columns: Sequence[str] | None = None,
    ) -> None:
        deltacells = _import_deltacells()
        self.uri = uri
        self.extra_obs_columns = list(obs_columns or [])
        self.dataset = deltacells.open_dataset(
            uri,
            max_cached_tiles=max_cached_tiles,
            max_prefetch_tiles=max_prefetch_tiles,
            io_threads=io_threads,
            decode_threads=decode_threads,
            cache_dir=cache_dir,
            verify=verify,
        )
        if self.dataset.var_names is None:
            raise ValueError(f"The deltacells dataset at {uri!r} has no variable names; cellarium needs `var_names`.")
        self.limits = [int(limit) for limit in self.dataset.limits]
        self.var_names = np.asarray(self.dataset.var_names)
        self._miss_base = 0

    def __repr__(self) -> str:
        return f"DistributedDeltaCellsCollection(uri={self.uri!r}) with n_obs × n_vars = {self.n_obs} × {self.n_vars}"

    @property
    def n_obs(self) -> int:
        return self.dataset.n_cells

    @property
    def n_vars(self) -> int:
        return self.dataset.n_genes

    @property
    def var(self) -> pd.DataFrame:
        var = self.dataset.var
        return var if var is not None else pd.DataFrame(index=pd.Index(self.var_names))

    @property
    def obs_columns_available(self) -> list[str]:
        """Names of the obs columns stored with the dataset."""
        return [] if self.dataset.obs is None else self.dataset.obs.columns

    def __getitem__(self, index: Any) -> DeltaCellsBatch:
        if isinstance(index, slice):
            indices = np.arange(*index.indices(self.n_obs), dtype=np.int64)
        elif isinstance(index, (int, np.integer)):
            indices = np.array([index if index >= 0 else index + self.n_obs], dtype=np.int64)
        else:
            indices = np.asarray(index)
            if indices.dtype == np.bool_:
                indices = np.flatnonzero(indices)
            indices = indices.astype(np.int64, copy=False).ravel()
        return DeltaCellsBatch(self, indices)

    def take_obs(self, indices: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
        """Obs columns for the given cells (fetched on first use if they were not made local in :meth:`prepare`)."""
        obs = self.dataset.obs
        if obs is None:
            raise ValueError(f"The deltacells dataset at {self.uri!r} has no obs.")
        return obs.take_pandas(indices, columns, warn=False)

    def prefetch(self, indices: np.ndarray) -> None:
        self.dataset.prefetch_cells(indices)

    def prepare(self, batch_keys: Any) -> None:
        obs = self.dataset.obs
        available = [] if obs is None else obs.columns
        columns = _obs_columns_needed(batch_keys, available)
        columns += [c for c in self.extra_obs_columns if c not in columns]
        if not columns:
            return
        if obs is None:
            raise ValueError(f"The batch_keys read obs columns {columns} but the dataset at {self.uri!r} has no obs.")
        missing = [c for c in columns if c not in available]
        if missing:
            raise KeyError(f"Obs columns {missing} are not in the dataset at {self.uri!r}; available: {available}")
        fetched = obs.localize(columns)
        if fetched:
            logger.info("Made obs columns %s local (cache: %s)", fetched, obs.cache_dir)
        obs.warmup()

    def reset_cache(self) -> None:
        self.dataset.close()
        self._miss_base = self.dataset.stats["tiles_fetched"]

    @property
    def cache_miss_count(self) -> int:
        return self.dataset.stats["tiles_fetched"] - self._miss_base
