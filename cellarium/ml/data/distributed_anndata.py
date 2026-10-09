# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import gc
from collections.abc import Iterable, Sequence
from contextlib import contextmanager

import numpy as np
import pandas as pd
from anndata import AnnData, concat
from anndata._core.index import _normalize_indices
from anndata.compat import Index, Index1D
from anndata.experimental.multi_files._anncollection import (
    AnnCollection,
    AnnCollectionView,
    ConvertType,
)
from boltons.cacheutils import LRU
from braceexpand import braceexpand

from cellarium.ml.data.distributed_collection import DistributedCollection
from cellarium.ml.data.fileio import read_h5ad_file
from cellarium.ml.data.schema import AnnDataSchema


class getattr_mode:
    lazy = False


_GETATTR_MODE = getattr_mode()


@contextmanager
def lazy_getattr():
    """
    When in lazy getattr mode, return a :class:`LazyAnnData` object attribute instead of
    the AnnData object attribute.
    """
    try:
        _GETATTR_MODE.lazy = True
        yield
    finally:
        _GETATTR_MODE.lazy = False


class DistributedAnnDataCollectionView(AnnCollectionView):
    """
    Distributed AnnData Collection View.

    This class is a wrapper around AnnCollectionView where adatas is a list
    of :class:`LazyAnnData` objects.
    """

    def __getitem__(self, index: Index) -> "DistributedAnnDataCollectionView":
        oidx, vidx = _normalize_indices(index, self.obs_names, self.var_names)
        resolved_idx = self._resolve_idx(oidx, vidx)

        return DistributedAnnDataCollectionView(self.reference, self.convert, resolved_idx)

    @property
    def obs_names(self) -> pd.Index:
        """
        Gather and return the obs_names from all AnnData objects in the collection.
        """
        indices = []
        for i, oidx in enumerate(self.adatas_oidx):
            if oidx is None:
                continue

            adata = self.adatas[i]
            indices.append(adata.obs_names[oidx])

        if len(indices) > 1:
            concat_indices = pd.concat([pd.Series(idx) for idx in indices], ignore_index=True)
            obs_names = pd.Index(concat_indices)
            obs_names = obs_names if self.reverse is None else obs_names[self.reverse]
        else:
            obs_names = indices[0]

        return obs_names


class DistributedAnnDataCollection(AnnCollection, DistributedCollection):
    r"""
    Distributed AnnData Collection.

    This class is a wrapper around AnnCollection where adatas is a list
    of LazyAnnData objects.

    Underlying anndata files must conform to the same schema
    (see :class:`~cellarium.ml.data.schema.AnnDataSchema.validate_anndata`).
    The schema is inferred from the first AnnData file in the collection. Individual AnnData files may
    otherwise vary in the number of cells, and the actual content stored in :attr:`X`, :attr:`layers`,
    :attr:`obs` and :attr:`obsm`.

    Example 1::

        >>> dadc = DistributedAnnDataCollection(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     shard_size=10000,  # use if shards are sized evenly
        ...     max_cache_size=2)

    Example 2::

        >>> dadc = DistributedAnnDataCollection(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     shard_size=10000,
        ...     last_shard_size=6000,  # use if the size of the last shard is different
        ...     max_cache_size=2)

    Example 3::

        >>> dadc = DistributedAnnDataCollection(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     limits=[500, 1000, 2000, 2500, 3000, 4000],  # use if shards are sized unevenly
        ...     max_cache_size=2)

    Args:
        filenames:
            Names of anndata files.
        limits:
            List of global cell indices (limits) for the last cells in each shard.
            If ``None``, the limits are inferred from ``shard_size`` and ``last_shard_size``.
        shard_size:
            The number of cells in each anndata file (shard).
            Must be specified if the ``limits`` is not provided.
        last_shard_size:
            Last shard size. If not ``None``, the last shard will have this size possibly
            different from ``shard_size``.
        max_cache_size:
            Max size of the cache.
        cache_size_strictly_enforced:
            Assert that the number of retrieved anndatas is not more than maxsize.
        label:
            Column in :attr:`obs` to place batch information in. If it's ``None``, no column is added.
        keys:
            Names for each object being added. These values are used for column values for
            ``label`` or appended to the index if ``index_unique`` is not ``None``.
            If ``None``, ``keys`` are set to ``filenames``.
        index_unique:
            Whether to make the index unique by using the keys. If provided, this
            is the delimeter between ``{orig_idx}{index_unique}{key}``. When ``None``,
            the original indices are kept.
        convert:
            You can pass a function or a Mapping of functions which will be applied
            to the values of attributes (:attr:`obs`, :attr:`obsm`, :attr:`layers`, :attr:`X`) or to specific
            keys of these attributes in the subset object.
            Specify an attribute and a key (if needed) as keys of the passed Mapping
            and a function to be applied as a value.
        indices_strict:
            If  ``True``, arrays from the subset objects will always have the same order
            of indices as in selection used to subset.
            This parameter can be set to ``False`` if the order in the returned arrays
            is not important, for example, when using them for stochastic gradient descent.
            In this case the performance of subsetting can be a bit better.
        obs_columns_to_validate:
            Subset of columns to validate in the :attr:`obs` attribute.
            If ``None``, all columns are validated.
    """

    def __init__(
        self,
        filenames: Sequence[str] | str,
        limits: Iterable[int] | None = None,
        shard_size: int | None = None,
        last_shard_size: int | None = None,
        max_cache_size: int = 1,
        cache_size_strictly_enforced: bool = True,
        label: str | None = None,
        keys: Sequence[str] | None = None,
        index_unique: str | None = None,
        convert: ConvertType | None = None,
        indices_strict: bool = True,
        obs_columns_to_validate: Sequence[str] | None = None,
    ):
        self.filenames = list(braceexpand(filenames) if isinstance(filenames, str) else filenames)
        if (shard_size is None) and (last_shard_size is not None):
            raise ValueError("If `last_shard_size` is specified then `shard_size` must also be specified.")
        if limits is None:
            if shard_size is None:
                raise ValueError("If `limits` is `None` then `shard_size` must be specified`")
            limits = [shard_size * (i + 1) for i in range(len(self.filenames))]
            if last_shard_size is not None:
                limits[-1] = limits[-1] - shard_size + last_shard_size
        else:
            limits = list(limits)
        if len(limits) != len(self.filenames):
            raise ValueError(
                f"The number of points in `limits` ({len(limits)}) must match "
                f"the number of `filenames` ({len(self.filenames)})."
            )
        # lru cache
        self.cache = LRU(max_cache_size)
        self.max_cache_size = max_cache_size
        self.cache_size_strictly_enforced = cache_size_strictly_enforced
        # schema
        adata0 = self.cache[self.filenames[0]] = read_h5ad_file(self.filenames[0])
        if len(adata0) != limits[0]:
            raise ValueError(
                f"The number of cells in the first anndata file ({len(adata0)}) "
                f"does not match the first limit ({limits[0]})."
            )
        self.obs_columns_to_validate = obs_columns_to_validate
        self.schema = AnnDataSchema(adata0, obs_columns_to_validate)
        # lazy anndatas
        lazy_adatas = [
            LazyAnnData(filename, (start, end), self.schema, self.cache)
            for start, end, filename in zip([0] + limits, limits, self.filenames)
        ]
        # use filenames as default keys
        if keys is None:
            keys = self.filenames
        if len(keys) != len(self.filenames):
            raise ValueError(
                f"The number of keys ({len(keys)}) must match the number of `filenames` ({len(filenames)})."
            )
        with lazy_getattr():
            super().__init__(
                adatas=lazy_adatas,
                join_obs=None,
                join_obsm=None,
                join_vars=None,
                label=label,
                keys=keys,
                index_unique=index_unique,
                convert=convert,
                harmonize_dtypes=False,
                indices_strict=indices_strict,
            )

    @property
    def var(self) -> pd.DataFrame:
        """Per-variable annotations, as in the first anndata file."""
        return self.schema.attr_values["var"]

    @property
    def reference_adata(self) -> "LazyAnnData":
        """The first anndata file (read lazily), for attributes that are the same in every file such as ``var``."""
        return self.adatas[0]

    def _first_obs(self, key: str) -> pd.Series:
        obs = self.adatas[0].obs
        if key not in obs:
            raise ValueError(f"Key '{key}' not found in obs columns: {list(obs.columns)}")
        return obs[key]

    def obs_categories(self, key: str) -> np.ndarray:
        """The categories of the categorical ``obs`` column ``key``, as in the first anndata file."""
        return np.asarray(self._first_obs(key).cat.categories)

    def obs_key_nunique(self, key: str) -> int:
        """Number of distinct values of the ``obs`` column ``key`` in the first anndata file."""
        return int(self._first_obs(key).nunique())

    def reset_cache(self) -> None:
        """Forget the cached anndata files."""
        self.cache.clear()

    @property
    def cache_miss_count(self) -> int:
        """Number of anndata files read so far by this process."""
        return self.cache.miss_count

    def __getitem__(self, index: Index) -> AnnData:
        """
        Materialize and gather anndata files at given indices from the list of lazy anndatas.

        :class:`LazyAnnData` instances corresponding to cells in the index are materialized.
        """
        return self._gather(index, in_groups=False)

    def read(self, index: Index) -> AnnData:
        """
        Like indexing, but cells that lie in more files than the cache holds (for example a batch of scattered cells)
        are read in groups of at most ``max_cache_size`` files, copying the cells of each group out before the next
        group is read, so no more than ``max_cache_size`` files are ever in memory (and
        ``cache_size_strictly_enforced`` is never violated).
        """
        return self._gather(index, in_groups=True)

    def _gather(self, index: Index, in_groups: bool) -> AnnData:
        oidx, vidx = _normalize_indices(index, self.obs_names, self.var_names)
        adatas_oidx, oidx, vidx, reverse = self._resolve_idx(oidx, vidx)
        used = [i for i, adata_oidx in enumerate(adatas_oidx) if adata_oidx is not None]
        if not in_groups or len(used) <= self.max_cache_size:
            adatas = self.materialize(adatas_oidx, vidx)
            adata = concat(adatas, merge="same")
        else:
            groups = []
            for start in range(0, len(used), self.max_cache_size):
                group = set(used[start : start + self.max_cache_size])
                group_oidx = [adata_oidx if i in group else None for i, adata_oidx in enumerate(adatas_oidx)]
                # `concat` copies the cells, which releases the (views of the) files of the group
                groups.append(concat(self.materialize(group_oidx, vidx), merge="same"))
            adata = concat(groups, merge="same")
        adata = adata if reverse is None else adata[reverse]
        # make sure that categorical dtypes are preserved. `astype` leaves a categorical column alone if its categories
        # are those of the schema in a different order (unordered categorical dtypes compare equal), which would make
        # the codes of a batch differ from those of other batches, so categories are set explicitly.
        obs = adata.obs
        for column, dtype in self.schema.attr_values["obs"].dtypes.items():
            if isinstance(dtype, pd.CategoricalDtype) and isinstance(obs[column].dtype, pd.CategoricalDtype):
                obs[column] = obs[column].cat.set_categories(dtype.categories, ordered=dtype.ordered)
            else:
                obs[column] = obs[column].astype(dtype)
        return adata

    def materialize(self, adatas_oidx: list[np.ndarray | None], vidx: Index1D) -> list[AnnData]:
        """
        Buffer and return anndata files at given indices from the list of lazy anndatas.

        This efficiently first retrieves cached files and only then caches new files.
        """
        adata_idx_to_oidx = {i: oidx for i, oidx in enumerate(adatas_oidx) if oidx is not None}
        n_adatas = len(adata_idx_to_oidx)
        if self.cache_size_strictly_enforced:
            if n_adatas > self.max_cache_size:
                raise ValueError(
                    f"Expected the number of anndata files ({n_adatas}) to be "
                    f"no more than the max cache size ({self.max_cache_size})."
                )
        adatas = [None] * n_adatas
        # first fetch cached anndata files
        # this ensures that they are not popped if they were lru
        for i, (adata_idx, oidx) in enumerate(adata_idx_to_oidx.items()):
            if self.adatas[adata_idx].cached:
                adatas[i] = self.adatas[adata_idx].adata[oidx, vidx]
        # only then cache new anndata files
        for i, (adata_idx, oidx) in enumerate(adata_idx_to_oidx.items()):
            if not self.adatas[adata_idx].cached:
                adatas[i] = self.adatas[adata_idx].adata[oidx, vidx]
        return adatas

    def __repr__(self) -> str:
        n_obs, n_vars = self.shape
        descr = f"DistributedAnnDataCollection object with n_obs × n_vars = {self.n_obs} × {self.n_vars}"
        descr += f"\n  constructed from {len(self.filenames)} AnnData objects"
        for attr, keys in self._view_attrs_keys.items():
            if len(keys) > 0:
                descr += f"\n    view of {attr}: {str(keys)[1:-1]}"
        for attr in self._attrs:
            keys = list(getattr(self, attr).keys())
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(keys)[1:-1]}"
        if "obs" in self._view_attrs_keys:
            keys = list(self.obs.keys())
            if len(keys) > 0:
                descr += f"\n    own obs: {str(keys)[1:-1]}"

        return descr

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["cache"]
        del state["adatas"]
        del state["obs_names"]
        del state["schema"]
        del state["_obs"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache = LRU(self.max_cache_size)
        adata0 = self.cache[self.filenames[0]] = read_h5ad_file(self.filenames[0])
        self.schema = AnnDataSchema(adata0, self.obs_columns_to_validate)
        self.adatas = [
            LazyAnnData(filename, (start, end), self.schema, self.cache)
            for start, end, filename in zip([0] + self.limits, self.limits, self.filenames)
        ]
        self.obs_names = pd.Index([f"cell_{i}" for i in range(self.limits[-1])])
        self._obs = pd.DataFrame(index=self.obs_names)


class LazyAnnData:
    """
    Lazy :class:`~anndata.AnnData` backed by a file.

    Accessing attributes under :func:`lazy_getattr` context returns schema attributes.

    Args:
        filename:
            Name of anndata file.
        limits:
            Limits of cell indices (inclusive, exclusive).
        schema:
            Schema used as a reference for lazy attributes.
        cache:
            Shared LRU cache storing buffered anndatas.
    """

    _lazy_attrs = ["obs", "obsm", "layers", "var", "varm", "varp", "var_names"]
    _all_attrs = [
        "obs",
        "var",
        "uns",
        "obsm",
        "varm",
        "layers",
        "obsp",
        "varp",
    ]

    def __init__(
        self,
        filename: str,
        limits: tuple[int, int],
        schema: AnnDataSchema,
        cache: LRU | None = None,
    ):
        self.filename = filename
        self.limits = limits
        self.schema = schema
        if cache is None:
            cache = LRU()
        self.cache = cache

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.limits[1] - self.limits[0]

    @property
    def n_vars(self) -> int:
        """Number of variables/features."""
        return len(self.schema.attr_values["var_names"])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the data matrix."""
        return self.n_obs, self.n_vars

    @property
    def obs_names(self) -> pd.Index:
        """Return the observation names."""
        if _GETATTR_MODE.lazy:
            # This is only used during the initialization of DistributedAnnDataCollection
            return pd.Index([f"cell_{i}" for i in range(*self.limits)])
        else:
            return self.adata.obs_names

    @property
    def cached(self) -> bool:
        """Return whether the anndata is cached."""
        return self.filename in self.cache

    @property
    def adata(self) -> AnnData:
        """Return backed anndata from the filename"""
        try:
            adata = self.cache[self.filename]
        except KeyError:
            # fetch anndata
            adata = read_h5ad_file(self.filename)
            # validate anndata
            if self.n_obs != adata.n_obs:
                raise ValueError(
                    "Expected `n_obs` for LazyAnnData object and backed anndata to match "
                    f"but found {self.n_obs} and {adata.n_obs}, respectively."
                )
            self.schema.validate_anndata(adata)
            # cache anndata
            if len(self.cache) < self.cache.max_size:
                self.cache[self.filename] = adata
            else:
                self.cache[self.filename] = adata
                # garbage collection of AnnData is not reliable
                # therefore we call garbage collection manually to free up the memory
                # https://github.com/scverse/anndata/issues/360
                gc.collect()
        return adata

    def __getattr__(self, attr):
        if _GETATTR_MODE.lazy:
            # This is only used during the initialization of DistributedAnnDataCollection
            if attr in self._lazy_attrs:
                return self.schema.attr_values[attr]
            raise AttributeError(f"Lazy AnnData object has no attribute '{attr}'")
        else:
            adata = self.adata
            if hasattr(adata, attr):
                return getattr(adata, attr)
            raise AttributeError(f"Backed AnnData object has no attribute '{attr}'")

    def __getitem__(self, idx) -> AnnData:
        return self.adata[idx]

    def __repr__(self) -> str:
        if self.cached:
            buffered = "Cached "
        else:
            buffered = ""
        backed_at = f" backed at {str(self.filename)!r}"
        descr = f"{buffered}LazyAnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}{backed_at}"
        if self.cached:
            for attr in self._all_attrs:
                keys = getattr(self, attr).keys()
                if len(keys) > 0:
                    descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr
