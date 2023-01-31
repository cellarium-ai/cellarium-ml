from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple, Union

import braceexpand
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._core.index import Index, _normalize_indices
from anndata.experimental.multi_files._anncollection import (
    AnnCollection,
    AnnCollectionView,
    ConvertType,
)
from boltons.cacheutils import LRU
from scvi.data._utils import get_anndata_attribute

from .read import read_h5ad_file
from .schema import AnnDataSchema


class getattr_mode:
    lazy = False


_GETATTR_MODE = getattr_mode()


@contextmanager
def lazy_getattr():
    try:
        _GETATTR_MODE.lazy = True
        yield
    finally:
        _GETATTR_MODE.lazy = False


class DistributedAnnDataCollection(AnnCollection):
    r"""
    Distributed AnnData Collection.

    This class is a wrapper around AnnCollection where adatas is a list
    of LazyAnnData objects.

    Args:
        filenames: Names of anndata files.
        limits: Limits of cell indices.
        shard_size: Shard size.
        maxsize: Max size of the cache.
        cachesize_strict: Assert that the number of retrieved anndatas is not more than maxsize.
        label: Column in `.obs` to place batch information in. If it's None, no column is added.
        keys: Names for each object being added. These values are used for column values for
            `label` or appended to the index if `index_unique` is not `None`. Defaults to filenames.
        index_unique: Whether to make the index unique by using the keys. If provided, this
            is the delimeter between "{orig_idx}{index_unique}{key}". When `None`,
            the original indices are kept.
        convert: You can pass a function or a Mapping of functions which will be applied
            to the values of attributes (`.obs`, `.obsm`, `.layers`, `.X`) or to specific
            keys of these attributes in the subset object.
            Specify an attribute and a key (if needed) as keys of the passed Mapping
            and a function to be applied as a value.
        indices_strict: If  `True`, arrays from the subset objects will always have the same order
            of indices as in selection used to subset.
            This parameter can be set to `False` if the order in the returned arrays
            is not important, for example, when using them for stochastic gradient descent.
            In this case the performance of subsetting can be a bit better.
    """

    def __init__(
        self,
        filenames: Union[Sequence[str], str],
        limits: Optional[Sequence[int]] = None,
        shard_size: Optional[int] = None,
        maxsize: Optional[int] = None,
        cachesize_strict: bool = True,
        label: Optional[str] = None,
        keys: Optional[Sequence[str]] = None,
        index_unique: Optional[str] = None,
        convert: Optional[ConvertType] = None,
        indices_strict: bool = True,
    ):
        self.filenames = expand_urls(filenames)
        assert isinstance(self.filenames[0], str)
        if (limits is None) == (shard_size is None):
            raise ValueError(
                "Either `limits` or `shard_size` must be specified, but not both."
            )
        if shard_size is not None:
            limits = [shard_size * (i + 1) for i in range(len(self.filenames))]
        else:
            limits = list(limits)
        # lru cache
        self.cache = LRU(maxsize)
        self.cachesize_strict = cachesize_strict
        # schema
        adata0 = self.cache[self.filenames[0]] = read_h5ad_file(self.filenames[0])
        self.schema = AnnDataSchema(adata0)
        # lazy anndatas
        limits0 = [0] + limits[:-1]
        lazy_adatas = [
            LazyAnnData(filename, (start, end), self.schema, self.cache)
            for start, end, filename in zip(limits0, limits, self.filenames)
        ]
        if keys is None:
            keys = self.filenames
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

    def __getitem__(self, index: Index) -> AnnCollectionView:
        oidx, vidx = _normalize_indices(index, self.obs_names, self.var_names)
        resolved_idx = self._resolve_idx(oidx, vidx)
        adatas_indices = [i for i, e in enumerate(resolved_idx[0]) if e is not None]
        # TODO: materialize at the last moment?
        self.materialize(adatas_indices)

        return AnnCollectionView(self, self.convert, resolved_idx)

    def materialize(self, indices) -> List[AnnData]:
        if isinstance(indices, int):
            indices = (indices,)
        if self.cachesize_strict:
            assert len(indices) <= self.cache.max_size
        adatas = [None] * len(indices)
        for i, idx in enumerate(indices):
            if self.adatas[idx].cached:
                adatas[i] = self.adatas[idx].adata

        for i, idx in enumerate(indices):
            if not self.adatas[idx].cached:
                adatas[i] = self.adatas[idx].adata
        return adatas

    def __repr__(self) -> str:
        n_obs, n_vars = self.shape
        descr = f"DistributedAnnCollection object with n_obs × n_vars = {self.n_obs} × {self.n_vars}"
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


class LazyAnnData:
    r"""
    Lazy AnnData backed by a file.

    Accessing attributes under `lazy_getattr` context returns schema attributes.

    Args:
        filename (str): Name of anndata file.
        limits (Tuple[int, int]): Limits of cell indices.
        schema (AnnDataSchema): Schema used as a reference for lazy attributes.
        cache (LRU): Shared LRU cache storing buffered anndatas.
    """

    lazy_attrs = ["obs", "obsm", "layers", "var", "varm", "varp", "var_names"]

    def __init__(
        self,
        filename: str,
        limits: Tuple[int, int],
        schema: AnnDataSchema,
        cache: Optional[LRU] = None,
    ):
        self.filename = filename
        self.limits = limits
        self.schema = schema
        if cache is None:
            cache = LRU()
        self.cache = cache

    @property
    def n_obs(self) -> int:
        return self.limits[1] - self.limits[0]

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_obs, self.n_vars

    @property
    def obs_names(self) -> pd.Index:
        """This is different from the backed anndata"""
        return pd.Index([f"cell_{i}" for i in range(*self.limits)])

    @property
    def cached(self) -> bool:
        return self.filename in self.cache

    @property
    def adata(self) -> AnnData:
        """Return backed anndata from the filename"""
        if not self.cached:
            # fetch anndata
            adata = read_h5ad_file(self.filename)
            print(f"DEBUG FETCHING {self.filename}")
            # validate anndata
            assert (
                self.n_obs == adata.n_obs
            ), "n_obs of LazyAnnData object and backed anndata must match."
            self.schema.validate_anndata(adata)
            # cache anndata
            self.cache[self.filename] = adata
        return self.cache[self.filename]

    def __getattr__(self, attr):
        if _GETATTR_MODE.lazy:
            # This is only used during the initialization of DistributedAnnDataCollection
            if attr in self.lazy_attrs:
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
            for attr in [
                "obs",
                "var",
                "uns",
                "obsm",
                "varm",
                "layers",
                "obsp",
                "varp",
            ]:
                keys = getattr(self, attr).keys()
                if len(keys) > 0:
                    descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr


@get_anndata_attribute.register
def _(
    adata: AnnCollection,
    attr_name: str,
    attr_key: Optional[str] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    return adata.lazy_attr(attr_name, attr_key)


# https://github.com/webdataset/webdataset/blob/ab8911ab3085949dce409646b96077e1c1448549/webdataset/shardlists.py#L25-L33
def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)
