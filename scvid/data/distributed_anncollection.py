from collections import OrderedDict
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
from scvi.data._utils import get_anndata_attribute

from .read import read_h5ad_gcs
from .schema import AnnDataSchema


class LRUCache:
    def __init__(self, maxsize: int):
        self.cache = OrderedDict()
        self._maxsize = maxsize

    def __getitem__(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        raise KeyError(f"{key} not found in the cache")

    def __setitem__(self, key, value):
        while len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.cache.move_to_end(key)

    def __contains__(self, key) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int):
        while len(self.cache) > value:
            lru_key, lru_value = self.cache.popitem(last=False)
            del lru_value
        self._maxsize = value


class LazyAnnData:
    """Lazy AnnData"""

    view_attrs = ["obs", "obsm", "layers", "var", "var_names"]

    def __init__(
        self,
        filename: str,
        cache: LRUCache = LRUCache(maxsize=1),
        n_obs: Optional[int] = None,
        idx: Optional[int] = None,
        reference: AnnDataSchema = None,
    ):
        self.filename = filename
        self.cache = cache
        self._n_obs = n_obs or self.adata.n_obs
        self.idx = idx or 0
        # TODO: allow to use index range instead of fixed size
        self._obs_names = pd.RangeIndex(
            self.n_obs * self.idx, self.n_obs * (self.idx + 1)
        )

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_obs, self.n_vars

    @property
    def obs_names(self) -> pd.Index:
        return self._obs_names

    @property
    def cached(self) -> bool:
        return self.filename in self.cache

    @property
    def adata(self) -> AnnData:
        """Return backed AnnData from the filename"""
        if not self.cached:
            print(f"DEBUG fetching {self.filename}")
            _adata = read_h5ad_gcs(self.filename)
            self.cache[self.filename] = _adata
            # TODO: assert that view_attrs indeed match to the cached values
            assert self.n_obs == _adata.n_obs
            self.reference.validate_adata(_adata)
        return self.cache[self.filename]

    def __getattr__(self, attr):
        if attr in self.view_attrs:
            # load the attribute from the least recently used adata
            # assuming this attribute is the same for all anndata files
            return getattr(self.reference, attr)
        adata = self.adata
        if hasattr(adata, attr):
            return getattr(adata, attr)
        raise AttributeError(f"Backed AnnData object has no attribute '{attr}'")

    def __getitem__(self, idx) -> AnnData:
        return self.adata[idx]

    def __repr__(self) -> str:
        if self.cached:
            buffered = "Buffered "
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


class DistributedAnnCollection(AnnCollection):
    """Distributed AnnData Collection with Lazy Attributes"""

    is_view = False
    isbacked = False

    def __init__(
        self,
        filenames,
        shard_size: Optional[int] = None,
        last_shard_size: Optional[int] = None,
        maxsize: int = 2,
        cachesize_strict: bool = True,
        label: Optional[str] = None,
        keys: Optional[Sequence[str]] = None,
        index_unique: Optional[str] = None,
        convert: Optional[ConvertType] = None,
        harmonize_dtypes: bool = False,
        indices_strict: bool = True,
    ):
        self.filenames = expand_urls(filenames)
        assert isinstance(self.filenames[0], str)
        self.cache = LRUCache(maxsize=maxsize)
        self.cachesize_strict = cachesize_strict
        if keys is None:
            keys = self.filenames
        print(f"DEBUG fetching {self.filenames[0]}")
        first_adata = self.cache[self.filenames[0]] = read_h5ad_gcs(self.filenames[0])
        # TODO: store shared AnnData properties as DistributedAnnCollection property
        self.schema = AnnDataSchema(first_adata)
        if shard_size is None:
            shard_size = len(first_adata)
        self.shard_size = shard_size
        self.last_shard_size = last_shard_size
        if self.last_shard_size is None:
            self.last_shard_size = self.shard_size
        self.adatas = [
            LazyAnnData(filename, self.cache, shard_size, idx, self.schema)
            for idx, filename in enumerate(self.filenames)
        ]
        self.uns = OrderedDict()
        super().__init__(
            adatas=self.adatas,
            join_obs=None,
            join_obsm=None,
            join_vars=None,
            label=label,
            keys=keys,
            index_unique=index_unique,
            convert=convert,
            harmonize_dtypes=harmonize_dtypes,
            indices_strict=indices_strict,
        )

    def __getitem__(self, index: Index):
        oidx, vidx = _normalize_indices(index, self.obs_names, self.var_names)
        resolved_idx = self._resolve_idx(oidx, vidx)
        adatas_indices = [i for i, e in enumerate(resolved_idx[0]) if e is not None]
        # TODO: materialize at the last moment?
        self.materialize(adatas_indices)

        return AnnCollectionView(self, self.convert, resolved_idx)

    def __repr__(self):
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

    def materialize(self, indices) -> List[AnnData]:
        if isinstance(indices, int):
            indices = (indices,)
        if self.cachesize_strict:
            assert len(indices) <= self.cache.maxsize
        adatas = [None] * len(indices)
        for i, idx in enumerate(indices):
            if self.adatas[idx].cached:
                adatas[i] = self.adatas[idx].adata

        for i, idx in enumerate(indices):
            if not self.adatas[idx].cached:
                adatas[i] = self.adatas[idx].adata
        return adatas


@get_anndata_attribute.register
def _(
    adata: AnnCollection,
    attr_name: str,
    attr_key: Optional[str],
    mod_key: Optional[str] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    if attr_name == "var":
        field = adata.var[attr_key]
        field = field.to_numpy().reshape(-1, 1)
        return field
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
