# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


class DistributedCollection(ABC):
    """
    Interface of the data sources that :class:`~cellarium.ml.data.IterableDistributedAnnDataCollectionDataset`
    and :class:`~cellarium.ml.core.CellariumAnnDataDataModule` read from.

    A collection is a sequence of ``n_obs`` cells split into consecutive *shards* (for example h5ad files or deltacells
    tiles). Indexing it with global cell indices returns an object that behaves like an :class:`~anndata.AnnData`
    for the attributes that :class:`~cellarium.ml.utilities.data.AnnDataField` reads (``X``, ``obs``, ``var_names`` ...).

    Subclasses must provide :attr:`n_obs`, :attr:`n_vars`, :attr:`var` and :meth:`__getitem__`, and should set the
    following attributes:

    Attributes:
        limits:
            Cumulative number of cells at the end of each shard, e.g. ``[10000, 20000, 25000]``. Shards are the units
            that the dataset shuffles and splits between workers, so cells of one shard should be cheap to read together.
        var_names:
            Names of the variables (genes), in column order.

    The remaining members have defaults that do nothing, so a source only overrides what it can use:
    :meth:`prefetch` (a hint about which cells will be requested soon), :meth:`prepare` (one-time preparation before
    training, e.g. making metadata local) and the cache hooks used by the dataset's ``test_mode``.
    """

    #: Whether :meth:`prefetch` does anything. The dataset only computes the look-ahead needed to call it if this is true.
    supports_prefetch: bool = False

    limits: Sequence[int]
    var_names: Any

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Number of cells."""

    @property
    @abstractmethod
    def n_vars(self) -> int:
        """Number of variables (genes)."""

    @property
    @abstractmethod
    def var(self) -> pd.DataFrame:
        """Per-variable annotations (one row per variable, in column order)."""

    @abstractmethod
    def __getitem__(self, index: Any) -> Any:
        """Cells with the given global indices (an int, a slice or a sequence of ints), as an AnnData-like object."""

    def __len__(self) -> int:
        return self.n_obs

    def prefetch(self, indices: np.ndarray) -> None:
        """
        Hint that cells ``indices`` will be requested soon (in about that order), so that their shards can be fetched
        in the background. Must be cheap, non-blocking and idempotent. Does nothing by default.
        """

    def prepare(self, batch_keys: Any) -> None:
        """
        One-time preparation for reading the fields in ``batch_keys`` (a pytree of
        :class:`~cellarium.ml.utilities.data.AnnDataField`), called by the datamodule's ``prepare_data``. Does nothing
        by default.
        """

    def reset_cache(self) -> None:
        """Forget cached shards (used by the dataset's ``test_mode``). Does nothing by default."""

    @property
    def cache_miss_count(self) -> int:
        """Number of shards fetched so far by this process (used by the dataset's ``test_mode``)."""
        return 0
