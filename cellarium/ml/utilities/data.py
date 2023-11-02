# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Data utilities
--------------
This module contains helper functions for data loading and processing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from anndata import AnnData
from anndata.experimental import AnnCollection


@dataclass
class AnnDataField:
    """
    Helper class for accessing fields of an AnnData-like object.

    Example::

        >>> from cellarium.ml.data import DistributedAnnDataCollection
        >>> from cellarium.ml.utilities.data import AnnDataField, densify, pandas_to_numpy

        >>> dadc = DistributedAnnDataCollection(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     shard_size=10_000,
        ...     max_cache_size=2)

        >>> field_X = AnnDataField(attr="X", convert_fn=densify)
        >>> X = field_X(dadc)[:100]  # densify(dadc[:100].X)

        >>> field_cell_type = AnnDataField(attr="obs", key="cell_type", convert_fn=pandas_to_numpy)
        >>> cell_type = field_cell_type(dadc)[:100]  # pandas_to_numpy(dadc[:100].obs["cell_type"])

    Args:
        attr:
            The attribute of the AnnData-like object to access.
        key:
            The key of the attribute to access. If ``None``, the entire attribute is returned.
        convert_fn:
            A function to apply to the attribute before returning it. If ``None``, no conversion is applied.
    """

    attr: str
    key: str | None = None
    convert_fn: Callable[[Any], np.ndarray] | None = None

    def __call__(self, adata: AnnData | AnnCollection) -> AnnDataField:
        self.adata = adata
        return self

    def __getitem__(self, idx: int | list[int] | slice) -> np.ndarray:
        if self.adata is None:
            raise ValueError("Must call AnnDataField with an AnnData-like object first")

        value = getattr(self.adata[idx], self.attr)
        if self.key is not None:
            value = value[self.key]

        if self.convert_fn is not None:
            value = self.convert_fn(value)

        if not isinstance(value, np.ndarray):
            raise ValueError(f"Expected {value} to be a numpy array. Got {type(value)}")

        return value

    @property
    def obs_column(self) -> str | None:
        result = None
        if self.attr == "obs":
            result = self.key
        return result
