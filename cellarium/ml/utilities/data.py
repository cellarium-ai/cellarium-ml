# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Data utilities
--------------

This module contains helper functions for data loading and processing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
from anndata import AnnData


@dataclass
class AnnDataField:
    """
    Helper class for accessing fields of an AnnData-like object.

    Example::

        >>> from cellarium.ml.data import DistributedAnnDataCollection
        >>> from cellarium.ml.utilities.data import AnnDataField, densify

        >>> dadc = DistributedAnnDataCollection(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     shard_size=10_000,
        ...     max_cache_size=2)

        >>> adata = dadc[:100]
        >>> field_X = AnnDataField(attr="X", convert_fn=densify)
        >>> X = field_X(adata)  # densify(adata.X)

        >>> field_total_mrna_umis = AnnDataField(attr="obs", key="total_mrna_umis")
        >>> total_mrna_umis = field_total_mrna_umis(adata)  # np.asarray(adata.obs["total_mrna_umis"])

    Args:
        attr:
            The attribute of the AnnData-like object to access.
        key:
            The key of the attribute to access. If ``None``, the entire attribute is returned.
        convert_fn:
            A function to apply to the attribute before returning it.
            If ``None``, :func:`np.asarray` is used.
    """

    attr: str
    key: str | None = None
    convert_fn: Callable[[Any], np.ndarray] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        value = attrgetter(self.attr)(adata)
        if self.key is not None:
            # TODO: could we put something here to accept a list of keys?
            value = value[self.key]

        if self.convert_fn is not None:
            value = self.convert_fn(value)
        else:
            value = np.asarray(value)

        return value


def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray | torch.Tensor]:
    """
    Collate function for the ``DataLoader``. This function assumes that the batch is a list of
    dictionaries, where each dictionary has the same keys. If the key ends with ``_g`` or
    ``_categories``, the value of that key is checked to be the same across all dictionaries in the
    batch and then taken from the first dictionary. Otherwise, the value of that key is concatenated
    along the first dimension.  Then the values which are not strings are converted to
    a :class:`torch.Tensor` and returned in a dictionary.

    Args:
        batch: List of dictionaries.

    Returns:
        Dictionary with the same keys as the input dictionaries, but with values concatenated along
        the batch dimension.
    """
    keys = batch[0].keys()
    collated_batch: dict[str, np.ndarray | torch.Tensor] = {}
    if len(batch) > 1:
        if not all(keys == data.keys() for data in batch[1:]):
            raise ValueError("All dictionaries in the batch must have the same keys.")
    for key in keys:
        if key.endswith("_g") or key.endswith("_categories"):
            # Check that all values are the same
            if len(batch) > 1:
                if not all(np.array_equal(batch[0][key], data[key]) for data in batch[1:]):
                    raise ValueError(f"All dictionaries in the batch must have the same {key}.")
            # If so, just take the first one
            value = batch[0][key]
        else:
            value = np.concatenate([data[key] for data in batch], axis=0)

        if not np.issubdtype(value.dtype, np.str_) and not np.issubdtype(value.dtype, np.object_):
            collated_batch[key] = torch.tensor(value, device="cpu")
        else:
            collated_batch[key] = value
    return collated_batch


def densify(x: scipy.sparse.csr_matrix) -> np.ndarray:
    """
    Convert a sparse matrix to a dense matrix.

    Args:
        x: Sparse matrix.

    Returns:
        Dense matrix.
    """
    return x.toarray()


def categories_to_codes(x: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series of categorical data to a numpy array of codes.
    Returned array is always a copy.

    Args:
        x: Pandas Series object.

    Returns:
        Numpy array.
    """
    return np.asarray(x.cat.codes)


def multiple_categories_to_codes(x: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series of categorical data to a numpy array of codes.
    Returned array is always a copy.

    Args:
        x: Pandas Series object.

    Returns:
        Numpy array.
    """
    return np.asarray(x.cat.codes)[:, None]  # TODO: this is a hack for now because we can only have one column


def get_categories(x: pd.Series) -> np.ndarray:
    """
    Get the categories of a pandas Series object.

    Args:
        x: Pandas Series object.

    Returns:
        Numpy array.
    """
    return np.asarray(x.cat.categories)
