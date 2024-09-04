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
import warnings

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
    convert_fn_kwargs: dict[str, Any] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        value = attrgetter(self.attr)(adata)
        if self.key is not None:
            if hasattr(value,"columns"):
                if self.key in value.columns:
                    value = value[self.key]
                else:
                    value = None
            else:
                try:
                    value = value[self.key]
                except:
                    warnings.warn("Attribute : {} not found in object {}".format(self.key,value))
                    value = None

        if self.convert_fn is not None:
            if (self.convert_fn_kwargs is not None) and ('var_name_key' in self.convert_fn_kwargs):
                gene_logic = adata[idx].var[self.convert_fn_kwargs['var_name_key']].isin(self.convert_fn_kwargs['gene_names'])
                kwargs = {'gene_logic': gene_logic}
            else:
                kwargs = {}
            value = self.convert_fn(value, **kwargs)
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
    batch = batch.copy()
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


def subset_genes_and_densify(x: scipy.sparse.csr_matrix, gene_logic: np.ndarray) -> np.ndarray:
    """
    Convert a sparse matrix to a dense matrix, using only a subset of genes.

    Args:
        x: Sparse matrix.
        gene_logic: logical array denoting which genes to include.

    Returns:
        Dense matrix.
    """
    if scipy.sparse.issparse(x):
        return x[:, np.asarray(gene_logic)].toarray()
    else:
        return x[np.asarray(gene_logic)]


def categories_to_codes(x: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series of categorical data to a numpy array of codes.
    Returned array is always a copy.

    Args:
        x: Pandas Series object.

    Returns:
        Numpy array.
    """

    if x is not None:
        return np.asarray(x.cat.codes)
    else:
        warnings.warn("Batch information not specified, setting number of batches/categories to 2")
        return np.asarray([3])


def get_categories(x: pd.Series) -> np.ndarray:
    """
    Get the categories of a pandas Series object.

    Args:
        x: Pandas Series object.

    Returns:
        Numpy array.
    """
    return np.asarray(x.cat.categories)
