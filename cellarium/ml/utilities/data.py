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
from torch.utils._pytree import tree_map
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
    key: list[str] | str | None = None
    convert_fn: Callable[[Any], np.ndarray] | None = None
    convert_fn_kwargs: dict[str, Any] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        value = attrgetter(self.attr)(adata)
        if self.key is not None:
            value = value[self.key]
        if self.convert_fn is not None:
            value = self.convert_fn(value)
        else:
            value = np.asarray(value)

        return value


def convert_to_tensor(value: np.ndarray) -> np.ndarray | torch.Tensor:
    if np.issubdtype(value.dtype, np.str_) or np.issubdtype(value.dtype, np.object_):
        return value
    return torch.tensor(value, device="cpu")


def collate_fn(
    batch: list[dict[str, dict[str, np.ndarray] | np.ndarray]],
) -> dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]:
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
    collated_batch: dict[str, dict[str, np.ndarray] | np.ndarray] = {}
    if len(batch) > 1 and not all(keys == data.keys() for data in batch[1:]):
        raise ValueError("All dictionaries in the batch must have the same keys.")
    for key in keys:
        if key.endswith("_g") or key.endswith("_categories"):
            # Check that all values are the same
            if len(batch) > 1:
                if not all(np.array_equal(batch[0][key], data[key]) for data in batch[1:]):  # type: ignore[arg-type]
                    raise ValueError(f"All dictionaries in the batch must have the same {key}.")
            # If so, just take the first one
            value = batch[0][key]
        elif isinstance(batch[0][key], dict):
            if not (key.endswith("_n") or key.endswith("_ng")):
                raise ValueError(f"Sub-dictionary '{key}' must have a batch dimension (end with '_n' or '_ng').")
            subkeys = batch[0][key].keys()  # type: ignore[union-attr]
            if len(batch) > 1 and not all(subkeys == data[key].keys() for data in batch[1:]):  # type: ignore[union-attr]
                raise ValueError(f"All '{key}' sub-dictionaries in the batch must have the same subkeys.")
            value = {subkey: np.concatenate([data[key][subkey] for data in batch], axis=0) for subkey in subkeys}
        else:
            value = np.concatenate([data[key] for data in batch], axis=0)

        collated_batch[key] = value

    return tree_map(convert_to_tensor, collated_batch)


def densify(x: scipy.sparse.csr_matrix) -> np.ndarray:
    """
    Convert a sparse matrix to a dense matrix.

    Args:
        x: Sparse matrix.

    Returns:
        Dense matrix.
    """
    return x.toarray()


def categories_to_codes(x: pd.Series | pd.DataFrame) -> np.ndarray:
    """
    Convert a pandas Series or DataFrame of categorical data to a numpy array of codes.
    Returned array is always a copy.

    Args:
        x: Pandas Series object or a pandas DataFrame containing multiple categorical Series.

    Returns:
        Numpy array.
    """
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: col.cat.codes).to_numpy()
    else:
        return np.asarray(x.cat.codes)

    if x is not None:
        return np.asarray(x.cat.codes)
    else:
        raise ValueError("batch_index_n information not provided")

def categories_to_product_codes(x: pd.Series | pd.DataFrame) -> np.ndarray:
    """
    Convert a pandas Series or DataFrame of categorical data to a numpy array of codes.
    If the input is a DataFrame, the output is created by first combining .
    Returned array is always a copy.

    Args:
        x: Pandas Series object or a pandas DataFrame containing multiple categorical Series.

    Returns:
        Numpy array.
    """
    if isinstance(x, pd.DataFrame):
        codes = x.apply(lambda col: col.cat.codes)
        n_cats = x.apply(lambda col: len(col.cat.categories))
        # compute codes as product of number of categories
        # like the code [1, 1] if there are 3 categories in the first column and 2 in the second
        # would be 1 + 1*3 = 4
        n_cats = n_cats.cumprod().shift(1).fillna(1)
        return np.asarray((n_cats.values[None, :] * codes).sum(axis=1).values).astype(int)
    else:
        return np.asarray(x.cat.codes)


def get_categories(x: pd.Series) -> np.ndarray:
    """
    Get the categories of a pandas Series object.

    Args:
        x: Pandas Series object.

    Returns:
        Numpy array.
    """

    return np.asarray(x.cat.categories)


def parse_gene_list(data):

    print(data)

    exit()

    genes = pd.read_csv(predict_genes,sep="\t")
    predict_gene_list = genes["ensembl_id"].tolist()

    print(predict_gene_list)
    exit()
    return predict_gene_list
