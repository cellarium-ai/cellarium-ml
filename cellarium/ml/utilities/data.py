# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Data utilities
--------------

This module contains helper functions for data loading and processing.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
import torch.distributed as dist
from anndata import AnnData
from anndata.experimental import AnnCollection
from torch.utils.data import get_worker_info as _get_worker_info


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

        >>> field_X = AnnDataField(attr="X", convert_fn=densify)
        >>> X = field_X(dadc)[:100]  # densify(dadc[:100].X)

        >>> field_cell_type = AnnDataField(attr="obs", key="cell_type")
        >>> cell_type = field_cell_type(dadc)[:100]  # np.asarray(dadc[:100].obs["cell_type"])

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

    def __call__(self, adata: AnnData | AnnCollection, idx: int | list[int] | slice) -> np.ndarray:
        value = getattr(adata[idx], self.attr)
        if self.key is not None:
            value = value[self.key]

        if self.convert_fn is not None:
            value = self.convert_fn(value)
        else:
            value = np.asarray(value)

        return value

    @property
    def obs_column(self) -> str | None:
        result = None
        if self.attr == "obs":
            result = self.key
        return result


def get_rank_and_num_replicas() -> tuple[int, int]:
    """
    This helper function returns the rank of the current process and
    the number of processes in the default process group. If distributed
    package is not available or default process group has not been initialized
    then it returns ``rank=0`` and ``num_replicas=1``.

    Returns:
        Tuple of ``rank`` and ``num_replicas``.
    """
    if not dist.is_available():
        num_replicas = 1
        rank = 0
    else:
        try:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        except (ValueError, RuntimeError):  # RuntimeError was changed to ValueError in PyTorch 2.2
            warnings.warn(
                "Distributed package is available but the default process group has not been initialized. "
                "Falling back to ``rank=0`` and ``num_replicas=1``.",
                UserWarning,
            )
            num_replicas = 1
            rank = 0
    if rank >= num_replicas or rank < 0:
        raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas-1}]")
    return rank, num_replicas


def get_worker_info() -> tuple[int, int]:
    """
    This helper function returns ``worker_id`` and ``num_workers``. If it is running
    in the main process then it returns ``worker_id=0`` and ``num_workers=1``.

    Returns:
        Tuple of ``worker_id`` and ``num_workers``.
    """
    worker_info = _get_worker_info()
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
    return worker_id, num_workers


def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray | torch.Tensor]:
    """
    Collate function for the ``DataLoader``. This function assumes that the batch is a list of
    dictionaries, where each dictionary has the same keys. The values of each key are converted
    to a :class:`torch.Tensor` and concatenated along the first dimension. If the key is ``obs_names``,
    the values are concatenated along the first dimension without converting to a :class:`torch.Tensor`.

    Args:
        batch: List of dictionaries.

    Returns:
        Dictionary with the same keys as the input dictionaries, but with values concatenated along
        the batch dimension.
    """
    keys = batch[0].keys()
    collated_batch = {}
    if len(batch) > 1:
        if not all(keys == data.keys() for data in batch[1:]):
            raise ValueError("All dictionaries in the batch must have the same keys.")
    for key in keys:
        if key == "obs_names_n":
            collated_batch[key] = np.concatenate([data[key] for data in batch], axis=0)
        elif key == "var_names_g":
            # Check that all var_names_g are the same
            if len(batch) > 1:
                if not all(np.array_equal(batch[0][key], data[key]) for data in batch[1:]):
                    raise ValueError("All dictionaries in the batch must have the same var_names_g.")
            # If so, just take the first one
            collated_batch[key] = batch[0][key]
        else:
            collated_batch[key] = torch.cat([torch.from_numpy(data[key]) for data in batch], dim=0)
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
        x: Pandas Index/Series/DataFrame object.

    Returns:
        Numpy array.
    """
    return np.asarray(x.cat.codes)
