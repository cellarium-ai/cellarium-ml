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


def series_to_str_list(x: pd.Series) -> list[str]:
    """
    Convert a pandas Series of strings to a list of strings.
    Args:
        x: Pandas Series object.

    Returns:
        List of strings.
    """
    return x.astype(str).to_list()


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
        return x.apply(lambda col: col.cat.codes).to_numpy(dtype=np.int32)
    else:
        return np.asarray(x.cat.codes, dtype=np.int32)


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


def get_var_names_g_indices(
    input_var_names_g: np.ndarray,
    stored_var_names_g: np.ndarray,
) -> np.ndarray:
    """
    Return integer indices that map each gene in ``input_var_names_g`` to its position in
    ``stored_var_names_g``.

    This allows parametric transforms (e.g. :class:`~cellarium.ml.transforms.ZScore`,
    :class:`~cellarium.ml.transforms.DivideByScale`) to accept any subset or reordering of
    the gene space they were initialized with, by looking up the per-gene statistics for only
    the genes present in the current batch.

    Args:
        input_var_names_g:
            Gene names arriving at the transform (may be a subset or reordering of
            ``stored_var_names_g``).
        stored_var_names_g:
            The full gene-name schema the transform was initialized with.

    Returns:
        A 1-D integer array of length ``len(input_var_names_g)`` where element ``i`` is the
        index of ``input_var_names_g[i]`` in ``stored_var_names_g``.

    Raises:
        ValueError: If any gene in ``input_var_names_g`` is absent from ``stored_var_names_g``.
    """
    stored_index: dict[str, int] = {name: idx for idx, name in enumerate(stored_var_names_g)}
    indices: list[int] = []
    missing: list[str] = []
    for gene in input_var_names_g:
        if gene in stored_index:
            indices.append(stored_index[gene])
        else:
            missing.append(gene)
    if missing:
        raise ValueError(f"The following genes in `var_names_g` are not present in the stored schema: {missing}")
    return np.array(indices, dtype=np.intp)


def compute_cl_distance_matrix(owl_uri: str) -> pd.DataFrame:
    """
    Compute an all-pairs shortest-path distance matrix over the Cell Ontology (CL).

    Nodes are all CL classes found in the OWL file. Distances are computed on the
    **undirected** ontology graph, so sibling cell types separated by a common parent
    get a finite distance (2 hops) rather than infinity.

    This is a slow, one-time offline pre-computation (typically ~1 minute for the full
    CL ontology). Save the result to disk and pass it to the scVI model constructor via
    ``ontology_distance_matrix``. Example::

        df = compute_cl_distance_matrix(
            "https://github.com/obophenotype/cell-ontology/releases/download/v2024-01-04/cl.owl"
        )
        df.to_parquet("cl_distance_matrix.parquet")

    Args:
        owl_uri: URI or local path of the CL OWL file (passed to
            ``owlready2.get_ontology(...).load()``).

    Returns:
        A symmetric square :class:`pandas.DataFrame` of ``float32`` values with CL ID
        strings (e.g. ``"CL:0000540"``) as both index and columns. Diagonal entries are
        ``0.0``; disconnected pairs have ``inf``.

    Raises:
        ImportError: If ``owlready2`` or ``networkx`` are not installed.
            Install with ``pip install cellarium-ml[ontology]``.
    """
    try:
        import networkx as nx
        import owlready2
    except ImportError as e:
        raise ImportError(
            "owlready2 and networkx are required for compute_cl_distance_matrix. "
            "Install them with: pip install cellarium-ml[ontology]"
        ) from e

    ontology = owlready2.get_ontology(owl_uri).load()
    all_classes = list(ontology.classes())

    # Filter to CL classes that have exactly one label
    cl_prefix = "CL_"
    classes = [c for c in all_classes if c.name.startswith(cl_prefix) and len(c.label) == 1]
    cl_ids = [c.name.replace("_", ":") for c in classes]
    name_to_idx: dict[str, int] = {cl_id: i for i, cl_id in enumerate(cl_ids)}
    classes_set = set(classes)

    # Build directed graph using parent/child relationships
    graph = nx.DiGraph()
    for cl_id in cl_ids:
        graph.add_node(cl_id)
    for cls in classes:
        cl_id = cls.name.replace("_", ":")
        for parent in ontology.get_parents_of(cls):
            if parent not in classes_set:
                continue
            parent_id = parent.name.replace("_", ":")
            graph.add_edge(parent_id, cl_id)
        for child in ontology.get_children_of(cls):
            if child not in classes_set:
                continue
            child_id = child.name.replace("_", ":")
            graph.add_edge(cl_id, child_id)

    # Use undirected graph so sibling pairs are reachable
    undirected = graph.to_undirected()

    n = len(cl_ids)
    dist_matrix = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(dist_matrix, 0.0)

    for source, lengths in nx.all_pairs_shortest_path_length(undirected):
        src_idx = name_to_idx[source]
        for target, d in lengths.items():
            tgt_idx = name_to_idx[target]
            dist_matrix[src_idx, tgt_idx] = float(d)

    df = pd.DataFrame(dist_matrix, index=cl_ids, columns=cl_ids)

    # Append an "unknown" label with inf distance to all other labels (and 0 to itself).
    # This allows datasets that include an "unknown" cell type to pass through the constructor
    # without errors; the inf distances will be excluded from all finite-only metric calculations.
    df.loc["unknown", :] = np.inf
    df.loc[:, "unknown"] = np.inf
    df.loc["unknown", "unknown"] = 0.0

    return df
