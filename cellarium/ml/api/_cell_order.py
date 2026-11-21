# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
A check that the cells are not grouped by an ``obs`` column (e.g. one experiment after another), as they are when the
data is a set of per-experiment files. Training draws every minibatch from one tile (or shard) of consecutive cells,
so in grouped data the minibatches are confounded with the groups.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd

_MIN_DIVERSITY = 0.1  # columns less diverse than this in the data (nearly one value) are not checked
_MAX_TILE_MIXING = 0.5  # warn if the tiles are less than this fraction as diverse as the data
_CHUNK_CELLS = 1 << 22  # cells counted at a time (rounded to whole tiles), to bound the memory of the counting
_MAX_DENSE_COUNTS = 1 << 24  # count with a dense (tiles x values) table up to this size, and by sorting beyond
_MAX_COLUMNS_SHOWN = 5


def _tile_sums_of_squares(codes: np.ndarray, n_values: int, tile_size: int) -> np.ndarray:
    """For each tile of ``tile_size`` consecutive cells of ``codes`` (from 0 to ``n_values - 1``), the sum over the
    values of the square of its number of cells in the tile."""
    tile = np.arange(len(codes), dtype=np.int64) // tile_size
    n_tiles = int(tile[-1]) + 1
    pairs = tile * n_values + codes
    if n_tiles * n_values <= _MAX_DENSE_COUNTS:
        counts = np.bincount(pairs, minlength=n_tiles * n_values).astype(np.float64)
        return (counts**2).reshape(n_tiles, n_values).sum(axis=1)
    pair_ids, pair_counts = np.unique(pairs, return_counts=True)
    return np.bincount(pair_ids // n_values, weights=pair_counts.astype(np.float64) ** 2, minlength=n_tiles)


def tile_mixing(codes: np.ndarray, tile_size: int) -> tuple[float, float]:
    """
    How mixed the tiles of ``tile_size`` consecutive cells are in the values ``codes`` (integer codes of a categorical,
    with -1, a missing value, as a value of its own), compared to all the cells.

    The diversity of all the cells is the Gini-Simpson index ``1 - sum_k p_k**2``, the probability that two cells
    drawn from them have different values. That of a tile of ``m`` cells, ``c_k`` of which have the value ``k``, is the
    unbiased ``1 - sum_k c_k (c_k - 1) / (m (m - 1))``, the probability that two different cells of the tile have
    different values: for cells in random order, its expectation is the diversity of all of them. Returns
    ``(diversity, mixing)``: the diversity of all the cells, and the mean diversity of the tiles (weighted by their
    numbers of cells; a tile of one cell has none, and is left out) as a fraction of it, which is about 1 if the cells
    are in random order and near 0 if they are grouped by the value (``nan`` if either is not defined).
    """
    codes = np.asarray(codes, dtype=np.int64) + 1
    n = len(codes)
    n_values = int(codes.max()) + 1 if n else 1
    value_counts = np.bincount(codes, minlength=n_values).astype(np.float64)
    diversity = 1.0 - float(((value_counts / max(n, 1)) ** 2).sum())
    if diversity <= 0 or tile_size < 2:
        return diversity, float("nan")
    step = max(1, _CHUNK_CELLS // tile_size) * tile_size  # whole tiles, so no tile is split between chunks
    weighted_sum, total_weight = 0.0, 0.0  # of the diversities of the tiles, weighted by their numbers of cells
    for start in range(0, n, step):
        chunk = codes[start : start + step]
        sizes = np.bincount(np.arange(len(chunk)) // tile_size).astype(np.float64)
        pairs_differing = sizes**2 - _tile_sums_of_squares(chunk, n_values, tile_size)  # m**2 - sum c_k**2
        kept = sizes >= 2
        tile_diversity = pairs_differing[kept] / (sizes[kept] * (sizes[kept] - 1))
        weighted_sum += float((tile_diversity * sizes[kept]).sum())
        total_weight += float(sizes[kept].sum())
    if total_weight == 0:
        return diversity, float("nan")
    return diversity, weighted_sum / total_weight / diversity


def grouped_obs_columns(obs: Any, tile_size: int) -> list[tuple[str, float]]:
    """
    The categorical columns of ``obs`` (a lazy ``obs`` of :class:`~cellarium.ml.api.CellariumData`) whose tiles of
    ``tile_size`` consecutive cells are less than half as diverse as all the cells (see :func:`tile_mixing`), with
    that fraction, from the most grouped. Columns with nearly one value are left out. Each categorical column is
    read once, one at a time; no other column and no counts are read.
    """
    if len(obs) <= tile_size or tile_size < 2:  # one tile, or tiles too small to have a diversity
        return []
    grouped = []
    for column in obs.categorical_columns:
        values = obs[column]
        if not isinstance(values.dtype, pd.CategoricalDtype):
            continue
        diversity, mixing = tile_mixing(values.cat.codes.to_numpy(), tile_size)
        if diversity >= _MIN_DIVERSITY and mixing < _MAX_TILE_MIXING:
            grouped.append((column, mixing))
    return sorted(grouped, key=lambda item: item[1])


def warn_if_cells_are_grouped(obs: Any, tile_size: int) -> None:
    """Warn if the cells, in tiles of ``tile_size`` in their order, are grouped by categorical columns of ``obs`` (see
    :func:`grouped_obs_columns`), naming them, since minibatches drawn from one tile would be confounded with them."""
    grouped = grouped_obs_columns(obs, tile_size)
    if not grouped:
        return
    shown = ", ".join(f"{column} ({mixing:.0%})" for column, mixing in grouped[:_MAX_COLUMNS_SHOWN])
    more = f", and {len(grouped) - _MAX_COLUMNS_SHOWN} more" if len(grouped) > _MAX_COLUMNS_SHOWN else ""
    warnings.warn(
        f"The cells are grouped by some obs columns: in tiles of {tile_size:,} consecutive cells, these columns are "
        "much less diverse than in all the cells (the diversity of a tile as a fraction of that of all the cells): "
        f"{shown}{more}. Training draws every minibatch from one tile, so the minibatches would be confounded with "
        "these columns. Shuffle the cells first, for example with `annslicer slice --shuffle` "
        "(https://github.com/cellarium-ai/annslicer) on the h5ad files, and make the dataset from the shuffled files.",
        UserWarning,
        stacklevel=3,
    )
