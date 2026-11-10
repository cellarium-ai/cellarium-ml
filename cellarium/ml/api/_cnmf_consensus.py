# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
The consensus step of consensus NMF (cNMF, Kotliar et al. 2019), on the factors of the replicate runs of a model:
filtering out outlier replicates by the density of their neighborhood, clustering the remaining ones, and taking the
median of each cluster as a consensus factor. Pure functions of tensors, used by
:class:`~cellarium.ml.api.NMFOutput`.

A consensus is a dictionary with the keys

- ``"consensus_D_kg"``: the L1-normalized consensus factors, of shape ``(k, g)``.
- ``"stability"``: the silhouette score of the clustering (1.0 if there is only one replicate).
- ``"all_neighbor_distances"``: for each of the ``r * k`` replicate factors, the distances to its nearest
  ``n_neighbors`` neighbors, of shape ``(r * k, n_neighbors)`` (``None`` if there is only one replicate).
- ``"n_neighbors"``, ``"local_neighborhood_size"`` and ``"density_threshold"``: the parameters that produced it.

The matrix of distances between the replicates that pass the density threshold (see
:func:`filtered_distance_matrix`) is quadratic in ``r * k``, so it is not part of a consensus.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_local_minima(mean_neighbor_distance_m: torch.Tensor, n_bins: int = 100) -> list[float]:
    """
    Helper function to find local minima in a histogram of distance values.

    Args:
        mean_neighbor_distance_m: 1D tensor of distance values
        n_bins: Number of bins for the histogram (default: 100)

    Returns:
        List of distance values corresponding to local minima (bin centers)
    """
    # Convert to numpy for histogram computation
    distances = mean_neighbor_distance_m.cpu().numpy()

    # Create histogram
    counts, bin_edges = np.histogram(distances, bins=n_bins, range=(0, 1))

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth the histogram slightly to reduce noise
    if len(counts) >= 3:
        # Simple smoothing: each bin becomes average of itself and neighbors
        smoothed_counts = counts.copy()
        for i in range(1, len(counts) - 1):
            smoothed_counts[i] = (counts[i - 1] + counts[i] + counts[i + 1]) / 3
        counts = smoothed_counts

    # Find local minima in the histogram counts
    # A local minimum is a point that is smaller than or equal to both neighbors
    # but at least one neighbor must be strictly larger
    local_minima_indices = []

    for i in range(1, len(counts) - 1):
        left_count = counts[i - 1]
        center_count = counts[i]
        right_count = counts[i + 1]

        # Check if this is a local minimum
        if (
            center_count <= left_count
            and center_count <= right_count
            and (center_count < left_count or center_count < right_count)
        ):
            local_minima_indices.append(i)

    # Convert indices to actual distance values (bin centers)
    candidate_values = [float(bin_centers[i]) for i in local_minima_indices]

    # Also consider the first and last bins as potential candidates
    # if they are at the boundaries and could be minima
    if len(counts) > 1:
        # Add first bin if it's a potential minimum
        if counts[0] <= counts[1]:
            candidate_values.insert(0, float(bin_centers[0]))

        # Add last bin if it's a potential minimum
        if counts[-1] <= counts[-2]:
            candidate_values.append(float(bin_centers[-1]))

    # Remove duplicates and sort
    candidate_values = sorted(list(set(candidate_values)))

    # Filter out values that are too close to each other (within 5% of range)
    min_distance = 0.05
    filtered_candidates: list[float] = []
    for val in candidate_values:
        if not any(abs(val - existing) < min_distance for existing in filtered_candidates):
            filtered_candidates.append(val)

    return filtered_candidates


@dataclass
class Neighbors:
    """
    The distances between the factors of the replicate runs of one value of k.

    Attributes:
        d_norm_mg: The L2-normalized factors of all replicates, of shape ``(m, g)`` with ``m = r * k``.
        euclidean_dist_mm: The Euclidean distances between them, of shape ``(m, m)``.
        n_nearest_dist_ml: For each factor, the distances to its ``n_neighbors`` nearest neighbors (not counting
            itself), of shape ``(m, n_neighbors)``.
        n_neighbors: The number of neighbors, ``int(r * local_neighborhood_size)``.
    """

    d_norm_mg: torch.Tensor
    euclidean_dist_mm: torch.Tensor
    n_nearest_dist_ml: torch.Tensor
    n_neighbors: int

    @property
    def mean_neighbor_distance_m(self) -> torch.Tensor:
        """The mean distance of each factor to its nearest neighbors, which the density threshold is applied to."""
        return self.n_nearest_dist_ml.mean(dim=1)


def _check_local_neighborhood_size(local_neighborhood_size: float) -> None:
    if not 0 < local_neighborhood_size < 1:
        raise ValueError(f"local_neighborhood_size must be between 0 and 1, got {local_neighborhood_size}")


def _check_density_threshold(density_threshold: float) -> None:
    if not 0 < density_threshold <= 1:
        raise ValueError(f"density_threshold must be > 0 and <= 1, got {density_threshold}")


def compute_neighbors(D_rkg: torch.Tensor, local_neighborhood_size: float) -> Neighbors:
    """
    The distances between the replicate factors ``D_rkg`` and each factor's nearest neighbors.

    Args:
        D_rkg: The factors of the ``r`` replicate runs, of shape ``(r, k, g)``. Requires ``r > 1``.
        local_neighborhood_size: The fraction of the replicates that are considered neighbors.

    Raises:
        ValueError: If ``local_neighborhood_size`` is not in (0, 1), or is so small that fewer than 2 replicates
            are neighbors.
    """
    _check_local_neighborhood_size(local_neighborhood_size)
    r, num_component, g = D_rkg.shape
    if r < 2:
        raise ValueError("The distances between replicates need at least 2 replicates.")
    n_neighbors = int(r * local_neighborhood_size)
    if n_neighbors < 2:
        raise ValueError(
            f"local_neighborhood_size {local_neighborhood_size} is too small for k={num_component}. "
            f"n_neighbors = int(replicates * local_neighborhood_size) = {n_neighbors}. "
            "We want n_neighbors >= 2. Increase local_neighborhood_size."
        )

    d_norm_mg = F.normalize(D_rkg, dim=-1, p=2).reshape(r * num_component, g)

    # euclidean distance to every other run
    euclidean_dist_mm = torch.cdist(d_norm_mg, d_norm_mg, p=2)
    euclidean_dist_mm.fill_diagonal_(0)  # correct for roundoff errors that may be present

    # top n_neighbors plus self (distance 0), and then the distances to the top n_neighbors
    n_nearest_dist_including_self_mL, _ = torch.topk(euclidean_dist_mm, n_neighbors + 1, largest=False)
    return Neighbors(
        d_norm_mg=d_norm_mg,
        euclidean_dist_mm=euclidean_dist_mm,
        n_nearest_dist_ml=n_nearest_dist_including_self_mL[:, 1:],
        n_neighbors=n_neighbors,
    )


def consensus_from_neighbors(
    neighbors: Neighbors,
    num_component: int,
    density_threshold: float,
    local_neighborhood_size: float,
) -> dict[str, Any]:
    """
    The consensus factors, given the ``neighbors`` of the replicates, which can be reused across density thresholds.

    Raises:
        ValueError: If ``density_threshold`` is not in (0, 1], or too few replicate factors pass it to be clustered.
    """
    _check_density_threshold(density_threshold)

    # filter out runs considered outliers based on threshold
    d_norm_fg = neighbors.d_norm_mg[neighbors.mean_neighbor_distance_m < density_threshold, :]
    if len(d_norm_fg) <= num_component:
        raise ValueError(
            f"Only {len(d_norm_fg)} replicate factors are left for k={num_component} after applying density "
            f"threshold {density_threshold}, which is too few to cluster into {num_component} programs. "
            "Increase density_threshold (see plot_density_histograms)."
        )

    # run k-means clustering on the remaining f programs
    df_d_norm_fg = pd.DataFrame(d_norm_fg.cpu().numpy())
    kmeans = KMeans(n_clusters=num_component, n_init=10, random_state=1)
    kmeans.fit(df_d_norm_fg)
    kmeans_cluster_labels_f = pd.Series(kmeans.labels_ + 1, index=df_d_norm_fg.index)

    # compute the consensus k programs as the median of each cluster
    median_D_kg = torch.tensor(df_d_norm_fg.groupby(kmeans_cluster_labels_f).median().values)

    silhouette = silhouette_score(df_d_norm_fg.values, kmeans_cluster_labels_f, metric="euclidean")

    return {
        "consensus_D_kg": F.normalize(median_D_kg, dim=-1, p=1),
        "stability": float(silhouette),
        "all_neighbor_distances": neighbors.n_nearest_dist_ml,
        "n_neighbors": neighbors.n_neighbors,
        "local_neighborhood_size": local_neighborhood_size,
        "density_threshold": density_threshold,
    }


def consensus_factors(
    D_rkg: torch.Tensor,
    density_threshold: float,
    local_neighborhood_size: float,
) -> dict[str, Any]:
    """
    The consensus of the replicate factors ``D_rkg``: filter out the replicate factors whose mean distance to their
    nearest neighbors is not less than ``density_threshold``, cluster the rest into ``k`` clusters, and take the median
    of each cluster. See the module docstring for the keys of the result.

    Args:
        D_rkg: The factors of the ``r`` replicate runs, of shape ``(r, k, g)``.
        density_threshold: The threshold on the mean distance to the nearest neighbors, in (0, 1].
        local_neighborhood_size: The fraction of the replicates that are considered neighbors, in (0, 1).

    Raises:
        ValueError: If a parameter is out of range, or the parameters leave too few replicate factors to cluster.
    """
    _check_local_neighborhood_size(local_neighborhood_size)
    _check_density_threshold(density_threshold)
    r, num_component, g = D_rkg.shape

    if r == 1:
        # a single replicate has nothing to cluster or filter: its factors are the consensus
        return {
            "consensus_D_kg": F.normalize(D_rkg.reshape(r * num_component, g), dim=-1, p=1),
            "stability": 1.0,
            "all_neighbor_distances": None,
            "n_neighbors": 0,
            "local_neighborhood_size": local_neighborhood_size,
            "density_threshold": density_threshold,
        }

    neighbors = compute_neighbors(D_rkg, local_neighborhood_size)
    return consensus_from_neighbors(neighbors, num_component, density_threshold, local_neighborhood_size)


def filtered_distance_matrix(
    D_rkg: torch.Tensor, density_threshold: float, local_neighborhood_size: float
) -> torch.Tensor:
    """
    The Euclidean distances between the replicate factors that pass the density threshold, of shape ``(f, f)``, for
    drawing the clustermap of the replicates. Recomputed on demand because it is quadratic in ``r * k``.
    """
    _check_density_threshold(density_threshold)
    neighbors = compute_neighbors(D_rkg, local_neighborhood_size)
    keep = neighbors.mean_neighbor_distance_m < density_threshold
    if not keep.any():
        raise ValueError(f"No replicate factors pass density_threshold {density_threshold}.")
    return neighbors.euclidean_dist_mm[keep, :][:, keep]


def neighbor_distances_for_plot(D_rkg: torch.Tensor, local_neighborhood_size: float) -> tuple[torch.Tensor, int]:
    """The mean neighbor distance of each replicate factor, and the number of neighbors it is the mean over."""
    neighbors = compute_neighbors(D_rkg, local_neighborhood_size)
    return neighbors.mean_neighbor_distance_m, neighbors.n_neighbors


def maximal_stability_consensus(
    D_rkg: torch.Tensor,
    local_neighborhood_size: float = 0.3,
    density_threshold_limits: tuple[float, float] = (0.05, 0.3),
    max_tolerable_fraction_eliminated: float = 0.3,
    search: Literal["fast", "exhaustive"] = "fast",
) -> dict[str, Any]:
    """
    The consensus at the density threshold that maximizes the stability (silhouette score), found by search.

    Args:
        D_rkg: The factors of the ``r`` replicate runs, of shape ``(r, k, g)``.
        local_neighborhood_size: The fraction of the replicates that are considered neighbors.
        density_threshold_limits: The lower and upper bounds (inclusive) of the density thresholds to try.
        max_tolerable_fraction_eliminated: In ``"fast"`` mode, do not consider a density threshold that eliminates more
            than this fraction of the replicate factors.
        search: ``"fast"`` tries only the local minima of the histogram of mean neighbor distances within the limits
            (and falls back to ``"exhaustive"`` if there are none), ``"exhaustive"`` tries every threshold from the
            lower to the upper limit in steps of 0.01.

    Returns:
        The consensus (see the module docstring) at the best density threshold.

    Raises:
        ValueError: If ``search`` is not ``"fast"`` or ``"exhaustive"``, or no density threshold within the limits
            leaves enough replicate factors to cluster.
    """
    if search not in ("fast", "exhaustive"):
        raise ValueError("search must be 'fast' or 'exhaustive'")
    low, high = density_threshold_limits
    r, num_component, _ = D_rkg.shape
    if r == 1:
        return consensus_factors(D_rkg, density_threshold=high, local_neighborhood_size=local_neighborhood_size)

    neighbors = compute_neighbors(D_rkg, local_neighborhood_size)  # the same for every density threshold
    exhaustive_candidates = np.arange(low, high + 1e-9, 0.01).tolist()

    candidates = exhaustive_candidates
    if search == "fast":
        mean_neighbor_distance_m = neighbors.mean_neighbor_distance_m
        candidates = [
            dt
            for dt in find_local_minima(mean_neighbor_distance_m)
            if low <= dt <= high and (mean_neighbor_distance_m > dt).float().mean() <= max_tolerable_fraction_eliminated
        ]
        if not candidates:
            warnings.warn(
                f"Unable to find local minima in the k={num_component} density histogram: the 'fast' search falls "
                "back to 'exhaustive'.",
                stacklevel=2,
            )
            candidates = exhaustive_candidates

    best: dict[str, Any] | None = None
    for density_threshold in candidates:
        try:
            result = consensus_from_neighbors(neighbors, num_component, density_threshold, local_neighborhood_size)
        except ValueError:  # too few replicate factors pass this threshold
            continue
        if best is None or result["stability"] > best["stability"]:
            best = result
    if best is None:
        raise ValueError(
            f"No density threshold between {low} and {high} leaves enough replicate factors to cluster for "
            f"k={num_component}. Widen density_threshold_limits or increase local_neighborhood_size."
        )
    return best
