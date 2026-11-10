# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
The plots of :class:`~cellarium.ml.api.NMFOutput`, as functions of the numbers that they show. They draw on new
``matplotlib`` figures (made with ``pyplot``, so that notebooks display them) and return them.
"""

from typing import Any

import numpy as np
import torch


def plot_density_histogram(
    mean_neighbor_distance_m: torch.Tensor,
    n_neighbors: int,
    k: int,
    density_threshold: float | None = None,
) -> Any:
    """
    A histogram of the mean distance of each replicate factor to its nearest ``n_neighbors`` neighbors, for choosing
    the density threshold. If ``density_threshold`` is given, it is drawn as a line, and the fraction of the replicate
    factors that it filters out is in the title.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 2))
    distances = mean_neighbor_distance_m.cpu().numpy()
    plt.hist(distances, bins=75)
    if density_threshold is None:
        plt.title(f"Local Neighborhood Distances: k = {k}")
    else:
        plt.vlines(density_threshold, 0, plt.gca().get_ylim()[1], colors="r")
        plt.title(
            f"Local Neighborhood Distances: k = {k}\ndensity_threshold filters "
            f"{100 * (distances > density_threshold).mean():.1f}% of runs"
        )
    plt.ylabel("Number of NMF factors\n(total is replicates times k)")
    plt.xlabel(f"Average distance to nearest {n_neighbors} neighbors")
    plt.xlim([-0.05, 1.05])
    return fig


def plot_k_selection(stability: dict[int, float], reconstruction_error: dict[int, float]) -> Any:
    """The stability and the reconstruction error as functions of k, on two y axes."""
    import matplotlib.pyplot as plt

    k_values = sorted(stability)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(k_values, [stability[k] for k in k_values], "o-", color="b")
    plt.grid(True)
    plt.ylabel("Stability", color="b")
    plt.xlabel("Number of components: k")
    plt.xticks(k_values)
    plt.gca().tick_params(axis="y", colors="b")
    plt.twinx()
    plt.plot(k_values, [reconstruction_error[k] for k in k_values], "o-", color="r")
    plt.ylabel("Reconstruction error", color="r")
    plt.gca().tick_params(axis="y", colors="r")
    plt.grid(False)
    return fig


def plot_clustermap(filtered_distance_matrix_ff: torch.Tensor, k: int) -> Any:
    """A clustermap of the Euclidean distances between the replicate factors that pass the density threshold."""
    try:
        import seaborn as sns
    except ImportError as error:
        raise ImportError("plot_clustermap needs seaborn: install it with `pip install seaborn`.") from error

    cg = sns.clustermap(
        np.asarray(filtered_distance_matrix_ff.cpu().numpy()),
        row_cluster=True,
        col_cluster=True,
        cbar_pos=(0.05, 0.25, 0.03, 0.15),
        figsize=(5, 5),
        xticklabels=False,
        yticklabels=False,
        rasterized=True,
    )
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    cg.cax.set_ylabel("Euclidean distance")
    cg.ax_heatmap.set_title(f"k = {k}")
    return cg.fig
