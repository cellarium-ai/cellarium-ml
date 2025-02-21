# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""General gene network analysis and validation."""

import itertools
import logging
import typing as t
import warnings
from dataclasses import dataclass
from functools import cached_property, lru_cache

import colorcet as cc
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymde
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.linalg import eigh
from scipy.stats import linregress, norm
from skopt import gp_minimize
from tqdm import tqdm

from cellarium.ml.utilities.inference.cellarium_gpt_inference import load_gene_info_table
from cellarium.ml.utilities.inference.gene_set_utils import (
    compute_function_on_gene_sets_given_clustering,
    compute_function_on_gene_sets_given_neighbors,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a handler
handler = logging.StreamHandler()

# Create and set a formatter
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


def quantile_normalize_select(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    top_k: int,
    min_x: int | float | None = None,
    max_x: int | float | None = None,
):
    """
    Filter, normalize, and select top_k elements.

    Args:
        x: 1D numpy array of covariate values.
        y: 1D numpy array of response values.
        n_bins: Number of bins to subdivide the range [min_x, max_x].
        top_k: Number of top largest normalized y values to select.
        min_x: Minimum x value (x must be > min_x).
        max_x: Maximum x value (x must be < max_x).

    Returns:
        selected_indices: indices in the original arrays corresponding to the top_k selected values
        top_normalized_y: the normalized y values corresponding to these selected indices
    """

    # Create bin edges (n_bins bins from min_x to max_x).
    bin_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)

    # Assign each x_valid to a bin.
    # np.digitize returns bin indices in 1..n_bins, so subtract 1 to have 0-indexed bins.
    bin_indices = np.digitize(x, bin_edges) - 1

    # Prepare an array for the normalized y values.
    y_normalized = np.empty_like(y, dtype=float)

    # Process each bin separately.
    for i in range(n_bins):
        # Find indices in x_valid that fall in bin i.
        in_bin = np.where(bin_indices == i)[0]
        if in_bin.size > 0:
            # Compute the mean of y values in this bin.
            bin_mean = np.mean(y[in_bin])
            # Avoid division by zero (if bin_mean happens to be zero, leave values unchanged).
            if bin_mean == 0:
                y_normalized[in_bin] = y[in_bin]
            else:
                y_normalized[in_bin] = y[in_bin] / bin_mean

    if min_x is None:
        min_x = np.min(x)
    if max_x is None:
        max_x = np.max(x)

    _y_normalized = y_normalized.copy()
    _y_normalized[x < min_x] = -np.inf
    _y_normalized[x > max_x] = -np.inf

    sorted_idx = np.argsort(_y_normalized)[::-1]
    top_k = min(top_k, int(np.sum(_y_normalized > -np.inf)))
    top_idx = sorted_idx[:top_k]

    return top_idx, y_normalized


@dataclass
class GeneNetworkAnalysisData:
    """Container for required inputs for a gene network analysis with type annotations."""

    matrix_qp: np.ndarray

    prompt_var_names: list[str]
    prompt_marginal_mean_p: np.ndarray
    prompt_marginal_std_p: np.ndarray

    query_var_names: list[str]
    query_marginal_mean_q: np.ndarray
    query_marginal_std_q: np.ndarray

    def __eq__(self, value):
        # for testing purposes
        return all(
            [
                (
                    (getattr(self, key).shape == getattr(value, key).shape)
                    and np.allclose(getattr(self, key), getattr(value, key))
                )
                for key in [
                    "matrix_qp",
                    "prompt_marginal_mean_p",
                    "prompt_marginal_std_p",
                    "query_marginal_mean_q",
                    "query_marginal_std_q",
                ]
            ]
        ) and all(
            [
                getattr(self, key) == getattr(value, key)
                for key in [
                    "prompt_var_names",
                    "query_var_names",
                ]
            ]
        )


def process_response_matrix(
    analysis_data: GeneNetworkAnalysisData,
    total_mrna_umis: float | None,
    response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
    feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
    min_prompt_gene_tpm: float = 0.0,
    min_query_gene_tpm: float = 0.0,
    query_response_amp_min_pct: float | None = None,
    feature_max_value: float | None = None,
    norm_pseudo_count: float = 1e-3,
    query_hv_top_k: int | None = None,
    query_hv_n_bins: int | None = 50,
    query_hv_min_x: float | None = 1e-2,
    query_hv_max_x: float | None = np.inf,
    z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
    eps: float = 1e-8,
    verbose: bool = True,
) -> GeneNetworkAnalysisData:
    """
    Process the response matrix for gene network analysis.

    Args:
        analysis_data: input data
            matrix_qp: response matrix
            prompt_var_names: prompt gene names
            prompt_marginal_mean_p: prompt marginal mean
            prompt_marginal_std_p: prompt marginal standard deviation
            query_var_names: query gene names
            query_marginal_mean_q: query marginal mean
            query_marginal_std_q: query marginal standard deviation
        total_mrna_umis: total mRNA UMIs
        response_normalization_strategy: matrix_qp normalization strategy
            - "mean": normalize by marginal mean
            - "std": normalize by marginal standard deviation
            - "none": no normalization
        feature_normalization_strategy: feature normalization strategy
            - "l2": L2 normalization per query feature
            - "z_score": z-score per feature
            - "none": no normalization
        min_prompt_gene_tpm: minimum prompt gene TPM
        min_query_gene_tpm: minimum query gene TPM
        query_response_amp_min_pct: minimum query response amplitude percentile
        feature_max_value: maximum feature value
        norm_pseudo_count: pseudo-count for normalization
        query_hv_top_k: number of top query genes to select
        query_hv_n_bins: number of bins for histogram equalization
        query_hv_min_x: minimum x value for histogram equalization
        query_hv_max_x: maximum x value for histogram equalization
        z_trans_func: transformation function for z values
        eps: epsilon value for numerical stability
        verbose: whether to print verbose output
    """
    response_qp = analysis_data.matrix_qp

    prompt_var_names = analysis_data.prompt_var_names
    prompt_marginal_mean_p = analysis_data.prompt_marginal_mean_p
    prompt_marginal_std_p = analysis_data.prompt_marginal_std_p

    query_var_names = analysis_data.query_var_names
    query_marginal_mean_q = analysis_data.query_marginal_mean_q
    query_marginal_std_q = analysis_data.query_marginal_std_q

    if response_normalization_strategy == "mean":
        z_p = prompt_marginal_mean_p
        z_q = query_marginal_mean_q
    elif response_normalization_strategy == "std":
        z_p = prompt_marginal_std_p
        z_q = query_marginal_std_q
    elif response_normalization_strategy == "none":
        z_p = np.ones_like(prompt_marginal_mean_p)
        z_q = np.ones_like(query_marginal_mean_q)
    else:
        raise ValueError("Invalid Jacobian normalization strategy")

    # linear proportional activation
    z_qp = response_qp * (z_p[None, :] + norm_pseudo_count) / (z_q[:, None] + norm_pseudo_count)
    assert isinstance(z_qp, np.ndarray)

    if verbose:
        logger.info(f"Maximum value of z_qp: {np.max(z_qp):.3f}")
        logger.info(f"Minimum value of z_qp: {np.min(z_qp):.3f}")

    if total_mrna_umis is not None:
        mask_q = (1e6 * query_marginal_mean_q / total_mrna_umis) >= min_query_gene_tpm
        mask_p = (1e6 * prompt_marginal_mean_p / total_mrna_umis) >= min_prompt_gene_tpm
    else:
        logger.warning("Total mRNA UMIs not provided, skipping TPM filtering")
        mask_q = np.ones_like(query_marginal_mean_q, dtype=bool)
        mask_p = np.ones_like(prompt_marginal_mean_p, dtype=bool)

    logger.info(f"Number of query genes after TPM filtering: {np.sum(mask_q)} / {len(mask_q)}")
    logger.info(f"Number of prompt genes after TPM filtering: {np.sum(mask_p)} / {len(mask_p)}")

    if query_response_amp_min_pct is not None:
        z_norm_q = np.linalg.norm(z_qp, axis=-1)
        z_norm_thresh = np.percentile(z_norm_q, query_response_amp_min_pct)
        mask_q = mask_q & (z_norm_q >= z_norm_thresh)
        logger.info(f"Number of query genes after z-norm filtering: {np.sum(mask_q)} / {len(mask_q)}")

    if query_hv_top_k is not None:
        assert query_hv_n_bins is not None
        assert query_hv_min_x is not None
        assert query_hv_max_x is not None
        top_idx, _ = quantile_normalize_select(
            x=np.log1p(query_marginal_mean_q),
            y=np.std(z_qp, axis=1),
            n_bins=query_hv_n_bins,
            top_k=query_hv_top_k,
            min_x=query_hv_min_x,
            max_x=query_hv_max_x,
        )
        hv_mask_q = np.zeros_like(mask_q, dtype=bool)
        hv_mask_q[top_idx] = True
        mask_q = mask_q & hv_mask_q
        logger.info(f"Number of query genes after highly-variable filtering: {np.sum(mask_q)} / {len(mask_q)}")

    # apply the mask to everything else
    prompt_var_names_masked = [prompt_var_names[i] for i in range(len(prompt_var_names)) if mask_p[i]]
    prompt_marginal_mean_p_masked = prompt_marginal_mean_p[mask_p]
    prompt_marginal_std_p_masked = prompt_marginal_std_p[mask_p]

    query_var_names_masked = [query_var_names[i] for i in range(len(query_var_names)) if mask_q[i]]
    query_marginal_mean_q_masked = query_marginal_mean_q[mask_q]
    query_marginal_std_q_masked = query_marginal_std_q[mask_q]

    # apply the mask to z_qp
    z_qp = z_qp[mask_q, :][:, mask_p]

    # clip and transform features
    assert isinstance(z_qp, np.ndarray)
    z_qp[np.isnan(z_qp)] = 0.0
    z_qp[np.isinf(z_qp)] = 0.0

    if feature_max_value is not None:
        assert feature_max_value > 0
        z_qp = np.clip(z_qp, -feature_max_value, feature_max_value)

    if z_trans_func is not None:
        z_qp = z_trans_func(z_qp)

    # handle feature normalization
    pearson = False  # True means that z_qp @ z_qp.T will be the pearson correlation matrix
    if feature_normalization_strategy == "z_score":
        # z-score each query gene separately in response to prompt genes
        z_qp = (z_qp - np.mean(z_qp, axis=1, keepdims=True)) / (eps + np.std(z_qp, axis=1, keepdims=True))
        if pearson:
            z_qp = z_qp / np.sqrt(z_qp.shape[1])
    elif feature_normalization_strategy == "l2":
        # l2-normalize query genes separately for each prompt gene
        z_qp = z_qp / (eps + np.linalg.norm(z_qp, axis=0, keepdims=True))
    elif feature_normalization_strategy == "none":
        pass
    else:
        raise ValueError("Invalid feature normalization strategy")

    z_qp[np.isnan(z_qp)] = 0.0
    z_qp[np.isinf(z_qp)] = 0.0

    return GeneNetworkAnalysisData(
        matrix_qp=z_qp,
        prompt_var_names=prompt_var_names_masked,
        prompt_marginal_mean_p=prompt_marginal_mean_p_masked,
        prompt_marginal_std_p=prompt_marginal_std_p_masked,
        query_var_names=query_var_names_masked,
        query_marginal_mean_q=query_marginal_mean_q_masked,
        query_marginal_std_q=query_marginal_std_q_masked,
    )


def compute_adjacency_matrix(
    z_qp: np.ndarray,
    adjacency_strategy: t.Literal[
        "shifted_correlation", "unsigned_correlation", "positive_correlation", "positive_correlation_binary"
    ] = "positive_correlation",
    n_neighbors: int | None = 50,
    self_loop: bool = False,
    scale_by_node_degree: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute an adjacency matrix from a query-by-prompt gene matrix of values.

    Args:
        z_qp: query-by-prompt gene matrix of values
        adjacency_strategy: adjacency strategy
        n_neighbors: number of neighbors to keep
        self_loop: whether to include self-loops
        scale_by_node_degree: whether to scale the adjacency matrix a_ij by the node degree d_i
            a_ij -> a_ij / sqrt(d_i * d_j)
        **kwargs: additional keyword arguments
            beta: power for correlation
            tau: threshold for binary adjacency

    Returns:
        query-by-query adjacency matrix for genes
    """
    n_query_genes = z_qp.shape[0]
    rho_pp = z_qp.T @ z_qp / n_query_genes

    if adjacency_strategy == "shifted_correlation":
        assert "beta" in kwargs, "Must provide beta for shifted correlation"
        beta = kwargs["beta"]
        a_pp = np.power(0.5 * (1 + rho_pp), beta)
    elif adjacency_strategy == "unsigned_correlation":
        assert "beta" in kwargs, "Must provide beta for unsigned correlation"
        beta = kwargs["beta"]
        a_pp = np.power(np.abs(rho_pp), beta)
    elif adjacency_strategy == "positive_correlation":
        assert "beta" in kwargs, "Must provide beta for positive correlation"
        beta = kwargs["beta"]
        a_pp = np.power(np.maximum(0, rho_pp), beta)
    elif adjacency_strategy == "positive_correlation_binary":
        assert n_neighbors is None, "n_neighbors must be None for binary adjacency"
        assert "tau" in kwargs, "Must provide correlation threshold for binary adjacency"
        tau = kwargs["tau"]
        a_pp = (np.maximum(0, rho_pp) > tau).astype(float)
    else:
        raise ValueError("Invalid adjacency strategy")

    assert np.isclose(a_pp, a_pp.T).all(), "Adjacency matrix must be symmetric -- something is wrong!"

    if n_neighbors is not None:
        assert n_neighbors > 0, "n_neighbors must be positive"
        t_qn = np.argsort(a_pp, axis=-1)[:, -n_neighbors:]  # take the top n_neighbors

        # make a mask for the top n_neighbors
        _a_pp = np.zeros_like(a_pp)
        for p in range(a_pp.shape[0]):
            _a_pp[p, t_qn[p]] = a_pp[p, t_qn[p]]
            _a_pp[t_qn[p], p] = a_pp[t_qn[p], p]
    else:
        _a_pp = a_pp
    a_pp = _a_pp

    if not self_loop:
        np.fill_diagonal(a_pp, 0)

    if scale_by_node_degree:
        d_p = np.sum(a_pp, axis=1)
        a_pp = a_pp / np.sqrt(d_p[:, None] * d_p[None, :])

    return a_pp


def compute_igraph_from_adjacency(
    a_pp: np.ndarray, node_names: list[str] | np.ndarray, directed: bool = False
) -> ig.Graph:
    """
    Convert an adjacency matrix to an :class:`ig.Graph` graph.

    Args:
        a_pp: (gene-gene) adjacency matrix
        node_names: names of the nodes (gene names)
        directed: whether the graph is directed

    Returns:
        :class:`ig.Graph` graph
    """
    assert len(node_names) == a_pp.shape[0], (
        f"Number of node names ({len(node_names)}) must match adjacency matrix shape ({a_pp.shape[0]})"
    )
    assert a_pp.shape[0] == a_pp.shape[1], "Adjacency matrix must be square"
    sources, targets = a_pp.nonzero()
    weights = a_pp[sources, targets]
    g = ig.Graph(directed=directed)
    g.add_vertices(node_names)
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights
    return g


def compute_leiden_communites(
    g: ig.Graph,
    resolution: float = 3.0,
    min_community_size: int = 2,
    random_seed: int = 0,
) -> np.ndarray:
    """
    Compute Leiden communities from an :class:`ig.Graph` graph by running
    :meth:`leidenalg.find_partition` with `RBConfigurationVertexPartition`.

    Args:
        g: :class:`ig.Graph` graph
        resolution: resolution parameter
        min_community_size: minimum community size
        random_seed: random seed

    Returns:
        array of community memberships
    """

    leiden_partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=random_seed
    )
    leiden_membership = np.array(leiden_partition.membership)

    # optionally remove small communities
    if min_community_size > 1:
        n_leiden = len(np.unique(leiden_membership))
        sizes = np.array([np.sum(leiden_membership == i) for i in range(n_leiden)])
        for i_leiden in range(n_leiden):
            if sizes[i_leiden] < min_community_size:
                leiden_membership[leiden_membership == i_leiden] = -1

    return leiden_membership


def compute_knn_from_adjacency(
    a_pp: np.ndarray,
    node_names_p: np.ndarray,
    n_neighbors: int,
) -> dict[str, set[str]]:
    """
    Compute the k-nearest neighbors for each node using a pre-computed adjacency matrix.

    Args:
        a_pp: adjacency matrix
        node_names_p: names of the nodes
        n_neighbors: number of neighbors to keep

    Returns:
        dictionary of k-nearest neighbors for each node
    """
    assert a_pp.shape[0] == len(node_names_p), "Number of node names must match the first dimension of a_qq"

    # compute the k-nearest neighbors
    knn: dict[str, set[str]] = {}
    for p in range(a_pp.shape[0]):
        t_p = np.argsort(a_pp[p], axis=-1)[-n_neighbors:].squeeze()  # take the top n_neighbors
        knn[node_names_p[p]] = set(node_names_p[t_p])
    return knn


def compute_spectral_dimension(
    a_pp: np.ndarray, offset: int = 2, n_lambda_for_estimation: int = 5
) -> dict[str, np.ndarray | float]:
    """
    Compute the spectral dimension of a graph from its adjacency matrix.

    Args:
        a_qq: adjacency matrix
        offset: offset for the first nonzero eigenvalue
        n_lambda_for_estimation: number of eigenvalues to use for estimation

    Returns:
        dictionary of spectral dimension results
    """

    # calculate normalized laplacian and its eigenvalues
    norm_p = 1.0 / (1e-9 + np.sqrt(a_pp.sum(0)))
    lap_pp = np.eye(a_pp.shape[0]) - norm_p[:, None] * norm_p[None, :] * a_pp
    eigs = eigh(lap_pp.astype(np.float64), eigvals_only=True)
    eigs[0] = 0
    eigs = np.clip(eigs, 0, np.inf)  # roundoff error guardrail

    n_lambda = np.cumsum(eigs)
    n_lambda = n_lambda / n_lambda[-1]
    first_nonzero = np.where(eigs > 0)[0][0] + offset
    xx = np.log(eigs[first_nonzero : first_nonzero + n_lambda_for_estimation])
    yy = np.log(n_lambda[first_nonzero : first_nonzero + n_lambda_for_estimation])

    lin = linregress(xx, yy)
    slope, intercept = lin.slope, lin.intercept

    # save a few thigs for later
    spectral_dim = 2 * linregress(xx, yy).slope
    spectral_results = {
        "spectral_dim": spectral_dim,
        "eigs": eigs,
        "n_lambda": n_lambda,
        "log_eigs_asymptotic": xx,
        "log_n_lambda_asymptotic": yy,
        "spectral_dim_slope": slope,
        "spectral_dim_intercept": intercept,
    }

    return spectral_results


def mde_embedding(
    z_qp: np.ndarray | torch.Tensor,
    n_neighbors: int = 7,
    repulsive_fraction: int = 5,
    attractive_penalty: pymde.functions.function.Function = pymde.penalties.Log1p,
    repulsive_penalty: pymde.functions.function.Function = pymde.penalties.InvPower,
    device: torch.device | str = "cuda",
    max_iter: int = 500,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Run :mod:`pymde` to compute a 2-dimensional MDE embedding.

    Args:
        z_qp: matrix to embed
        others: passthroughs to :func:`pymde.preserve_neighbors`, except
            for `max_iter` which is passed to :meth:`pymde.MDE.embed`

    Returns:
        2-dimensional MDE embedding
    """
    mde = pymde.preserve_neighbors(
        torch.tensor(z_qp),
        device=device,
        verbose=verbose,
        n_neighbors=n_neighbors,
        repulsive_fraction=repulsive_fraction,
        attractive_penalty=attractive_penalty,
        repulsive_penalty=repulsive_penalty,
        **kwargs,
    )
    embedding_q2 = mde.embed(verbose=verbose, max_iter=max_iter).cpu().numpy()
    return embedding_q2


def mde_embedding_replogle(
    z_qp: np.ndarray,
    embedding_dimension: int = 2,
    do_row_zscore: bool = True,
    do_gene_zscore_preprocessing: bool = False,
    spectral_embedding_kwargs: dict = dict(
        n_components=20, affinity="nearest_neighbors", n_neighbors=7, eigen_solver="arpack"
    ),
    mde_kwargs: dict = dict(device="cuda", embedding_dim=2, n_neighbors=7, repulsive_fraction=5, verbose=True),
    scaling_max_value: float = 10.0,
) -> np.ndarray:
    """
    Compute a minimal distortion embedding of a dataset, following the recipe from Replogle et al. [1].

    Args:
        adata: AnnData object
        layer: layer in adata.layers to use
        embedding_dimension: dimension of the embedding
        do_row_zscore: whether to z-score rows just before computing the spectral embedding, as in [1]
        do_gene_zscore_preprocessing: whether to z-score genes as a preprocessing step (this should be done
            per batch. if already done in adata.layers[layer], set to False)
        spectral_embedding_kwargs: keyword arguments for SpectralEmbedding
        mde_kwargs: keyword arguments for pymde.preserve_neighbors
        scaling_max_value: maximum value when scaling the data

    References:
    [1] Replogle, Joseph M. et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq.
        Cell, Volume 185, Issue 14, 2559 - 2575.e28
    """

    try:
        import pymde
    except ImportError:
        raise ImportError("pymde must be installed to run compute_mde. do `pip install pymde`")

    try:
        from sklearn.manifold import SpectralEmbedding
    except ImportError:
        raise ImportError("scikit-learn must be installed to run compute_mde. do `pip install scikit-learn`")

    # handle embedding dimension
    if "n_components" in spectral_embedding_kwargs:
        logger.warning("ignoring n_components in spectral_embedding_kwargs and replacing with embedding_dimension")
    spectral_embedding_kwargs["n_components"] = embedding_dimension

    if "embedding_dim" in mde_kwargs:
        logger.warning("ignoring embedding_dim in mde_kwargs and replacing with embedding_dimension")
    mde_kwargs["embedding_dim"] = embedding_dimension

    adata = sc.AnnData(X=z_qp)

    # z-score each gene (this is the original scaling of the data done in replogle... per batch)
    if do_gene_zscore_preprocessing:
        sc.pp.scale(adata, zero_center=True, max_value=scaling_max_value)

    # as per replogle, scale rows (so that euclidean distance becomes proportional to correlation)
    if do_row_zscore:
        adata.X = sc.pp.scale(adata.X.transpose(), max_value=scaling_max_value, zero_center=True).transpose()
    x = adata.X.copy()
    adata.obsm["X_moderated"] = x.copy()

    # spectral embedding for initialization (Replogle does 20 dimensions)
    np.random.seed(0)
    se = SpectralEmbedding(**spectral_embedding_kwargs).fit_transform(x)

    # minimal distortion embedding
    pymde.seed(0)
    mde = pymde.preserve_neighbors(x, **mde_kwargs)  # MDE runs on the actual data
    x_init = torch.tensor(se).contiguous()  # .contiguous() sidesteps a bug
    embedding = mde.embed(X=x_init, verbose=True)  # uses the spectral embedding as initialization

    return embedding.detach().cpu().numpy()


class NetworkAnalysisBase:
    """Base class for network analysis.

    Provides methods for computing adjacency matrix, graph, Leiden clustering, and spectral dimension.

    Attributes:
        z_qp: input matrix of values
        node_names_p: input names of the prompt genes (nodes -- the perturbations)
        leiden_membership: Leiden clustering results
        spectral: dictionary of spectral dimension results

    Properties:
        adjacency_matrix: computed adjacency matrix
        leiden_communities: computed Leiden communities
        spectral_dim: computed spectral dimension and related results
    """

    def __init__(self, z_qp: np.ndarray, node_names_p: list[str] | np.ndarray):
        assert z_qp.shape[-1] == len(node_names_p), "Number of node names must match the second dimension of z_qp"
        assert np.isnan(z_qp).sum() == 0, "There are NaNs in z_qp, which is not allowed"
        assert np.isinf(z_qp).sum() == 0, "There are infs in z_qp, which is not allowed"
        assert (z_qp != 0).sum() > 0, "There are no nonzero values in z_qp, which is not allowed"
        self.z_qp = z_qp
        self.node_names_p = node_names_p
        self.a_pp: np.ndarray | None = None  # adjacency matrix
        self.adjacency_kwargs: dict[str, t.Any] = {}  # used by validation methods to re-compute adjacency matrix
        self.leiden_membership: np.ndarray | None = None
        self.spectral: dict[str, np.ndarray | float] = {}
        self.embedding_p2: np.ndarray | None = None

        # prevent false cache hits after reprocessing
        self.igraph.cache_clear()
        self.compute_leiden_communites.cache_clear()

    @property
    def adjacency_matrix(self) -> np.ndarray:
        if self.a_pp is None:
            raise UserWarning("Compute adjacency matrix by calling obj.compute_adjacency_matrix()")
        return self.a_pp

    @property
    def leiden_communities(self) -> np.ndarray:
        if self.leiden_membership is None:
            raise UserWarning("Compute Leiden clustering by calling obj.compute_leiden_communites()")
        return self.leiden_membership

    @property
    def spectral_dim(self) -> dict:
        if "spectral_dim" not in self.spectral:
            raise UserWarning("Compute spectral dimension by calling obj.compute_spectral_dimension()")
        return self.spectral

    def compute_adjacency_matrix(
        self,
        adjacency_strategy: t.Literal[
            "shifted_correlation", "unsigned_correlation", "positive_correlation", "positive_correlation_binary"
        ],
        n_neighbors: int | None = 50,
        self_loop: bool = False,
        scale_by_node_degree: bool = False,
        **kwargs,
    ) -> np.ndarray:
        self.adjacency_kwargs = kwargs | {
            "adjacency_strategy": adjacency_strategy,
            "n_neighbors": n_neighbors,
            "self_loop": self_loop,
        }
        a_pp = compute_adjacency_matrix(
            z_qp=self.z_qp,
            adjacency_strategy=adjacency_strategy,
            n_neighbors=n_neighbors,
            self_loop=self_loop,
            scale_by_node_degree=scale_by_node_degree,
            **kwargs,
        )
        self.a_pp = a_pp

        # clear previously computed properties
        self.igraph.cache_clear()
        self.compute_leiden_communites.cache_clear()

        return a_pp

    @lru_cache(maxsize=2)
    def igraph(self, directed: bool = False) -> ig.Graph:
        if self.a_pp is None:
            raise UserWarning("Compute adjacency matrix first by calling obj.compute_adjacency_matrix()")
        return compute_igraph_from_adjacency(
            a_pp=self.a_pp,
            node_names=self.node_names_p,
            directed=directed,
        )

    @lru_cache(maxsize=30)
    def compute_leiden_communites(
        self,
        resolution: float,
        min_community_size: int = 2,
    ) -> np.ndarray:
        self.leiden_membership = compute_leiden_communites(
            g=self.igraph(),
            resolution=resolution,
            min_community_size=min_community_size,
        )
        return self.leiden_membership

    def compute_spectral_dimension(self, offset: int = 2, n_lambda_for_estimation: int = 5) -> float:
        self.spectral = compute_spectral_dimension(
            a_pp=self.adjacency_matrix,
            offset=offset,
            n_lambda_for_estimation=n_lambda_for_estimation,
        )
        return float(self.spectral["spectral_dim"])

    def make_mde_embedding(
        self,
        n_neighbors: int = 7,
        repulsive_fraction: int = 5,
        attractive_penalty: pymde.functions.function.Function = pymde.penalties.Log1p,
        repulsive_penalty: pymde.functions.function.Function = pymde.penalties.InvPower,
        device: torch.device = torch.device("cpu"),
        max_iter: int = 500,
        verbose: bool = True,
        **kwargs,
    ):
        self.embedding_p2 = mde_embedding(
            self.z_qp.T,  # we are embedding the prompts (perturbations)
            n_neighbors=n_neighbors,
            repulsive_fraction=repulsive_fraction,
            attractive_penalty=attractive_penalty,
            repulsive_penalty=repulsive_penalty,
            device=device,
            max_iter=max_iter,
            verbose=verbose,
            **kwargs,
        )


class GeneNetworkAnalysisBase(NetworkAnalysisBase):
    def __init__(
        self,
        adata_obs: pd.DataFrame | None,
        total_mrna_umis: float | None,
        gene_info_tsv_path: str,
        query_var_names: list[str],
        prompt_var_names: list[str],
        response_qp: np.ndarray,
        prompt_marginal_mean_p: np.ndarray,
        prompt_marginal_std_p: np.ndarray,
        query_marginal_mean_q: np.ndarray,
        query_marginal_std_q: np.ndarray,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
        min_prompt_gene_tpm: float = 0.0,
        min_query_gene_tpm: float = 0.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float | None = None,
        norm_pseudo_count: float = 1e-3,
        query_hv_top_k: int | None = None,
        query_hv_n_bins: int | None = 50,
        query_hv_min_x: float | None = 1e-2,
        query_hv_max_x: float | None = np.inf,
        z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        eps: float = 1e-8,
        verbose: bool = True,
    ):
        self.verbose = verbose

        n_query_vars = len(query_var_names)
        n_prompt_vars = len(prompt_var_names)

        assert response_qp.shape == (n_query_vars, n_prompt_vars)
        assert prompt_marginal_mean_p.shape == (n_prompt_vars,)
        assert prompt_marginal_std_p.shape == (n_prompt_vars,)
        assert query_marginal_mean_q.shape == (n_query_vars,)
        assert query_marginal_std_q.shape == (n_query_vars,)

        self.adata_obs = adata_obs

        # handle total mrna umis
        self.total_mrna_umis_: float | None = None
        if total_mrna_umis is not None:
            self.total_mrna_umis_ = total_mrna_umis
        elif adata_obs is not None:
            if "total_mrna_umis" in adata_obs.columns:
                self.total_mrna_umis_ = adata_obs["total_mrna_umis"].values[0]

        self.unprocessed = GeneNetworkAnalysisData(
            matrix_qp=response_qp,
            prompt_var_names=prompt_var_names,
            prompt_marginal_mean_p=prompt_marginal_mean_p,
            prompt_marginal_std_p=prompt_marginal_std_p,
            query_var_names=query_var_names,
            query_marginal_mean_q=query_marginal_mean_q,
            query_marginal_std_q=query_marginal_std_q,
        )

        self.gene_info_tsv_path = gene_info_tsv_path
        self.gene_info_df, self.gene_symbol_to_gene_id_map, self.gene_id_to_gene_symbol_map = load_gene_info_table(
            gene_info_tsv_path, query_var_names + prompt_var_names
        )

        self.processed: GeneNetworkAnalysisData
        self.reprocess(
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
            norm_pseudo_count=norm_pseudo_count,
            query_hv_top_k=query_hv_top_k,
            query_hv_n_bins=query_hv_n_bins,
            query_hv_min_x=query_hv_min_x,
            query_hv_max_x=query_hv_max_x,
            z_trans_func=z_trans_func,
            eps=eps,
        )

        super().__init__(z_qp=self.processed.matrix_qp, node_names_p=self.processed.prompt_var_names)

    @property
    def cell_type(self) -> str | None:
        if self.adata_obs is None:
            return None
        return self.adata_obs["cell_type"].values[0]

    @property
    def tissue(self) -> str | None:
        if self.adata_obs is None:
            return None
        return self.adata_obs["tissue"].values[0]

    @property
    def disease(self) -> str | None:
        if self.adata_obs is None:
            return None
        return self.adata_obs["disease"].values[0]

    @property
    def development_stage(self) -> str | None:
        if self.adata_obs is None:
            return None
        return self.adata_obs["development_stage"].values[0]

    @property
    def sex(self) -> str | None:
        if self.adata_obs is None:
            return None
        return self.adata_obs["sex"].values[0]

    @property
    def total_mrna_umis(self) -> float | None:
        return self.total_mrna_umis_

    @property
    def query_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.processed.query_var_names]

    @property
    def query_gene_ids(self) -> list[str]:
        return self.processed.query_var_names

    @property
    def prompt_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.processed.prompt_var_names]

    @property
    def prompt_gene_ids(self) -> list[str]:
        return self.processed.prompt_var_names

    @cached_property
    def query_gene_id_to_idx_map(self) -> dict[str, int]:
        return {gene_id: idx for idx, gene_id in enumerate(self.processed.query_var_names)}

    @cached_property
    def query_gene_symbol_to_idx_map(self) -> dict[str, int]:
        return {gene_symbol: idx for idx, gene_symbol in enumerate(self.query_gene_symbols)}

    @cached_property
    def query_gene_symbol_to_id_map(self) -> dict[str, str]:
        return {gene_symbol: gene_id for gene_symbol, gene_id in zip(self.query_gene_symbols, self.query_gene_ids)}

    @cached_property
    def query_gene_id_to_symbol_map(self) -> dict[str, str]:
        return {gene_id: gene_symbol for gene_symbol, gene_id in self.query_gene_symbol_to_id_map.items()}

    @cached_property
    def prompt_gene_id_to_idx_map(self) -> dict[str, int]:
        return {gene_id: idx for idx, gene_id in enumerate(self.processed.prompt_var_names)}

    @cached_property
    def prompt_gene_symbol_to_idx_map(self) -> dict[str, int]:
        return {gene_symbol: idx for idx, gene_symbol in enumerate(self.prompt_gene_symbols)}

    @cached_property
    def prompt_gene_symbol_to_id_map(self) -> dict[str, str]:
        return {gene_symbol: gene_id for gene_symbol, gene_id in zip(self.prompt_gene_symbols, self.prompt_gene_ids)}

    @cached_property
    def prompt_gene_id_to_symbol_map(self) -> dict[str, str]:
        return {gene_id: gene_symbol for gene_symbol, gene_id in self.prompt_gene_symbol_to_id_map.items()}

    def __str__(self) -> str:
        return (
            f"GeneNetworkAnalysisBase({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__

    def reprocess(
        self,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
        min_prompt_gene_tpm: float = 0.0,
        min_query_gene_tpm: float = 0.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float | None = None,
        norm_pseudo_count: float = 1e-3,
        query_hv_top_k: int | None = None,
        query_hv_n_bins: int | None = 50,
        query_hv_min_x: float | None = 1e-2,
        query_hv_max_x: float | None = np.inf,
        z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.processed = process_response_matrix(
            analysis_data=self.unprocessed,
            total_mrna_umis=self.total_mrna_umis,
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
            norm_pseudo_count=norm_pseudo_count,
            query_hv_top_k=query_hv_top_k,
            query_hv_n_bins=query_hv_n_bins,
            query_hv_min_x=query_hv_min_x,
            query_hv_max_x=query_hv_max_x,
            z_trans_func=z_trans_func,
            eps=eps,
            verbose=self.verbose,
        )

        # re-initialize the object and zero out pre-computed and cached properties
        for attr in [
            "query_gene_id_to_idx_map",
            "prompt_gene_id_to_idx_map",
            "query_gene_symbol_to_idx_map",
            "prompt_gene_symbol_to_idx_map",
            "query_gene_symbol_to_id_map",
            "query_gene_id_to_symbol_map",
            "prompt_gene_symbol_to_id_map",
            "prompt_gene_id_to_symbol_map",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)  # clear cached property
        super().__init__(z_qp=self.processed.matrix_qp, node_names_p=self.processed.prompt_var_names)

    def plot_mde_embedding(
        self,
        marker_size: int = 2,
        highlight_marker_size: int = 4,
        width: int = 800,
        height: int = 800,
        highlight_gene_sets: dict[str, t.Tuple[list[str], list[str], str]] | None = None,
    ) -> go.Figure:
        assert self.embedding_p2 is not None, "Must compute MDE embedding first"
        assert self.leiden_membership is not None, "Must compute Leiden communities first"

        plot_title = f"""{self.cell_type}<br>{self.tissue}<br>{self.disease}"""

        # Create a color map for the memberships
        memberships_p = self.leiden_membership
        unique_memberships = np.unique(memberships_p)

        # Convert memberships to strings for categorical mapping
        unique_memberships.astype(str)

        # Create the color map with string keys
        colormap = {str(label): cc.glasbey[i % len(cc.glasbey)] for i, label in enumerate(unique_memberships)}

        # Create a DataFrame for Plotly
        df = pd.DataFrame(
            {
                "x": self.embedding_p2[:, 0],
                "y": self.embedding_p2[:, 1],
                "label": self.prompt_gene_symbols,
                "membership": memberships_p.astype(str),  # Convert to string
            }
        )

        # Create the scatter plot
        fig = px.scatter(
            df, x="x", y="y", hover_name="label", title=plot_title, color="membership", color_discrete_map=colormap
        )

        # Update marker size
        fig.update_traces(marker=dict(size=marker_size))  # Adjust the size as needed

        if highlight_gene_sets is not None:
            for gene_set_name, (gene_ids, gene_symbols, color) in highlight_gene_sets.items():
                prompt_gene_indices = [self.prompt_gene_id_to_idx_map[gene_id] for gene_id in gene_ids]

                # show a scatter plot and color the markers in red
                fig.add_scatter(
                    x=self.embedding_p2[prompt_gene_indices, 0],
                    y=self.embedding_p2[prompt_gene_indices, 1],
                    mode="markers",
                    marker=dict(color=color, size=highlight_marker_size),
                    text=gene_symbols,
                    showlegend=True,
                    name=gene_set_name,
                )

        # Update layout to decrease the width of the plot
        fig.update_layout(
            width=width,  # Adjust the width as needed
            height=height,
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, showticklabels=False, title="MDE_1"),
            yaxis=dict(showgrid=False, showticklabels=False, title="MDE_2"),
        )

        return fig

    def plot_spectral_dimension(self, ax: plt.Axes) -> None:
        assert self.spectral["eigs"] is not None, "Must compute spectral dimension first"
        ax.scatter(self.spectral["log_eigs_asymptotic"], self.spectral["log_n_lambda_asymptotic"])
        ax.plot(
            self.spectral["log_eigs_asymptotic"],
            self.spectral["spectral_dim_slope"] * self.spectral["log_eigs_asymptotic"]
            + self.spectral["spectral_dim_intercept"],
            color="red",
            label=f"$d_S$ = {self.spectral['spectral_dim']:.2f}",
        )
        ax.set_xlabel(r"ln $\lambda$")
        ax.set_ylabel(r"ln N($\lambda$)")
        if self.cell_type is not None:
            ax.set_title(self.cell_type)
        ax.legend()


class BaseClassProtocol(t.Protocol):
    """Tell the mixin what it can count on inheriting from the base class (for typechecking)"""

    z_qp: np.ndarray
    node_names_p: list[str] | np.ndarray
    adjacency_kwargs: dict[str, t.Any]

    @property
    def adjacency_matrix(self) -> np.ndarray: ...

    @property
    def query_gene_symbol_to_idx_map(self) -> dict[str, int]: ...

    @property
    def query_gene_id_to_idx_map(self) -> dict[str, int]: ...

    @property
    def prompt_gene_id_to_idx_map(self) -> dict[str, int]: ...

    @property
    def prompt_gene_symbol_to_idx_map(self) -> dict[str, int]: ...

    @property
    def query_gene_symbols(self) -> list[str]: ...

    @property
    def prompt_gene_symbols(self) -> list[str]: ...

    compute_leiden_communites: t.Callable[..., np.ndarray]


class ValidationMixin(BaseClassProtocol):
    """Mixin with methods for bio-inspired validation of gene networks."""

    def _get_reference_gene_sets_as_inds_and_filter(
        self,
        reference_gene_sets: dict[str, set[str]],
        gene_naming: t.Literal["id", "symbol"] = "symbol",
        reference_set_exclusion_fraction: float = 0.25,
        min_set_size: int = 3,
        verbose: bool = False,
    ) -> dict[str, set[int]]:
        """
        Convert reference gene sets to indices. Omits gene sets entirely if they do not have enough
        members in :meth:`GeneNetworkAnalysisBase.igraph`.

        Args:
            reference_gene_sets: dictionary of reference gene sets {set1_name: {gene_A, gene_B, ...}, ...}
            gene_naming: whether the genes in the reference sets are gene IDs or gene symbols
            reference_set_exclusion_fraction: gene sets with fewer than this fraction in graph are excluded
            min_set_size: minimum size of a gene set present in the graph to be included

        Returns:
            dictionary of reference gene sets as indices {set1_name: {gene_A_idx, gene_B_idx, ...}, ...}
        """
        # handle gene naming
        if gene_naming == "symbol":
            gene_ind_lookup: dict[str, int] = self.prompt_gene_symbol_to_idx_map
        elif gene_naming == "id":
            gene_ind_lookup = self.prompt_gene_id_to_idx_map
        else:
            raise ValueError("Invalid gene_naming")

        # convert reference gene sets to indices
        reference_gene_sets_as_inds = {
            set_name: {gene_ind_lookup[g] for g in gene_set if g in gene_ind_lookup.keys()}
            for set_name, gene_set in reference_gene_sets.items()
        }

        # filter out gene sets that are too small
        final_ref_gene_sets_as_inds: dict[str, set[int]] = {}
        for set_name, gene_set_inds in reference_gene_sets_as_inds.items():
            if len(gene_set_inds) < min_set_size:
                if verbose:
                    logger.warning(
                        f"Reference gene set {set_name} has < {min_set_size} members in graph and will be skipped"
                    )
                continue
            fraction_of_set_in_graph = len(gene_set_inds) / len(reference_gene_sets[set_name])
            if fraction_of_set_in_graph < reference_set_exclusion_fraction:
                if verbose:
                    logger.warning(
                        f"Reference gene set {set_name} has < {reference_set_exclusion_fraction:.2f} "
                        "fraction of members in graph and will be skipped"
                    )
                continue
            final_ref_gene_sets_as_inds[set_name] = gene_set_inds

        return final_ref_gene_sets_as_inds

    def compute_network_adjacency_auc_metric(
        self,
        reference_gene_sets: dict[str, set[str]],
        reference_set_exclusion_fraction: float = 0.25,
        min_set_size: int = 3,
        min_adjacency: float = 1e-5,
        gene_naming: t.Literal["id", "symbol"] = "symbol",
        verbose: bool = False,
    ) -> tuple[float, pd.DataFrame]:
        """
        Compute a metric for the overall concordance between the concordance matrix in
        :meth:`GeneNetworkAnalysisBase.adjacency_matrix` and reference gene sets defined by a dictionary.
        The metric is based on the adjacency matrix and the reference gene sets, and is computed
        by calculating an AUC by varying the threshold for the adjacency matrix and considering
        true positives as the number of edges (above the threshold) that are in a reference set,
        and false positives as the number of edges that are not in a reference set. The adjacency
        can be considered to be the metric which prioritizes candidate edges, and the union of
        reference gene sets constitutes ground truth.

        Args:
            reference_gene_sets: dictionary of reference gene sets {set1_name: {gene_A, gene_B, ...}, ...}
            reference_set_exclusion_fraction: gene sets with fewer than this fraction of genes present in the graph
                are excluded from the concordance metric
            min_set_size: minimum size of a gene set present in the graph to be included in the concordance metric
            min_adjacency: act as though all adjacencies below this value are zero
            gene_naming: whether to use gene IDs or gene symbols
            verbose: whether to print warnings for excluded gene sets

        Returns:
            concordance AUC, DataFrame of thresholds, true positive rate, false positive rate
        """
        # adjacency matrix without diagonal elements and with small values set to zero
        a_qq = self.adjacency_matrix
        assert isinstance(a_qq, np.ndarray)
        q = a_qq.shape[0]
        zero_diag_a_qq = a_qq * (1.0 - np.eye(q))
        mask_qq = zero_diag_a_qq > min_adjacency
        zero_diag_a_qq = zero_diag_a_qq * mask_qq  # helps sparse calcs below

        # decide which gene sets to include and convert to indices
        final_ref_gene_sets_as_inds = self._get_reference_gene_sets_as_inds_and_filter(
            reference_gene_sets=reference_gene_sets,
            gene_naming=gene_naming,
            reference_set_exclusion_fraction=reference_set_exclusion_fraction,
            min_set_size=min_set_size,
            verbose=False,
        )
        if verbose:
            logger.info(f"{len(final_ref_gene_sets_as_inds)} gene sets meet criteria")

        # compute ground truth for edge evidence in any reference gene set
        row: list[int] = []
        col: list[int] = []
        for _, gene_set_inds in final_ref_gene_sets_as_inds.items():
            # all pairs of inds in the gene set
            gene_set_pairs = list(itertools.combinations(gene_set_inds, 2))
            # convert list of tuples to row list and column list, and append
            row_inds, col_inds = zip(*gene_set_pairs)
            row.extend(row_inds)
            col.extend(col_inds)
        edges_in_reference_df = pd.DataFrame(
            {
                "row": row,
                "col": col,
            }
        ).drop_duplicates()
        edges_in_reference_df["reference"] = 1

        # set up dataframe used for ROC curve
        edges_in_adjacency_coo = sp.coo_matrix(zero_diag_a_qq)
        edges_in_adjacency_coo.eliminate_zeros()
        edges_in_adjacency_df = pd.DataFrame(
            {
                "row": edges_in_adjacency_coo.row,
                "col": edges_in_adjacency_coo.col,
                "adjacency": edges_in_adjacency_coo.data,
            }
        )
        edges_in_adjacency_df = pd.merge(
            left=edges_in_adjacency_df,
            right=edges_in_reference_df,
            on=["row", "col"],
            how="left",
        )
        edges_in_adjacency_df["reference"] = edges_in_adjacency_df["reference"].fillna(0)

        # compute tpr and fpr
        # tpr = tp / (tp + fn) = tp / total positives
        # fpr = fp / (fp + tn) = fp / total negatives
        edges_in_adjacency_df = edges_in_adjacency_df.sort_values("adjacency", ascending=False)
        edges_in_adjacency_df["true_positives"] = edges_in_adjacency_df["reference"].cumsum().astype(int)
        edges_in_adjacency_df["false_positives"] = (1 - edges_in_adjacency_df["reference"]).cumsum().astype(int)
        edges_in_adjacency_df["true_positive_rate"] = (
            edges_in_adjacency_df["true_positives"] / edges_in_adjacency_df["reference"].sum()
        )
        edges_in_adjacency_df["false_positive_rate"] = (
            edges_in_adjacency_df["false_positives"] / (1 - edges_in_adjacency_df["reference"]).sum()
        )

        # compute AUC
        auc = np.trapz(edges_in_adjacency_df["true_positive_rate"], edges_in_adjacency_df["false_positive_rate"])

        return auc, edges_in_adjacency_df

    @staticmethod
    def plot_roc(
        fpr: pd.Series | np.ndarray | list[float],
        tpr: pd.Series | np.ndarray | list[float],
        max_datapoints: int = 10_000,
    ) -> None:
        """
        Plot a ROC curve.

        Args:
            fpr: false positive rate
            tpr: true positive rate
            max_datapoints: maximum number of data points to plot
        """
        assert len(fpr) == len(tpr), "fpr and tpr must have the same length"
        auc = np.trapz(tpr, fpr)
        if len(fpr) > max_datapoints:
            step = len(fpr) // max_datapoints
            fpr = fpr[::step]
            tpr = tpr[::step]
        plt.fill_between(fpr, tpr, color="gray", alpha=0.25)
        plt.fill_between([0, 1], [0, 1], color="white", alpha=1.0)
        plt.plot([0, 1], [0, 1], "--", color="red")
        plt.plot(fpr, tpr, ".k")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.annotate(f"AUC: {auc:.3f}", (0.5, 0.05))
        plt.grid(False)
        plt.gca().set_aspect("equal")

    def plot_gene_set_adjacency_heatmap(
        self,
        gene_set: set[str],
        gene_naming: t.Literal["id", "symbol"] = "symbol",
        n_random_control_genes: int = 0,
    ) -> None:
        """
        Plot a heatmap of the adjacency matrix for a set of genes.

        Args:
            gene_set: set of gene names
            gene_naming: whether to use gene IDs or gene symbols
            n_random_control_genes: number of random genes to include as a control
        """
        import seaborn as sns

        gene_list = list(gene_set)

        # handle gene naming
        if gene_naming == "symbol":
            gene_ind_lookup: dict[str, int] = self.prompt_gene_symbol_to_idx_map
        elif gene_naming == "id":
            gene_ind_lookup = self.prompt_gene_id_to_idx_map
        else:
            raise ValueError("Invalid gene_naming")

        # convert reference gene sets to indices
        gene_inds = [gene_ind_lookup[g] for g in gene_list if g in gene_ind_lookup.keys()]

        # add random genes as a control if called for
        if n_random_control_genes > 0:
            random_gene_inds = np.random.choice(
                np.array(list(set(np.arange(self.z_qp.shape[0])) - set(gene_inds))),
                n_random_control_genes,
                replace=False,
            )
            gene_inds.extend(random_gene_inds)
            gene_list.extend(self.prompt_gene_symbols[i] for i in random_gene_inds)

        a_qq = self.adjacency_matrix
        a_qq = a_qq[gene_inds][:, gene_inds]

        # seaborn clustermap that labels the genes
        g = sns.clustermap(
            a_qq,
            cmap="Oranges",
            figsize=(4, 4),
            xticklabels=gene_list,
            yticklabels=gene_list,
        )
        g.ax_heatmap.grid(False)
        # plt.imshow(a_qq, cmap="Oranges", aspect="auto")
        # plt.colorbar()
        # plt.xticks(range(len(gene_inds)), [f"{g}\n{i}" for i, g in enumerate(gene_set)], rotation=90)
        # plt.yticks(range(len(gene_inds)), [f"{g}\n{i}" for i, g in enumerate(gene_set)])
        # plt.title("Adjacency matrix for gene set")
        # plt.show()

    def compute_network_adjacency_concordance_metric(
        self,
        reference_gene_sets: dict[str, set[str]],
        reference_set_exclusion_fraction: float = 0.25,
        min_set_size: int = 3,
        p_value_threshold: float = 0.5,
        gene_naming: t.Literal["id", "symbol"] = "symbol",
        verbose: bool = False,
    ) -> tuple[float, pd.DataFrame]:
        """
        Compute a metric for the overall concordance between the adjacency matrix in
        :meth:`GeneNetworkAnalysisBase.adjacency_matrix` and reference gene sets defined by a dictionary.
        The metric is based on the adjacency matrix and the reference gene sets, and is computed
        by comparing the mean adjacency element for a set of genes to the mean adjacency element
        for a random set of genes and computing a p-value. The concordance metric is the sum of
        -log10(p-value) for all reference gene sets with p-value below a threshold.

        Args:
            reference_gene_sets: dictionary of reference gene sets {set1_name: {gene_A, gene_B, ...}, ...}
            reference_set_exclusion_fraction: gene sets with fewer than this fraction of genes present in the graph
                are excluded from the concordance metric
            min_set_size: minimum size of a gene set present in the graph to be included in the concordance metric
            adjacency_strategy: adjacency strategy
            p_value_threshold: p-values above this threshold are considered totally insignificant
                and are not included in the final concordance metric, which is the sum of -log10(p-value)
            gene_naming: whether to use gene IDs or gene symbols
            verbose: whether to print warnings for excluded gene sets

        Returns:
            concordance, DataFrame of values for each reference gene set
        """
        # adjacency matrix without diagonal elements
        a_qq = self.adjacency_matrix
        assert isinstance(a_qq, np.ndarray)
        q = a_qq.shape[0]
        zero_diag_a_qq = a_qq * (1.0 - np.eye(q))
        mean_element_value = zero_diag_a_qq.sum() / (q * q - q)

        # decide which gene sets to include and convert to indices
        final_ref_gene_sets_as_inds = self._get_reference_gene_sets_as_inds_and_filter(
            reference_gene_sets=reference_gene_sets,
            gene_naming=gene_naming,
            reference_set_exclusion_fraction=reference_set_exclusion_fraction,
            min_set_size=min_set_size,
            verbose=False,
        )

        if verbose:
            logger.info(f"{len(final_ref_gene_sets_as_inds)} gene sets meet criteria")

        def _effect_of_set(gene_inds: set[int]) -> float:
            assert len(gene_inds) > 1, "Gene set must have at least 2 members"
            # compute the mean adjacency element for a set of genes
            inds = np.array(list(gene_inds))
            element_sum = zero_diag_a_qq[inds][:, inds].sum()
            effect = element_sum / (len(gene_inds) * len(gene_inds) - len(gene_inds))
            return effect

        n_samples = 1000

        @lru_cache(maxsize=1000)
        def _random_set_effects(size_of_random_set: int, n_samples: int = n_samples) -> np.ndarray:
            # compute the standard deviation of the effect of a random set of genes
            effects = [
                _effect_of_set(set(np.random.choice(a_qq.shape[0], size_of_random_set, replace=False)))
                for _ in range(n_samples)
            ]
            return np.array(effects)

        # go through each reference gene set and compute the effect
        dfs = []
        for set_name, gene_set_inds in tqdm(final_ref_gene_sets_as_inds.items()):
            if len(gene_set_inds) < min_set_size:
                if verbose:
                    logger.warning(
                        f"Reference gene set {set_name} has < {min_set_size} members in graph and will be skipped"
                    )
                continue
            fraction_of_set_in_graph = len(gene_set_inds) / len(reference_gene_sets[set_name])
            if fraction_of_set_in_graph < reference_set_exclusion_fraction:
                if verbose:
                    logger.warning(
                        f"Reference gene set {set_name} has < {reference_set_exclusion_fraction:.2f} "
                        "fraction of members in graph and will be skipped"
                    )
                continue
            effect = _effect_of_set(gene_set_inds)
            normalized_effect = effect - mean_element_value

            # p-value via permutation test
            random_control_effects = _random_set_effects(size_of_random_set=len(gene_set_inds))
            pval = (random_control_effects > effect).mean()
            pval = np.clip(pval, a_min=1.0 / n_samples, a_max=None)  # cannot be less than 1/n_samples

            # p-value via analytical approximation using law of large numbers
            scale = 1 / len(gene_set_inds) / 22  # the factor 1/22 is empirical
            pval_analytic = 1.0 - norm.cdf(effect, loc=mean_element_value, scale=scale)

            # TODO permutation test for small gene sets, then analytic approx for large gene sets

            dfs.append(
                pd.DataFrame(
                    {
                        "gene_set": [set_name],
                        "effect": [effect],
                        "normalized_effect": [normalized_effect],
                        "pval": [pval],
                        "pval_analytic": [pval_analytic],
                        "fraction_of_set_in_graph": [fraction_of_set_in_graph],
                        "n_genes_in_set": [len(reference_gene_sets[set_name])],
                        "n_genes_in_set_in_graph": [len(gene_set_inds)],
                    }
                )
            )

        df = pd.concat(dfs, axis=0, ignore_index=True)
        concordance = (
            df["pval"][df["pval"] < p_value_threshold].apply(lambda x: -1 * np.log10(x)).sum()
        )  # sum of negative log p-values
        return concordance, df

    def compute_network_cluster_concordance_metric(
        self,
        reference_gene_sets: dict[str, set[str]],
        resolution_range: tuple[float, float] = (0.2, 8.0),
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
        optimization_strategy: t.Literal["gridsearch", "bayesopt"] = "gridsearch",
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> float:
        """
        Compute a metric for the overall concordance between the network in
        :meth:`GeneNetworkAnalysisBase.igraph` and reference gene sets defined by a dictionary.
        Distills the :meth:`bayesopt_optimal_resolution_communities_given_gene_sets` or
        :meth:`gridsearch_optimal_resolution_communities_given_gene_sets` output
        into a single concordance metric.

        Args:
            reference_gene_sets: dictionary of reference gene sets {set1_name: {gene_A, gene_B, ...}, ...}
            resolution_range: range of Leiden clustering resolutions to consider, e.g. (0.2, 8.0)
            metric_name: metric to use for concordance
            optimization_strategy: optimization strategy, either "gridsearch" or "bayesopt"
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            concordance value
        """
        if optimization_strategy == "gridsearch":
            _, _, _, best_metrics_mean = self.gridsearch_optimal_resolution_communities_given_gene_sets(
                reference_gene_sets=reference_gene_sets,
                resolutions=np.linspace(*resolution_range, 20),
                metric_name=metric_name,
                gene_naming=gene_naming,
            )
        elif optimization_strategy == "bayesopt":
            _, _, _, best_metrics_mean = self.bayesopt_optimal_resolution_communities_given_gene_sets(
                reference_gene_sets=reference_gene_sets,
                resolution_range=resolution_range,
                metric_name=metric_name,
                gene_naming=gene_naming,
            )
        else:
            raise ValueError("Invalid optimization strategy")

        return best_metrics_mean

    def compute_network_knn_concordance_metric(
        self,
        reference_gene_sets: dict[str, set[str]],
        k_values: list[int] | np.ndarray = [2, 3, 4, 5, 6, 7, 8, 9, 10],
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> float:
        """
        Compute a metric for the overall concordance between the network in
        :meth:`GeneNetworkAnalysisBase.igraph` and reference gene sets defined by a dictionary.
        Distills the :meth:`gridsearch_optimal_k_neighbors_given_gene_sets` output
        into a single concordance metric.

        Args:
            reference_gene_sets: dictionary of reference gene sets {set1_name: {gene_A, gene_B, ...}, ...}
            k_values: kNN k values to consider
            metric_name: metric to use for concordance
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            concordance value
        """
        _, _, _, best_metrics_mean = self.gridsearch_optimal_k_neighbors_given_gene_sets(
            reference_gene_sets=reference_gene_sets,
            k_values=k_values,
            metric_name=metric_name,
            gene_naming=gene_naming,
        )
        return best_metrics_mean

    def cluster_and_compute_metrics(
        self,
        resolution: float,
        reference_gene_sets: dict[str, set[str]],
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"],
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Run Leiden clustering on the network and compute the mean of a concordance metric
        over reference gene sets.

        Args:
            resolution: Leiden resolution
            reference_gene_sets: dictionary of reference gene sets
            metric_name: metric to use for concordance
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            (Leiden clustering, DataFrame of best reference gene set metrics for all clusters)
        """
        gene_names = np.asarray(self.node_names_p) if (gene_naming == "id") else np.asarray(self.prompt_gene_symbols)

        # compute the Leiden clustering
        clustering = self.compute_leiden_communites(resolution=resolution)

        # compute the best reference gene set for each cluster and record the metric
        metrics_df = compute_function_on_gene_sets_given_clustering(
            clustering=clustering,
            gene_names=gene_names,
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
        )

        return clustering, metrics_df

    def knn_and_compute_metrics(
        self,
        a_pp: np.ndarray,
        k: int,
        reference_gene_sets: dict[str, set[str]],
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"],
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> tuple[dict[str, set[str]], pd.DataFrame]:
        """
        Compute k-nearest-neighbors on the network and compute the mean of a concordance metric
        over reference gene sets.

        Args:
            a_pp: adjacency matrix
            k: number of neighbors for kNN
            reference_gene_sets: dictionary of reference gene sets
            metric_name: metric to use for concordance
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            (neighbor dictionary, DataFrame of best reference gene set metrics for all genes)
        """

        # compute the neighbors (cheap given adjacency)
        neighbor_lookup = compute_knn_from_adjacency(
            a_pp=a_pp,
            node_names_p=(
                np.asarray(self.node_names_p) if (gene_naming == "id") else np.asarray(self.prompt_gene_symbols)
            ),
            n_neighbors=k,
        )

        # compute the best reference gene set for each cluster and record the metric
        metrics_df = compute_function_on_gene_sets_given_neighbors(
            neighbor_lookup=neighbor_lookup,
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
        )

        return neighbor_lookup, metrics_df

    @staticmethod
    def _mean_metric_knn(
        metrics_df: pd.DataFrame,
        metric_name: str,
        all_genes_in_reference_sets: set[str],
    ) -> float:
        """
        How to take a mean over the computed neighborhood metrics. This method only averages
        over neighborhoods whose root node is in the union of all reference gene sets. The
        thought experiment is to imagine reference sets that do not "cover" the space of genes
        in the graph (imagine one gene set). In this case, we do not want one giant neighboorhood
        to be "most concordant".
        """
        return metrics_df[metric_name][metrics_df["gene"].isin(all_genes_in_reference_sets)].mean()

    @staticmethod
    def _mean_metric_clustering(
        metrics_df: pd.DataFrame,
        metric_name: str,
        cluster_label_p: np.ndarray,
        gene_names_p: list[str] | np.ndarray,
        all_genes_in_reference_sets: set[str],
    ) -> float:
        """
        How to take a mean over the computed cluster metrics. This method only averages
        over clusters which contain a gene in the union of all reference gene sets. The
        thought experiment is to imagine reference sets that do not "cover" the space of genes
        in the graph (imagine one gene set). In this case, we do not want one giant cluster
        to be "most concordant".
        """
        # figure out which cluster labels to exclude from mean calculation
        clusters_with_genes_in_reference_sets = set(
            [label for gene, label in zip(gene_names_p, cluster_label_p) if gene in all_genes_in_reference_sets]
        )
        clusters_with_no_genes_in_reference_sets = set(cluster_label_p) - clusters_with_genes_in_reference_sets
        # excluded_cluster_labels = clusters_with_no_genes_in_reference_sets.union({-1})
        excluded_cluster_labels = clusters_with_no_genes_in_reference_sets
        return metrics_df[metric_name][~metrics_df["cluster"].isin(excluded_cluster_labels)].mean()

    # TODO consider a method which computes the below for (gene, k) and then takes the k for each gene
    # that maximizes the metric for that gene. (also ignore node genes that do not appear in any reference set).
    # this could deal with the fact that having the same k everywhere is not ever going to be perfectly concordant
    # with reference gene sets of multiple sizes.

    def gridsearch_optimal_k_neighbors_given_gene_sets(
        self,
        reference_gene_sets: dict[str, set[str]],
        k_values: list[int] | np.ndarray,
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> tuple[int, dict[str, set[str]], pd.DataFrame, float]:
        """
        Compute concordance metrics between the k-nearest-neighbor communities of this graph
        (of which there are n_genes number) and reference gene sets. Use these metrics to
        determine an optimal value for k which maximizes concordance.
        Optimization is performed by checking each of the input `k_values` and finding the max.

        .. note:
            When computing a mean metric over the computed neighborhoods, instead of averaging
            over all neighborhoods, we exclude the gene neighborhoods whose gene is not included
            in the union of all the reference sets. Otherwise large k values would always be
            favored due to an increased likelihood of containing anything in the union of the
            reference sets.

        Args:
            reference_gene_sets: dictionary of reference gene sets
            k_values: list of k values to test when constructing k-nearest-neighbor graph
            metric_name: metric to use for concordance
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            (optimal k, optimal neighborhoods, dataframe with genes and reference gene set metrics, mean metric)
        """
        all_genes_in_reference_sets = set().union(*reference_gene_sets.values())  # union of all sets
        mean_metrics_dfs: list[pd.DataFrame] = []
        metrics_dfs: list[pd.DataFrame] = []

        # precompute adjacency matrix once using max k
        a_pp = compute_adjacency_matrix(z_qp=self.z_qp, **self.adjacency_kwargs)  # same kwargs as user invocation

        for k in k_values:
            # compute neighbors and metrics
            _, metrics_df = self.knn_and_compute_metrics(
                a_pp=a_pp,
                k=k,
                reference_gene_sets=reference_gene_sets,
                metric_name=metric_name,
                gene_naming=gene_naming,
            )

            # compute mean metric
            mean_metric = self._mean_metric_knn(
                metrics_df=metrics_df,
                metric_name=metric_name,
                all_genes_in_reference_sets=all_genes_in_reference_sets,
            )

            # mean of best matches over all gene neighborhoods
            mean_metrics_dfs.append(pd.DataFrame({"k": [k], "mean_of_best_match_metric": [mean_metric]}))

            # store all metrics for later inspection
            metrics_dfs.append(metrics_df)

        mean_metrics_df = pd.concat(mean_metrics_dfs, axis=0)
        metrics_df = pd.concat(metrics_dfs, axis=0)

        # choose the resolution with the highest mean metric
        best_k = mean_metrics_df.set_index("k")["mean_of_best_match_metric"].idxmax()

        # re-compute the kNN at the best resolution (cached) and metrics
        best_knn, best_metrics_df = self.knn_and_compute_metrics(
            a_pp=a_pp,
            k=best_k,
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
            gene_naming=gene_naming,
        )

        # compute best mean metric
        best_mean_metric = self._mean_metric_knn(
            metrics_df=best_metrics_df,
            metric_name=metric_name,
            all_genes_in_reference_sets=all_genes_in_reference_sets,
        )

        # return the best resolution and the Leiden communities
        return best_k, best_knn, metrics_df, best_mean_metric

    def gridsearch_optimal_resolution_communities_given_gene_sets(
        self,
        reference_gene_sets: dict[str, set[str]],
        resolutions: list[float] | np.ndarray,
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> tuple[float, np.ndarray, pd.DataFrame, float]:
        """
        Compute an "optimal" Leiden clustering by choosing the Leiden resolution by
        maximizing the mean of a concordance metric over reference gene sets.
        Optimization is performed by grid search over the input resolutions.
        The resolution which results in the best concordance with the reference gene sets
        is chosen and used to compute the Leiden communities.

        .. note:
            When computing a mean metric over the computed clusters, instead of averaging
            over all clusters, we exclude the clusters with no genes included
            in the union of all the reference sets. Otherwise low-resolution clusters would always be
            favored due to an increased number of clusters with metric=0 at high resolution
            (even if the high resolution clusters that do overlap the reference are more concordant).

        Args:
            reference_gene_sets: dictionary of reference gene sets
            resolutions: list of resolutions to consider, e.g. np.linspace(0.5, 5.0, 20)
            metric_name: metric to use for concordance
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            (optimal resolution, Leiden clusters, dataframe with clusters and reference gene set metrics, mean metric)
        """
        all_genes_in_reference_sets = set().union(*reference_gene_sets.values())  # union of all sets
        gene_names_p = np.asarray(self.node_names_p) if (gene_naming == "id") else np.asarray(self.prompt_gene_symbols)
        mean_metrics_dfs: list[pd.DataFrame] = []

        for res in resolutions:
            # compute clustering and metrics
            cluster_label_p, metrics_df = self.cluster_and_compute_metrics(
                resolution=res,
                metric_name=metric_name,
                reference_gene_sets=reference_gene_sets,
                gene_naming=gene_naming,
            )

            # compute mean metric
            mean_metric = self._mean_metric_clustering(
                metrics_df=metrics_df,
                metric_name=metric_name,
                cluster_label_p=cluster_label_p,
                gene_names_p=gene_names_p,
                all_genes_in_reference_sets=all_genes_in_reference_sets,
            )

            # mean of best matches over all clusters in clustering
            mean_metrics_dfs.append(pd.DataFrame({"resolution": [res], "mean_of_best_match_metric": [mean_metric]}))

        mean_metrics_df = pd.concat(mean_metrics_dfs, axis=0)

        # choose the resolution with the highest mean metric
        best_resolution = mean_metrics_df.set_index("resolution")["mean_of_best_match_metric"].idxmax()

        # re-compute the Leiden clustering at the best resolution (cached) and metrics
        best_clustering, best_metrics_df = self.cluster_and_compute_metrics(
            resolution=best_resolution,
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
            gene_naming=gene_naming,
        )

        # compute best mean metric
        best_mean_metric = self._mean_metric_clustering(
            metrics_df=best_metrics_df,
            metric_name=metric_name,
            cluster_label_p=best_clustering,
            gene_names_p=gene_names_p,
            all_genes_in_reference_sets=all_genes_in_reference_sets,
        )

        # return the best resolution and the Leiden communities
        return best_resolution, best_clustering, best_metrics_df, best_mean_metric

    def bayesopt_optimal_resolution_communities_given_gene_sets(
        self,
        reference_gene_sets: dict[str, set[str]],
        resolution_range: tuple[float, float] = (0.2, 10.0),
        num_clusterings_to_compute: int = 20,
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> tuple[float, np.ndarray, pd.DataFrame, float]:
        """
        Compute an "optimal" Leiden clustering by choosing the Leiden resolution by
        maximizing the mean of a concordance metric over reference gene sets.
        Optimization is performed by Bayesian optimization over the input resolution range.
        The resolution which results in the best concordance with the reference gene sets
        is chosen and used to compute the Leiden communities.

        .. note:
            When computing a mean metric over the computed clusters, instead of averaging
            over all clusters, we exclude the clusters with no genes included
            in the union of all the reference sets. Otherwise low-resolution clusters would always be
            favored due to an increased number of clusters with metric=0 at high resolution
            (even if the high resolution clusters that do overlap the reference are more concordant).

        Args:
            reference_gene_sets: dictionary of reference gene sets
            resolution_range: range of resolutions to consider, e.g. (-2.0, 2.0)
            num_clusterings_to_compute: number of clusterings to compute during optimization
            metric_name: metric to use for concordance
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            (optimal resolution, Leiden clusters, dataframe with clusters and reference gene set metrics, mean metric)
        """
        assert num_clusterings_to_compute >= 15, "num_clusterings_to_compute must be >= 15"
        gene_names_p = np.asarray(self.node_names_p) if (gene_naming == "id") else np.asarray(self.prompt_gene_symbols)
        all_genes_in_reference_sets = set().union(*reference_gene_sets.values())  # union of all sets

        # function to be minimized
        def _compute_inv_mean_metric(x: list[float]) -> float:
            resolution = x[0]

            # compute clustering and metrics
            cluster_label_p, metrics_df = self.cluster_and_compute_metrics(
                resolution=resolution,
                metric_name=metric_name,
                reference_gene_sets=reference_gene_sets,
                gene_naming=gene_naming,
            )

            # compute mean metric
            mean_metric = self._mean_metric_clustering(
                metrics_df=metrics_df,
                metric_name=metric_name,
                cluster_label_p=cluster_label_p,
                gene_names_p=gene_names_p,
                all_genes_in_reference_sets=all_genes_in_reference_sets,
            )

            return -1 * mean_metric

        # use scikit-optimize Bayesian optimization to find the best resolution
        res = gp_minimize(
            _compute_inv_mean_metric,
            [resolution_range],
            acq_func="EI",
            n_calls=num_clusterings_to_compute - 10,
            n_initial_points=10,
            initial_point_generator="random",
            random_state=1234,
        )
        best_resolution = res.x[0]

        # re-compute the Leiden clustering at the best resolution (cached) and metrics
        best_clustering, best_metrics_df = self.cluster_and_compute_metrics(
            resolution=best_resolution,
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
            gene_naming=gene_naming,
        )

        # compute best mean metric
        best_mean_metric = self._mean_metric_clustering(
            metrics_df=best_metrics_df,
            metric_name=metric_name,
            cluster_label_p=best_clustering,
            gene_names_p=gene_names_p,
            all_genes_in_reference_sets=all_genes_in_reference_sets,
        )

        # return the best resolution and the Leiden communities
        return best_resolution, best_clustering, best_metrics_df, best_mean_metric


class GeneralContext(NetworkAnalysisBase, ValidationMixin):
    pass


class JacobianContext(GeneNetworkAnalysisBase, ValidationMixin):
    def __init__(
        self,
        adata_obs: pd.DataFrame,
        gene_info_tsv_path: str,
        jacobian_point: str,
        query_var_names: list[str],
        prompt_var_names: list[str],
        jacobian_qp: np.ndarray,
        prompt_marginal_mean_p: np.ndarray,
        prompt_marginal_std_p: np.ndarray,
        query_marginal_mean_q: np.ndarray,
        query_marginal_std_q: np.ndarray,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
        min_prompt_gene_tpm: float = 0.0,
        min_query_gene_tpm: float = 0.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float | None = None,
        norm_pseudo_count: float = 1e-3,
        query_hv_top_k: int | None = None,
        query_hv_n_bins: int | None = 50,
        query_hv_min_x: float | None = 1e-2,
        query_hv_max_x: float | None = np.inf,
        z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        eps: float = 1e-8,
        verbose: bool = True,
    ):
        self.jacobian_point = jacobian_point

        super().__init__(
            adata_obs=adata_obs,
            gene_info_tsv_path=gene_info_tsv_path,
            total_mrna_umis=None,
            query_var_names=query_var_names,
            prompt_var_names=prompt_var_names,
            response_qp=jacobian_qp,
            prompt_marginal_mean_p=prompt_marginal_mean_p,
            prompt_marginal_std_p=prompt_marginal_std_p,
            query_marginal_mean_q=query_marginal_mean_q,
            query_marginal_std_q=query_marginal_std_q,
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
            norm_pseudo_count=norm_pseudo_count,
            query_hv_top_k=query_hv_top_k,
            query_hv_n_bins=query_hv_n_bins,
            query_hv_min_x=query_hv_min_x,
            query_hv_max_x=query_hv_max_x,
            z_trans_func=z_trans_func,
            eps=eps,
            verbose=verbose,
        )

    @staticmethod
    def from_old_jacobian_pt_dump(
        jacobian_pt_path: str,
        adata_path: str,
        gene_info_tsv_path: str,
        device: torch.device | str,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
        min_prompt_gene_tpm: float = 0.0,
        min_query_gene_tpm: float = 0.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float | None = None,
        norm_pseudo_count: float = 1e-3,
        query_hv_top_k: int | None = None,
        query_hv_n_bins: int | None = 50,
        query_hv_min_x: float | None = 1e-2,
        query_hv_max_x: float | None = np.inf,
        z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        eps: float = 1e-8,
        verbose: bool = True,
    ) -> "JacobianContext":
        # suppres FutureWarning in a context manager
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            adata = sc.read_h5ad(adata_path)
            old_jac_dict = torch.load(jacobian_pt_path, weights_only=False, map_location=device)

        # make a metacell
        X_meta_g = np.asarray(adata.X.sum(0))

        # set total mrna umis to the mean of the dataset
        target_total_mrna_umis = adata.obs["total_mrna_umis"].mean()
        X_meta_g = X_meta_g * target_total_mrna_umis / X_meta_g.sum()

        # make a metacell anndata
        adata_meta = adata[0, :].copy()
        adata_meta.X = X_meta_g
        adata_meta.obs["total_mrna_umis"] = [target_total_mrna_umis]

        assert (len(old_jac_dict["query_var_names"]), len(old_jac_dict["prompt_var_names"])) == old_jac_dict[
            "jacobian_qg"
        ].shape, "Jacobian shape mismatch"

        return JacobianContext(
            adata_obs=adata_meta.obs,
            gene_info_tsv_path=gene_info_tsv_path,
            jacobian_point=old_jac_dict["jacobian_point"],
            query_var_names=old_jac_dict["query_var_names"],
            prompt_var_names=old_jac_dict["prompt_var_names"],
            jacobian_qp=old_jac_dict["jacobian_qg"].cpu().numpy(),
            prompt_marginal_mean_p=old_jac_dict["prompt_marginal_dict"]["gene_marginal_means_q"].cpu().numpy(),
            prompt_marginal_std_p=old_jac_dict["prompt_marginal_dict"]["gene_marginal_std_q"].cpu().numpy(),
            query_marginal_mean_q=old_jac_dict["query_marginal_dict"]["gene_marginal_means_q"].cpu().numpy(),
            query_marginal_std_q=old_jac_dict["query_marginal_dict"]["gene_marginal_std_q"].cpu().numpy(),
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
            norm_pseudo_count=norm_pseudo_count,
            query_hv_top_k=query_hv_top_k,
            query_hv_n_bins=query_hv_n_bins,
            query_hv_min_x=query_hv_min_x,
            query_hv_max_x=query_hv_max_x,
            z_trans_func=z_trans_func,
            eps=eps,
            verbose=verbose,
        )

    def __str__(self) -> str:
        return (
            f"JacobianContext({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__


class EmpiricalCorrelationContext(GeneNetworkAnalysisBase, ValidationMixin):
    def __init__(
        self,
        gene_info_tsv_path: str,
        total_mrna_umis: float | None,
        var_names_g: list[str],
        correlation_gg: np.ndarray,
        marginal_mean_g: np.ndarray,
        marginal_std_g: np.ndarray,
        metadata: dict[str, str] = {},
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
        min_prompt_gene_tpm: float = 0.0,
        min_query_gene_tpm: float = 0.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float | None = None,
        norm_pseudo_count: float = 1e-3,
        query_hv_top_k: int | None = None,
        query_hv_n_bins: int | None = 50,
        query_hv_min_x: float | None = 1e-2,
        query_hv_max_x: float | None = np.inf,
        z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        eps: float = 1e-8,
        verbose: bool = True,
    ):
        self.metadata = metadata

        super().__init__(
            adata_obs=None,
            gene_info_tsv_path=gene_info_tsv_path,
            total_mrna_umis=total_mrna_umis,
            query_var_names=var_names_g,
            prompt_var_names=var_names_g,
            response_qp=correlation_gg,
            prompt_marginal_mean_p=marginal_mean_g,
            prompt_marginal_std_p=marginal_std_g,
            query_marginal_mean_q=marginal_mean_g,
            query_marginal_std_q=marginal_std_g,
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
            norm_pseudo_count=norm_pseudo_count,
            query_hv_top_k=query_hv_top_k,
            query_hv_n_bins=query_hv_n_bins,
            query_hv_min_x=query_hv_min_x,
            query_hv_max_x=query_hv_max_x,
            z_trans_func=z_trans_func,
            eps=eps,
            verbose=verbose,
        )

    @staticmethod
    def from_model_ckpt(
        ckpt_path: str,
        gene_info_tsv_path: str,
        total_mrna_umis: float | None,
        device: torch.device | str,
        medata: dict[str, str] = {},
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
        min_prompt_gene_tpm: float = 0.0,
        min_query_gene_tpm: float = 0.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float | None = None,
        norm_pseudo_count: float = 1e-3,
        query_hv_top_k: int | None = None,
        query_hv_n_bins: int | None = 50,
        query_hv_min_x: float | None = 1e-2,
        query_hv_max_x: float | None = np.inf,
        z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        eps: float = 1e-8,
        verbose: bool = True,
    ) -> "EmpiricalCorrelationContext":
        model = torch.load(ckpt_path, weights_only=False, map_location=device)

        return EmpiricalCorrelationContext(
            gene_info_tsv_path=gene_info_tsv_path,
            total_mrna_umis=total_mrna_umis,
            var_names_g=model.var_names_g,
            correlation_gg=model.correlation_gg.cpu().numpy(),
            marginal_mean_g=model.mean_g.cpu().numpy(),
            marginal_std_g=model.std_g.cpu().numpy(),
            metadata=medata,
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
            norm_pseudo_count=norm_pseudo_count,
            query_hv_top_k=query_hv_top_k,
            query_hv_n_bins=query_hv_n_bins,
            query_hv_min_x=query_hv_min_x,
            query_hv_max_x=query_hv_max_x,
            z_trans_func=z_trans_func,
            eps=eps,
            verbose=verbose,
        )

    def __str__(self) -> str:
        return f"EmpiricalCorrelationContext(\n{self.metadata}\n)"

    __repr__ = __str__
