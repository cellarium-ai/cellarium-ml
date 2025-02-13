# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""General gene network analysis and validation."""

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
import torch
from scipy.linalg import eigh
from scipy.stats import linregress, norm
from skopt import gp_minimize
from tqdm import tqdm

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


def load_gene_info_table(
    gene_info_tsv_path: str,
    included_gene_ids: list[str] | np.ndarray,
) -> t.Tuple[pd.DataFrame, dict, dict]:
    gene_info_df = pd.read_csv(gene_info_tsv_path, sep="\t")

    gene_symbol_to_gene_id_map = dict()
    for gene_symbol, gene_id in zip(gene_info_df["Gene Symbol"], gene_info_df["ENSEMBL Gene ID"]):
        if gene_symbol != float("nan"):
            gene_symbol_to_gene_id_map[gene_symbol] = gene_id

    gene_id_to_gene_symbol_map = {gene_id: gene_symbol for gene_symbol, gene_id in gene_symbol_to_gene_id_map.items()}
    for gene_id in included_gene_ids:
        if gene_id not in gene_id_to_gene_symbol_map:
            gene_id_to_gene_symbol_map[gene_id] = gene_id

    return gene_info_df, gene_symbol_to_gene_id_map, gene_id_to_gene_symbol_map


@dataclass
class GeneNetworkAnalysisData:
    """Container for required inputs for a gene network analysis with type annotations."""

    matrix_qp: np.ndarray

    prompt_var_names: list[str]
    prompt_marginal_mean_p: np.ndarray
    prompt_marginal_std_p: np.ndarray
    prompt_empirical_mean_p: np.ndarray

    query_var_names: list[str]
    query_marginal_mean_q: np.ndarray
    query_marginal_std_q: np.ndarray
    query_empirical_mean_q: np.ndarray

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
                    "prompt_empirical_mean_p",
                    "query_marginal_mean_q",
                    "query_marginal_std_q",
                    "query_empirical_mean_q",
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
    total_mrna_umis: float,
    response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
    feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
    min_prompt_gene_tpm: float = 10.0,
    min_query_gene_tpm: float = 10.0,
    query_response_amp_min_pct: float | None = None,
    feature_max_value: float = 10.0,
    eps: float = 1e-8,
    verbose: bool = True,
) -> GeneNetworkAnalysisData:
    """
    Process the response matrix for gene network analysis.

    Args:
        response_qp: response matrix
        prompt_var_names: prompt gene names
        prompt_marginal_mean_p: prompt marginal mean
        prompt_marginal_std_p: prompt marginal standard deviation
        prompt_empirical_mean_p: prompt empirical mean
        query_var_names: query gene names
        query_marginal_mean_q: query marginal mean
        query_marginal_std_q: query marginal standard deviation
        query_empirical_mean_q: query empirical mean
        total_mrna_umis: total mRNA UMIs
        response_normalization_strategy: response_qp normalization strategy
            - "mean": normalize by marginal mean
            - "std": normalize by marginal standard deviation
            - "none": no normalization
        feature_normalization_strategy: feature normalization strategy
            - "l2": L2 normalization per query feature
            - "query_z_score": z-score per query feature
            - "prompt_z_score": z-score per prompt feature
        min_prompt_gene_tpm: minimum prompt gene TPM
        min_query_gene_tpm: minimum query gene TPM
        query_response_amp_min_pct: minimum query response amplitude percentile
        feature_max_value: maximum feature value
        eps: epsilon value for numerical stability
        verbose: whether to print verbose output
    """
    response_qp = analysis_data.matrix_qp

    prompt_var_names = analysis_data.prompt_var_names
    prompt_marginal_mean_p = analysis_data.prompt_marginal_mean_p
    prompt_marginal_std_p = analysis_data.prompt_marginal_std_p
    prompt_empirical_mean_p = analysis_data.prompt_empirical_mean_p

    query_var_names = analysis_data.query_var_names
    query_marginal_mean_q = analysis_data.query_marginal_mean_q
    query_marginal_std_q = analysis_data.query_marginal_std_q
    query_empirical_mean_q = analysis_data.query_empirical_mean_q

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
    z_qp = response_qp * (z_p[None, :] + eps) / (z_q[:, None] + eps)
    assert isinstance(z_qp, np.ndarray)

    if verbose:
        logger.info(f"Maximum value of z_qp: {np.max(z_qp):.3f}")
        logger.info(f"Minimum value of z_qp: {np.min(z_qp):.3f}")

    mask_q = (1e6 * query_marginal_mean_q / total_mrna_umis) >= min_query_gene_tpm
    mask_p = (1e6 * prompt_marginal_mean_p / total_mrna_umis) >= min_prompt_gene_tpm

    logger.info(f"Number of query genes after TPM filtering: {np.sum(mask_q)} / {len(mask_q)}")
    logger.info(f"Number of prompt genes after TPM filtering: {np.sum(mask_p)} / {len(mask_p)}")

    if query_response_amp_min_pct is not None:
        z_norm_q = np.linalg.norm(z_qp, axis=-1)
        z_norm_thresh = np.percentile(z_norm_q, query_response_amp_min_pct)
        mask_q = mask_q & (z_norm_q >= z_norm_thresh)
        logger.info(f"Number of query genes after z-norm filtering: {np.sum(mask_q)} / {len(mask_q)}")

    # apply the mask to everything else
    prompt_var_names_masked = [prompt_var_names[i] for i in range(len(prompt_var_names)) if mask_p[i]]
    prompt_empirical_mean_p_masked = prompt_empirical_mean_p[mask_p]
    prompt_marginal_mean_p_masked = prompt_marginal_mean_p[mask_p]
    prompt_marginal_std_p_masked = prompt_marginal_std_p[mask_p]

    query_var_names_masked = [query_var_names[i] for i in range(len(query_var_names)) if mask_q[i]]
    query_empirical_mean_q_masked = query_empirical_mean_q[mask_q]
    query_marginal_mean_q_masked = query_marginal_mean_q[mask_q]
    query_marginal_std_q_masked = query_marginal_std_q[mask_q]

    # apply the mask to z_qp
    z_qp = z_qp[mask_q, :][:, mask_p]

    pearson = True  # True means that z_qp @ z_qp.T will be the pearson correlation matrix
    if feature_normalization_strategy == "prompt_z_score":
        z_qp = (z_qp - np.mean(z_qp, axis=0, keepdims=True)) / (eps + np.std(z_qp, axis=0, keepdims=True))
        if pearson:
            z_qp = z_qp / np.sqrt(z_qp.shape[0])
    elif feature_normalization_strategy == "query_z_score":
        z_qp = (z_qp - np.mean(z_qp, axis=1, keepdims=True)) / (eps + np.std(z_qp, axis=1, keepdims=True))
        if pearson:
            z_qp = z_qp / np.sqrt(z_qp.shape[1])
    elif feature_normalization_strategy == "l2":
        z_qp = z_qp / (eps + np.linalg.norm(z_qp, axis=-1, keepdims=True))
    else:
        raise ValueError("Invalid feature normalization strategy")

    # clip features
    assert isinstance(z_qp, np.ndarray)
    z_qp[np.isnan(z_qp)] = 0.0
    z_qp[np.isinf(z_qp)] = 0.0
    z_qp = np.clip(z_qp, -feature_max_value, feature_max_value)

    return GeneNetworkAnalysisData(
        matrix_qp=z_qp,
        prompt_var_names=prompt_var_names_masked,
        prompt_marginal_mean_p=prompt_marginal_mean_p_masked,
        prompt_marginal_std_p=prompt_marginal_std_p_masked,
        prompt_empirical_mean_p=prompt_empirical_mean_p_masked,
        query_var_names=query_var_names_masked,
        query_marginal_mean_q=query_marginal_mean_q_masked,
        query_marginal_std_q=query_marginal_std_q_masked,
        query_empirical_mean_q=query_empirical_mean_q_masked,
    )


def compute_adjacency_matrix(
    z_qp: np.ndarray,
    adjacency_strategy: t.Literal[
        "shifted_correlation", "unsigned_correlation", "positive_correlation", "positive_correlation_binary"
    ] = "positive_correlation",
    n_neighbors: int | None = 50,
    self_loop: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute an adjacency matrix from a query-by-prompt gene matrix of values.

    Args:
        z_qp: query-by-prompt gene matrix of values
        adjacency_strategy: adjacency strategy
        n_neighbors: number of neighbors to keep
        self_loop: whether to include self-loops
        **kwargs: additional keyword arguments
            beta: power for correlation
            tau: threshold for binary adjacency

    Returns:
        query-by-query adjacency matrix for genes
    """
    # mat_prod_qq = np.dot(z_qp, z_qp.T)  # older code
    mat_prod_qq = z_qp @ z_qp.T
    if adjacency_strategy == "shifted_correlation":
        assert "beta" in kwargs, "Must provide beta for shifted correlation"
        beta = kwargs["beta"]
        a_qq = np.power(0.5 * (1 + mat_prod_qq), beta)
    elif adjacency_strategy == "unsigned_correlation":
        assert "beta" in kwargs, "Must provide beta for unsigned correlation"
        beta = kwargs["beta"]
        a_qq = np.power(np.abs(mat_prod_qq), beta)
    elif adjacency_strategy == "positive_correlation":
        assert "beta" in kwargs, "Must provide beta for positive correlation"
        beta = kwargs["beta"]
        a_qq = np.power(np.maximum(0, mat_prod_qq), beta)
    elif adjacency_strategy == "positive_correlation_binary":
        assert n_neighbors is None, "n_neighbors must be None for binary adjacency"
        assert "tau" in kwargs, "Must provide correlation threshold for binary adjacency"
        tau = kwargs["tau"]
        a_qq = (np.maximum(0, mat_prod_qq) > tau).astype(float)
    else:
        raise ValueError("Invalid adjacency strategy")

    assert np.isclose(a_qq, a_qq.T).all(), "Adjacency matrix must be symmetric -- something is wrong!"

    if n_neighbors is not None:
        assert n_neighbors > 0, "n_neighbors must be positive"
        t_qn = np.argsort(a_qq, axis=-1)[:, -n_neighbors:]  # take the top n_neighbors

        # make a mask for the top n_neighbors
        _a_qq = np.zeros_like(a_qq)
        for q in range(a_qq.shape[0]):
            _a_qq[q, t_qn[q]] = a_qq[q, t_qn[q]]
            _a_qq[t_qn[q], q] = a_qq[t_qn[q], q]
    else:
        _a_qq = a_qq
    a_qq = _a_qq

    if not self_loop:
        np.fill_diagonal(a_qq, 0)

    return a_qq


def compute_igraph_from_adjacency(
    a_qq: np.ndarray, node_names: list[str] | np.ndarray, directed: bool = False
) -> ig.Graph:
    """
    Convert an adjacency matrix to an :class:`ig.Graph` graph.

    Args:
        a_qq: (gene-gene) adjacency matrix
        node_names: names of the nodes (gene names)
        directed: whether the graph is directed

    Returns:
        :class:`ig.Graph` graph
    """
    assert len(node_names) == a_qq.shape[0], (
        f"Number of node names ({len(node_names)}) must match adjacency matrix shape ({a_qq.shape[0]})"
    )
    assert a_qq.shape[0] == a_qq.shape[1], "Adjacency matrix must be square"
    sources, targets = a_qq.nonzero()
    weights = a_qq[sources, targets]
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
    a_qq: np.ndarray,
    node_names_q: np.ndarray,
    n_neighbors: int,
) -> dict[str, set[str]]:
    """
    Compute the k-nearest neighbors for each node using a pre-computed adjacency matrix.

    Args:
        a_qq: adjacency matrix
        node_names_q: names of the nodes
        n_neighbors: number of neighbors to keep

    Returns:
        dictionary of k-nearest neighbors for each node
    """
    assert a_qq.shape[0] == len(node_names_q), "Number of node names must match the first dimension of a_qq"

    # compute the k-nearest neighbors
    knn: dict[str, set[str]] = {}
    for q in range(a_qq.shape[0]):
        t_q = np.argsort(a_qq[q], axis=-1)[-n_neighbors:].squeeze()  # take the top n_neighbors
        knn[node_names_q[q]] = set(node_names_q[t_q])
    return knn


def compute_spectral_dimension(
    a_qq: np.ndarray, offset: int = 2, n_lambda_for_estimation: int = 5
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
    norm_q = 1.0 / (1e-9 + np.sqrt(a_qq.sum(0)))
    lap_qq = np.eye(a_qq.shape[0]) - norm_q[:, None] * norm_q[None, :] * a_qq
    eigs = eigh(lap_qq.astype(np.float64), eigvals_only=True)
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
    device: torch.device = torch.device("cpu"),
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
        node_names_q: input names of the rows
        leiden_membership: Leiden clustering results
        spectral: dictionary of spectral dimension results

    Properties:
        adjacency_matrix: computed adjacency matrix
        leiden_communities: computed Leiden communities
        spectral_dim: computed spectral dimension and related results
    """

    def __init__(self, z_qp: np.ndarray, node_names_q: list[str] | np.ndarray):
        assert z_qp.shape[0] == len(node_names_q), "Number of node names must match the first dimension of z_qp"
        assert np.isnan(z_qp).sum() == 0, "There are NaNs in z_qp, which is not allowed"
        assert np.isinf(z_qp).sum() == 0, "There are infs in z_qp, which is not allowed"
        assert (z_qp != 0).sum() > 0, "There are no nonzero values in z_qp, which is not allowed"
        self.z_qp = z_qp
        self.node_names_q = node_names_q
        self.a_qq: np.ndarray | None = None  # adjacency matrix
        self.adjacency_kwargs: dict[str, t.Any] = {}  # used by validation methods to re-compute adjacency matrix
        self.leiden_membership: np.ndarray | None = None
        self.spectral: dict[str, np.ndarray | float] = {}
        self.embedding_q2: np.ndarray | None = None

        # prevent false cache hits after reprocessing
        self.igraph.cache_clear()
        self.compute_leiden_communites.cache_clear()

    @property
    def adjacency_matrix(self) -> np.ndarray:
        if self.a_qq is None:
            raise UserWarning("Compute adjacency matrix by calling obj.compute_adjacency_matrix()")
        return self.a_qq

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
        **kwargs,
    ) -> np.ndarray:
        self.adjacency_kwargs = kwargs | {
            "adjacency_strategy": adjacency_strategy,
            "n_neighbors": n_neighbors,
            "self_loop": self_loop,
        }
        a_qq = compute_adjacency_matrix(
            z_qp=self.z_qp,
            adjacency_strategy=adjacency_strategy,
            n_neighbors=n_neighbors,
            self_loop=self_loop,
            **kwargs,
        )
        self.a_qq = a_qq

        # clear previously computed properties
        self.igraph.cache_clear()
        self.compute_leiden_communites.cache_clear()

        return a_qq

    @lru_cache(maxsize=2)
    def igraph(self, directed: bool = False) -> ig.Graph:
        if self.a_qq is None:
            raise UserWarning("Compute adjacency matrix first by calling obj.compute_adjacency_matrix()")
        return compute_igraph_from_adjacency(
            a_qq=self.a_qq,
            node_names=self.node_names_q,
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
            a_qq=self.adjacency_matrix,
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
        self.embedding_q2 = mde_embedding(
            self.z_qp,
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
        adata_obs: pd.DataFrame,
        gene_info_tsv_path: str,
        query_var_names: list[str],
        prompt_var_names: list[str],
        response_qp: np.ndarray,
        prompt_marginal_mean_p: np.ndarray,
        prompt_marginal_std_p: np.ndarray,
        prompt_empirical_mean_p: np.ndarray,
        query_marginal_mean_q: np.ndarray,
        query_marginal_std_q: np.ndarray,
        query_empirical_mean_q: np.ndarray,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
        min_prompt_gene_tpm: float = 10.0,
        min_query_gene_tpm: float = 10.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float = 10.0,
        eps: float = 1e-8,
        verbose: bool = True,
    ):
        self.verbose = verbose

        n_query_vars = len(query_var_names)
        n_prompt_vars = len(prompt_var_names)

        assert response_qp.shape == (n_query_vars, n_prompt_vars)
        assert prompt_marginal_mean_p.shape == (n_prompt_vars,)
        assert prompt_marginal_std_p.shape == (n_prompt_vars,)
        assert prompt_empirical_mean_p.shape == (n_prompt_vars,)
        assert query_marginal_mean_q.shape == (n_query_vars,)
        assert query_marginal_std_q.shape == (n_query_vars,)
        assert query_empirical_mean_q.shape == (n_query_vars,)

        self.adata_obs = adata_obs

        self.unprocessed = GeneNetworkAnalysisData(
            matrix_qp=response_qp,
            prompt_var_names=prompt_var_names,
            prompt_marginal_mean_p=prompt_marginal_mean_p,
            prompt_marginal_std_p=prompt_marginal_std_p,
            prompt_empirical_mean_p=prompt_empirical_mean_p,
            query_var_names=query_var_names,
            query_marginal_mean_q=query_marginal_mean_q,
            query_marginal_std_q=query_marginal_std_q,
            query_empirical_mean_q=query_empirical_mean_q,
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
            eps=eps,
        )

        super().__init__(z_qp=self.processed.matrix_qp, node_names_q=self.processed.query_var_names)

    @property
    def cell_type(self) -> str:
        return self.adata_obs["cell_type"].values[0]

    @property
    def tissue(self) -> str:
        return self.adata_obs["tissue"].values[0]

    @property
    def disease(self) -> str:
        return self.adata_obs["disease"].values[0]

    @property
    def development_stage(self) -> str:
        return self.adata_obs["development_stage"].values[0]

    @property
    def sex(self) -> str:
        return self.adata_obs["sex"].values[0]

    @property
    def total_mrna_umis(self) -> float:
        return self.adata_obs["total_mrna_umis"].values[0]

    @property
    def query_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.processed.query_var_names]

    @property
    def query_gene_ids(self) -> list[str]:
        return self.processed.query_var_names

    @property
    def prompt_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.processed.prompt_var_names]

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

    def __str__(self) -> str:
        return (
            f"GeneNetworkAnalysisBase({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__

    def reprocess(
        self,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
        min_prompt_gene_tpm: float = 10.0,
        min_query_gene_tpm: float = 10.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float = 10.0,
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
            eps=eps,
            verbose=self.verbose,
        )

        # re-initialize the object and zero out pre-computed and cached properties
        if hasattr(self, "query_gene_id_to_idx_map"):
            del self.query_gene_id_to_idx_map  # clear cached property
        super().__init__(z_qp=self.processed.matrix_qp, node_names_q=self.processed.query_var_names)

    def plot_mde_embedding(
        self,
        marker_size: int = 2,
        highlight_marker_size: int = 4,
        width: int = 800,
        height: int = 800,
        highlight_gene_sets: dict[str, t.Tuple[list[str], list[str], str]] | None = None,
    ) -> go.Figure:
        assert self.embedding_q2 is not None, "Must compute MDE embedding first"
        assert self.leiden_membership is not None, "Must compute Leiden communities first"

        plot_title = f"""{self.cell_type}<br>{self.tissue}<br>{self.disease}"""

        # Create a color map for the memberships
        memberships_q = self.leiden_membership
        unique_memberships = np.unique(memberships_q)

        # Convert memberships to strings for categorical mapping
        unique_memberships.astype(str)

        # Create the color map with string keys
        colormap = {str(label): cc.glasbey[i % len(cc.glasbey)] for i, label in enumerate(unique_memberships)}

        # Create a DataFrame for Plotly
        df = pd.DataFrame(
            {
                "x": self.embedding_q2[:, 0],
                "y": self.embedding_q2[:, 1],
                "label": self.query_gene_symbols,
                "membership": memberships_q.astype(str),  # Convert to string
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
                query_gene_indices = [self.query_gene_id_to_idx_map[gene_id] for gene_id in gene_ids]

                # show a scatter plot and color the markers in red
                fig.add_scatter(
                    x=self.embedding_q2[query_gene_indices, 0],
                    y=self.embedding_q2[query_gene_indices, 1],
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
        ax.set_title(self.cell_type)
        ax.legend()


class BaseClassProtocol(t.Protocol):
    """Tell the mixin what it can count on inheriting from the base class (for typechecking)"""

    z_qp: np.ndarray
    node_names_q: list[str] | np.ndarray
    adjacency_kwargs: dict[str, t.Any]

    @property
    def adjacency_matrix(self) -> np.ndarray: ...

    @property
    def query_gene_symbol_to_idx_map(self) -> dict[str, int]: ...

    @property
    def query_gene_id_to_idx_map(self) -> dict[str, int]: ...

    @property
    def query_gene_symbols(self) -> list[str]: ...

    compute_leiden_communites: t.Callable[..., np.ndarray]


class ValidationMixin(BaseClassProtocol):
    """Mixin with methods for bio-inspired validation of gene networks."""

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
        Compute a metric for the overall concordance between the concordance matrix in
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
        # precompute sums of adjacency matrix rows without diagonal elements
        a_qq = self.adjacency_matrix
        assert isinstance(a_qq, np.ndarray)
        q = a_qq.shape[0]
        zero_diag_a_qq = a_qq * (1.0 - np.eye(q))
        mean_element_value = zero_diag_a_qq.sum() / (q * q - q)

        # handle gene naming
        if gene_naming == "symbol":
            gene_ind_lookup: dict[str, int] = self.query_gene_symbol_to_idx_map
        elif gene_naming == "id":
            gene_ind_lookup = self.query_gene_id_to_idx_map
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

            # p-value via analytical approximation using law of large numbers
            scale = 1 / len(gene_set_inds) / 22
            pval_analytic = 1.0 - norm.cdf(effect, loc=mean_element_value, scale=scale)

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
            df["pval"][df["pval"] < p_value_threshold]
            .apply(lambda x: -1 * np.log10(np.clip(x, a_min=1.0 / n_samples, a_max=None)))
            .sum()
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
        gene_names = np.asarray(self.node_names_q) if (gene_naming == "id") else np.asarray(self.query_gene_symbols)

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
        a_qq: np.ndarray,
        k: int,
        reference_gene_sets: dict[str, set[str]],
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"],
        gene_naming: t.Literal["id", "symbol"] = "symbol",
    ) -> tuple[dict[str, set[str]], pd.DataFrame]:
        """
        Compute k-nearest-neighbors on the network and compute the mean of a concordance metric
        over reference gene sets.

        Args:
            k: number of neighbors for kNN
            reference_gene_sets: dictionary of reference gene sets
            metric_name: metric to use for concordance
            gene_naming: whether to use gene IDs or gene symbols

        Returns:
            (neighbor dictionary, DataFrame of best reference gene set metrics for all genes)
        """

        # compute the neighbors (cheap given adjacency)
        neighbor_lookup = compute_knn_from_adjacency(
            a_qq=a_qq,
            node_names_q=(
                np.asarray(self.node_names_q) if (gene_naming == "id") else np.asarray(self.query_gene_symbols)
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
        cluster_label_q: np.ndarray,
        gene_names_q: list[str] | np.ndarray,
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
            [label for gene, label in zip(gene_names_q, cluster_label_q) if gene in all_genes_in_reference_sets]
        )
        clusters_with_no_genes_in_reference_sets = set(cluster_label_q) - clusters_with_genes_in_reference_sets
        excluded_cluster_labels = clusters_with_no_genes_in_reference_sets.union({-1})
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
        a_qq = compute_adjacency_matrix(z_qp=self.z_qp, **self.adjacency_kwargs)  # same kwargs as user invocation

        for k in k_values:
            # compute neighbors and metrics
            _, metrics_df = self.knn_and_compute_metrics(
                a_qq=a_qq,
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
            a_qq=a_qq,
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
        gene_names_q = np.asarray(self.node_names_q) if (gene_naming == "id") else np.asarray(self.query_gene_symbols)
        mean_metrics_dfs: list[pd.DataFrame] = []

        for res in resolutions:
            # compute clustering and metrics
            cluster_label_q, metrics_df = self.cluster_and_compute_metrics(
                resolution=res,
                metric_name=metric_name,
                reference_gene_sets=reference_gene_sets,
                gene_naming=gene_naming,
            )

            # compute mean metric
            mean_metric = self._mean_metric_clustering(
                metrics_df=metrics_df,
                metric_name=metric_name,
                cluster_label_q=cluster_label_q,
                gene_names_q=gene_names_q,
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
            cluster_label_q=best_clustering,
            gene_names_q=gene_names_q,
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
        gene_names_q = np.asarray(self.node_names_q) if (gene_naming == "id") else np.asarray(self.query_gene_symbols)
        all_genes_in_reference_sets = set().union(*reference_gene_sets.values())  # union of all sets

        # function to be minimized
        def _compute_inv_mean_metric(x: list[float]) -> float:
            resolution = x[0]

            # compute clustering and metrics
            cluster_label_q, metrics_df = self.cluster_and_compute_metrics(
                resolution=resolution,
                metric_name=metric_name,
                reference_gene_sets=reference_gene_sets,
                gene_naming=gene_naming,
            )

            # compute mean metric
            mean_metric = self._mean_metric_clustering(
                metrics_df=metrics_df,
                metric_name=metric_name,
                cluster_label_q=cluster_label_q,
                gene_names_q=gene_names_q,
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
            cluster_label_q=best_clustering,
            gene_names_q=gene_names_q,
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
        prompt_empirical_mean_p: np.ndarray,
        query_empirical_mean_q: np.ndarray,
        prompt_marginal_mean_p: np.ndarray,
        prompt_marginal_std_p: np.ndarray,
        query_marginal_mean_q: np.ndarray,
        query_marginal_std_q: np.ndarray,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
        min_prompt_gene_tpm: float = 10.0,
        min_query_gene_tpm: float = 10.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float = 10.0,
        eps: float = 1e-8,
        verbose: bool = True,
    ):
        self.jacobian_point = jacobian_point

        super().__init__(
            adata_obs=adata_obs,
            gene_info_tsv_path=gene_info_tsv_path,
            query_var_names=query_var_names,
            prompt_var_names=prompt_var_names,
            response_qp=jacobian_qp,
            prompt_marginal_mean_p=prompt_marginal_mean_p,
            prompt_marginal_std_p=prompt_marginal_std_p,
            prompt_empirical_mean_p=prompt_empirical_mean_p,
            query_marginal_mean_q=query_marginal_mean_q,
            query_marginal_std_q=query_marginal_std_q,
            query_empirical_mean_q=query_empirical_mean_q,
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
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
        feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
        min_prompt_gene_tpm: float = 10.0,
        min_query_gene_tpm: float = 10.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float = 10.0,
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

        prompt_empirical_mean_p = adata_meta[0, old_jac_dict["prompt_var_names"]].X.flatten()
        query_empirical_mean_q = adata_meta[0, old_jac_dict["query_var_names"]].X.flatten()

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
            prompt_empirical_mean_p=prompt_empirical_mean_p,
            query_empirical_mean_q=query_empirical_mean_q,
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
            eps=eps,
            verbose=verbose,
        )

    def __str__(self) -> str:
        return (
            f"JacobianContext({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__
