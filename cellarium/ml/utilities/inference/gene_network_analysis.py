# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""General gene network analysis and validation."""

import logging
import typing as t
import warnings
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
from scipy.stats import linregress
from skopt import gp_minimize

from cellarium.ml.utilities.inference.gene_set_utils import (
    compute_function_on_gene_sets_given_clustering,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a handler
handler = logging.StreamHandler()

# Create and set a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


def compute_adjacency_matrix(
    z_qp: np.ndarray,
    adjacency_strategy: t.Literal[
        "shifted_correlation", "unsigned_correlation", "positive_correlation", "positive_correlation_binary"
    ],
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
    if adjacency_strategy == "shifted_correlation":
        assert "beta" in kwargs, "Must provide beta for shifted correlation"
        beta = kwargs["beta"]
        a_qq = np.power(0.5 * (1 + np.dot(z_qp, z_qp.T)), beta)
    elif adjacency_strategy == "unsigned_correlation":
        assert "beta" in kwargs, "Must provide beta for unsigned correlation"
        beta = kwargs["beta"]
        a_qq = np.power(np.abs(np.dot(z_qp, z_qp.T)), beta)
    elif adjacency_strategy == "positive_correlation":
        assert "beta" in kwargs, "Must provide beta for positive correlation"
        beta = kwargs["beta"]
        a_qq = np.power(np.maximum(0, np.dot(z_qp, z_qp.T)), beta)
    elif adjacency_strategy == "positive_correlation_binary":
        assert n_neighbors is None, "n_neighbors must be None for binary adjacency"
        assert "tau" in kwargs, "Must provide correlation threshold for binary adjacency"
        tau = kwargs["tau"]
        a_qq = (np.maximum(0, np.dot(z_qp, z_qp.T)) > tau).astype(float)
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
    Convert an adjacency matrix to an igraph graph.

    Args:
        a_qq: (gene-gene) adjacency matrix
        node_names: names of the nodes (gene names)
        directed: whether the graph is directed

    Returns:
        igraph graph
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
) -> np.ndarray:
    """
    Compute Leiden communities from an igraph graph by running leidenalg with RBConfigurationVertexPartition.

    Args:
        g: igraph graph
        resolution: resolution parameter
        min_community_size: minimum community size

    Returns:
        array of community memberships
    """

    leiden_partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution
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


def mde_embedding(
    z_qp: np.ndarray,
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
    Run pymde to compute a 2-dimensional MDE embedding.

    Args:
        passthroughs to pymde

    Returns:
        2-dimensional MDE embedding
    """
    mde = pymde.preserve_neighbors(
        z_qp,
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
        print("WARNING: ignoring n_components in spectral_embedding_kwargs and replacing with embedding_dimension")
    spectral_embedding_kwargs["n_components"] = embedding_dimension

    if "embedding_dim" in mde_kwargs:
        print("WARNING: ignoring embedding_dim in mde_kwargs and replacing with embedding_dimension")
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
    def __init__(self, z_qp: np.ndarray, node_names_q: list[str] | np.ndarray):
        assert z_qp.shape[0] == len(node_names_q), "Number of node names must match the first dimension of z_qp"
        self.z_qp = z_qp
        self.node_names_q = node_names_q

        # network
        self.a_qq: np.ndarray | None = None  # adjacency matrix
        self.leiden_membership: np.ndarray | None = None

        # spectral analysis
        self.eigs: np.ndarray | None = None
        self.spectral_dim: np.ndarray | None = None

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

    def compute_adjacency_matrix(
        self,
        adjacency_strategy: t.Literal[
            "shifted_correlation", "unsigned_correlation", "positive_correlation", "positive_correlation_binary"
        ],
        n_neighbors: int | None = 50,
        self_loop: bool = False,
        **kwargs,
    ) -> np.ndarray:
        a_qq = compute_adjacency_matrix(
            z_qp=self.z_qp,
            adjacency_strategy=adjacency_strategy,
            n_neighbors=n_neighbors,
            self_loop=self_loop,
            **kwargs,
        )
        self.a_qq = a_qq
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
        resolution: float = 3.0,
        min_community_size: int = 2,
    ) -> np.ndarray:
        self.leiden_membership = compute_leiden_communites(
            g=self.igraph(),
            resolution=resolution,
            min_community_size=min_community_size,
        )
        return self.leiden_membership

    def compute_spectral_dimension(self, offset: int = 2, n_lambda_for_estimation: int = 5) -> float:
        if self.a_qq is None:
            raise UserWarning("Compute adjacency matrix first by calling obj.compute_adjacency_matrix()")

        # calculate normalized laplacian and its eigenvalues
        norm_q = 1.0 / (1e-9 + np.sqrt(self.a_qq.sum(0)))
        lap_qq = np.eye(self.a_qq.shape[0]) - norm_q[:, None] * norm_q[None, :] * self.a_qq
        eigs = eigh(lap_qq.astype(np.float64), eigvals_only=True)
        eigs[0] = 0
        eigs = np.clip(eigs, 0, np.inf)  # roundoff error guardrail
        self.eigs = eigs

        n_lambda = np.cumsum(eigs)
        n_lambda = n_lambda / n_lambda[-1]
        first_nonzero = np.where(eigs > 0)[0][0] + offset
        xx = np.log(eigs[first_nonzero : first_nonzero + n_lambda_for_estimation])
        yy = np.log(n_lambda[first_nonzero : first_nonzero + n_lambda_for_estimation])

        lin = linregress(xx, yy)
        slope, intercept = lin.slope, lin.intercept

        # save a few thigs for later
        spectral_dim = 2 * linregress(xx, yy).slope
        self.spectral_dim = spectral_dim
        self.eigs = eigs
        self.n_lambda = n_lambda
        self.log_eigs_asymptotic = xx
        self.log_n_lambda_asymptotic = yy
        self.spectral_dim_slope = slope
        self.spectral_dim_intercept = intercept

        return float(spectral_dim)


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
        self.verbose = verbose

        n_query_vars = len(query_var_names)
        n_prompt_vars = len(prompt_var_names)

        assert response_qp.shape == (n_query_vars, n_prompt_vars)
        assert prompt_marginal_mean_p.shape == (n_prompt_vars,)
        assert prompt_marginal_std_p.shape == (n_prompt_vars,)
        assert query_marginal_mean_q.shape == (n_query_vars,)
        assert query_marginal_std_q.shape == (n_query_vars,)

        self.adata_obs = adata_obs
        self.query_var_names = query_var_names
        self.prompt_var_names = prompt_var_names
        self.response_qp = response_qp
        self.prompt_marginal_mean_p = prompt_marginal_mean_p
        self.prompt_marginal_std_p = prompt_marginal_std_p
        self.query_marginal_mean_q = query_marginal_mean_q
        self.query_marginal_std_q = query_marginal_std_q

        self.gene_info_df, self.gene_symbol_to_gene_id_map, self.gene_id_to_gene_symbol_map = load_gene_info_table(
            gene_info_tsv_path, query_var_names + prompt_var_names
        )

        self._process(
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy=feature_normalization_strategy,
            min_prompt_gene_tpm=min_prompt_gene_tpm,
            min_query_gene_tpm=min_query_gene_tpm,
            query_response_amp_min_pct=query_response_amp_min_pct,
            feature_max_value=feature_max_value,
            eps=eps,
        )

        super().__init__(z_qp=self.z_qp, node_names_q=query_var_names)

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
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.query_var_names]

    @property
    def prompt_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.prompt_var_names]

    @cached_property
    def query_gene_id_to_idx_map(self) -> dict[str, int]:
        return {gene_id: idx for idx, gene_id in enumerate(self.query_var_names)}

    def __str__(self) -> str:
        return (
            f"GeneNetworkAnalysisBase({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__

    def _process(
        self,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
        min_prompt_gene_tpm: float = 10.0,
        min_query_gene_tpm: float = 10.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        if response_normalization_strategy == "mean":
            z_p = self.prompt_marginal_mean_p
            z_q = self.query_marginal_mean_q
        elif response_normalization_strategy == "std":
            z_p = self.prompt_marginal_std_p
            z_q = self.query_marginal_std_q
        elif response_normalization_strategy == "none":
            z_p = np.ones_like(self.prompt_marginal_mean_p)
            z_q = np.ones_like(self.query_marginal_mean_q)
        else:
            raise ValueError("Invalid Jacobian normalization strategy")

        # linear proportional activation
        self.z_qp = self.response_qp * (z_p[None, :] + eps) / (z_q[:, None] + eps)
        assert isinstance(self.z_qp, np.ndarray)

        if self.verbose:
            logger.info(f"Maximum value of z_qp: {np.max(self.z_qp):.3f}")
            logger.info(f"Minimum value of z_qp: {np.min(self.z_qp):.3f}")

        self.mask_q = (1e6 * self.query_marginal_mean_q / self.total_mrna_umis) >= min_query_gene_tpm
        self.mask_p = (1e6 * self.prompt_marginal_mean_p / self.total_mrna_umis) >= min_prompt_gene_tpm

        logger.info(f"Number of query genes after TPM filtering: {np.sum(self.mask_q)} / {len(self.mask_q)}")
        logger.info(f"Number of prompt genes after TPM filtering: {np.sum(self.mask_p)} / {len(self.mask_p)}")

        if query_response_amp_min_pct is not None:
            z_norm_q = np.linalg.norm(self.z_qp, axis=-1)
            z_norm_thresh = np.percentile(z_norm_q, query_response_amp_min_pct)
            self.mask_q = self.mask_q & (z_norm_q >= z_norm_thresh)
            logger.info(f"Number of query genes after z-norm filtering: {np.sum(self.mask_q)} / {len(self.mask_q)}")

        # apply the mask to everything else
        self.prompt_var_names = [self.prompt_var_names[i] for i in range(len(self.prompt_var_names)) if self.mask_p[i]]
        # self.prompt_empirical_mean_p = self.prompt_empirical_mean_p[self.mask_p]
        self.prompt_marginal_mean_p = self.prompt_marginal_mean_p[self.mask_p]
        self.prompt_marginal_std_p = self.prompt_marginal_std_p[self.mask_p]

        self.query_var_names = [self.query_var_names[i] for i in range(len(self.query_var_names)) if self.mask_q[i]]
        # self.query_empirical_mean_q = self.query_empirical_mean_q[self.mask_q]
        self.query_marginal_mean_q = self.query_marginal_mean_q[self.mask_q]
        self.query_marginal_std_q = self.query_marginal_std_q[self.mask_q]

        # apply the mask to z_qp
        self.z_qp = self.z_qp[self.mask_q, :][:, self.mask_p]

        if feature_normalization_strategy == "prompt_z_score":
            self.z_qp = (self.z_qp - np.mean(self.z_qp, axis=0, keepdims=True)) / (
                np.sqrt(self.z_qp.shape[0]) * (eps + np.std(self.z_qp, axis=0, keepdims=True))
            )
        elif feature_normalization_strategy == "query_z_score":
            self.z_qp = (self.z_qp - np.mean(self.z_qp, axis=1, keepdims=True)) / (
                np.sqrt(self.z_qp.shape[1]) * (eps + np.std(self.z_qp, axis=1, keepdims=True))
            )
        elif feature_normalization_strategy == "l2":
            self.z_qp = self.z_qp / (eps + np.linalg.norm(self.z_qp, axis=-1, keepdims=True))
        else:
            raise ValueError("Invalid feature normalization strategy")

        # clip features
        assert isinstance(self.z_qp, np.ndarray)
        self.z_qp[np.isnan(self.z_qp)] = 0.0
        self.z_qp[np.isinf(self.z_qp)] = 0.0
        self.z_qp = np.clip(self.z_qp, -feature_max_value, feature_max_value)

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
        assert isinstance(self.z_qp, np.ndarray), "Must call obj.process() before computing MDE embedding"
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
        assert self.eigs is not None, "Must compute spectral dimension first"
        ax.scatter(self.log_eigs_asymptotic, self.log_n_lambda_asymptotic)
        ax.plot(
            self.log_eigs_asymptotic,
            self.spectral_dim_slope * self.log_eigs_asymptotic + self.spectral_dim_intercept,
            color="red",
            label=f"$d_S$ = {self.spectral_dim:.2f}",
        )
        ax.set_xlabel(r"ln $\lambda$")
        ax.set_ylabel(r"ln N($\lambda$)")
        ax.set_title(self.cell_type)
        ax.legend()


class ValidationMixin:
    """Mixin with methods for bio-inspired validation of gene networks."""

    # make this mixin aware that these are defined in the base class (GeneNetworkAnalysisBase)
    node_names_q: list[str] | np.ndarray
    compute_leiden_communites: t.Callable[..., np.ndarray]

    def compute_network_concordance_metric(
        self,
        reference_gene_sets: dict[str, set[str]],
        resolution_range: tuple[float, float] = (0.2, 8.0),
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
        optimization_strategy: t.Literal["gridsearch", "bayesopt"] = "bayesopt",
    ) -> float:
        """
        Compute a metric for the overall concordance between the network in
        GeneNetworkAnalysisBase.igraph() and reference gene sets defined by a dictionary.
        Distills the bayesopt_optimal_resolution_communities_given_gene_sets() output
        into a single concordance metric.

        Args:
            reference_gene_sets: dictionary of reference gene sets {set1_name: {gene_A, gene_B, ...}, ...}
            resolution_range: range of Leiden clustering resolutions to consider, e.g. (0.2, 8.0)
            metric_name: metric to use for concordance
            optimization_strategy: optimization strategy, either "gridsearch" or "bayesopt"

        Returns:
            concordance value
        """
        if optimization_strategy == "gridsearch":
            _, _, best_metrics_df = self.gridsearch_optimal_resolution_communities_given_gene_sets(
                reference_gene_sets=reference_gene_sets,
                resolutions=np.linspace(*resolution_range, 20),
                metric_name=metric_name,
            )
        elif optimization_strategy == "bayesopt":
            _, _, best_metrics_df = self.bayesopt_optimal_resolution_communities_given_gene_sets(
                reference_gene_sets=reference_gene_sets,
                resolution_range=resolution_range,
                metric_name=metric_name,
            )
        else:
            raise ValueError("Invalid optimization strategy")

        print(f"best_metrics_df:\n{best_metrics_df}")

        # distill a single number
        return best_metrics_df[metric_name].mean()

    def cluster_and_compute_metrics(
        self,
        resolution: float,
        reference_gene_sets: dict[str, set[str]],
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"],
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Run Leiden clustering on the network and compute the mean of a concordance metric
        over reference gene sets.

        Args:
            resolution: Leiden resolution
            reference_gene_sets: dictionary of reference gene sets
            metric_name: metric to use for concordance

        Returns:
            (Leiden clustering, DataFrame of best reference gene set metrics for all clusters)
        """

        # compute the Leiden clustering
        clustering = self.compute_leiden_communites(resolution=resolution)

        # compute the best reference gene set for each cluster and record the metric
        metrics_df = compute_function_on_gene_sets_given_clustering(
            clustering=clustering,
            gene_names=np.asarray(self.node_names_q),
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
        )

        return clustering, metrics_df

    def gridsearch_optimal_resolution_communities_given_gene_sets(
        self,
        reference_gene_sets: dict[str, set[str]],
        resolutions: list[float] | np.ndarray,
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
    ) -> tuple[float, np.ndarray, pd.DataFrame]:
        """
        Compute an "optimal" Leiden clustering by choosing the Leiden resolution by
        maximizing the mean of a concordance metric over reference gene sets.
        Optimization is performed by grid search over the input resolutions.
        The resolution which results in the best concordance with the reference gene sets
        is chosen and used to compute the Leiden communities.

        Args:
            reference_gene_sets: dictionary of reference gene sets
            resolutions: list of resolutions to consider, e.g. np.linspace(0.5, 5.0, 20)
            metric_name: metric to use for concordance

        Returns:
            (optimal resolution, Leiden clusters, dataframe with clusters and reference gene set metrics)
        """
        mean_metrics_df = pd.DataFrame(columns=["resolution", "mean_of_best_match_metric"])

        for res in resolutions:
            _, metrics_df = self.cluster_and_compute_metrics(
                resolution=res,
                reference_gene_sets=reference_gene_sets,
                metric_name=metric_name,
            )
            mean_metric = metrics_df[metric_name][metrics_df["cluster"] != -1].mean()

            # mean of best matches over all clusters in clustering
            mean_metrics_df = pd.concat(
                [mean_metrics_df, pd.DataFrame({"resolution": [res], "mean_of_best_match_metric": [mean_metric]})],
                axis=0,
            )

        # choose the resolution with the highest mean metric
        best_resolution = mean_metrics_df.set_index("resolution")["mean_of_best_match_metric"].idxmax()

        # re-compute the Leiden clustering at the best resolution (cached) and metrics
        best_clustering, best_metrics_df = self.cluster_and_compute_metrics(
            resolution=best_resolution,
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
        )

        # return the best resolution and the Leiden communities
        return best_resolution, best_clustering, best_metrics_df

    def bayesopt_optimal_resolution_communities_given_gene_sets(
        self,
        reference_gene_sets: dict[str, set[str]],
        resolution_range: tuple[float, float] = (0.2, 10.0),
        num_clusterings_to_compute: int = 20,
        metric_name: t.Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "f1",
    ) -> tuple[float, np.ndarray, pd.DataFrame]:
        """
        Compute an "optimal" Leiden clustering by choosing the Leiden resolution by
        maximizing the mean of a concordance metric over reference gene sets.
        Optimization is performed by Bayesian optimization over the input resolution range.
        The resolution which results in the best concordance with the reference gene sets
        is chosen and used to compute the Leiden communities.

        Args:
            reference_gene_sets: dictionary of reference gene sets
            resolution_range: range of resolutions to consider, e.g. (-2.0, 2.0)
            num_clusterings_to_compute: number of clusterings to compute during optimization
            metric_name: metric to use for concordance

        Returns:
            (optimal resolution, Leiden clusters, dataframe with clusters and reference gene set metrics)
        """
        assert num_clusterings_to_compute >= 15, "num_clusterings_to_compute must be >= 15"

        # function to be minimized
        def _compute_inv_mean_metric(x: list[float]) -> float:
            resolution = x[0]
            _, metrics_df = self.cluster_and_compute_metrics(
                resolution=resolution,
                reference_gene_sets=reference_gene_sets,
                metric_name=metric_name,
            )
            return -1 * metrics_df[metric_name][metrics_df["cluster"] != -1].mean()

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
        )

        # return the best resolution and the Leiden communities
        return best_resolution, best_clustering, best_metrics_df


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
        verbose: bool = True,
    ):
        super().__init__(
            adata_obs=adata_obs,
            gene_info_tsv_path=gene_info_tsv_path,
            query_var_names=query_var_names,
            prompt_var_names=prompt_var_names,
            response_qp=jacobian_qp,
            prompt_marginal_mean_p=prompt_marginal_mean_p,
            prompt_marginal_std_p=prompt_marginal_std_p,
            query_marginal_mean_q=query_marginal_mean_q,
            query_marginal_std_q=query_marginal_std_q,
            verbose=verbose,
        )

        n_query_vars = len(query_var_names)
        n_prompt_vars = len(prompt_var_names)

        assert prompt_empirical_mean_p.shape == (n_prompt_vars,)
        assert query_empirical_mean_q.shape == (n_query_vars,)

        self.jacobian_point = jacobian_point
        self.prompt_empirical_mean_p = prompt_empirical_mean_p
        self.query_empirical_mean_q = query_empirical_mean_q

    @staticmethod
    def from_old_jacobian_pt_dump(jacobian_pt_path: str, adata_path: str, gene_info_tsv_path: str) -> "JacobianContext":
        # suppres FutureWarning in a context manager
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            adata = sc.read_h5ad(adata_path)
            old_jac_dict = torch.load(jacobian_pt_path)

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
        )

    def __str__(self) -> str:
        return (
            f"JacobianContext({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__
