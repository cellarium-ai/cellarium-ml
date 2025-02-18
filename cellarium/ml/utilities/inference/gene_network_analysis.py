# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""General gene network analysis and validation."""

import logging
import typing as t
import warnings
from functools import cached_property

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


def load_gene_info_table(gene_info_tsv_path: str, included_gene_ids: list[str]) -> t.Tuple[pd.DataFrame, dict, dict]:
    gene_info_df = pd.read_csv(gene_info_tsv_path, sep="\t")

    gene_symbol_to_gene_id_map = dict()
    for gene_symbol, gene_id in zip(gene_info_df['Gene Symbol'], gene_info_df['ENSEMBL Gene ID']):
        if gene_symbol != float('nan'):
            gene_symbol_to_gene_id_map[gene_symbol] = gene_id

    gene_id_to_gene_symbol_map = {
        gene_id: gene_symbol for gene_symbol, gene_id in gene_symbol_to_gene_id_map.items()}
    for gene_id in included_gene_ids:
        if gene_id not in gene_id_to_gene_symbol_map:
            gene_id_to_gene_symbol_map[gene_id] = gene_id

    return gene_info_df, gene_symbol_to_gene_id_map, gene_id_to_gene_symbol_map



def quantile_normalize_select(
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int,
        top_k: int,
        min_x: int | None = None,
        max_x: int | None = None):
    """
    Filter, normalize, and select top_k elements.
    
    Parameters:
    -----------
    x : np.ndarray
        1D numpy array of covariate values.
    y : np.ndarray
        1D numpy array of response values.
    n_bins : int
        Number of bins to subdivide the range [min_x, max_x].
    top_k : int
        Number of top largest normalized y values to select.
    min_x : float
        Minimum x value (x must be > min_x).
    max_x : float
        Maximum x value (x must be < max_x).

    Returns:
    --------
    selected_indices : np.ndarray
        Indices in the original arrays corresponding to the top_k selected values.
    top_normalized_y : np.ndarray
        The normalized y values corresponding to these selected indices.
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
    top_k = min(top_k, np.sum(_y_normalized > -np.inf))
    top_idx = sorted_idx[:top_k]

    return top_idx, y_normalized


class GeneNetworkAnalysisBase:
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
            verbose: bool = True):
        
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

        self.gene_info_df, self.gene_symbol_to_gene_id_map, self.gene_id_to_gene_symbol_map = \
            load_gene_info_table(gene_info_tsv_path, query_var_names + prompt_var_names)

        self.processed = False

    @property
    def cell_type(self) -> str:
        return self.adata_obs['cell_type'].values[0]
    
    @property
    def tissue(self) -> str:
        return self.adata_obs['tissue'].values[0]

    @property
    def disease(self) -> str:
        return self.adata_obs['disease'].values[0]

    @property
    def development_stage(self) -> str:
        return self.adata_obs['development_stage'].values[0]

    @property
    def sex(self) -> str:
        return self.adata_obs['sex'].values[0]

    @property
    def total_mrna_umis(self) -> float:
        return self.adata_obs['total_mrna_umis'].values[0]

    @property
    def query_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.query_var_names]
    
    @property
    def prompt_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.prompt_var_names]
    
    @cached_property
    def query_gene_id_to_idx_map(self) -> dict[str, int]:
        assert self.processed, "Must process before accessing"
        return {gene_id: idx for idx, gene_id in enumerate(self.query_var_names)}
    
    @cached_property
    def prompt_gene_id_to_idx_map(self) -> dict[str, int]:
        assert self.processed, "Must process before accessing"
        return {gene_id: idx for idx, gene_id in enumerate(self.prompt_var_names)}

    def __str__(self) -> str:
        return (
            f"GeneNetworkAnalysisBase({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__

    def process(
            self,
            response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
            feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
            min_prompt_gene_tpm: float = 0.,
            min_query_gene_tpm: float = 0.,
            query_response_amp_min_pct: float | None = None,
            feature_max_value: float | None = None,
            norm_pseudo_count: float = 1e-3,
            query_hv_top_k: int | None = None,
            query_hv_n_bins: int | None = 50,
            query_hv_min_x: float | None = 1e-2,
            query_hv_max_x: float | None = np.inf,
            eps: float = 1e-8,
            z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        ) -> None:

        assert not self.processed, "Already processed -- please create a new instance"
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
        self.z_qp = self.response_qp * (z_p[None, :] + norm_pseudo_count) / (z_q[:, None] + norm_pseudo_count)

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
        
        if query_hv_top_k is not None:
            assert query_hv_n_bins is not None
            assert query_hv_min_x is not None
            assert query_hv_max_x is not None
            top_idx, _ = quantile_normalize_select(
                x=np.log1p(self.query_marginal_mean_q),
                y=np.std(self.z_qp, axis=1),
                n_bins=query_hv_n_bins,
                top_k=query_hv_top_k,
                min_x=query_hv_min_x,
                max_x=query_hv_max_x)
            hv_mask_q = np.zeros_like(self.mask_q, dtype=bool)
            hv_mask_q[top_idx] = True
            self.mask_q = self.mask_q & hv_mask_q
            logger.info(f"Number of query genes after highly-variable filtering: {np.sum(self.mask_q)} / {len(self.mask_q)}")

        # apply the mask to everything else
        self.prompt_var_names = [self.prompt_var_names[i] for i in range(len(self.prompt_var_names)) if self.mask_p[i]]
        self.prompt_marginal_mean_p = self.prompt_marginal_mean_p[self.mask_p]
        self.prompt_marginal_std_p = self.prompt_marginal_std_p[self.mask_p]

        self.query_var_names = [self.query_var_names[i] for i in range(len(self.query_var_names)) if self.mask_q[i]]
        self.query_marginal_mean_q = self.query_marginal_mean_q[self.mask_q]
        self.query_marginal_std_q = self.query_marginal_std_q[self.mask_q]

        # apply the mask to z_qp
        self.z_qp = self.z_qp[self.mask_q, :][:, self.mask_p]

        # clip and transform features
        self.z_qp[np.isnan(self.z_qp)] = 0.
        self.z_qp[np.isinf(self.z_qp)] = 0.

        if feature_max_value is not None:
            assert feature_max_value > 0
            self.z_qp = np.clip(self.z_qp, -feature_max_value, feature_max_value)

        if z_trans_func is not None:
            self.z_qp = z_trans_func(self.z_qp)

        if feature_normalization_strategy == "z_score":
            # z-score each query gene separately in response to prompt genes
            self.z_qp = (self.z_qp - np.mean(self.z_qp, axis=0, keepdims=True)) / (
                eps + np.std(self.z_qp, axis=0, keepdims=True))
        elif feature_normalization_strategy == "l2":
            # l2-normalize query genes separately for each prompt gene
            self.z_qp = self.z_qp / (eps + np.linalg.norm(self.z_qp, axis=0, keepdims=True))
        elif feature_normalization_strategy == "none":
            pass
        else:
            raise ValueError("Invalid feature normalization strategy")

        self.processed = True

        # adj
        self.a_pp = None
        
        # leiden
        self.leiden_membership = None
        
        # spectral analysis
        self.eigs = None
        self.spectral_dim = None


    def compute_adjacency_matrix(
            self,
            adjacency_strategy: str = t.Literal[
                "shifted_correlation", "unsigned_correlation", "positive_correlation", "binary"],
            n_neighbors: int | None = 50,
            self_loop: bool = False,
            **kwargs) -> None:

        n_query_genes = self.z_qp.shape[0]
        rho_pp = self.z_qp.T @ self.z_qp / n_query_genes

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
            t_pp = np.argsort(a_pp, axis=-1)[:, -n_neighbors:]  # take the top n_neighbors

            # make a mask for the top n_neighbors
            _a_pp = np.zeros_like(a_pp)
            for p in range(a_pp.shape[0]):
                _a_pp[p, t_pp[p]] = a_pp[p, t_pp[p]]
                _a_pp[t_pp[p], p] = a_pp[t_pp[p], p]
        else:
            _a_pp = a_pp
        a_pp = _a_pp
        
        if not self_loop:
            np.fill_diagonal(a_pp, 0)

        self.a_pp = a_pp

    def _compute_igraph_from_adjacency(self, directed: bool = False) -> ig.Graph:
        assert self.a_pp is not None, "Must compute adjacency matrix first"
        sources, targets = self.a_pp.nonzero()
        weights = self.a_pp[sources, targets]
        g = ig.Graph(directed=directed)
        g.add_vertices(self.a_pp.shape[0])  # this adds adjacency.shape[0] vertices
        g.add_edges(list(zip(sources, targets)))
        g.es["weight"] = weights
        return g

    def compute_leiden_communites(
            self,
            resolution: float = 3.0,
            min_community_size: int = 2,
        ):

        g = self._compute_igraph_from_adjacency()
        
        leiden_partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
        
        self.leiden_membership = np.array(leiden_partition.membership)
        
        # remove small communities
        n_leiden = len(np.unique(self.leiden_membership))
        sizes = np.array([np.sum(self.leiden_membership == i) for i in range(n_leiden)])
        for i_leiden in range(n_leiden):
            if sizes[i_leiden] < min_community_size:
                self.leiden_membership[self.leiden_membership == i_leiden] = -1
    
    def compute_spectral_dimension(
            self,
            offset: int = 2,
            n_lambda_for_estimation: int = 5) -> float:
        assert self.a_pp is not None, "Must compute adjacency matrix first"
        
        # calculate normalized laplacian and its eigenvalues
        norm_p = 1. / (1e-9 + np.sqrt(self.a_pp.sum(0)))
        lap_pp = np.eye(self.a_pp.shape[0]) - norm_p[:, None] * norm_p[None, :] * self.a_pp
        eigs = eigh(lap_pp.astype(np.float64), eigvals_only=True)
        eigs[0] = 0
        eigs = np.clip(eigs, 0, np.inf)  # roundoff error guardrail
        self.eigs = eigs

        n_lambda = np.cumsum(eigs)
        n_lambda = n_lambda / n_lambda[-1]
        first_nonzero = np.where(eigs > 0)[0][0] + offset
        xx = np.log(eigs[first_nonzero:first_nonzero + n_lambda_for_estimation])
        yy = np.log(n_lambda[first_nonzero:first_nonzero + n_lambda_for_estimation])

        lin = linregress(xx, yy)
        slope, intercept = lin.slope, lin.intercept
 
        # save a few thigs for later
        self.spectral_dim = 2 * linregress(xx, yy).slope
        self.eigs = eigs
        self.n_lambda = n_lambda
        self.log_eigs_asymptotic = xx
        self.log_n_lambda_asymptotic = yy
        self.spectral_dim_slope = slope
        self.spectral_dim_intercept = intercept

    def make_mde_embedding(
            self,
            n_neighbors: int = 7,
            repulsive_fraction: int = 5,
            attractive_penalty: pymde.functions.function.Function = pymde.penalties.Log1p,
            repulsive_penalty: pymde.functions.function.Function = pymde.penalties.InvPower,
            device: torch.device = torch.device("cpu"),
            max_iter: int = 500,
            verbose: bool = True,
            **kwargs
        ):
        
        mde = pymde.preserve_neighbors(
            self.z_qp.T,  # we are embedding the prompts (perturbations)
            device=device,
            verbose=verbose,
            n_neighbors=n_neighbors,
            repulsive_fraction=repulsive_fraction,
            attractive_penalty=attractive_penalty,
            repulsive_penalty=repulsive_penalty,
            **kwargs)

        self.embedding_p2 = mde.embed(verbose=verbose, max_iter=max_iter).cpu().numpy()

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

        # Create the color map with string keys
        colormap = {str(label): cc.glasbey[i % len(cc.glasbey)] for i, label in enumerate(unique_memberships)}

        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'x': self.embedding_p2[:, 0],
            'y': self.embedding_p2[:, 1],
            'label': self.prompt_gene_symbols,
            'membership': memberships_p.astype(str)  # Convert to string
        })

        # Create the scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            hover_name='label',
            title=plot_title,
            color='membership',
            color_discrete_map=colormap
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
                    mode='markers',
                    marker=dict(color=color, size=highlight_marker_size),
                    text=gene_symbols,
                    showlegend=True,
                    name=gene_set_name
                )

        # Update layout to decrease the width of the plot
        fig.update_layout(
            width=width,  # Adjust the width as needed
            height=height,
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                title='MDE_1'
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                title='MDE_2'
            )
        )

        return fig

    def plot_spectral_dimension(self, ax: plt.Axes) -> None:
        assert self.eigs is not None, "Must compute spectral dimension first"
        ax.scatter(self.log_eigs_asymptotic, self.log_n_lambda_asymptotic)
        ax.plot(
            self.log_eigs_asymptotic,
            self.spectral_dim_slope * self.log_eigs_asymptotic + self.spectral_dim_intercept,
            color='red',
            label=f"$d_S$ = {self.spectral_dim:.2f}")
        ax.set_xlabel("ln $\lambda$")
        ax.set_ylabel("ln N($\lambda$)")
        ax.set_title(self.cell_type)
        ax.legend()


class JacobianContext(GeneNetworkAnalysisBase):
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
            verbose: bool = True):

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
            verbose=verbose)
        
        n_query_vars = len(query_var_names)
        n_prompt_vars = len(prompt_var_names)

        assert prompt_empirical_mean_p.shape == (n_prompt_vars,)
        assert query_empirical_mean_q.shape == (n_query_vars,)

        self.jacobian_point = jacobian_point
        self.prompt_empirical_mean_p = prompt_empirical_mean_p
        self.query_empirical_mean_q = query_empirical_mean_q
    
    @staticmethod
    def from_old_jacobian_pt_dump(
            jacobian_pt_path: str,
            adata_path: str,
            gene_info_tsv_path: str) -> 'JacobianContext':
        
        # suppres FutureWarning in a context manager
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            adata = sc.read_h5ad(adata_path)
            old_jac_dict = torch.load(jacobian_pt_path)

        # make a metacell
        X_meta_g = np.asarray(adata.X.sum(0))

        # set total mrna umis to the mean of the dataset
        target_total_mrna_umis = adata.obs['total_mrna_umis'].mean()
        X_meta_g = X_meta_g * target_total_mrna_umis / X_meta_g.sum()

        # make a metacell anndata
        adata_meta = adata[0, :].copy()
        adata_meta.X = X_meta_g
        adata_meta.obs['total_mrna_umis'] = [target_total_mrna_umis]

        prompt_empirical_mean_p = adata_meta[0, old_jac_dict['prompt_var_names']].X.flatten()
        query_empirical_mean_q = adata_meta[0, old_jac_dict['query_var_names']].X.flatten()

        return JacobianContext(
            adata_obs=adata_meta.obs,
            gene_info_tsv_path=gene_info_tsv_path,
            jacobian_point=old_jac_dict['jacobian_point'],
            query_var_names=old_jac_dict['query_var_names'],
            prompt_var_names=old_jac_dict['prompt_var_names'],
            jacobian_qp=old_jac_dict['jacobian_qg'].cpu().numpy(),
            prompt_empirical_mean_p=prompt_empirical_mean_p,
            query_empirical_mean_q=query_empirical_mean_q,
            prompt_marginal_mean_p=old_jac_dict['prompt_marginal_dict']['gene_marginal_means_q'].cpu().numpy(),
            prompt_marginal_std_p=old_jac_dict['prompt_marginal_dict']['gene_marginal_std_q'].cpu().numpy(),
            query_marginal_mean_q=old_jac_dict['query_marginal_dict']['gene_marginal_means_q'].cpu().numpy(),
            query_marginal_std_q=old_jac_dict['query_marginal_dict']['gene_marginal_std_q'].cpu().numpy(),
        )

    def __str__(self) -> str:
        return (
            f"JacobianContext({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__
