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

        self.processed = False

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
        assert self.processed, "Must process before accessing"
        return {gene_id: idx for idx, gene_id in enumerate(self.query_var_names)}

    def __str__(self) -> str:
        return (
            f"GeneNetworkAnalysisBase({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__

    def process(
        self,
        response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
        feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
        min_prompt_gene_tpm: float = 10.0,
        min_query_gene_tpm: float = 10.0,
        query_response_amp_min_pct: float | None = None,
        feature_max_value: float = 10.0,
        eps: float = 1e-8,
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
        self.z_qp = self.response_qp * (z_p[None, :] + eps) / (z_q[:, None] + eps)

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
        self.z_qp[np.isnan(self.z_qp)] = 0.0
        self.z_qp[np.isinf(self.z_qp)] = 0.0
        self.z_qp = np.clip(self.z_qp, -feature_max_value, feature_max_value)

        self.processed = True

        # adj
        self.a_qq: np.ndarray | None = None

        # leiden
        self.leiden_membership: np.ndarray | None = None

        # spectral analysis
        self.eigs: np.ndarray | None = None
        self.spectral_dim: np.ndarray | None = None

    def compute_adjacency_matrix(
        self,
        adjacency_strategy: t.Literal[
            "shifted_correlation", "unsigned_correlation", "positive_correlation", "positive_correlation_binary"
        ],
        n_neighbors: int | None = 50,
        self_loop: bool = False,
        **kwargs,
    ) -> None:
        if adjacency_strategy == "shifted_correlation":
            assert "beta" in kwargs, "Must provide beta for shifted correlation"
            beta = kwargs["beta"]
            a_qq = np.power(0.5 * (1 + np.dot(self.z_qp, self.z_qp.T)), beta)
        elif adjacency_strategy == "unsigned_correlation":
            assert "beta" in kwargs, "Must provide beta for unsigned correlation"
            beta = kwargs["beta"]
            a_qq = np.power(np.abs(np.dot(self.z_qp, self.z_qp.T)), beta)
        elif adjacency_strategy == "positive_correlation":
            assert "beta" in kwargs, "Must provide beta for positive correlation"
            beta = kwargs["beta"]
            a_qq = np.power(np.maximum(0, np.dot(self.z_qp, self.z_qp.T)), beta)
        elif adjacency_strategy == "positive_correlation_binary":
            assert n_neighbors is None, "n_neighbors must be None for binary adjacency"
            assert "tau" in kwargs, "Must provide correlation threshold for binary adjacency"
            tau = kwargs["tau"]
            a_qq = (np.maximum(0, np.dot(self.z_qp, self.z_qp.T)) > tau).astype(float)
        else:
            raise ValueError("Invalid adjacency strategy")

        assert np.isclose(a_qq, a_qq.T).all(), "Adjacency matrix must be symmetric -- something is wrong!"

        if n_neighbors is not None:
            assert n_neighbors > 0, "n_neighbors must be positive"
            t_qq = np.argsort(a_qq, axis=-1)[:, -n_neighbors:]  # take the top n_neighbors

            # make a mask for the top n_neighbors
            _a_qq = np.zeros_like(a_qq)
            for q in range(a_qq.shape[0]):
                _a_qq[q, t_qq[q]] = a_qq[q, t_qq[q]]
                _a_qq[t_qq[q], q] = a_qq[t_qq[q], q]
        else:
            _a_qq = a_qq
        a_qq = _a_qq

        if not self_loop:
            np.fill_diagonal(a_qq, 0)

        self.a_qq = a_qq

    def _compute_igraph_from_adjacency(self, directed: bool = False) -> ig.Graph:
        assert self.a_qq is not None, "Must compute adjacency matrix first by calling obj.compute_adjacency_matrix()"

        sources, targets = self.a_qq.nonzero()
        weights = self.a_qq[sources, targets]
        g = ig.Graph(directed=directed)
        g.add_vertices(self.a_qq.shape[0])  # this adds adjacency.shape[0] vertices
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
            g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution
        )

        self.leiden_membership = np.array(leiden_partition.membership)

        # remove small communities
        assert self.leiden_membership is not None  # mypy
        n_leiden = len(np.unique(self.leiden_membership))
        sizes = np.array([np.sum(self.leiden_membership == i) for i in range(n_leiden)])
        for i_leiden in range(n_leiden):
            if sizes[i_leiden] < min_community_size:
                self.leiden_membership[self.leiden_membership == i_leiden] = -1

    def compute_spectral_dimension(self, offset: int = 2, n_lambda_for_estimation: int = 5) -> float:
        assert self.a_qq is not None, "Must compute adjacency matrix first"

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
        mde = pymde.preserve_neighbors(
            self.z_qp,
            device=device,
            verbose=verbose,
            n_neighbors=n_neighbors,
            repulsive_fraction=repulsive_fraction,
            attractive_penalty=attractive_penalty,
            repulsive_penalty=repulsive_penalty,
            **kwargs,
        )

        self.embedding_q2 = mde.embed(verbose=verbose, max_iter=max_iter).cpu().numpy()

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
