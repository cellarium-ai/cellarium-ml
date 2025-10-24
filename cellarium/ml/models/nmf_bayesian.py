# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from types import EllipsisType
from typing import Callable, List, Optional, Sequence, Type, Union

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.nn.module import PyroParam, _unconstrain
from torch.distributions import constraints

from cellarium.ml.models.nmf import NonNegativeMatrixFactorization
from cellarium.ml.transforms import Filter

logger = logging.getLogger(__name__)


class Exp(torch.nn.Module):
    """Exponential activation function as a torch module"""

    def __init__(self, eps: float = 1e-5):
        """Exponential activation function with numerical stabilization, useful
        for outputs that must be > 0

        NOTE: output = torch.exp(input) + eps

        Args:
            eps: Numerical stability additive constant.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.exp(x) + self.eps

    def __repr__(self):
        return f"torch.exp() + {self.eps}"


class FullyConnectedLayer(torch.nn.Module):
    """
    Neural network unit made of a fully connected linear layer, but
    customizable including shapes, activations, batch norm, layer norm, and
    dropout.

    Args:
        input_dim: Number of features for input
        output_dim: Number of features for output
        activation: Activation function to be applied to each hidden layer
            (default :py:class:`torch.nn.ReLU`)
        use_batch_norm: True to apply batch normalization using
            :py:class:`torch.nn.BatchNorm1d` with ``momentum=0.01``, ``eps=0.001``
            (default False)
        use_layer_norm: True to apply layer normalization (after optional batch
            normalization) using :py:class:`torch.nn.LayerNorm` with
            ``elementwise_affine=False`` (default False)
        dropout_rate: Dropout rate to use in :py:class:`torch.nn.Dropout` before
            linear layer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: torch.nn.Module | None = torch.nn.ReLU(),
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # set up layers as a list of Linear modules with appropriate extras
        modules = torch.nn.ModuleList()
        if dropout_rate is not None:
            modules.append(torch.nn.Dropout(p=dropout_rate))
        modules.append(torch.nn.Linear(in_features=input_dim, out_features=output_dim))
        if use_batch_norm:
            modules.append(torch.nn.BatchNorm1d(num_features=output_dim, momentum=0.01, eps=0.001))
        if use_layer_norm:
            modules.append(torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False))
        if activation is not None:
            modules.append(activation)

        # concatenate Linear layers using Sequential
        self.layer = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FullyConnectedNetwork(torch.nn.Module):
    """
    Neural network made of fully connected linear layers,
    :py:class:`FullyConnectedLayer`. Architecture is customizable including
    shapes, activations, batch norm, layer norm, and dropout.

    Args:
        input_dim: Number of features for input
        hidden_dims: List of hidden layer sizes, can be empty list []
        output_dim: Number of features for output
        hidden_activation: Activation function to be applied to each hidden layer
            (default :py:class:`torch.nn.ReLU`)
        output_activation: Activation function to be applied to output (default None)
        use_batch_norm: True to apply batch normalization using
            :py:class:`torch.nn.BatchNorm1d` with ``momentum=0.01``, ``eps=0.001``
            (default False)
        use_layer_norm: True to apply layer normalization (after optional batch
            normalization) using :py:class:`torch.nn.LayerNorm` with
            ``elementwise_affine=False`` (default False)
        norm_output: True to apply normalization to output layer before output
            activation (default False)
        dropout_rate: Dropout rate to use in :py:class:`torch.nn.Dropout` for each
            hidden layer (applied before each layer)
        dropout_input: True to apply dropout before first layer (default False)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        output_activation: Optional[torch.nn.Module] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        norm_output: bool = False,
        dropout_rate: Optional[float] = None,
        dropout_input: bool = False,
    ):
        super().__init__()

        if use_layer_norm and use_batch_norm:
            raise UserWarning("You are trying to use both batch norm and layer norm. That's probably too much norm.")

        # set up layers as a list of Linear modules with appropriate extras
        dim_ins_and_outs = zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        n_layers = 1 + len(hidden_dims)
        layers = [
            FullyConnectedLayer(
                input_dim=i,
                output_dim=j,
                activation=hidden_activation if (layer < n_layers - 1) else output_activation,
                use_batch_norm=use_batch_norm if ((layer < n_layers - 1) or norm_output) else False,
                use_layer_norm=use_layer_norm if ((layer < n_layers - 1) or norm_output) else False,
                dropout_rate=None if ((layer == 0) and not dropout_input) else dropout_rate,
            )
            for layer, (i, j) in enumerate(dim_ins_and_outs)
        ]

        # concatenate Linear layers using Sequential
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        return self.network(x_ng)


@torch.no_grad()
def initialize_matrix(
    rows: int,
    cols: int,
    alpha_cols: torch.Tensor,
    simplex: bool = True,
    seed: torch.Tensor | int | None = None,
):
    """
    Initialize a parameter matrix for NMF: [rows, cols]

    Args:
        rows: Rows for matrix
        cols: Cols for matrix
        alpha_cols: Concentration of gamma distribution associated with cols.
            Small values < 1 lead to sparsity.
        simplex: True to initialize values so that matrix.sum(dim=-1) = 1
        seed: Random seed

    Returns:
        matrix: [rows, cols] matrix of initial values for NMF
    """

    # Bayesian NMF, Cemgil 2009... ish
    alpha = alpha_cols.expand((cols,))
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=alpha / cols)

    if seed:
        torch.manual_seed(seed)

    matrix = gamma_dist.sample([rows])

    if simplex:
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)

    return matrix.detach()


def ard_regularization(loading_matrix_nk: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Automatic relevance determination over factors, which induces some
    factor loadings (over all cells) to go to zero.

    NOTE: There is some justification for alpha being of order k_max. There is
    less justification for an upper bound, but it seems to speed up learning
    by limiting the gains produced by focusing on sending one entry to infinity.

    NOTE: This could equivalently be written as a pyro.sample statement where
    the prior in the model is Normal(0, 1 / alpha) and the posterior in the
    guide is Delta(loadings). This implementation is just a bit easier since it
    only requires one if-statement in the model and nothing in the guide.
    """
    # k_max = loading_matrix_nk.shape[-1]
    # log_ard_alpha_k = pyro.param(
    #     "log_ard_alpha_k",
    #     np.log(k_max) * torch.ones(k_max).to(loading_matrix_nk.device),
    #     constraint=constraints.less_than(upper_bound=6.0),
    # )
    log_ard_alpha_k = pyro.param("log_ard_alpha_k")
    ard_alpha_k = log_ard_alpha_k.exp()

    # TODO: what about using a zero-mean Beta distribution (matches support)?
    # TODO: but that's not exactly ARD then is it?
    with pyro.poutine.scale(scale=scale):
        log_prob_sum_ard_reg = dist.Normal(loc=0.0, scale=1.0 / ard_alpha_k).log_prob(loading_matrix_nk).sum()
        pyro.factor("ard_regularization", log_prob_sum_ard_reg)
    # log_metric("ARD_alpha_k", ard_alpha_k)
    return log_prob_sum_ard_reg


def gene_graph_concordance_statistic(
    similarity_matrix_gg: torch.Tensor, factor_matrix_kg: torch.Tensor
) -> torch.Tensor:
    """Compute the gene graph concordance statistic (a scalar)

    concordance_score = tr(F S F^T)

    which is a score we want to maximize

    Parameters
    ----------
    similarity_matrix_gg: Gene-gene similarity matrix, symmetric, with max
        value 1, where large values indicate more similarity
    factor_matrix_kg: The current inferred factors
    """

    return torch.trace(
        torch.matmul(factor_matrix_kg, torch.matmul(similarity_matrix_gg, factor_matrix_kg.t())),
    )


def gene_graph_regularization(
    similarity_matrix_gg: torch.Tensor,
    factor_matrix_kg: torch.Tensor,
    gamma_loc: torch.Tensor,
    gamma_scale: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply regularization to the gene expression programs themselves

    concordance_score = tr(F S F^T)
    is a score which we want to maximize

    Args:
        similarity_matrix_gg: Gene-gene similarity matrix, symmetric, with max
            value 1, where large values indicate more similarity
        factor_matrix_kg: The current inferred factors

    Returns:
        log_prob: The log probability of the concordance score under the
            null distribution N(gamma_loc, gamma_scale)
    """
    # compute the concordance score
    gamma = gene_graph_concordance_statistic(
        similarity_matrix_gg=similarity_matrix_gg,
        factor_matrix_kg=factor_matrix_kg,
    )
    log_prob = torch.distributions.Normal(gamma_loc, gamma_scale).cdf(gamma).log()
    with pyro.poutine.scale(scale=scale):
        pyro.factor("graph_regularization", log_prob)
    # log_metric("graph_regularization", log_prob)
    return log_prob


def solve_nnls_fista(A, B, max_iter=1000, tol=1e-6):
    """
    FISTA algorithm for NNLS solving Ax = B for x >= 0
    
    Args:
        A: Coefficient matrix of shape (..., m, n)
        B: Right-hand side of shape (..., m, k) 
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        x: Solution of shape (..., n, k) with x >= 0
    """
    # Handle batch dimensions
    *batch_dims, m, n = A.shape
    *batch_dims_B, m_B, k = B.shape
    
    assert m == m_B, f"Incompatible dimensions: A has {m} rows, B has {m_B} rows"
    
    # Precompute AtA and AtB for efficiency
    AtA = A.transpose(-2, -1) @ A  # (..., n, n)
    AtB = A.transpose(-2, -1) @ B  # (..., n, k)
    
    # Compute Lipschitz constant (largest eigenvalue of AtA)
    eigenvals = torch.linalg.eigvals(AtA).real  # (..., n)
    L = eigenvals.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (..., 1, 1)
    
    # Initialize variables
    x = torch.zeros(*batch_dims, n, k, device=A.device, dtype=A.dtype)
    y = x.clone()
    t = torch.ones(*batch_dims, 1, 1, device=A.device, dtype=A.dtype)
    
    for i in range(max_iter):
        x_old = x.clone()
        
        # Gradient step: grad = AtA @ y - AtB
        grad = AtA @ y - AtB
        x_new = torch.clamp(y - grad / L, min=0)
        
        # Momentum update
        t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
        
        # Check convergence
        if torch.norm(x - x_old) < tol:
            break
            
    return x


def solve_nnls_coordinate_descent(A, B, max_iter=1000, tol=1e-6):
    """
    Coordinate descent for NNLS solving Ax = B for x >= 0
    
    Args:
        A: Coefficient matrix of shape (..., m, n)
        B: Right-hand side of shape (..., m, k)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        x: Solution of shape (..., n, k) with x >= 0
    """
    # Handle batch dimensions
    *batch_dims, m, n = A.shape
    *batch_dims_B, m_B, k = B.shape
    
    assert m == m_B, f"Incompatible dimensions: A has {m} rows, B has {m_B} rows"
    
    # Precompute AtA and AtB for efficiency
    AtA = A.transpose(-2, -1) @ A  # (..., n, n)
    AtB = A.transpose(-2, -1) @ B  # (..., n, k)
    
    # Extract diagonal elements of AtA for coordinate updates
    AtA_diag = torch.diagonal(AtA, dim1=-2, dim2=-1)  # (..., n)
    
    # Initialize solution
    x = torch.zeros(*batch_dims, n, k, device=A.device, dtype=A.dtype)
    
    for iteration in range(max_iter):
        x_old = x.clone()
        
        # Update each coordinate
        for j in range(n):
            # Compute residual for coordinate j
            # residual = AtB[j] - AtA[j, :] @ x + AtA[j, j] * x[j]
            AtA_j = AtA[..., j:j+1, :]  # (..., 1, n)
            residual = AtB[..., j:j+1, :] - AtA_j @ x + AtA_diag[..., j:j+1, None] * x[..., j:j+1, :]
            
            # Update x[j] with non-negativity constraint
            x[..., j:j+1, :] = torch.clamp(residual / AtA_diag[..., j:j+1, None], min=0)
        
        # Check convergence
        if torch.norm(x - x_old) < tol:
            break
            
    return x


class InferenceStrategies:
    """Static methods for different inference strategies"""

    @staticmethod
    def _get_likelihood_distribution(lam_ng: torch.Tensor, likelihood_dist: Type) -> dist.Distribution:
        """Create the appropriate likelihood distribution"""
        if likelihood_dist == dist.Poisson:
            return dist.Poisson(rate=lam_ng + 1e-5).to_event(1)
        elif likelihood_dist == dist.Normal:
            return dist.Normal(loc=lam_ng, scale=1.0).to_event(1)
        else:
            raise ValueError(f"Unsupported likelihood distribution: {likelihood_dist}")

    @staticmethod
    def maximum_likelihood_model(
        x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor, model: "BayesianNonNegativeMatrixFactorization"
    ) -> dict[str, torch.Tensor]:
        """
        Maximum likelihood model implementation

        Args:
            x_ng: Count data minibatch, rows are cells and columns are genes
            minibatch_indices_n: Indices of each cell in this minibatch (used to index full dataset)
            model: The main model instance

        Returns:
            Dictionary of tensors relevant for the generative model
        """
        # Access PyroParam objects from the model
        loading_matrix_mk: torch.Tensor = model.loading_matrix_mk  # type: ignore[assignment]
        factor_matrix_kg: torch.Tensor = model.factor_matrix_kg  # type: ignore[assignment]
        assert isinstance(loading_matrix_mk, torch.Tensor)
        assert isinstance(factor_matrix_kg, torch.Tensor)

        # subset the loading matrix to the cells in this minibatch
        loading_matrix_nk = loading_matrix_mk[minibatch_indices_n, :]

        # apply ARD regularization, if called for
        if model.use_ard:
            log_prob_sum_ard_reg = ard_regularization(loading_matrix_nk=loading_matrix_nk)
        else:
            log_prob_sum_ard_reg = torch.tensor(-float("inf")).to(x_ng.device)

        # the normalized poisson rate, where chi.sum(dim=-1) = 1
        chi_ng = torch.matmul(loading_matrix_nk, factor_matrix_kg)

        # the poisson rate
        cell_size_factors_n1 = x_ng.sum(dim=-1, keepdim=True)
        lam_ng = chi_ng * cell_size_factors_n1

        # compare to observed count data
        with pyro.plate("obs_plate", size=x_ng.shape[0]):
            likelihood_dist = InferenceStrategies._get_likelihood_distribution(lam_ng, model.likelihood_dist)
            pyro.sample("obs", likelihood_dist, obs=x_ng)  # type: ignore[arg-type]

        return {"ard_regularization": log_prob_sum_ard_reg}

    @staticmethod
    def maximum_likelihood_guide(
        x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor, model: "BayesianNonNegativeMatrixFactorization"
    ):
        """Maximum likelihood inference has empty guide - not Bayesian"""
        pass

    @staticmethod
    @torch.no_grad()
    def maximum_likelihood_get_loadings(
        x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor, model: "BayesianNonNegativeMatrixFactorization"
    ) -> torch.Tensor:
        """Get factor loadings for maximum likelihood strategy"""

        # solve an auxiliary OLS problem (does better on tests)
        factor_matrix_kg: torch.Tensor = model.factor_matrix_kg  # type: ignore[assignment]

        # option 1: non-negativity is a hack after OLS
        # loading_matrix_nk = torch.linalg.lstsq(factor_matrix_kg.t(), x_ng.t()).solution.t()
        # loading_matrix_nk = torch.clamp(loading_matrix_nk, min=0)

        # option 2: fista
        loading_matrix_nk = solve_nnls_fista(factor_matrix_kg.t(), x_ng.t()).t()  # best of these three

        # option 3: coordinate descent
        # loading_matrix_nk = solve_nnls_coordinate_descent(factor_matrix_kg.t(), x_ng.t()).t()
        
        return loading_matrix_nk
    
        # believe the fit loading param
        # loading_matrix_mk: torch.Tensor = model.loading_matrix_mk  # type: ignore[assignment]
        # total_mrna_umis_n1 = x_ng.sum(dim=-1, keepdim=True)
        # return loading_matrix_mk[minibatch_indices_n, :] * total_mrna_umis_n1


    @staticmethod
    def amortized_loadings_model(
        x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor, model: "BayesianNonNegativeMatrixFactorization"
    ) -> dict[str, torch.Tensor]:
        """Generative model for amortized inference"""

        with pyro.plate("obs_plate"):
            # prior on loadings is Dirichlet
            prior_alpha_nk = model.loading_alpha * torch.ones([x_ng.shape[0], model.k_max]).to(x_ng.device)
            loading_matrix_nk = pyro.sample(
                "amortized_loading_prior_nk",
                dist.Dirichlet(prior_alpha_nk).to_event(1),
            )

            # apply ARD regularization, if called for
            if model.use_ard:
                log_prob_sum_ard_reg = ard_regularization(
                    loading_matrix_nk=loading_matrix_nk,
                    scale=1.0,
                )
            else:
                log_prob_sum_ard_reg = torch.tensor(-float("inf")).to(x_ng.device)

            factor_matrix_kg: torch.Tensor = model.factor_matrix_kg  # type: ignore[assignment]

            # apply gene graph regularization, if called for
            if model.use_gene_graph_prior:
                assert isinstance(model.similarity_matrix_gg, torch.Tensor)
                log_prob_sum_gene_graph_reg = gene_graph_regularization(
                    similarity_matrix_gg=model.similarity_matrix_gg,
                    factor_matrix_kg=factor_matrix_kg,
                    gamma_loc=model.null_concordance_loc,
                    gamma_scale=model.null_concordance_scale,
                    scale=1.0,
                )
            else:
                log_prob_sum_gene_graph_reg = torch.tensor(-float("inf")).to(x_ng.device)

            # Compute Poisson rate
            chi_ng = torch.matmul(loading_matrix_nk, factor_matrix_kg)
            cell_size_factors_m = x_ng.sum(dim=-1, keepdim=True)
            lam_ng = chi_ng * cell_size_factors_m

            # Observe data
            likelihood_dist = InferenceStrategies._get_likelihood_distribution(lam_ng, model.likelihood_dist)
            pyro.sample("obs", likelihood_dist, obs=x_ng)  # type: ignore[arg-type]

        return {
            "ard_regularization": log_prob_sum_ard_reg,
            "gene_graph_regularization": log_prob_sum_gene_graph_reg,
        }

    @staticmethod
    def amortized_loadings_guide(
        x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor, model: "BayesianNonNegativeMatrixFactorization"
    ):
        """Variational guide using neural network encoder"""
        if model.loading_encoder is None:
            raise RuntimeError("Encoder not initialized.")

        with pyro.plate("obs_plate"):
            # encode the loadings per cell
            loading_matrix_concentration_nk = model.loading_encoder(x_ng)

            # sample from approximate posterior
            loading_matrix_nk = pyro.sample(
                "amortized_loading_prior_nk",
                dist.Dirichlet(loading_matrix_concentration_nk).to_event(1),
            )

        return loading_matrix_nk

    @staticmethod
    @torch.no_grad()
    def amortized_loadings_get_loadings(
        x_ng: torch.Tensor, minibatch_indices_n: torch.Tensor, model: "BayesianNonNegativeMatrixFactorization"
    ) -> torch.Tensor:
        """Get factor loadings using encoder"""
        if model.loading_encoder is None:
            raise RuntimeError("Encoder not initialized.")

        with torch.no_grad():
            concentration = model.loading_encoder(x_ng)
            return concentration / concentration.sum(dim=-1, keepdim=True)


class BayesianNonNegativeMatrixFactorization(NonNegativeMatrixFactorization):
    """
    Pyro model for optionally-regularized, optionally-amortized Bayesian NMF.

    Args:
        total_n_cells: Total number of cells in the entire anndata
        var_names_g: The variable names schema for the input data validation.
        n_genes: Number of genes
        k_max: Maximum allowed number of 'factors', i.e. gene expression programs
        gene_alpha: Concentration parameter for the Gamma distribution that is used
            to initialize the coefficients of genes in a factor. One value per
            factor. Small values << 1 induce sparsity of genes in factors.
        loading_alpha: Concentration parameter for the Gamma distribution that is
            used to initialize the coefficients of loadings of the factors (in
            each cell). Small values << 1 induce sparsity, but sparsity is not
            necessarily desirable here.
        similarity_matrix_gg: Similarity matrix used as a graph-based prior for
            gene relatedness (only supplied if use_gene_graph_prior is True)
        use_gene_graph_prior: True to apply a regularization to the gene
            programs (factors) so that they are encouraged to become concordant
            with the supplied similarity_matrix_gg
        use_ard: True to use automatic relevance determination to induce sparsity
            over factor loadings, so that some factors become 'irrelevant'.
        encoder_type: The type of encoder to use in ['mlp', 'linear']
        likelihood_dist: The data distribution to use for observations, either
            dist.Poisson (default) or dist.Normal
    """

    def __init__(
        self,
        total_n_cells: int,
        var_names_g: Sequence[str],
        n_genes: int,
        similarity_matrix_gg: Optional[torch.Tensor],
        use_gene_graph_prior: bool,
        k_max: int,
        gene_alpha_k: List[float] | float = 0.1,
        loading_alpha: float = 0.5,
        use_ard: bool = True,
        encoder_type: Optional[str] = None,
        encoder_hidden_dims: List[int] = [512],
        encoder_dropout_rate: float = 0.1,
        likelihood_dist: str = "normal",
        seed: int = 0,
    ):
        super().__init__(var_names_g=var_names_g, k_values=[k_max])
        self.total_n_cells = total_n_cells
        self.cell_ind_lookup: dict[str, int] = {}
        self.g = n_genes
        self.k_max = k_max
        if isinstance(gene_alpha_k, float):
            gene_alpha_k = [gene_alpha_k] * k_max  # list with k_max elements
        self.gene_alpha_k = torch.tensor(gene_alpha_k).expand([k_max]).detach()
        self.loading_alpha = loading_alpha
        self.similarity_matrix_gg = similarity_matrix_gg
        self.use_gene_graph_prior = use_gene_graph_prior
        self.use_ard = use_ard
        self.seed = seed
        self.elbo = pyro.infer.Trace_ELBO()

        match likelihood_dist:
            case "poisson":
                self.likelihood_dist: Type[Union[dist.Poisson, dist.Normal]] = dist.Poisson
            case "normal":
                self.likelihood_dist = dist.Normal
            case _:
                raise ValueError("likelihood_dist must be 'poisson' or 'normal'")

        # Initialize the filter transform for HVGs
        self.transform__filter_to_hvgs = Filter([str(s) for s in var_names_g])

        # properties
        self.initial_factors_kg: torch.Tensor
        self.null_concordance_loc: torch.Tensor
        self.null_concordance_scale: torch.Tensor

        # set up pyro params
        self.factor_matrix_kg = PyroParam(torch.empty(self.k_max, self.g), constraint=constraints.simplex)
        self.loading_matrix_mk = PyroParam(
            torch.empty(self.total_n_cells, self.k_max), constraint=constraints.nonnegative
        )
        self.log_ard_alpha_k = PyroParam(torch.empty(self.k_max), constraint=constraints.less_than(upper_bound=6.0))

        # Only create encoder for amortized strategy
        self.loading_encoder: torch.nn.Module | None = None
        if encoder_type is not None:
            self.loading_encoder = self._build_encoder(n_genes, encoder_type, encoder_hidden_dims, encoder_dropout_rate)
            pyro.module("loadings_encoder", self.loading_encoder)

        # Set up strategy functions
        if encoder_type is not None:
            self.strategy: str = "amortized"
            self.model_fn: Callable = InferenceStrategies.amortized_loadings_model
            self.guide_fn: Callable = InferenceStrategies.amortized_loadings_guide
            self.get_loadings_fn: Callable = InferenceStrategies.amortized_loadings_get_loadings
            logger.info("Using an amortized encoder to estimate factor loadings")
        else:
            self.strategy = "mle"
            self.model_fn = InferenceStrategies.maximum_likelihood_model
            self.guide_fn = InferenceStrategies.maximum_likelihood_guide
            self.get_loadings_fn = InferenceStrategies.maximum_likelihood_get_loadings
            logger.info("Using local latent variables factor loadings")

        self.reset_parameters()

    def _build_encoder(
        self, n_genes: int, encoder_type: str, encoder_hidden_dims: List[int], encoder_dropout_rate: float
    ) -> torch.nn.Module:
        """Build the encoder network for loading inference."""
        if encoder_type == "linear":
            return FullyConnectedNetwork(
                input_dim=n_genes,
                hidden_dims=[],
                output_dim=self.k_max,
                output_activation=Exp(),
                dropout_rate=None,
            )
        elif encoder_type == "mlp":
            return FullyConnectedNetwork(
                input_dim=n_genes,
                hidden_dims=encoder_hidden_dims,
                output_dim=self.k_max,
                hidden_activation=torch.nn.ReLU(),
                output_activation=Exp(),
                use_layer_norm=False,
                use_batch_norm=True,
                dropout_rate=encoder_dropout_rate,
            )
        else:
            raise ValueError(f"encoder_type must be in ['linear', 'mlp'], got {encoder_type}")

    def reset_parameters(self) -> None:
        """
        Reset model parameters.
        """
        torch.manual_seed(self.seed)

        # Initialize factors
        initial_factors = initialize_matrix(
            cols=self.k_max,
            rows=self.g,
            alpha_cols=self.gene_alpha_k,
            simplex=True,
            seed=torch.randint(0, 1_000_000, (1,)),
        ).t()

        self.factor_matrix_kg_unconstrained.data.copy_(
            _unconstrain(
                initial_factors,
                constraint=constraints.simplex,
            )
        )

        # Initialize loadings for ML strategy only
        if self.loading_matrix_mk is not None:
            initial_loadings = initialize_matrix(
                rows=self.total_n_cells,
                cols=self.k_max,
                alpha_cols=torch.tensor(self.loading_alpha),
                simplex=True,
                seed=torch.randint(0, 1_000_000, (1,)),
            )
            self.loading_matrix_mk_unconstrained.data.copy_(
                _unconstrain(initial_loadings, constraint=constraints.nonnegative)
            )

        # Initialize ARD alphas
        if self.use_ard:
            self.log_ard_alpha_k_unconstrained.data.copy_(
                _unconstrain(
                    (np.log(self.k_max) * torch.ones(self.k_max)).detach(),
                    constraint=constraints.less_than(upper_bound=6.0),
                ).detach()
            )

        # set up the initial factors based on the gene graph prior
        if self.use_gene_graph_prior:
            assert self.similarity_matrix_gg is not None, (
                "use_gene_graph_prior is True: you must supply similarity_matrix_gg"
            )
            logger.info(
                "Initializing gene expression factors in a manner consistent with the supplied similarity matrix"
            )
            assert isinstance(self.similarity_matrix_gg, torch.Tensor)
            gammas = []
            best_init_factors_kg: torch.Tensor | None = None
            alpha_kg = torch.ones([self.g]).unsqueeze(0) * torch.tensor(self.gene_alpha_k).unsqueeze(1)
            for _ in range(1000):
                # sample a factor matrix
                random_factors_kg = torch.distributions.Dirichlet(alpha_kg).sample()

                # compute the graph prior concordance statistic gamma
                gamma = gene_graph_concordance_statistic(
                    similarity_matrix_gg=self.similarity_matrix_gg,
                    factor_matrix_kg=random_factors_kg,
                ).item()
                gammas.append(gamma)

                # keep track of highest-scoring random factor matrix
                if gamma == max(gammas):
                    best_init_factors_kg = random_factors_kg

            assert isinstance(best_init_factors_kg, torch.Tensor)

            # set these properties
            gamma_vals: torch.Tensor = torch.tensor(gammas)
            self.initial_factors_kg = best_init_factors_kg
            self.null_concordance_loc = gamma_vals.mean().detach()
            self.null_concordance_scale = gamma_vals.std(dim=-1).detach()
            # self.initial_factors_kg = torch.nn.Parameter(
            #     best_init_factors_kg,
            #     requires_grad=False,
            # )
            # self.null_concordance_loc = torch.nn.Parameter(
            #     gamma_vals.mean(),
            #     requires_grad=False,
            # )
            # self.null_concordance_scale = torch.nn.Parameter(
            #     gamma_vals.std(dim=-1),
            #     requires_grad=False,
            # )

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        obs_names_n: np.ndarray,
    ) -> dict[str, torch.Tensor | None]:
        """
        Forward pass for the Bayesian NMF model.

        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.
            obs_names_n: Unique identifier of each cell.

        Returns:
            A dictionary with the ELBO loss value.
        """
        # Validate inputs
        from cellarium.ml.utilities.testing import (
            assert_arrays_equal,
            assert_columns_and_array_lengths_equal,
        )

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        # Create minibatch indices for this batch
        # if any obs_names_n value is not a key of self.cell_ind_lookup, add it
        # and assign it an unused integer index
        new_obs_names = set(obs_names_n) - set(self.cell_ind_lookup.keys())
        if len(new_obs_names) > 0:
            available_inds = set(range(self.total_n_cells)) - set(self.cell_ind_lookup.values())
            self.cell_ind_lookup |= dict(zip(new_obs_names, available_inds))
        minibatch_indices_n = torch.tensor(
            [self.cell_ind_lookup[obs_name] for obs_name in obs_names_n],
            dtype=torch.long,
            device=x_ng.device,
        )

        # Compute the differentiable ELBO loss
        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng, minibatch_indices_n)

        return {"loss": loss}

    def model(self, x: torch.FloatTensor, minibatch_indices_n: torch.Tensor):
        return self.model_fn(x_ng=x, minibatch_indices_n=minibatch_indices_n, model=self)

    def guide(self, x: torch.FloatTensor, minibatch_indices_n: torch.Tensor):
        return self.guide_fn(x_ng=x, minibatch_indices_n=minibatch_indices_n, model=self)

    def _get_ard_cutoff_logic(
        self,
        log_alpha_ard_cutoff: float | None,
    ) -> torch.Tensor | EllipsisType:
        """
        Get a logical vector that can be used to index the K dimension of the
        factorized matrices.

        Args:
            log_alpha_ard_cutoff: Numerical cutoff for determining whether a factor should be
                included, based on automatic relevance determination.

        Returns:
            relevant_dimension_logic: Logical vector where True denotes relevant factors.
        """
        if log_alpha_ard_cutoff is None:
            return Ellipsis
        log_ard_alpha_k = self.get_ard_alphas()
        if log_ard_alpha_k is None:
            return Ellipsis
        else:
            return log_ard_alpha_k < log_alpha_ard_cutoff

    def get_ard_alphas(self) -> Optional[torch.Tensor]:
        """
        Get the learned vector of alphas for automatic relevance determination.

        Returns
        -------
        log_ard_alpha_k
            If ARD was used, the k_max-length vector of learned coefficients,
            where large values indicate that a given k is not relevant.
        """
        if self.use_ard:
            log_ard_alpha_k: torch.Tensor = self.log_ard_alpha_k  # type: ignore[assignment]
            return log_ard_alpha_k.detach()
        else:
            return None

    @torch.no_grad()
    def get_factors(self, log_alpha_ard_cutoff: Optional[float] = None) -> torch.Tensor:
        factor_matrix_kg: torch.Tensor = self.factor_matrix_kg  # type: ignore[assignment]
        logic = self._get_ard_cutoff_logic(log_alpha_ard_cutoff=log_alpha_ard_cutoff)
        factor_matrix_kg = factor_matrix_kg[logic, :]
        return factor_matrix_kg

    @torch.no_grad()
    def get_factor_loadings(self, dataloader=None, log_alpha_ard_cutoff: Optional[float] = None):
        logic = self._get_ard_cutoff_logic(log_alpha_ard_cutoff=log_alpha_ard_cutoff)

        if self.strategy == "mle":
            # Maximum likelihood strategy
            loading_matrix_nk: torch.Tensor = self.loading_matrix_nk  # type: ignore[assignment]
            loading_matrix_nk = loading_matrix_nk[:, logic]
        elif self.strategy == "amortized":
            # Amortized strategy
            loading_matrices_mk = []
            for tensors in dataloader:
                x_ng = tensors["x_ng"]
                minibatch_indices_n = torch.arange(x_ng.shape[0], dtype=torch.long)
                loading_matrix_mk = self.get_loadings_fn(x_ng, minibatch_indices_n, self)
                loading_matrix_mk = loading_matrix_mk[:, logic]
                # loading_matrix_mk = loading_matrix_mk / loading_matrix_mk.sum(dim=-1, keepdim=True)
                loading_matrices_mk.append(loading_matrix_mk)
            loading_matrix_nk = torch.cat(loading_matrices_mk, dim=0)
        else:
            raise ValueError("only allowed strategies are 'mle' and 'amortized': ", self.strategy)
        return loading_matrix_nk

    def _get_loading(self, batch):
        x_ng = batch["x_ng"].to(self.device)
        assert self.loading_encoder is not None, (
            "You are trying to get amortized loadings, but loading encoder is not initialized"
        )
        dirichlet_alphas_nk = self.loading_encoder(x_ng)
        return dirichlet_alphas_nk

    # @property
    # def device(self):
    #     if self._device is None:
    #         if isinstance(self.strategy, AmortizedLoadingsStrategy) and self.strategy.loading_encoder is not None:
    #             devices = list({p.device for p in self.strategy.loading_encoder.parameters()})
    #             if len(devices) > 1:
    #                 raise RuntimeError(
    #                     f"Encoder {str(self.strategy.loading_encoder)} has parameters on more than one device"
    #                 )
    #             else:
    #                 device = devices[0]
    #             self._device = device
    #     return self._device

    @property
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """
        Return the learned factors for each k value.

        Returns:
            Dictionary mapping k -> factor tensor of shape (r, k, g) where:
            - r is number of replicates (1 for Bayesian NMF)
            - k is number of factors
            - g is number of genes
        """
        # For Bayesian NMF, we have a single set of factors, so r=1
        factors = self.get_factors()  # shape (k, g)
        return {self.k_max: factors.unsqueeze(0)}  # shape (1, k, g)

    def infer_loadings(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
        k: int,
        normalize: bool = True,
        obs_names_n: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Infer the loadings of each program for the input count matrix.

        Args:
            x_ng: Gene counts matrix
            var_names_g: Variable names
            obs_names_n: Cell unique names
            consensus_factors: Consensus factors from consensus computation
            k: Number of factors
            normalize: Whether to normalize loadings

        Returns:
            Loadings tensor of shape (n, k)
        """
        assert obs_names_n is not None, "Must provide obs_names_n"

        # Use the transform to filter to the right genes
        x_filtered_ng = self.transform__filter_to_hvgs(x_ng, var_names_g)["x_ng"]

        # Get the consensus factors
        D_kg = consensus_factors[k]["consensus_D_kg"]
        assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"

        minibatch_indices_n = torch.tensor(
            [self.cell_ind_lookup[obs_name] for obs_name in obs_names_n],
            dtype=torch.long,
            device=x_ng.device,
        )

        # # Create a batch with the filtered data
        # batch = {"x_ng": x_filtered_ng}

        loading_matrix_nk = self.get_loadings_fn(x_filtered_ng, minibatch_indices_n, self)

        if normalize:
            loading_matrix_nk = loading_matrix_nk / loading_matrix_nk.sum(dim=-1, keepdim=True)

        return loading_matrix_nk

    def reconstruction_error(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
    ) -> dict[int, float]:
        """
        Compute reconstruction error for each k value.

        Args:
            x_ng: Gene counts matrix
            var_names_g: Variable names
            consensus_factors: Consensus factors from consensus computation

        Returns:
            Dictionary mapping k -> reconstruction error
        """
        x_filtered_ng = self.transform__filter_to_hvgs(x_ng, var_names_g)["x_ng"]

        rec_error = {}
        for k in consensus_factors.keys():
            D_kg = consensus_factors[k]["consensus_D_kg"]
            assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"

            # Get loadings for this data
            loadings_nk = self.infer_loadings(
                x_ng=x_ng,
                var_names_g=var_names_g,
                consensus_factors=consensus_factors,
                k=k,
                normalize=False,
            )

            # Compute reconstruction error
            reconstruction_ng = torch.matmul(loadings_nk, D_kg)
            error = torch.nn.functional.mse_loss(x_filtered_ng, reconstruction_ng, reduction="sum")
            rec_error[k] = error.item()

        return rec_error
