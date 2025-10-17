# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from types import EllipsisType
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from cellarium.ml.models.nmf import NonNegativeMatrixFactorization

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


def initialize_matrix(rows: int, cols: int, alpha: Union[float, torch.Tensor], simplex: bool = True):
    """
    Initialize a parameter matrix for NMF: [rows, cols]

    Args:
        rows: Total number of cells in the entire dataset
        cols: Maximum number of factors to model
        alpha: Concentration of gamma distribution used to sample values. Small
            values < 1 lead to sparsity.
        simplex: True to initialize values so that matrix.sum(dim=-1) = 1

    Returns:
        matrix: [rows, cols] matrix of initial values for NMF
    """

    # Bayesian NMF, Cemgil 2009... ish
    if isinstance(alpha, torch.Tensor) and (len(alpha) > 1):
        matrix = dist.Gamma(concentration=alpha, rate=alpha / cols).sample(torch.Size([cols])).t()
    else:
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()  # having it in a tensor messes up sample size
        matrix = dist.Gamma(concentration=alpha, rate=alpha / cols).sample(torch.Size([rows, cols]))
    if simplex:
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
    return matrix


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
    k_max = loading_matrix_nk.shape[-1]
    log_ard_alpha_k = pyro.param(
        "log_ard_alpha_k",
        np.log(k_max) * torch.ones(k_max).to(loading_matrix_nk.device),
        constraint=constraints.less_than(upper_bound=6.0),
    )
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


class MaximumLikelihood:
    """
    Maximum likelihood inference - i.e. 'typical' NMF, but with Poisson
    likelihood. This is said to match the Lee and Seung 2000 (NeurIPS) algorithm
    with the KL divergence loss function (Eqn 3). See Cemgil 2009, Bayesian
    inference for NMF models.
    """

    loading_encoder = None

    @staticmethod
    def model(
        x_ng: torch.Tensor,
        minibatch_indices: torch.LongTensor,
        n_cells: int,
        k_max: int,
        gene_alpha: torch.Tensor,
        loading_alpha: float,
        use_ard: bool,
        obs: bool = True,
        factor_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Generative model p(x|z)p(z) for non-negative matrix factorization

        Args:
            x_ng: Count data minibatch, rows are cells and columns are genes
            minibatch_indices: Indices of each cell in this minibatch (used to index full dataset)
            n_cells: Total number of cells in the full dataset
            k_max: Number of factors in the model
            gene_alpha: Gamma concentration parameter controlling gene sparsity for each factor.
                Small value < 1 initializes factors as sparse in gene space.
            loading_alpha: Gamma concentration parameter controlling sparsity of
                usage of each factor by each cell. Large value >> 1 initialize
                the factor loadings as approximately uniform.
            use_ard: True to use automatic relevance determination to induce
                sparsity over factor loadings, so that some factors become
                'irrelevant'.
            obs: True to observe data (simulation), False to simulate
            factor_matrix: Used for simulations only

        Returns:
            out: Dictionary of tensors relevant for the generative model
        """

        # factors and loadings
        factor_matrix_kg = pyro.param(
            "factor_matrix_kg",
            (
                initialize_matrix(rows=k_max, cols=x_ng.shape[-1], alpha=gene_alpha, simplex=True)
                if factor_matrix is None
                else factor_matrix
            ),
            constraint=constraints.simplex,  # in the g dimension
        )
        loading_matrix_mk = pyro.param(
            "loading_matrix_mk",
            initialize_matrix(rows=n_cells, cols=k_max, alpha=loading_alpha, simplex=True),
            constraint=constraints.simplex,  # in the k dimension
        )

        # subset the loading matrix to the cells in this minibatch
        loading_matrix_nk = loading_matrix_mk[minibatch_indices, :]

        # apply ARD regularization, if called for
        if use_ard:
            log_prob_sum_ard_reg = ard_regularization(loading_matrix_nk=loading_matrix_nk)
        else:
            log_prob_sum_ard_reg = torch.tensor(-float("inf")).to(x_ng.device)

        # the normalized poisson rate, where chi.sum(dim=-1) = 1
        # print(f'loading_matrix_mk.shape is {loading_matrix_mk.shape}')
        # print(f'factor_matrix_kg.shape is {factor_matrix_kg.shape}')
        chi_ng = torch.matmul(loading_matrix_nk, factor_matrix_kg)
        # print(f'chi_mg.shape is {chi_mg.shape}')

        # the poisson rate
        cell_size_factors_m = x_ng.sum(dim=-1, keepdim=True)
        lam_ng = chi_ng * cell_size_factors_m

        # compare to observed count data
        with pyro.plate("obs_plate", size=x_ng.shape[0]):
            # print(f'lam_ng.shape is {lam_ng.shape}')
            # print(f'x.shape is {x.shape}')
            c_ng = pyro.sample(
                "obs",
                dist.Poisson(rate=lam_ng + 1e-10).to_event(1),
                obs=x_ng if obs else None,
            )

        return {
            "counts_mg": c_ng,
            "chi_mg": chi_ng,
            "loadings_mk": loading_matrix_nk,
            "factors_kg": factor_matrix_kg,
            "ard_regularization": log_prob_sum_ard_reg,
        }

    @staticmethod
    def guide(*args, **kwargs):
        """
        Maximum likelihood inference is just... do nothing... not Bayesian
        """
        pass


class AmortizedLoadings:
    """
    Infer the loadings via amortization using an encoder neural network.
    """

    loading_encoder: torch.nn.Module

    @classmethod
    def initialize_loading_encoder(cls, n_genes: int, k: int, encoder_type: str):
        if encoder_type == "linear":
            cls.loading_encoder = FullyConnectedNetwork(
                input_dim=n_genes,
                hidden_dims=[],
                output_dim=k,
                output_activation=Exp(),
                dropout_rate=None,
            )
        elif encoder_type == "mlp":
            cls.loading_encoder = FullyConnectedNetwork(
                input_dim=n_genes,
                hidden_dims=[512],
                output_dim=k,
                hidden_activation=torch.nn.ReLU(),
                output_activation=Exp(),
                use_layer_norm=False,
                use_batch_norm=True,
                dropout_rate=None,
            )
        else:
            raise ValueError("encoder_type must be in ['linear', 'mlp']")

    @staticmethod
    def model(
        x_ng: torch.Tensor,
        k_max: int,
        gene_alpha: torch.Tensor,
        loading_alpha: float,
        similarity_matrix_gg: Optional[torch.Tensor],
        gamma_loc: Optional[torch.Tensor],
        gamma_scale: Optional[torch.Tensor],
        use_ard: bool,
        use_gene_graph_prior: bool,
        obs: bool = True,
        factor_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Generative model p(x|z)p(z) for non-negative matrix factorization

        Args:
            x_ng: Count data minibatch, rows are cells and columns are genes
            k_max: Number of factors in the model
            gene_alpha: Gamma concentration parameter controlling gene sparsity
                for each factor.
                Small value < 1 initializes factors as sparse in gene space.
            loading_alpha: Gamma concentration parameter controlling sparsity of
                usage of each factor by each cell. Large value >> 1 initialize
                the factor loadings as approximately uniform.
            similarity_matrix_gg: Similarity matrix used as a graph-based prior for
                gene relatedness (only supplied if use_gene_graph_prior is True)
            gamma_loc: Null prior mean for the graph concordance statistic gamma
            gamma_scale: Null prior std for the graph concordance statistic gamma
            use_gene_graph_prior: True to apply a regularization to the gene
                programs (factors) so that they are encouraged to become concordant
                with the supplied similarity_matrix_gg
            use_ard: True to use automatic relevance determination to induce
                sparsity over factor loadings, so that some factors become
                'irrelevant'.
            obs: True to observe data (simulation), False to simulate
            factor_matrix: Used for simulations, or for initialization based on the
                gene graph prior

        Returns:
            out: Dictionary of tensors relevant for the generative model
        """

        # factors
        factor_matrix_kg = pyro.param(
            "factor_matrix_kg",
            initialize_matrix(rows=k_max, cols=x_ng.shape[-1], alpha=gene_alpha, simplex=True)
            if (factor_matrix is None)
            else factor_matrix,
            constraint=constraints.simplex,  # in the g dimension
        )

        with pyro.plate("obs_plate"):
            # prior on loadings is Dirichlet
            prior_alpha_nk = loading_alpha * torch.ones([x_ng.shape[0], k_max]).to(x_ng.device)
            loading_matrix_nk = pyro.sample(
                "loading_matrix_nk",
                dist.Dirichlet(prior_alpha_nk).to_event(1),
            )

            # apply ARD regularization, if called for
            if use_ard:
                log_prob_sum_ard_reg = ard_regularization(
                    loading_matrix_nk=loading_matrix_nk,
                    scale=1.0,
                )
            else:
                log_prob_sum_ard_reg = torch.tensor(-float("inf")).to(x_ng.device)

            # apply gene graph regularization, if called for
            if use_gene_graph_prior:
                assert isinstance(similarity_matrix_gg, torch.Tensor), (
                    "use_gene_graph_prior is True: you must supply similarity_matrix_gg"
                )
                assert isinstance(gamma_loc, torch.Tensor), "use_gene_graph_prior is True: you must supply gamma_loc"
                assert isinstance(gamma_scale, torch.Tensor), (
                    "use_gene_graph_prior is True: you must supply gamma_scale"
                )
                log_prob_sum_gene_graph_reg = gene_graph_regularization(
                    similarity_matrix_gg=similarity_matrix_gg,
                    factor_matrix_kg=factor_matrix_kg,
                    gamma_loc=gamma_loc,
                    gamma_scale=gamma_scale,
                    scale=10.0,
                )
            else:
                log_prob_sum_gene_graph_reg = torch.tensor(-float("inf")).to(x_ng.device)

            # the normalized poisson rate, where chi.sum(dim=-1) = 1
            # print(f'loading_matrix_mk.shape is {loading_matrix_mk.shape}')
            # print(f'factor_matrix_kg.shape is {factor_matrix_kg.shape}')
            chi_ng = torch.matmul(loading_matrix_nk, factor_matrix_kg)
            # print(f'chi_mg.shape is {chi_mg.shape}')

            # the poisson rate
            cell_size_factors_m = x_ng.sum(dim=-1, keepdim=True)
            lam_ng = chi_ng * cell_size_factors_m

            # compare to observed count data
            c_ng = pyro.sample(
                "obs",
                dist.Poisson(rate=lam_ng + 1e-10).to_event(1),
                obs=x_ng if obs else None,
            )

        return {
            "counts_mg": c_ng,
            "chi_mg": chi_ng,
            "loadings_mk": loading_matrix_nk,
            "factors_kg": factor_matrix_kg,
            "ard_regularization": log_prob_sum_ard_reg,
            "gene_graph_regularization": log_prob_sum_gene_graph_reg,
        }

    @classmethod
    def guide(cls, x_ng: torch.Tensor, **kwargs):
        """
        Amortize the loadings via a neural network.
        """

        # register PyTorch modules with Pyro
        pyro.module("loadings_encoder", cls.loading_encoder)

        with pyro.plate("obs_plate"):
            # encode the loadings per cell
            loading_matrix_concentration_nk = cls.loading_encoder(x_ng)

            # subset the loading matrix to the cells in this minibatch
            loading_matrix_nk = pyro.sample(
                "loading_matrix_nk",
                dist.Dirichlet(loading_matrix_concentration_nk).to_event(1),
            )

        return loading_matrix_nk


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
        encoder_type: The type of encoder to use in ['mlp', 'transformer']
    """

    def __init__(
        self,
        total_n_cells: int,
        var_names_g: Sequence[str],
        n_genes: int,
        similarity_matrix_gg: Optional[torch.Tensor],
        use_gene_graph_prior: bool,
        k_max: int = 20,
        gene_alpha_k: List[float] = [0.1] * 20,
        loading_alpha: float = 0.5,
        use_ard: bool = True,
        encoder_type: Optional[str] = None,
    ):
        super().__init__(var_names_g=var_names_g, k_values=[k_max])
        self.n = total_n_cells
        self.g = n_genes
        self.k_max = k_max
        self.gene_alpha_k = gene_alpha_k
        self.loading_alpha = loading_alpha
        self.similarity_matrix_gg = similarity_matrix_gg
        self.use_gene_graph_prior = use_gene_graph_prior
        self.use_ard = use_ard

        # properties
        self._device: torch.device | None = None
        self.initial_factors_kg: torch.nn.Parameter | None = None
        self.null_concordance_loc: torch.nn.Parameter | None = None
        self.null_concordance_scale: torch.nn.Parameter | None = None

        # set up the initial factors based on the gene graph prior
        if use_gene_graph_prior:
            assert similarity_matrix_gg is not None, (
                "use_gene_graph_prior is True: you must supply similarity_matrix_gg"
            )
            logger.info(
                "Initializing gene expression factors in a manner consistent with the supplied similarity matrix"
            )
            self._sample_null_graph_concordance_stats_and_initialize_factors()

        # choose model
        if encoder_type is not None:
            AmortizedLoadings.initialize_loading_encoder(
                n_genes=n_genes,
                k=k_max,
                encoder_type=encoder_type,
            )
            self.model_class: type[AmortizedLoadings | MaximumLikelihood] = AmortizedLoadings
            logger.info("Using an amortized encoder to estimate factor loadings")
        else:
            self.model_class = MaximumLikelihood
            logger.info("Using local latent variables factor loadings")
            logger.info("Not using gene graph prior, as it is not implemented here")

    def model(self, x: torch.FloatTensor, minibatch_indices: torch.LongTensor):
        return self.model_class.model(
            x=x,
            minibatch_indices=minibatch_indices,
            **self._get_kwargs(),
        )

    def guide(self, x: torch.FloatTensor, minibatch_indices: torch.LongTensor):
        return self.model_class.guide(
            x=x,
            minibatch_indices=minibatch_indices,
            **self._get_kwargs(),
        )

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
        try:
            return pyro.param("log_ard_alpha_k").detach()
        except KeyError:
            logger.warning("Attempted to access log_ard_alpha_k, but it is not in the Pyro param store")
            return None

    @torch.no_grad()
    def get_factors(self, log_alpha_ard_cutoff: Optional[float] = None) -> torch.Tensor:
        factor_matrix_kg = pyro.param("factor_matrix_kg").detach()
        logic = self._get_ard_cutoff_logic(log_alpha_ard_cutoff=log_alpha_ard_cutoff)
        factor_matrix_kg = factor_matrix_kg[logic, :]
        return factor_matrix_kg

    @torch.no_grad()
    def get_factor_loadings(self, dataloader=None, log_alpha_ard_cutoff: Optional[float] = None):
        logic = self._get_ard_cutoff_logic(log_alpha_ard_cutoff=log_alpha_ard_cutoff)

        if self.model_class == MaximumLikelihood:
            loading_matrix_nk = pyro.param("loading_matrix_nk").detach()
            loading_matrix_nk = loading_matrix_nk[:, logic]
        else:
            loading_matrices_mk = []
            for tensors in dataloader:
                dirichlet_alphas_mk = self._get_loading(tensors)
                loading_matrix_mk = dist.Dirichlet(dirichlet_alphas_mk).mean
                loading_matrix_mk = loading_matrix_mk[:, logic]
                loading_matrix_mk = loading_matrix_mk / loading_matrix_mk.sum(dim=-1, keepdim=True)
                loading_matrices_mk.append(loading_matrix_mk)
            loading_matrix_nk = torch.cat(loading_matrices_mk, dim=0)
        return loading_matrix_nk

    def _get_loading(self, batch):
        x_ng = batch["x_ng"].to(self.device)
        assert isinstance(self.model_class.loading_encoder, torch.nn.Module), (
            "You are trying to get amortized loadings, but the model_class has no loading_encoder"
        )
        dirichlet_alphas_nk = self.model_class.loading_encoder(x_ng)
        return dirichlet_alphas_nk

    def _sample_null_graph_concordance_stats_and_initialize_factors(
        self,
        n: int = 1000,
    ):
        """Sample from the null distribution of the graph prior concordance
        statistic gamma (once and for all).

        Args:
            n: Number of random samples to draw

        Returns:
            None
            (Sets null_concordance_loc, null_concordance_scale, initial_factors_kg)
        """
        assert isinstance(self.similarity_matrix_gg, torch.Tensor)
        gammas = []
        best_init_factors_kg: torch.Tensor | None = None
        alpha_kg = torch.ones([self.g]).unsqueeze(0) * torch.tensor(self.gene_alpha_k).unsqueeze(1)
        for _ in range(n):
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
        self.initial_factors_kg = torch.nn.Parameter(
            best_init_factors_kg,
            requires_grad=False,
        )
        self.null_concordance_loc = torch.nn.Parameter(
            gamma_vals.mean(),
            requires_grad=False,
        )
        self.null_concordance_scale = torch.nn.Parameter(
            gamma_vals.std(dim=-1),
            requires_grad=False,
        )

    @property
    def device(self):
        if self._device is None:
            if isinstance(self.model_class.loading_encoder, torch.nn.Module):
                devices = list({p.device for p in self.model_class.loading_encoder.parameters()})
                if len(devices) > 1:
                    raise RuntimeError(
                        f"Encoder {str(self.model_class.loading_encoder)} has parameters on more than one device"
                    )
                else:
                    device = devices[0]
                self._device = device
        return self._device
