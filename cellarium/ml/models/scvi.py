# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Flexible modified version of single-cell variational inference (scVI) re-implemented in Cellarium ML."""

from typing import Literal, Sequence
import importlib

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Poisson
from torch.distributions import kl_divergence as kl

from cellarium.ml.distributions import NegativeBinomial
from cellarium.ml.models.common.nn import DressedLayer, FullyConnectedLinear
from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


def class_from_class_path(class_path: str):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ref = getattr(module, class_name)
    return class_ref


def instantiate_from_class_path(class_path, *args, **kwargs):
    class_ = class_from_class_path(class_path)
    return class_(*args, **kwargs)


def weights_init(m):
    if isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Linear) or isinstance(m, LinearWithBatch):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class LinearWithBatch(torch.nn.Linear):
    """A `torch.nn.Linear` layer where batch indices are given as input to the forward pass.

    Args:
        in_features: passed to `torch.nn.Linear`
        out_features: passed to `torch.nn.Linear`
        n_batch: the dimensionality of the batch representation
        categorical_covariate_dimensions: a list of integers containing the number of categories for each categorical covariate
        batch_to_bias_hidden_layers: a list of hidden layer sizes for the batch-to-bias decoder
        bias: passed to `torch.nn.Linear` (True is like the scvi-tools implementation)
        batch_to_bias_dressing_init_kwargs: a dictionary of keyword arguments to pass to the `DressedLayer` constructor
    """

    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            n_batch: int, 
            batch_to_bias_hidden_layers: list[int], 
            categorical_covariate_dimensions: list[int] = [],
            bias: bool = True,
            batch_to_bias_dressing_init_kwargs: dict[str, any] = {},
        ):
        super().__init__(in_features, out_features, bias=bias)
        self.bias_decoder = FullyConnectedLinear(
            in_features=n_batch + sum(categorical_covariate_dimensions),
            out_features=out_features,
            n_hidden=batch_to_bias_hidden_layers,
            dressing_init_kwargs=batch_to_bias_dressing_init_kwargs,
        )
    
    def compute_bias(self, batch_nb: torch.Tensor, categorical_covariate_np: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns the bias given batch representations.

        Args:
            batch_nb: a tensor of batch representations (could be one-hot) of shape (n, batch_latent_dim)
            categorical_covariates_nd: a tensor of categorical covariates of shape (n, sum(n_categories_per_covariate))

        Returns:
            a tensor of shape (n, out_features)
        """
        if categorical_covariate_np is None:
            return self.bias_decoder(batch_nb)
        else:
            return self.bias_decoder(torch.cat([batch_nb, categorical_covariate_np], dim=-1))

    def forward(self, x_ng: torch.Tensor, batch_nb: torch.Tensor, categorical_covariate_np: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes the forward pass of the layer as
        out = x @ self.weight.T + self.bias + bias

        where bias is computed as
        bias = bias_encoder(batch)

        Args:
            x_ng: a tensor of shape (n, in_features)
            batch_nb: a tensor of batch indices of shape (n, batch_latent_dim)
            categorical_covariates_nd: a tensor of categorical covariates of shape (n, sum(n_categories_per_covariate))
        """
        return super().forward(x_ng) + self.compute_bias(batch_nb=batch_nb, categorical_covariate_np=categorical_covariate_np)


class FullyConnectedWithBatchArchitecture(torch.nn.Module):
    """
    Fully connected block of layers (can be empty) that can include LinearWithBatch layers. 
    The forward pass takes per-cell batches.

    Args:
        in_features: The dimensionality of the input
        layers: A list of dictionaries, each containing the following keys:
            * ``class_path``: the class path of the layer to use
            * ``init_args``: a dictionary of keyword arguments to pass to the layer's constructor
                - must contain "out_features"
    """

    def __init__(
        self,
        in_features: int, 
        layers: list[dict],
    ) -> tuple[torch.nn.ModuleList, int]:
        super().__init__()
        for layer in layers:
            assert "out_features" in layer["init_args"], \
            """
            "out_features" must be specified in init_args for hidden layers, e.g.

            - class_path: cellarium.ml.models.scvi.LinearWithBatch
              init_args:
                out_features: 128
            """
        
        if len(layers) == 0:
            module_list = torch.nn.ModuleList([torch.nn.Identity()])
            out_features = in_features
        else:
            module_list = []
            n_hidden = [layer["init_args"].pop("out_features") for layer in layers]
            for layer, n_in, n_out in zip(layers, [in_features] + n_hidden, n_hidden):
                module_list.append(
                    DressedLayer(
                        instantiate_from_class_path(
                            layer["class_path"], 
                            in_features=n_in, 
                            out_features=n_out, 
                            bias=True,
                            **layer["init_args"],
                        ),
                        **layer["dressing_init_args"],
                    )
                )
            module_list = torch.nn.ModuleList(module_list)
            out_features = module_list[-1].layer.out_features
        self.module_list = module_list
        self.out_features = out_features

    def forward(self, x_ng: torch.Tensor, batch_nb: torch.Tensor, categorical_covariate_np: torch.Tensor | None) -> torch.Tensor:
        x_ = x_ng
        for dressed_layer in self.module_list:
            x_ = (dressed_layer(x_, batch_nb=batch_nb, categorical_covariate_np=categorical_covariate_np) 
                  if (hasattr(dressed_layer, "layer") and isinstance(dressed_layer.layer, LinearWithBatch)) 
                  else dressed_layer(x_))
        return x_


class EncoderSCVI(torch.nn.Module):
    """
    Encode data of ``in_features`` dimensions into a latent space of ``out_features`` dimensions.

    Args:
        in_features: The dimensionality of the input (data space)
        out_features: The dimensionality of the output (latent space)
        hidden_layers: A list of dictionaries, each containing the following keys:
            * ``class_path``: the class path of the layer to use
            * ``init_args``: a dictionary of keyword arguments to pass to the layer's constructor
                - must contain "out_features"
        final_layer: Same as hidden_layers, but for the final layer
        output_bias: If True, the output layer will have a batch-specific bias added 
            (scvi-tools does not include this)
        var_eps: Minimum value for the variance; used for numerical stability
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        hidden_layers: list[dict],
        final_layer: dict,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.fully_connected = FullyConnectedWithBatchArchitecture(in_features, hidden_layers)
        self.mean_encoder = instantiate_from_class_path(
            final_layer["class_path"], 
            in_features=self.fully_connected.out_features, 
            out_features=out_features, 
            bias=final_layer["init_args"].pop("bias", True),
            **final_layer["init_args"],
        )
        self.var_encoder = instantiate_from_class_path(
            final_layer["class_path"], 
            in_features=self.fully_connected.out_features, 
            out_features=out_features, 
            bias=final_layer["init_args"].pop("bias", True),
            **final_layer["init_args"],
        )
        self.mean_encoder_takes_batch = isinstance(self.mean_encoder, LinearWithBatch)
        self.var_eps = var_eps

    def forward(self, x_ng: torch.Tensor, batch_nb: torch.Tensor, categorical_covariate_np: torch.Tensor | None) -> torch.distributions.Distribution:
        q_nh = self.fully_connected(x_ng, batch_nb=batch_nb, categorical_covariate_np=categorical_covariate_np)
        q_mean_nk = self.mean_encoder(q_nh, batch_nb) if self.mean_encoder_takes_batch else self.mean_encoder(q_nh)
        q_var_nk = torch.exp(self.var_encoder(q_nh, batch_nb) if self.mean_encoder_takes_batch else self.var_encoder(q_nh)) + self.var_eps
        return torch.distributions.Normal(q_mean_nk, q_var_nk.sqrt())


class DecoderSCVI(torch.nn.Module):
    """
    Decode data of ``in_features`` latent dimensions into data space of ``out_features`` dimensions.

    Args:
        in_features: The dimensionality of the input (latent space)
        out_features: The dimensionality of the output (data space)
        hidden_layers: A list of dictionaries, each containing the following keys:
            * ``class_path``: the class path of the layer to use
            * ``init_args``: a dictionary of keyword arguments to pass to the layer's constructor
                - must contain "out_features"
        final_layer: Same as hidden_layers, but for the final layer
        dispersion: Granularity at which the overdispersion of the negative binomial distribution is computed
        gene_likelihood: Distribution to use for reconstruction in the generative process
        scale_activation: Activation layer to use to compute normalized counts (before multiplying by library size)
        final_additive_bias: If True, the final layer will have a batch-specific bias added after the activation.
            If final_layer is a LinearWithBatch layer and final_additive_bias is True, the last layer of the decoder 
            will act as a batch-specific affine transformation.
        eps: Numerical stability factor added to mean and inverse overdispersion of negative binomial
    """
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        hidden_layers: list[dict],
        final_layer: dict,
        n_batch: int,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        final_additive_bias: bool = False,
        n_cats_per_cov: list[int] | None = None,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.eps = eps
        self.n_batch = n_batch
        self.n_cats_per_cov = n_cats_per_cov
        if gene_likelihood == "zinb":
            raise NotImplementedError("Zero-inflated negative binomial not yet implemented")
        self.gene_likelihood = gene_likelihood
        self.fully_connected = FullyConnectedWithBatchArchitecture(in_features, hidden_layers)
        self.inverse_overdispersion_decoder = (
            torch.nn.Linear(self.fully_connected.out_features, out_features) 
            if ((gene_likelihood != "poisson") and (dispersion == "gene-cell")) 
            else None
        )
        self.dropout_decoder = (
            torch.nn.Linear(self.fully_connected.out_features, out_features) 
            if (gene_likelihood == "zinb") 
            else None
        )
        self.normalized_count_decoder = instantiate_from_class_path(
            final_layer["class_path"], 
            in_features=self.fully_connected.out_features, 
            out_features=out_features, 
            bias=final_layer["init_args"].pop("bias", True),
            **final_layer["init_args"],
        )
        self.count_decoder_takes_batch = isinstance(self.normalized_count_decoder, LinearWithBatch)
        self.normalized_count_activation = torch.nn.Softmax(dim=-1) if (scale_activation == "softmax") else torch.nn.Softplus()
        self.final_additive_bias = final_additive_bias
        if self.final_additive_bias:
            self.final_additive_bias_layer = torch.nn.Sequential(
                FullyConnectedLinear(
                    in_features=self.n_batch + sum(self.n_cats_per_cov),
                    out_features=out_features,
                    n_hidden=[],
                    dressing_init_kwargs={},
                ),
                torch.nn.ReLU(),
            )
        else:
            self.final_additive_bias_layer = None

    def forward(
        self, 
        z_nk: torch.Tensor, 
        batch_nb: torch.Tensor, 
        inverse_overdispersion: torch.Tensor | None, 
        library_size_n1: torch.Tensor,
        categorical_covariate_np: torch.Tensor | None = None,
    ) -> torch.distributions.Distribution:
        
        # bulk of the network
        q_nh = self.fully_connected(z_nk, batch_nb=batch_nb, categorical_covariate_np=categorical_covariate_np)

        # mean counts
        unnormalized_chi_ng = (self.normalized_count_decoder(q_nh, batch_nb=batch_nb, categorical_covariate_np=categorical_covariate_np) 
                               if self.count_decoder_takes_batch 
                               else self.normalized_count_decoder(q_nh))
        chi_ng = self.normalized_count_activation(unnormalized_chi_ng)
        if self.final_additive_bias:
            count_mean_ng = torch.exp(library_size_n1) * chi_ng + self.final_additive_bias_layer(
                torch.cat([batch_nb, categorical_covariate_np], dim=-1)
                if categorical_covariate_np is not None
                else batch_nb
            )
        else:
            count_mean_ng = torch.exp(library_size_n1) * chi_ng

        # optional inverse overdispersion per cell
        if inverse_overdispersion is None:
            assert self.inverse_overdispersion_decoder is not None, \
                "inverse_overdispersion must be provided when not using Poisson or gene-cell dispersion"
            inverse_overdispersion = self.inverse_overdispersion_decoder(q_nh).exp()

        # construct the count distribution
        match self.gene_likelihood:
            case "nb":
                dist = NegativeBinomial(count_mean_ng + self.eps, inverse_overdispersion + self.eps)
            case "poisson":
                dist = Poisson(count_mean_ng + self.eps)
            case "zinb":
                raise NotImplementedError("ZINB is not currently implemented")
                # dist = ZeroInflatedNegativeBinomial(count_mean_ng, inverse_overdispersion, self.dropout_decoder(q_nh))

        return dist


class SingleCellVariationalInference(CellariumModel, PredictMixin):
    """
    Flexible version of single-cell variational inference (scVI) [1] re-implemented in Cellarium ML.

    **References:**

    1. `Deep generative modeling for single-cell transcriptomics (Lopez et al.)
       <https://www.nature.com/articles/s41592-018-0229-2>`_.

    Args:
        var_names_g: The variable names schema for the input data validation.
        encoder: Dict specifying the encoder configuration.
        decoder: Dict specifying the decoder configuration.
        n_latent: Dimension of the latent space.
        n_batch: Number of total batches in the dataset.
        batch_bais_sampled: True to sample the batch-specific biases from their distributions.
        n_continuous_cov: Number of continuous covariates.
        n_cats_per_cov: A list of integers containing the number of categories for each categorical covariate.
        dropout_rate: Dropout rate for hidden units in the encoder only.
        dispersion: Flexibility of the dispersion parameter when ``gene_likelihood`` is either ``"nb"`` or
            ``"zinb"``. One of the following:
                * ``"gene"``: parameter is constant per gene across cells.
                * ``"gene-batch"``: parameter is constant per gene per batch.
                * ``"gene-label"``: parameter is constant per gene per label.
                * ``"gene-cell"``: parameter is constant per gene per cell.
        log_variational: If ``True``, use :func:`~torch.log1p` on input data before encoding for numerical stability
            (not normalization).
        gene_likelihood: Distribution to use for reconstruction in the generative process. One of the following:
                * ``"nb"``: :class:`~scvi.distributions.NegativeBinomial`.
                * ``"zinb"``: :class:`~scvi.distributions.ZeroInflatedNegativeBinomial`. (not implemented)
                * ``"poisson"``: :class:`~scvi.distributions.Poisson`.
        latent_distribution: Distribution to use for the latent space. One of the following:
                * ``"normal"``: isotropic normal.
                * ``"ln"``: logistic normal with normal params N(0, 1). (not implemented)
        use_batch_norm: Specifies where to use :class:`~torch.nn.BatchNorm1d` in the model. One of the following:
                * ``"none"``: don't use batch norm in either encoder(s) or decoder.
                * ``"encoder"``: use batch norm only in the encoder(s).
                * ``"decoder"``: use batch norm only in the decoder.
                * ``"both"``: use batch norm in both encoder(s) and decoder.
            Note: if ``use_layer_norm`` is also specified, both will be applied (first
            :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
        use_layer_norm: Specifies where to use :class:`~torch.nn.LayerNorm` in the model. One of the following:
                * ``"none"``: don't use layer norm in either encoder(s) or decoder.
                * ``"encoder"``: use layer norm only in the encoder(s).
                * ``"decoder"``: use layer norm only in the decoder.
                * ``"both"``: use layer norm in both encoder(s) and decoder.
            Note: if ``use_batch_norm`` is also specified, both will be applied (first
            :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
        use_size_factor_key: If ``True``, use the :attr:`~anndata.AnnData.obs` column as defined by the
            ``size_factor_key`` parameter in the model's ``setup_anndata`` method as the scaling
            factor in the mean of the conditional distribution. Takes priority over
            ``use_observed_lib_size``.
        use_observed_lib_size: If ``True``, use the observed library size for RNA as the scaling factor in the mean of the
            conditional distribution. (currently must be ``True``)
        library_log_means: :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of means of the log library sizes that
            parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
            ``use_observed_lib_size`` is ``False``.
        library_log_vars: :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of variances of the log library sizes
            that parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
            ``use_observed_lib_size`` is ``False``.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        encoder: dict[str, list[dict] | dict | bool],
        decoder: dict[str, list[dict] | dict | bool],
        n_batch: int = 0,
        n_latent: int = 10,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] = [],
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        batch_embedded: bool = False,
        batch_representation_sampled: bool = False,
        n_latent_batch: int | None = None,
        batch_kl_weight: float = 0.0,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
    ):
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.n_input = len(self.var_names_g)
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.n_cats_per_cov = n_cats_per_cov
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.batch_embedded = batch_embedded
        self.batch_representation_sampled = batch_representation_sampled
        self.n_latent_batch = n_latent_batch
        assert batch_kl_weight >= 0.0, "batch_kl_weight must be non-negative"
        self.batch_kl_weight = batch_kl_weight

        if n_continuous_cov > 0:
            raise NotImplementedError("Continuous covariates are not yet implemented")
        
        if gene_likelihood == "zinb":
            raise NotImplementedError("Zero-inflated negative binomial not yet implemented")
        
        if not use_observed_lib_size:
            raise NotImplementedError("use_observed_lib_size=False is not yet implemented")
        
        if latent_distribution == "ln":
            raise NotImplementedError("Logistic normal latent distribution is not yet implemented")

        # if you use one-hot and try to specify a different latent batch than n_batch, raise an error
        if (not self.batch_embedded) and (self.n_latent_batch is not None) and (self.n_latent_batch != self.n_batch):
            raise ValueError("n_latent_batch must be equal to n_batch if batch_embedded is False")

        # if n_latent_batch not specified, set it to n_batch
        if self.n_latent_batch is None:
            self.n_latent_batch = self.n_batch  # same dim as one-hot would be

        # handle the embedded batch posterior
        if self.batch_embedded:
            # initialize the means as one-hot, std as 1 (after exp)
            self.batch_representation_mean_bd = torch.nn.Parameter(torch.eye(self.n_batch, self.n_latent_batch))
            self.batch_representation_std_unconstrained_bd = torch.nn.Parameter(torch.zeros(self.n_batch, self.n_latent_batch))
        else:
            self.batch_representation_mean_bd = None
            self.batch_representation_std_unconstrained_bd = None

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, " "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.n_input))
        elif self.dispersion == "gene-label":
            raise NotImplementedError
            # self.px_r = torch.nn.Parameter(torch.randn(self.n_input, n_labels))
        elif self.dispersion == "gene-cell":
            self.px_r = torch.nn.Parameter(torch.zeros(1))  # dummy
        else:
            raise ValueError(
                "dispersion must be one of ['gene', "
                " 'gene-label', 'gene-cell'], but input was "
                "{}".format(self.dispersion)
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # encoder layers
        for layer in encoder["hidden_layers"]:
            if layer["class_path"] == "cellarium.ml.models.scvi.LinearWithBatch":
                layer["init_args"]["n_batch"] = self.n_latent_batch
                layer["init_args"]["categorical_covariate_dimensions"] = n_cats_per_cov
            if "dressing_init_args" not in layer:
                layer["dressing_init_args"] = {}
            layer["dressing_init_args"]["use_batch_norm"] = use_batch_norm_encoder
            layer["dressing_init_args"]["use_layer_norm"] = use_layer_norm_encoder
            layer["dressing_init_args"]["dropout_rate"] = dropout_rate
        if encoder["final_layer"]["class_path"] == "cellarium.ml.models.scvi.LinearWithBatch":
            encoder["final_layer"]["init_args"]["n_batch"] = self.n_latent_batch
            encoder["final_layer"]["init_args"]["categorical_covariate_dimensions"] = n_cats_per_cov

        # decoder layers
        for layer in decoder["hidden_layers"]:
            if layer["class_path"] == "cellarium.ml.models.scvi.LinearWithBatch":
                layer["init_args"]["n_batch"] = self.n_latent_batch
                layer["init_args"]["categorical_covariate_dimensions"] = n_cats_per_cov
            if "dressing_init_args" not in layer:
                layer["dressing_init_args"] = {}
            layer["dressing_init_args"]["use_batch_norm"] = use_batch_norm_decoder
            layer["dressing_init_args"]["use_layer_norm"] = use_layer_norm_decoder
            layer["dressing_init_args"]["dropout_rate"] = 0.0  # scvi-tools does not use dropout in the decoder
        if decoder["final_layer"]["class_path"] == "cellarium.ml.models.scvi.LinearWithBatch":
            decoder["final_layer"]["init_args"]["n_batch"] = self.n_latent_batch
            decoder["final_layer"]["init_args"]["categorical_covariate_dimensions"] = n_cats_per_cov

        self.z_encoder = EncoderSCVI(
            in_features=self.n_input,
            out_features=self.n_latent,
            hidden_layers=encoder["hidden_layers"],
            final_layer=encoder["final_layer"],
        )
        
        self.decoder = DecoderSCVI(
            in_features=self.n_latent,
            out_features=self.n_input,
            hidden_layers=decoder["hidden_layers"],
            final_layer=decoder["final_layer"],
            dispersion=self.dispersion,
            gene_likelihood=self.gene_likelihood,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            final_additive_bias=decoder["final_additive_bias"],
            n_batch=self.n_latent_batch,  # currently used only for the (optional) sizing of the final additive bias layer
            n_cats_per_cov=self.n_cats_per_cov,  # currently used only for the (optional) sizing of the final additive bias layer
        )

        print(self)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            m.apply(weights_init)
        torch.nn.init.normal_(self.px_r, mean=0.0, std=1.0)
        if self.batch_representation_mean_bd is not None:
            with torch.no_grad():
                self.batch_representation_mean_bd.data.copy_(torch.eye(self.n_batch, self.n_latent_batch))
                self.batch_representation_std_unconstrained_bd.data.fill_(0.0)

    def batch_embedding_distribution(self, batch_index_n: torch.Tensor) -> Distribution:
        return Normal(
            self.batch_representation_mean_bd[batch_index_n.long(), :], 
            self.batch_representation_std_unconstrained_bd[batch_index_n.long(), :].exp() + 1e-5,
        )

    def batch_representation_from_batch_index(self, batch_index_n: torch.Tensor) -> torch.Tensor:
        """Compute a batch representation from batch indices.
        
        If self.batch_embedded is False, the batch representation will be one-hot (like scvi-tools)
        If self.batch_embedded is True:
            If self.batch_representation_sampled is True, the batch representation is sampled from a normal distribution
            If self.batch_representation_sampled is False, the batch representation is a point estimate
        
        """
        if not self.batch_embedded:
            batch_nb = torch.nn.functional.one_hot(batch_index_n.squeeze().long(), num_classes=self.n_batch).float()
        else:
            if self.batch_representation_sampled:
                batch_nb = self.batch_embedding_distribution(batch_index_n=batch_index_n).rsample()
            else:
                batch_nb = self.batch_representation_mean_bd[batch_index_n.long(), :]
        return batch_nb
    
    def categorical_onehot_from_categorical_index(self, categorical_covariate_index_nd: torch.Tensor | None) -> torch.Tensor | None:
        """Compute one-hot encoding of categorical covariates from integer category indices.

        Args:
            categorical_covariate_index_nd: a tensor of shape (n, n_categorical_covariates)
        
        """
        if categorical_covariate_index_nd is not None:

            # make the categorical covariates one-hot
            categorical_covariate_np = torch.cat(
                [torch.nn.functional.one_hot(categorical_covariate_index_nd[:, i].long(), num_classes=n_cats).float()
                 for i, n_cats in enumerate(self.n_cats_per_cov)], 
                 dim=1,
            )
            return categorical_covariate_np

        return None

    def inference(
        self,
        x_ng: torch.Tensor,
        batch_nb: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_np: torch.Tensor | None = None,
    ):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """

        encoder_input_ng = x_ng
        if self.use_observed_lib_size:
            library_size_n1 = torch.log(x_ng.sum(dim=-1, keepdim=True))
        if self.log_variational:
            encoder_input_ng = torch.log1p(encoder_input_ng)

        # if continuous_covariates_nc is not None and self.encode_covariates:
        #     encoder_input_ng = torch.cat((encoder_input_ng, continuous_covariates_nc), dim=-1)

        qz = self.z_encoder(x_ng=encoder_input_ng, batch_nb=batch_nb, categorical_covariate_np=categorical_covariate_np)
        z = qz.rsample()

        outputs = dict(
            z=z,
            qz=qz,
            library_size_n1=library_size_n1,
        )

        return outputs

    def generative(
        self,
        z_nk: torch.Tensor,
        library_size_n1: torch.Tensor,
        batch_nb: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_np: torch.Tensor | None = None,
        size_factor_n1: torch.Tensor | None = None,
        # y: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Runs the generative model."""

        if not self.use_size_factor_key:
            size_factor_n1 = library_size_n1

        match self.dispersion:
            case "gene":
                inverse_overdispersion = self.px_r.exp()
            case "gene-cell":
                inverse_overdispersion = None
            case "gene-batch":
                inverse_overdispersion = torch.nn.functional.linear(
                    batch_nb, 
                    self.px_r,
                ).exp()
            case "gene-label":
                inverse_overdispersion = None
                raise NotImplementedError
                # px_r = linear(
                #     torch.nn.functional.one_hot(y.squeeze().long(), self.n_labels).float(), self.px_r
                # )  # px_r gets transposed - last dimension is nb genes

        count_distribution = self.decoder(
            z_nk=z_nk,
            batch_nb=batch_nb, 
            categorical_covariate_np=categorical_covariate_np,
            inverse_overdispersion=inverse_overdispersion, 
            library_size_n1=size_factor_n1,
        )

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            local_library_log_means, local_library_log_vars = self._compute_local_library_params(batch_nb)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z_nk), torch.ones_like(z_nk))

        return dict(px=count_distribution, pl=pl, pz=pz)

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,  # d is the number of categorical covariates; tensor is integer membership
        size_factor_n1: torch.Tensor | None = None,
    ):
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
            batch_index_n:
                Batch indices of input cells as integers.
            continuous_covariates_nc:
                Continuous covariates for each cell (c-dimensional).
            categorical_covariate_index_nd:
                Categorical covariates for each cell (d-dimensional). Integer membership categorical codes.
            size_factor_n1:
                Library size factor for each cell.

        Returns:
            A dictionary with the loss value.
        """

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        batch_nb = self.batch_representation_from_batch_index(batch_index_n)
        categorical_covariate_np = self.categorical_onehot_from_categorical_index(categorical_covariate_index_nd)

        inference_outputs = self.inference(
            x_ng=x_ng,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )
        generative_outputs = self.generative(
            z_nk=inference_outputs["z"],
            library_size_n1=inference_outputs["library_size_n1"],
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
            size_factor_n1=size_factor_n1,
        )

        # KL divergence for z
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=1)

        # optional KL divergence for batch representation
        if self.batch_representation_sampled and (self.batch_kl_weight > 0):
            kl_divergence_batch = self.batch_kl_weight * kl(
                self.batch_embedding_distribution(batch_index_n=batch_index_n), 
                Normal(torch.zeros_like(batch_nb), torch.ones_like(batch_nb))
            ).sum(dim=1)
        else:
            kl_divergence_batch = 0

        # reconstruction loss
        rec_loss = -generative_outputs["px"].log_prob(x_ng).sum(-1)

        # full loss
        loss = torch.mean(rec_loss + kl_divergence_z + kl_divergence_batch)

        return {"loss": loss}

    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,  # d is the number of categorical covariates; tensor is integer membership
    ):
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
            batch_index_n:
                Batch indices of input cells as integers.
            continuous_covariates_nc:
                Continuous covariates for each cell (c-dimensional).
            categorical_covariate_index_nd:
                Categorical covariates for each cell (d-dimensional where d is number of categorical variables).

        Returns:
            A dictionary with the loss value.
        """

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        batch_nb = self.batch_representation_from_batch_index(batch_index_n)
        categorical_covariate_np = self.categorical_onehot_from_categorical_index(categorical_covariate_index_nd)

        return self.inference(
            x_ng=x_ng,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )
    
    def reconstruct(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
        size_factor_n: torch.Tensor | None = None,
        transform_batch: str | int | None = None,
        sample: bool = True,
    ):
        """
        Reconstruct the data using the VAE, optionally transforming the batch.

        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
            batch_index_n:
                Batch indices of input cells as integers.
            continuous_covariates_nc:
                Continuous covariates for each cell (c-dimensional).
            categorical_covariate_index_nd:
                Categorical covariates for each cell (d-dimensional where d is the number of categorical variables).
            size_factor_n:
                Library size factor for each cell.
            transform_batch:
                If not None, transform the batch to this index before reconstruction.
            sample:
                If True, sample from the generative model. If False, use the mean.
        """

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        if transform_batch is None:
            transformed_batch_index_n_list = [batch_index_n]  # make this a list of size one with the measured values as default: an actual reconstruction
        else:
            transformed_batch_index_n_list = []
            if isinstance(transform_batch, str):
                if transform_batch != 'mean':
                    raise ValueError('transform_batch must be an integer or the string "mean" which will project counts into each batch and compute the mean')
                for i in range(self.n_batch):
                    transformed_batch_index_n_list.append(torch.ones_like(batch_index_n) * i)
            else:
                if transform_batch >= self.n_batch:
                    raise ValueError(f"transform_batch must be less than self.n_batch: {self.n_batch}")
                transformed_batch_index_n_list = [torch.ones_like(batch_index_n) * transform_batch]

        batch_nb = self.batch_representation_from_batch_index(batch_index_n)
        categorical_covariate_np = self.categorical_onehot_from_categorical_index(categorical_covariate_index_nd)

        inference_outputs = self.inference(
            x_ng=x_ng,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )

        output_counts_ng_list = []

        # go through each output batch projection (just one unless transform_batch == "mean")
        for transformed_batch_index_n in transformed_batch_index_n_list:

            batch_nb = self.batch_representation_from_batch_index(transformed_batch_index_n)

            generative_outputs = self.generative(
                z_nk=inference_outputs["z"],
                library_size_n1=inference_outputs["library_size_n1"],
                batch_nb=batch_nb,
                continuous_covariates_nc=continuous_covariates_nc,
                categorical_covariate_np=categorical_covariate_np,
                size_factor_n1=size_factor_n,
            )

            if sample:
                counts_ng = generative_outputs["px"].sample()
            else:
                counts_ng = generative_outputs["px"].mean

            output_counts_ng_list.append(counts_ng)

        x_ng = torch.mean(torch.stack(output_counts_ng_list), dim=0)
        return x_ng
