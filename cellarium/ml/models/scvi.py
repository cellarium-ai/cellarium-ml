# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Poisson
from torch.distributions import kl_divergence as kl

from cellarium.ml.models.common.decoder import DecoderSCVI
from cellarium.ml.models.common.distributions import NegativeBinomial
from cellarium.ml.models.common.encoder import Encoder
from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""

    # TODO: do not repeat code, and probably use pytorch's implementation anyway

    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


class SingleCellVariationalInference(CellariumModel, PredictMixin):
    """
    Flexible version of single-cell variational inference (scVI) [1] re-implemented in Cellarium ML.

    **References:**

    1. `Deep generative modeling for single-cell transcriptomics (Lopez et al.)
       <https://www.nature.com/articles/s41592-018-0229-2>`_.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        n_hidden:
            Number of hidden units in all hidden layers of the encoder and decoder.
        n_latent:
            Dimension of the latent space.
        n_batch:
            Number of total batches in the dataset.
        n_layers:
            Number of hidden layers in the encoder and decoder.
        n_continuous_cov:
            Number of continuous covariates.
        n_cats_per_cov:
            A list of integers containing the number of categories for each categorical covariate.
        dropout_rate:
            Dropout rate for hidden units in the encoder only.
        dispersion:
            Flexibility of the dispersion parameter when ``gene_likelihood`` is either ``"nb"`` or
            ``"zinb"``. One of the following:
                * ``"gene"``: parameter is constant per gene across cells.
                * ``"gene-batch"``: parameter is constant per gene per batch.
                * ``"gene-label"``: parameter is constant per gene per label.
                * ``"gene-cell"``: parameter is constant per gene per cell.
        log_variational:
            If ``True``, use :func:`~torch.log1p` on input data before encoding for numerical stability
            (not normalization).
        gene_likelihood:
            Distribution to use for reconstruction in the generative process. One of the following:
                * ``"nb"``: :class:`~scvi.distributions.NegativeBinomial`.
                * ``"zinb"``: :class:`~scvi.distributions.ZeroInflatedNegativeBinomial`.
                * ``"poisson"``: :class:`~scvi.distributions.Poisson`.
        latent_distribution:
            Distribution to use for the latent space. One of the following:
                * ``"normal"``: isotropic normal.
                * ``"ln"``: logistic normal with normal params N(0, 1).
        encode_covariates:
            If ``True``, covariates are concatenated to gene expression prior to passing through
            the encoder(s). Else, only gene expression is used.
        deeply_inject_covariates:
            If ``True`` and ``n_layers > 1``, covariates are concatenated to the outputs of hidden
            layers in the encoder(s) (if ``encoder_covariates`` is ``True``) and the decoder prior to
            passing through the next layer.
        batch_representation:
            ``EXPERIMENTAL`` Method for encoding batch information. One of the following:
                * ``"one-hot"``: represent batches with one-hot encodings.
                * ``"embedding"``: represent batches with continuously-valued embeddings using
                :class:`~scvi.nn.Embedding`.
            Note that batch representations are only passed into the encoder(s) if
            ``encode_covariates`` is ``True``.
        use_batch_norm:
            Specifies where to use :class:`~torch.nn.BatchNorm1d` in the model. One of the following:
                * ``"none"``: don't use batch norm in either encoder(s) or decoder.
                * ``"encoder"``: use batch norm only in the encoder(s).
                * ``"decoder"``: use batch norm only in the decoder.
                * ``"both"``: use batch norm in both encoder(s) and decoder.
            Note: if ``use_layer_norm`` is also specified, both will be applied (first
            :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
        use_layer_norm:
            Specifies where to use :class:`~torch.nn.LayerNorm` in the model. One of the following:
                * ``"none"``: don't use layer norm in either encoder(s) or decoder.
                * ``"encoder"``: use layer norm only in the encoder(s).
                * ``"decoder"``: use layer norm only in the decoder.
                * ``"both"``: use layer norm in both encoder(s) and decoder.
            Note: if ``use_batch_norm`` is also specified, both will be applied (first
            :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
        use_size_factor_key:
            If ``True``, use the :attr:`~anndata.AnnData.obs` column as defined by the
            ``size_factor_key`` parameter in the model's ``setup_anndata`` method as the scaling
            factor in the mean of the conditional distribution. Takes priority over
            ``use_observed_lib_size``.
        use_observed_lib_size:
            If ``True``, use the observed library size for RNA as the scaling factor in the mean of the
            conditional distribution.
        library_log_means:
            :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of means of the log library sizes that
            parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
            ``use_observed_lib_size`` is ``False``.
        library_log_vars:
            :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of variances of the log library sizes
            that parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
            ``use_observed_lib_size`` is ``False``.
        var_activation:
            Callable used to ensure positivity of the variance of the variational distribution. Passed
            into :class:`~scvi.nn.Encoder`. Defaults to :func:`~torch.exp`.
        extra_encoder_kwargs:
            Additional keyword arguments passed into :class:`~scvi.nn.Encoder`.
        extra_decoder_kwargs:
            Additional keyword arguments passed into :class:`~scvi.nn.DecoderSCVI`.
        batch_embedding_kwargs:
            Keyword arguments passed into :class:`~scvi.nn.Embedding` if ``batch_representation`` is
            set to ``"embedding"``.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        batch_embedding_kwargs: dict | None = None,
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
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size

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
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', "
                " 'gene-label', 'gene-cell'], but input was "
                "{}".format(self.dispersion)
            )

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            raise NotImplementedError
            # self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            # batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")
        if batch_embedding_kwargs is not None:
            raise NotImplementedError

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = self.n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            raise NotImplementedError
            # n_input_encoder += batch_dim * encode_covariates
            # cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}

        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            # return_dist=True,
            **_extra_encoder_kwargs,
        )

        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            raise NotImplementedError
            # n_input_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            self.n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.z_encoder.encoder.fc_layers.apply(weights_init)
        self.z_encoder.mean_encoder.apply(weights_init)
        self.z_encoder.var_encoder.apply(weights_init)
        self.decoder.px_decoder.fc_layers.apply(weights_init)

    def inference(
        self,
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """

        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if self.batch_representation == "embedding" and self.encode_covariates:
            raise NotImplementedError
            # batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            # encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            # qz, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))

        outputs = dict(
            z=z,
            qz=qz,
            library=library,
        )

        return outputs

    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        # y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Runs the generative model."""
        from torch.nn.functional import linear

        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat([z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        if self.batch_representation == "embedding":
            raise NotImplementedError
            # batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            # decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            # px_scale, px_r, px_rate, px_dropout = self.decoder(
            #     self.dispersion,
            #     decoder_input,
            #     size_factor,
            #     *categorical_input,
            #     y,
            # )
        else:
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                batch_index,
                *categorical_input,
                # y,
            )

        if self.dispersion == "gene-label":
            raise NotImplementedError
            # px_r = linear(
            #     one_hot(y, self.n_labels), self.px_r
            # )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            raise NotImplementedError
            # px = ZeroInflatedNegativeBinomial(
            #     mu=px_rate,
            #     theta=px_r,
            #     zi_logits=px_dropout,
            #     scale=px_scale,
            # )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            px = Poisson(rate=px_rate)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(px=px, pl=pl, pz=pz)

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        cont_covs_nc: torch.Tensor | None = None,
        cat_covs_nd: torch.Tensor | None = None,
        size_factor_n: torch.Tensor | None = None,
    ):
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
            batch_index_n:
                Batch indices of input cells as integers.
            cont_covs_nc:
                Continuous covariates for each cell (c-dimensional).
            cat_covs_nd:
                Categorical covariates for each cell (d-dimensional).
            size_factor_n:
                Library size factor for each cell.

        Returns:
            A dictionary with the loss value.
        """

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        inference_outputs = self.inference(
            x=x_ng,
            batch_index=batch_index_n,
            cont_covs=cont_covs_nc,
            cat_covs=cat_covs_nd,
            n_samples=1,
        )
        generative_outputs = self.generative(
            z=inference_outputs["z"],
            library=inference_outputs["library"],
            batch_index=batch_index_n,
            cont_covs=cont_covs_nc,
            cat_covs=cat_covs_nd,
            size_factor=size_factor_n,
            # y=y,
            # transform_batch=transform_batch_n,  # see self.predict()
        )

        # KL divergence
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=1)

        # reconstruction loss
        rec_loss = -generative_outputs["px"].log_prob(x_ng).sum(-1)

        loss = torch.mean(rec_loss + kl_divergence_z)

        print('generative_outputs_px')
        print(generative_outputs["px"].mu)
        print(generative_outputs["px"].theta)

        print('rec_loss')
        print(torch.mean(rec_loss))
        print('kl_divergence_z')
        print(torch.mean(kl_divergence_z))
        print('total_loss')
        print(loss.item())

        return {"loss": loss}

    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        cont_covs_nc: torch.Tensor | None = None,
        cat_covs_nd: torch.Tensor | None = None,
        size_factor_n: torch.Tensor | None = None,
    ):
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
            batch_index_n:
                Batch indices of input cells as integers.
            cont_covs_nc:
                Continuous covariates for each cell (c-dimensional).
            cat_covs_nd:
                Categorical covariates for each cell (d-dimensional).
            size_factor_n:
                Library size factor for each cell.

        Returns:
            A dictionary with the loss value.
        """
        return self.inference(
            x=x_ng,
            batch_index=batch_index_n,
            cont_covs=cont_covs_nc,
            cat_covs=cat_covs_nd,
            n_samples=1,
        )

