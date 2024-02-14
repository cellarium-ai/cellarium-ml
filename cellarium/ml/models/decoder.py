from typing import Literal
from collections.abc import Iterable

import torch
from torch import nn

from cellarium.ml.FCLayer import FCLayers

class DecoderSCVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        **kwargs,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class LinearDecoderSCVI(nn.Module):
    """Linear decoder for scVI."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        # mean gamma
        self.factor_regressor = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

        # dropout
        self.px_dropout_decoder = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        """Forward pass."""
        # The decoder returns values for the parameters of the ZINB distribution
        raw_px_scale = self.factor_regressor(z, *cat_list)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z, *cat_list)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout

    class Decoder(nn.Module):
        """Decodes data from latent space to data space.

        ``n_input`` dimensions to ``n_output``
        dimensions using a fully-connected neural network of ``n_hidden`` layers.
        Output is the mean and variance of a multivariate Gaussian

        Parameters
        ----------
        n_input
            The dimensionality of the input (latent space)
        n_output
            The dimensionality of the output (data space)
        n_cat_list
            A list containing the number of categories
            for each category of interest. Each category will be
            included using a one-hot encoding
        n_layers
            The number of fully-connected hidden layers
        n_hidden
            The number of nodes per hidden layer
        dropout_rate
            Dropout rate to apply to each of the hidden layers
        kwargs
            Keyword args for :class:`~scvi.module._base.FCLayers`
        """

        def __init__(
                self,
                n_input: int,
                n_output: int,
                n_cat_list: Iterable[int] = None,
                n_layers: int = 1,
                n_hidden: int = 128,
                **kwargs,
        ):
            super().__init__()
            self.decoder = FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
                **kwargs,
            )

            self.mean_decoder = nn.Linear(n_hidden, n_output)
            self.var_decoder = nn.Linear(n_hidden, n_output)

        def forward(self, x: torch.Tensor, *cat_list: int):
            """The forward computation for a single sample.

             #. Decodes the data from the latent space using the decoder network
             #. Returns tensors for the mean and variance of a multivariate distribution

            Parameters
            ----------
            x
                tensor with shape ``(n_input,)``
            cat_list
                list of category membership(s) for this sample

            Returns
            -------
            2-tuple of :py:class:`torch.Tensor`
                Mean and variance tensors of shape ``(n_output,)``

            """
            # Parameters for latent distribution
            p = self.decoder(x, *cat_list)
            p_m = self.mean_decoder(p)
            p_v = torch.exp(self.var_decoder(p))
            return p_m, p_v