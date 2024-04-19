# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import collections
from collections.abc import Iterable

import torch


class LinearWithBatch(torch.nn.Linear):
    """A `torch.nn.Linear` layer where batch indices are given as input to the forward pass.

    Args:
        in_features: passed to `torch.nn.Linear`
        out_features: passed to `torch.nn.Linear`
        n_batch: total number of batches in dataset
        sample: True to sample the bias weight matrix from a distribution
        bias: passed to `torch.nn.Linear` (True is like the scvi-tools implementation)
    """

    def __init__(self, in_features: int, out_features: int, n_batch: int, sample: bool = False, bias: bool = True, precomputed_bias: torch.Tensor | None = None):
        super().__init__(in_features, out_features, bias=bias)
        self.sample = sample
        self.cached_biases = None
        self.n_batch = n_batch
        self.precomputed_bias = None
        if self.precomputed_bias:
            self.batch_bias_fx = self.load_precomputed_bias
        else:
            self.batch_bias_fx = self.compute_bias
        self.bias_mean_layer = torch.nn.Linear(in_features=n_batch, out_features=out_features, bias=False)
        if sample:
            self.bias_std_unconstrained_layer = torch.nn.Linear(in_features=n_batch, out_features=out_features, bias=False)
    
    def compute_bias(self, batch_n: torch.Tensor) -> torch.Tensor:
        """
        Returns the bias for a given list of batch indices.

        Args:
            batch_n: a tensor of batch indices of shape (n)

        Returns:
            a tensor of shape (n, out_features)
        """
        one_hot_batch_nb = torch.nn.functional.one_hot(batch_n.squeeze().long(), num_classes=self.n_batch).float()
        mean_bias_nh = self.bias_mean_layer(one_hot_batch_nb)
        if self.sample:
            std_bias_nh = self.bias_std_unconstrained_layer(one_hot_batch_nb).exp()
            self.cached_biases = mean_bias_nh + std_bias_nh * torch.randn_like(std_bias_nh)
        else:
            self.cached_biases = mean_bias_nh
        return self.cached_biases

    def load_precomputed_bias(self, batch: torch.Tensor) -> torch.Tensor:
        return self.precomputed_bias[0]


    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the layer as
        out = x @ self.weight.T + self.bias + bias

        where bias is computed as
        bias = batch_one_hot @ weight_batch.T

        or is sampled from a distribution if sample=True
        """

        return super().forward(x) + self.batch_bias_fx(batch)


class DressedLayer(torch.nn.Module):
    """
    Small block comprising a `~torch.nn.Module` with optional batch/layer normalization 
    and configurable activation and dropout.

    Similar to
    torch.nn.Sequential(
        layer,
        optional batch normalization,
        optional layer normalization,
        optional activation,
        optional dropout,
    )
    but the `layer` can take multiple inputs.

    Note that batch normalization and layer normalization are mutually exclusive options.

    Args:
        layer: single layer `torch.nn.Module`, such as an instance of `torch.nn.Linear`
        use_batch_norm: whether to use batch normalization
        use_layer_norm: whether to use layer normalization
        activation_fn: the activation function to use
        dropout_rate: dropout rate, can be zero
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        use_batch_norm: bool = False,
        batch_norm_kwargs: dict = {'momentum': 0.01, 'eps': 0.001},
        use_layer_norm: bool = False,
        layer_norm_kwargs: dict = {'elementwise_affine': False},
        activation_fn: torch.nn.Module | None = torch.nn.ReLU,
        dropout_rate: float = 0,
    ):
        assert not (use_batch_norm and use_layer_norm), "Cannot use both batch and layer normalization."
        super().__init__()
        out_features = layer.out_features
        batch_norm = torch.nn.BatchNorm1d(out_features, **batch_norm_kwargs) if use_batch_norm else None
        layer_norm = torch.nn.LayerNorm(out_features, **layer_norm_kwargs) if use_layer_norm else None
        activation = activation_fn() if (activation_fn is not None) else None
        dropout = torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        module_list = [batch_norm, layer_norm, activation, dropout]
        self.layer = layer
        self.dressing = torch.nn.Sequential(*[m for m in module_list if m is not None])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes the forward pass of the block.
        """
        x = self.layer(*args, **kwargs)
        return self.dressing(x)
