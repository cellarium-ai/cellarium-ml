# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import collections
from collections.abc import Iterable

import torch


class LinearInputBias(torch.nn.Linear):
    """A `torch.nn.Linear` layer where bias is an input to the forward pass.

    Args:
        in_features: passed to `torch.nn.Linear`
        out_features: passed to `torch.nn.Linear`
        bias: passed to `torch.nn.Linear`
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the layer as
        out = x @ self.weight.T + self.bias + bias
        """
        return super().forward(x) + bias


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
