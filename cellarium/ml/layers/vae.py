# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Type

import torch


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
        batch_norm_kwargs: dict = {"momentum": 0.01, "eps": 0.001},
        use_layer_norm: bool = False,
        layer_norm_kwargs: dict = {"elementwise_affine": False},
        activation_fn: Type[torch.nn.Module] | None = torch.nn.ReLU,
        dropout_rate: float = 0,
    ):
        assert not (use_batch_norm and use_layer_norm), "Cannot use both batch and layer normalization."
        super().__init__()
        out_features = layer.out_features
        assert isinstance(out_features, int), "The layer must have an `out_features` attribute of type `int`."
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


class FullyConnectedLinear(torch.nn.Module):
    """
    Fully connected block of layers (can be empty).

    Args:
        in_features: The dimensionality of the input
        out_features: The dimensionality of the output
        n_hidden: A list of sizes of torch.nn.Linear hidden layers
        dressing_init_kwargs: A dictionary of keyword arguments to pass ``DressedLayer``'s constructor
        bias: True to include a bias in the final linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden: list[int],
        dressing_init_kwargs: dict[str, Any] = {},
        bias: bool = False,
    ):
        super().__init__()
        module_list = torch.nn.ModuleList()
        layer_size = in_features
        if len(n_hidden) > 0:
            for n_in, n_out in zip([in_features] + n_hidden, n_hidden):
                module_list.append(
                    DressedLayer(
                        torch.nn.Linear(in_features=n_in, out_features=n_out, bias=True),
                        **dressing_init_kwargs,
                    )
                )
            layer_size = n_out
        module_list.append(torch.nn.Linear(layer_size, out_features, bias=bias))
        self.module_list = module_list
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x
        for layer in self.module_list:
            x_ = layer(x_)
        return x_
