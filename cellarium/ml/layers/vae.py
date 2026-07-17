# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Any, Type

import torch


class SwiGLUActivation(torch.nn.Module):
    """
    Stateless SwiGLU gated activation.

    Chunks the input in half along the last dimension and returns
    ``x_val * F.silu(gate)``. When used as ``activation_fn`` in
    :class:`DressedLayer`, the upstream linear layer should produce
    ``2 × inner_dim`` outputs; :class:`~cellarium.ml.models.scvi.FullyConnectedWithBatchArchitecture`
    handles this automatically when ``out_features`` is set to the desired
    post-gating width.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_val, gate = x.chunk(2, dim=-1)
        return x_val * torch.nn.functional.silu(gate)


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
        use_residual: if ``True``, adds the input as a residual when the output shape matches the
            input shape. Silently disabled when shapes differ (e.g. dimension-changing layers).
            Intended for constant-width hidden stacks.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        use_batch_norm: bool = False,
        batch_norm_kwargs: dict | None = None,
        use_layer_norm: bool = False,
        layer_norm_kwargs: dict | None = None,
        activation_fn: Type[torch.nn.Module] | None = torch.nn.ReLU,
        dropout_rate: float = 0,
        use_residual: bool = False,
    ):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {"momentum": 0.01, "eps": 0.001}
        if layer_norm_kwargs is None:
            layer_norm_kwargs = {"elementwise_affine": False}
        assert not (use_batch_norm and use_layer_norm), "Cannot use both batch and layer normalization."
        super().__init__()
        out_features = getattr(layer, "out_features", None)
        if out_features is None:
            raise ValueError(f"attempted to use {layer} in DressedLayer, but it does not define out_features")
        assert isinstance(out_features, int), "The layer must have an `out_features` attribute of type `int`."
        batch_norm = torch.nn.BatchNorm1d(out_features, **batch_norm_kwargs) if use_batch_norm else None
        layer_norm = torch.nn.LayerNorm(out_features, **layer_norm_kwargs) if use_layer_norm else None
        activation = activation_fn() if (activation_fn is not None) else None
        dropout = torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        module_list = [batch_norm, layer_norm, activation, dropout]
        self.layer = layer
        self.use_residual = use_residual
        self.out_features = out_features // 2 if (activation_fn is SwiGLUActivation) else out_features
        if use_residual:
            in_features = getattr(layer, "in_features", None)
            if in_features is not None and in_features != self.out_features:
                warnings.warn(
                    f"use_residual=True but in_features={in_features} != out_features={self.out_features}; "
                    "residual connection will be disabled for this layer.",
                    UserWarning,
                    stacklevel=2,
                )
        self.dressing = torch.nn.Sequential(*[m for m in module_list if m is not None])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes the forward pass of the block.
        """
        residual = args[0]
        h = self.dressing(self.layer(*args, **kwargs))
        if self.use_residual and h.shape == residual.shape:
            h = h + residual
        return h


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
        dressing_init_kwargs: dict[str, Any] | None = None,
        bias: bool = False,
    ):
        super().__init__()
        if dressing_init_kwargs is None:
            dressing_init_kwargs = {}
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
