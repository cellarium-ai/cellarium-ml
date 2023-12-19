# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class abdParameter:
    data: torch.Tensor | None = None
    width: int = 1
    a: float = 0.0
    b: float = 0.0
    d: float = 0.0
    mult_scale: float = 1.0
    init_scale: float = 1.0
    base_d_width: int = 128


class MuLinear(nn.Module):
    """
    Linear layer with a maximal update parametrization.

    abcd-Parametrization multiplier scaling:

    .. math::

        W = \\mathrm{mult\\_scale} \\cdot n^{-a} \\cdot w

    Learnable parameter initialization scaling:

    .. math::

        w \\sim \\mathcal{N}(\\mu = 0, \\sigma = \\mathrm{init\\_scale} \\cdot n^{-b})

    The maximal update parametrization for SGD is defined by

    +-----------+----------------+----------------+----------------+
    |           | Input & Biases | Hidden         | Output         |
    +-----------+----------------+----------------+----------------+
    | :math:`a` | :math:`-0.5`   | :math:`0`      | :math:`0.5`    |
    +-----------+----------------+----------------+----------------+
    | :math:`b` | :math:`0.5`    | :math:`0.5`    | :math:`0.5`    |
    +-----------+----------------+----------------+----------------+
    | :math:`c` | :math:`0`      | :math:`0`      | :math:`0`      |
    +-----------+----------------+----------------+----------------+
    | width     | out_features   | in_features    | in_features    |
    +-----------+----------------+----------------+----------------+

    The maximal update parametrization for Adam and AdamW is defined by

    +-----------+----------------+----------------+----------------+
    |           | Input & Biases | Hidden         | Output         |
    +-----------+----------------+----------------+----------------+
    | :math:`a` | :math:`0`      | :math:`1`      | :math:`1`      |
    +-----------+----------------+----------------+----------------+
    | :math:`b` | :math:`0`      | :math:`-0.5`   | :math:`0`      |
    +-----------+----------------+----------------+----------------+
    | :math:`c` | :math:`0`      | :math:`0`      | :math:`0`      |
    +-----------+----------------+----------------+----------------+
    | :math:`d` | :math:`1`      | :math:`2`      | :math:`1`      |
    +-----------+----------------+----------------+----------------+
    | width     | out_features   | in_features    | in_features    |
    +-----------+----------------+----------------+----------------+

    Since in this implementation :math:`c` always equals 0, regular PyTorch optimizers
    can be used.

    **References:**

    1. `Feature Learning in Infinite-Width Neural Networks (Yang et al.)
       <https://arxiv.org/pdf/2011.14522.pdf>`_.
    2. `Tensor Programs IVb: Adaptive Optimization in the ∞-Width Limit (Yang et al.)
       <https://arxiv.org/pdf/2308.01814.pdf>`_.

    Args:
        in_features:
            Size of each input sample.
        out_features:
            Size of each output sample.
        bias:
            If set to ``False``, the layer will not learn an additive bias.
        layer:
            Layer type. One of ``"input"``, ``"hidden"``, or ``"output"``.
        optimizer:
            Optimizer type. One of ``"sgd"``, ``"adam"``, or ``"adamw"``.
        mult_scale:
            Scaling factor for the parameter multiplier.
        init_scale:
            Scaling factor for the parameter initialization.
        base_d_width:
            Base width for the parameter ``d``.

    Attributes:
        weight:
            The weights of the module of shape ``(out_features, in_features)``. The parameter
            multiplier scales according to the table above.
        weight_unscaled:
            The learnable weights of the module of shape ``(out_features, in_features)``.
            The values are initialized as according to the table above.
        bias:
            The bias of the module of shape ``(out_features)``. The parameter multiplier
            scales according to the table above.
        bias_unscaled:
            The learnable bias of the module of shape ``(out_features)``. If :attr:`bias` is ``True``,
            the values are initialized as according to the table above with ``init_scale = 0.0``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        layer: Literal["input", "hidden", "output"],
        optimizer: Literal["sgd", "adam", "adamw"],
        mult_scale: float = 1.0,
        init_scale: float = 1.0,
        base_d_width: int = 128,
    ) -> None:
        super().__init__()

        # inf dim
        bias_width = out_features
        if layer == "input":
            weight_width = out_features
        else:
            weight_width = in_features

        # c = 0 for all layers by design

        # a, b, d
        if optimizer in ["adam", "adamw"]:
            # bias scaling: 1
            bias_a = 0.0
            bias_b = 0.0
            bias_d = 1.0
            # weight
            if layer == "input":
                # scaling: 1
                weight_a = 0.0
                weight_b = 0.0
                weight_d = 1.0
            elif layer == "hidden":
                # scaling: 1 / sqrt(n)
                weight_a = 1.0
                weight_b = -0.5
                weight_d = 2.0
            elif layer == "output":
                # scaling: 1 / n
                weight_a = 1.0
                weight_b = 0.0
                weight_d = 1.0
        elif optimizer == "sgd":
            # bias scaling: 1
            bias_a = -0.5
            bias_b = 0.5
            bias_d = 0.0
            # weight
            if layer == "input":
                # scaling: 1
                weight_a = -0.5
                weight_b = 0.5
                weight_d = 0.0
            elif layer == "hidden":
                # scaling: 1 / sqrt(n)
                weight_a = 0.0
                weight_b = 0.5
                weight_d = 0.0
            elif layer == "output":
                # scaling: 1 / n
                weight_a = 0.5
                weight_b = 0.5
                weight_d = 0.0
        else:
            raise ValueError(f"Optimizer must be either 'sgd', 'adam', or 'adamw'. Got {optimizer!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.layer = layer
        self.optimizer = optimizer
        self.mult_scale = mult_scale
        self.init_scale = init_scale
        self.base_d_width = base_d_width
        self.weight = abdParameter(  # type: ignore[assignment]
            torch.empty(out_features, in_features),
            width=weight_width,
            a=weight_a,
            b=weight_b,
            d=weight_d,
            mult_scale=mult_scale,
            init_scale=init_scale,
            base_d_width=base_d_width,
        )
        if bias:
            self.bias = abdParameter(  # type: ignore[assignment]
                torch.empty(out_features),
                width=bias_width,
                a=bias_a,
                b=bias_b,
                d=bias_d,
                init_scale=0.0,
                base_d_width=base_d_width,
            )
        else:
            self.bias = abdParameter(None)  # type: ignore[assignment]

    @staticmethod
    def scale_grad(base_width: int, width: int, d: float) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if d == 0:
                return grad
            return grad * (width / base_width) ** d

        return hook

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_multiplier * self.weight_unscaled

    @weight.setter
    def weight(self, value: abdParameter) -> None:
        assert value.data is not None
        self.weight_unscaled = nn.Parameter(value.data)
        self.weight_unscaled.register_hook(self.scale_grad(value.base_d_width, value.width, value.d))
        if value.init_scale == 0:
            self.weight_unscaled.data.zero_()
        else:
            std = value.init_scale / value.width**value.b
            self.weight_unscaled.data.normal_(mean=0.0, std=std)
        self.weight_multiplier = value.mult_scale / value.width**value.a

    @property
    def bias(self) -> torch.Tensor | None:
        if self.bias_unscaled is None:
            return None
        return self.bias_multiplier * self.bias_unscaled

    @bias.setter
    def bias(self, value: abdParameter) -> None:
        if value.data is None:
            self.register_parameter("bias_unscaled", None)
            return
        self.bias_unscaled = nn.Parameter(value.data)
        self.bias_unscaled.register_hook(self.scale_grad(value.base_d_width, value.width, value.d))
        if value.init_scale == 0:
            self.bias_unscaled.data.zero_()
        else:
            std = value.init_scale / value.width**value.b
            self.bias_unscaled.data.normal_(mean=0.0, std=std)
        self.bias_multiplier = value.mult_scale / value.width**value.a

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
            f", layer={self.layer}, optimizer={self.optimizer}"
            f", mult_scale={self.mult_scale}, init_scale={self.init_scale}, base_d_width={self.base_d_width}"
        )
