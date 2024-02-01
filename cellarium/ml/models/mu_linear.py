# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class abcdParameter:
    """
    An *abcd*-parametrization describes the scaling of a parameter :math:`W` with width :math:`n`. The parameter
    is initialized with a standard deviation :math:`\\sigma` and a parameter-specific learning rate scaling factor
    :math:`\\alpha` at base width :math:`n_0`. The scaling of the parameterization with width :math:`n` is described
    by a set of numbers :math:`\\{a, b, c, d\\}` such that:

    a. Parameter is given as :math:`W = \\sqrt{\\alpha} \\cdot (n_0 / n)^a \\cdot w` where :math:`w`
       is the learnable parameter.
    b. Learnable parameter is initialized as
       :math:`w \\sim \\mathcal{N}(0, \\sigma \\cdot (n_0 / n)^b / \\sqrt{\\alpha})`.
    c. The effective learning rate for :math:`W` is
       :math:`\\alpha \\cdot (n_0 / n)^{2a} \\cdot (n_0 / n)^c \\cdot \\eta` for some global learning rate
       :math:`\\eta`. In this implementation, :math:`c` is equal to :math:`0`.
    d. The gradients of :math:`w` are scaled by :math:`(n / n_0)^d`.

    Args:
        data:
            The tensor data.
        width:
            The width :math:`n` of the tensor.
        a:
            The :math:`a` parameter.
        b:
            The :math:`b` parameter.
        d:
            The :math:`d` parameter.
        init_std:
            The initialization standard deviation :math:`\\sigma` at base width :math:`n_0`.
        lr_scale:
            The learning rate scale factor :math:`\\alpha` at base width :math:`n_0`.
        base_width:
            The base width :math:`n_0`.
    """

    data: torch.Tensor | None = None
    width: int = 1
    a: float = 0.0
    b: float = 0.0
    d: float = 0.0
    init_std: float = 1.0
    lr_scale: float = 1.0
    base_width: int = 1


class MuLinear(nn.Module):
    r"""
    Linear layer with a maximal update parametrization.

    The maximal update parametrization for SGD is defined by:

    +-----------+----------------+----------------+----------------+
    |           | Input & Biases | Hidden         | Output         |
    +-----------+----------------+----------------+----------------+
    | :math:`a` | :math:`-0.5`   | :math:`0`      | :math:`0.5`    |
    +-----------+----------------+----------------+----------------+
    | :math:`b` | :math:`0.5`    | :math:`0.5`    | :math:`0.5`    |
    +-----------+----------------+----------------+----------------+
    | :math:`c` | :math:`0`      | :math:`0`      | :math:`0`      |
    +-----------+----------------+----------------+----------------+
    | :math:`d` | :math:`0`      | :math:`0`      | :math:`0`      |
    +-----------+----------------+----------------+----------------+
    | :math:`n` | out_features   | in_features    | in_features    |
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
    | :math:`n` | out_features   | in_features    | in_features    |
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
            Layer type.
        optimizer:
            Optimizer type.
        weight_init_std:
            The standard deviation of the weight initialization at base width.
        bias_init_std:
            The standard deviation of the bias initialization at base width.
        lr_scale:
            The learning rate scaling factor for the weight and the bias.
        base_width:
            The base width of the layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        layer: Literal["input", "hidden", "output"],
        optimizer: Literal["sgd", "adam", "adamw"],
        weight_init_std: float = 1.0,
        bias_init_std: float = 0.0,
        lr_scale: float = 1.0,
        base_width: int = 1,
    ) -> None:
        super().__init__()

        # inf dim
        if layer == "output":
            bias_width = base_width
        else:
            bias_width = out_features
        if layer == "input":
            weight_width = out_features
        else:
            weight_width = in_features

        # c = 0 for all layers by design

        # a, b, d
        if optimizer == "sgd":
            # bias scaling: Θ(1)
            bias_a = -0.5
            bias_b = 0.5
            bias_d = 0.0
            # weight
            if layer == "input":
                # scaling: Θ(1)
                weight_a = -0.5
                weight_b = 0.5
                weight_d = 0.0
            elif layer == "hidden":
                # scaling: Θ(1 / sqrt(n))
                weight_a = 0.0
                weight_b = 0.5
                weight_d = 0.0
            elif layer == "output":
                # scaling: Θ(1 / n)
                weight_a = 0.5
                weight_b = 0.5
                weight_d = 0.0

        elif optimizer in ["adam", "adamw"]:
            # bias scaling: Θ(1)
            bias_a = 0.0
            bias_b = 0.0
            bias_d = 1.0
            # weight
            if layer == "input":
                # scaling: Θ(1)
                weight_a = 0.0
                weight_b = 0.0
                weight_d = 1.0
            elif layer == "hidden":
                # scaling: Θ(1 / sqrt(n))
                weight_a = 1.0
                weight_b = -0.5
                weight_d = 2.0
            elif layer == "output":
                # scaling: Θ(1 / n)
                weight_a = 1.0
                weight_b = 0.0
                weight_d = 1.0

        else:
            raise ValueError(f"Optimizer must be either 'sgd', 'adam', or 'adamw'. Got {optimizer!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.layer = layer
        self.optimizer = optimizer
        self.weight_init_std = weight_init_std
        self.bias_init_std = bias_init_std
        self.lr_scale = lr_scale
        self.base_width = base_width
        self.weight = abcdParameter(  # type: ignore[assignment]
            torch.empty(out_features, in_features),
            width=weight_width,
            a=weight_a,
            b=weight_b,
            d=weight_d,
            init_std=weight_init_std,
            lr_scale=lr_scale,
            base_width=base_width,
        )
        if bias:
            self.bias = abcdParameter(  # type: ignore[assignment]
                torch.empty(out_features),
                width=bias_width,
                a=bias_a,
                b=bias_b,
                d=bias_d,
                init_std=bias_init_std,
                lr_scale=lr_scale,
                base_width=base_width,
            )
        else:
            self.bias = abcdParameter(None)  # type: ignore[assignment]

    @staticmethod
    def scale_grad(base_width: int, width: int, d: float) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if d == 0:
                return grad
            return grad * (width / base_width) ** d

        return hook

    @property
    def weight(self) -> torch.Tensor:
        """
        The weights of the module of shape ``(out_features, in_features)``. The weight-specific learning rate and
        the initialization standard deviation are scaled with the width of the layer according to the table above.
        """
        return self.weight_multiplier * self.weight_unscaled

    @weight.setter
    def weight(self, value: abcdParameter) -> None:
        assert value.data is not None
        self.weight_multiplier = value.lr_scale**0.5 * (value.base_width / value.width) ** value.a
        self.weight_unscaled = nn.Parameter(value.data)
        if value.init_std == 0:
            self.weight_unscaled.data.zero_()
        else:
            std = value.init_std * (value.base_width / value.width) ** value.b / value.lr_scale**0.5
            self.weight_unscaled.data.normal_(mean=0.0, std=std)
        self.weight_unscaled.register_hook(self.scale_grad(value.base_width, value.width, value.d))

    @property
    def bias(self) -> torch.Tensor | None:
        """
        The bias of the module of shape ``(out_features)``. If :attr:`bias` is ``True``,
        the bias-specific learning rate and the initialization standard deviation are scaled with the width of the
        layer according to the table above.
        """
        if self.bias_unscaled is None:
            return None
        return self.bias_multiplier * self.bias_unscaled

    @bias.setter
    def bias(self, value: abcdParameter) -> None:
        self.bias_unscaled: nn.Parameter | None
        if value.data is None:
            self.register_parameter("bias_unscaled", None)
            return
        self.bias_multiplier = value.lr_scale**0.5 * (value.base_width / value.width) ** value.a
        self.bias_unscaled = nn.Parameter(value.data)
        if value.init_std == 0:
            self.bias_unscaled.data.zero_()
        else:
            std = value.init_std * (value.base_width / value.width) ** value.b / value.lr_scale**0.5
            self.bias_unscaled.data.normal_(mean=0.0, std=std)
        self.bias_unscaled.register_hook(self.scale_grad(value.base_width, value.width, value.d))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
            f", layer={self.layer}, optimizer={self.optimizer}, weight_init_std={self.weight_init_std}"
            f", bias_init_std={self.bias_init_std}, lr_scale={self.lr_scale}, base_width={self.base_width}"
        )
