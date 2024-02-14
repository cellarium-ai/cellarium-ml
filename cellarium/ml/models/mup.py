# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from operator import attrgetter

import torch
from torch.utils._pytree import tree_map


class abcdParameter(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, multiplier=1.0) -> None:
        ret = super().__new__(cls, data, requires_grad)
        ret._multiplier = multiplier
        return ret

    def __repr__(self):
        return "abcd" + super().__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.nn.functional.linear, torch.nn.functional.embedding, torch.nn.functional.layer_norm):
            args = tree_map(lambda t: t * t._multiplier if isinstance(t, cls) else t, args)
            kwargs = tree_map(lambda t: t * t._multiplier if isinstance(t, cls) else t, kwargs)

        return super().__torch_function__(func, types, args, kwargs)


def get_param_type_and_base_width(param_shape: tuple[int], base_param_shape: tuple[int]) -> tuple[str, int, int]:
    assert len(param_shape) == len(base_param_shape)
    assert len(param_shape) in (1, 2)
    if len(param_shape) == 1:
        param_type = "bias"
        width = param_shape[0]
        base_width = base_param_shape[0]
    elif len(param_shape) == 2:
        if param_shape[0] == base_param_shape[0]:
            param_type = "output"
            width = param_shape[1]  # fan_in
            base_width = base_param_shape[1]
        elif param_shape[1] == base_param_shape[1]:
            param_type = "input"
            width = param_shape[0]  # fan_out
            base_width = base_param_shape[0]
        else:
            param_type = "hidden"
            width = param_shape[1]
            base_width = base_param_shape[1]
    return param_type, width, base_width


def scale_grad(base_width: int, width: int, d: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook(grad: torch.Tensor) -> torch.Tensor:
        if d == 0:
            return grad
        return grad * (width / base_width) ** d

    return hook


def apply_mup(module, base_module, optimizer, scale="init"):
    for (name, param), (base_name, base_param) in zip(module.named_parameters(), base_module.named_parameters()):
        assert name == base_name
        if param.shape != base_param.shape:
            param_type, width, base_width = get_param_type_and_base_width(param.shape, base_param.shape)
            if optimizer == "sgd":
                if param_type in ["bias", "input"]:
                    # scaling: Θ(1)
                    a = -0.5
                    b = 0.5
                    d = 0.0
                elif param_type == "hidden":
                    # scaling: Θ(1 / sqrt(n))
                    a = 0.0
                    b = 0.5
                    d = 0.0
                elif param_type == "output":
                    # scaling: Θ(1 / n)
                    a = 0.5
                    b = 0.5
                    d = 0.0
            elif optimizer in ["adam", "adamw"]:
                if param_type in ["bias", "input"]:
                    # scaling: Θ(1)
                    a = 0.0
                    b = 0.0
                    d = 1.0
                elif param_type == "hidden":
                    # scaling: Θ(1 / sqrt(n))
                    a = 1.0
                    b = -0.5
                    d = 2.0
                elif param_type == "output":
                    # scaling: Θ(1 / n)
                    a = 1.0
                    b = 0.0
                    d = 1.0
            else:
                raise ValueError(f"Optimizer must be either 'sgd', 'adam', or 'adamw'. Got {optimizer!r}")

            multiplier = (base_width / width) ** a
            if scale == "scale_init":
                data = param.data * (base_width / width) ** b
            elif scale == "init":
                data = param.data / multiplier
            elif scale == "load":
                data = param.data
            attr = ".".join(name.split(".")[:-1])
            _name = name.split(".")[-1]
            if attr:
                _module = attrgetter(attr)(module)
            else:
                _module = module
            if not isinstance(_module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
                raise ValueError(f"Module {module} is not an instance of torch.nn.Linear or torch.nn.Embedding")
            setattr(_module, _name, abcdParameter(data, multiplier=multiplier))
            getattr(_module, _name).register_hook(scale_grad(base_width, width, d))
