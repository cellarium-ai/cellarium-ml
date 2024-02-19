# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from operator import attrgetter
from typing import Literal

import torch
from torch.utils._pytree import tree_map


class abcdParameter(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        requires_grad=True,
        a: float = 0.0,
        b: float = 0.0,
        d: float = 0.0,
        base_width: int = 1,
        width: int = 1,
        mode: Literal["init", "load"] = "init",
    ) -> None:
        if mode == "init":
            data = data * (base_width / width) ** b
        t = torch.Tensor._make_subclass(cls, data, requires_grad)
        t.multiplier = (base_width / width) ** a
        t.register_hook(cls.scale_grad(base_width, width, d))
        return t

    @property
    def scaled(self):
        return self * self.multiplier

    @staticmethod
    def scale_grad(base_width: int, width: int, d: float) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if d == 0:
                return grad
            return grad * (width / base_width) ** d

        return hook

    # def __repr__(self):
    #     return "abcdParameter containing:\n" + torch._tensor_str._str(self, tensor_contents=None)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in {
            torch.nn.functional.linear,
            torch.nn.functional.embedding,
            torch.nn.functional.layer_norm,
        }:
            args = tree_map(lambda t: t.scaled if isinstance(t, cls) else t, args)
            kwargs = tree_map(lambda t: t.scaled if isinstance(t, cls) else t, kwargs)

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


def apply_mup(base_module, module, optimizer, mode="init"):
    for (base_name, base_param), (name, param) in zip(base_module.named_parameters(), module.named_parameters()):
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

            child_name = name.split(".")[-1]
            attr = ".".join(name.split(".")[:-1])
            child_module = attrgetter(attr)(module) if attr else module
            if not isinstance(child_module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
                raise ValueError(f"Module {module} is not an instance of torch.nn.Linear or torch.nn.Embedding")
            setattr(
                child_module,
                child_name,
                abcdParameter(param.data, a=a, b=b, d=d, base_width=base_width, width=width, mode=mode),
            )
