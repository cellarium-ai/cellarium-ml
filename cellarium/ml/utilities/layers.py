# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def create_initializer(initializer: dict[str, Any]) -> Callable[[torch.Tensor], None]:
    initializer_fn = getattr(nn.init, initializer["name"])
    initializer_kwargs = initializer.copy()
    del initializer_kwargs["name"]
    return lambda x: initializer_fn(x, **initializer_kwargs)


def scale_initializers_by_dimension(
    initializers: dict[str, Any] | list[dict[str, Any]],
    width_scale: float | None = None,
    depth_scale: float | None = None,
) -> None:
    """
    Scales the std of an initializer or list of initializers by the specified
    width and depth scalars.
    """
    if not width_scale:
        width_scale = 1.0
    if not depth_scale:
        depth_scale = 1.0
    mup_scalar = width_scale * depth_scale

    if not isinstance(initializers, list):
        initializers = [initializers]

    for initializer in initializers:
        if "name" not in initializer:
            raise ValueError("Initializer name must be provided")
        initializer_name = initializer["name"].lower()

        if initializer_name == "normal_":
            initializer["std"] = initializer.get("std", 1.0) * mup_scalar
        elif initializer_name == "trunc_normal_":
            std = initializer.get("std", 1.0)
            initializer["std"] = std * mup_scalar
            initializer["a"] = initializer.get("a", -2 * std) * mup_scalar
            initializer["b"] = initializer.get("b", 2 * std) * mup_scalar
            std = None
        else:
            raise ValueError(f"Initializer {initializer_name} is not supported for muP")
