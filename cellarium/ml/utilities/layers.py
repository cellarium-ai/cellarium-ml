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
