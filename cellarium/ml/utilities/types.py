# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import torch

ConvertType = dict[str, Callable | dict[str, Callable]]


class BatchDict(TypedDict, total=False):
    x_ng: torch.Tensor
    feature_g: np.ndarray
    loss: torch.Tensor
