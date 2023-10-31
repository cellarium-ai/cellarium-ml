# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, TypedDict

import numpy as np
import torch


class BatchDict(TypedDict, total=False):
    feature_g: np.ndarray
    kwargs: Any
    loss: torch.Tensor
    output_attentions: bool
    output_hidden_states: bool
    total_mrna_umis_n: torch.Tensor
    x_ng: torch.Tensor
    z_nk: torch.Tensor
