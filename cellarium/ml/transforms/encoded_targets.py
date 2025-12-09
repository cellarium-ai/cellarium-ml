# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from torch import nn


class EncodedTargets(nn.Module):
    """
    when called, assigns multilabel targets. All parents of the target cell type get assigned as targets.
    """

    def __init__(
        self,
        unique_cell_types_nparray: np.ndarray,
    ) -> None:
        super().__init__()
        self.unique_cell_types_nparray = unique_cell_types_nparray

    def forward(self, y_n: np.ndarray) -> dict[str, torch.Tensor | np.ndarray]:
        """ """
        indices = np.searchsorted(self.unique_cell_types_nparray, y_n)
        return {"y_n": torch.tensor(indices), "y_n_predict": indices}  # using for prediction
