# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
    assert_nonnegative,
)


class DivideByScale(nn.Module):
    """
    Divide gene counts by a scale.

    .. math::

        y_{ng} = \\frac{x_{ng}}{\\mathrm{scale}_g + \\mathrm{eps}}

    Args:
        scale_g:
            A scale for each gene.
        var_names_g:
            The variable names schema for the input data validation.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(self, scale_g: torch.Tensor, var_names_g: Sequence[str], eps: float = 1e-6) -> None:
        super().__init__()
        self.scale_g: torch.Tensor
        self.register_buffer("scale_g", scale_g)
        self.var_names_g = np.array(var_names_g)
        assert_nonnegative("eps", eps)
        self.eps = eps

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng:
                Gene counts.
            var_names_g:
                The list of the variable names in the input data. If ``None``, no validation is performed.

        Returns:
            A dictionary with the following keys:

            - ``x_ng``: The gene counts divided by the scale.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        x_ng = x_ng / (self.scale_g + self.eps)

        return {"x_ng": x_ng}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_g={self.scale_g}, var_names_g={self.var_names_g}, eps={self.eps})"
