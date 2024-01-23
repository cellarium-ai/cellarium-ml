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


class ZScore(nn.Module):
    """
    ZScore gene counts with  mean and standard deviation.

    .. math::

        y_{ng} = \\frac{x_{ng} - \\mathrm{mean}_g}{\\mathrm{std}_g + \\mathrm{eps}}

    Args:
        mean_g:
            Means for each gene.
        std_g:
            Standard deviations for each gene.
        var_names_g:
            The variable names schema for the input data validation.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        mean_g: torch.Tensor | float,
        std_g: torch.Tensor | float,
        var_names_g: Sequence[str],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mean_g: torch.Tensor
        self.std_g: torch.Tensor
        self.register_buffer("mean_g", torch.as_tensor(mean_g))
        self.register_buffer("std_g", torch.as_tensor(std_g))
        self.var_names_g = np.array(var_names_g)
        assert_nonnegative("eps", eps)
        self.eps = eps

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """
        .. note::

            When used with :class:`~cellarium.ml.core.CellariumModule` or :class:`~cellarium.ml.core.CellariumPipeline`,
            ``x_ng`` key in the input dictionary will be overwritten with the z-scored values.

        Args:
            x_ng:
                Gene counts.
            var_names_g:
                The list of the variable names in the input data. If ``None``, no validation is performed.

        Returns:
            A dictionary with the following keys:

            - ``x_ng``: The z-scored gene counts.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        x_ng = (x_ng - self.mean_g) / (self.std_g + self.eps)
        return {"x_ng": x_ng}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean_g={self.mean_g}, std_g={self.std_g}, "
            f"var_names_g={self.var_names_g}), eps={self.eps}"
        )
