# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
    assert_nonnegative,
)
from cellarium.ml.utilities.types import BatchDict


class DivideByScale(nn.Module):
    """
    Divide gene counts by a scale.

    .. math::

        y_{ng} = \\frac{x_{ng}}{\\mathrm{scale}_g + \\mathrm{eps}}

    Args:
        scale_g:
            A scale for each gene.
        feature_schema:
            The variable names schema for the input data validation.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(self, scale_g: torch.Tensor, feature_schema: Sequence[str], eps: float = 1e-6) -> None:
        super().__init__()
        self.scale_g: torch.Tensor
        self.register_buffer("scale_g", scale_g)
        self.feature_schema = np.array(feature_schema)
        assert_nonnegative("eps", eps)
        self.eps = eps

    def forward(
        self,
        x_ng: torch.Tensor,
        feature_g: np.ndarray | None = None,
    ) -> BatchDict:
        """
        Args:
            x_ng:
                Gene counts.
            feature_g:
                The list of the variable names in the input data. If ``None``, no validation is performed.

        Returns:
            Gene counts divided by scale.
        """
        if feature_g is not None:
            assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
            assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        x_ng = x_ng / (self.scale_g + self.eps)

        return BatchDict(x_ng=x_ng)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale_g={self.scale_g}, feature_schema={self.feature_schema}, eps={self.eps})"
        )
