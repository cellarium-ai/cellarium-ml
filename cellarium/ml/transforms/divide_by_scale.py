# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from cellarium.ml.models import CellariumPipelineUpdatable
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
    assert_nonnegative,
)


class DivideByScale(nn.Module, CellariumPipelineUpdatable):
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

    def update_input_tensors_from_previous_module(self, batch: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Update feature schema, and scale factor according to a new batch dimension.

        Args:
             batch: The batch forwarded from the previous module.
        """
        mask = np.isin(element=self.feature_schema, test_elements=batch["feature_g"])
        mask_tensor = torch.Tensor(mask)
        mask_tensor.to(self.scale_g.device)

        self.feature_schema = self.feature_schema[mask]
        self.scale_g = self.scale_g[mask_tensor]

    def forward(
        self,
        x_ng: torch.Tensor,
        feature_g: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng:
                Gene counts.
            feature_g:
                The list of the variable names in the input data. If ``None``, no validation is performed.

        Returns:
            A dictionary with the following keys:

            - ``x_ng``: The gene counts divided by the scale.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        x_ng = x_ng / (self.scale_g + self.eps)

        return {"x_ng": x_ng}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale_g={self.scale_g}, feature_schema={self.feature_schema}, eps={self.eps})"
        )
