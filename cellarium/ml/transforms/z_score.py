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


class ZScore(nn.Module, CellariumPipelineUpdatable):
    """
    ZScore gene counts with  mean and standard deviation.

    .. math::

        y_{ng} = \\frac{x_{ng} - \\mathrm{mean}_g}{\\mathrm{std}_g + \\mathrm{eps}}

    Args:
        mean_g:
            Means for each gene.
        std_g:
            Standard deviations for each gene.
        feature_schema:
            The variable names schema for the input data validation.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        mean_g: torch.Tensor | float,
        std_g: torch.Tensor | float,
        feature_schema: Sequence[str],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mean_g: torch.Tensor
        self.std_g: torch.Tensor
        self.register_buffer("mean_g", torch.as_tensor(mean_g))
        self.register_buffer("std_g", torch.as_tensor(std_g))
        self.feature_schema = np.array(feature_schema)
        assert_nonnegative("eps", eps)
        self.eps = eps

    def update_input_tensors_from_previous_module(self, batch: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Update feature schema, g_genes, mean and std according to a new batch dimension.

        Args:
             batch: The batch forwarded from the previous module.
        """
        mask = np.isin(element=self.feature_schema, test_elements=batch["feature_g"])
        mask_tensor = torch.Tensor(mask)
        mask_tensor.to(self.mean_g.device)
        self.feature_schema = self.feature_schema[mask]
        self.mean_g = self.mean_g[mask_tensor]
        self.std_g = self.std_g[mask_tensor]

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

            - ``x_ng``: The z-scored gene counts.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        x_ng = (x_ng - self.mean_g) / (self.std_g + self.eps)
        return {"x_ng": x_ng}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean_g={self.mean_g}, std_g={self.std_g}, "
            f"feature_schema={self.feature_schema}), eps={self.eps}"
        )
