# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from torch import nn


class NormalizeTotal(nn.Module):
    """
    Normalize total gene counts per cell to target count.

    .. math::

        total\\_mrna\\_umis_n = \\sum_{g=1}^G x_{ng}

        x_{ng} = \\frac{target\\_count \\times x_{ng}}{total\\_mrna\\_umis_n + eps}

    Args:
        target_count:
            Target gene epxression count.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        target_count: int = 10_000,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.target_count = target_count
        self.eps = eps

    def forward(
        self,
        x_ng: torch.Tensor,
        total_mrna_umis_n: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_ng:
                Gene counts.
            total_mrna_umis_n:
                Total mRNA UMI counts per cell. If ``None``, it is computed from ``x_ng``.

        Returns:
            Gene counts normalized to target count.
        """
        if total_mrna_umis_n is None:
            total_mrna_umis_n = x_ng.sum(dim=-1)
        return self.target_count * x_ng / (total_mrna_umis_n[:, None] + self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_count={self.target_count}, eps={self.eps})"


class DivideByScale(nn.Module):
    """
    Divide gene counts by a scale.

    .. math::

        x_{ng} = \\frac{x_{ng}}{scale_g + eps}

    Args:
        scale_g:
            A scale for each gene.
        eps:
            A value added to the denominator for numerical stability.
        feature_schema:
            The variable names schema. If ``None``, no validation is performed.
    """

    def __init__(self, scale_g: torch.Tensor, eps: float = 1e-6, feature_schema: np.ndarray | None = None) -> None:
        super().__init__()
        self.scale_g: torch.Tensor
        self.register_buffer("scale_g", scale_g)
        self.eps = eps
        self.feature_schema = feature_schema

    def forward(
        self,
        x_ng: torch.Tensor,
        feature_list: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.
            feature_list:
                The list of the variable names in the input data. If ``None``, no validation is performed.

        Returns:
            Gene counts divided by scale.
        """
        if self.feature_schema is not None and feature_list is not None:
            assert x_ng.shape[1] == len(feature_list), "The number of x_ng columns must match the feature_list length."
            assert np.array_equal(self.feature_schema, feature_list), "feature_list must match the feature_schema."

        return x_ng / (self.scale_g + self.eps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale_g={self.scale_g}, eps={self.eps}, feature_schema={self.feature_schema})"
        )


class Log1p(nn.Module):
    """
    Log1p transform gene counts.

    .. math::

        x_{ng} = \\log(1 + x_{ng})
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Log1p transformed gene counts.
        """
        return torch.log1p(x_ng)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ZScore(nn.Module):
    """
    ZScore gene counts with  mean and standard deviation.

    .. math::

        x_{ng} = \\frac{x_{ng} - mean_g}{std_g + eps}

    Args:
        mean_g:
            Means for each gene.
        std_g:
            Standard deviations for each gene.
        eps:
            A value added to the denominator for numerical stability.
        feature_schema:
            The variable names schema. If ``None``, no validation is performed.
    """

    def __init__(
        self,
        mean_g: torch.Tensor | float,
        std_g: torch.Tensor | float,
        eps: float = 1e-6,
        feature_schema: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.mean_g = mean_g
        self.std_g = std_g
        self.eps = eps
        self.feature_schema = feature_schema

    def forward(
        self,
        x_ng: torch.Tensor,
        feature_list: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts.
            feature_list:
                The list of the variable names in the input data. If ``None``, no validation is performed.

        Returns:
            Z-scored gene counts.
        """
        if self.feature_schema is not None and feature_list is not None:
            assert x_ng.shape[1] == len(feature_list), "The number of x_ng columns must match the feature_list length."
            assert np.array_equal(self.feature_schema, feature_list), "feature_list must match the feature_schema."

        return (x_ng - self.mean_g) / (self.std_g + self.eps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean_g={self.mean_g}, std_g={self.std_g}, eps={self.eps}, "
            f"feature_schema={self.feature_schema})"
        )
