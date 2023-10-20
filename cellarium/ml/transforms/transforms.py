# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from functools import cache
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
    assert_nonnegative,
    assert_positive,
)


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

    def __init__(self, scale_g: torch.Tensor, feature_schema: ArrayLike, eps: float = 1e-6) -> None:
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
    ) -> torch.Tensor:
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

        return x_ng / (self.scale_g + self.eps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale_g={self.scale_g}, feature_schema={self.feature_schema}, eps={self.eps})"
        )


class Filter(nn.Module):
    """
    Filter gene counts by a list of features.

    .. math::

        \\mathrm{mask}_g = \\mathrm{feature}_g \\in \\mathrm{filter\\_list}

        y_{ng} = x_{ng}[:, \\mathrm{mask}_g]

    Args:
        filter_list: A list of features to filter by.
    """

    def __init__(self, filter_list: ArrayLike) -> None:
        super().__init__()
        self.filter_list = np.array(filter_list)
        if len(self.filter_list) == 0:
            raise ValueError(f"`filter_list` must not be empty. Got {self.filter_list}")

    @cache
    def filter(self, feature_g: tuple) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """
        Args:
            feature_g: The list of the variable names in the input data.

        Returns:
            A boolean mask of the features to filter by.
        """
        mask = np.isin(feature_g, self.filter_list)
        if not np.any(mask):
            raise AssertionError("No features in `feature_g` matched the `filter_list`")
        return mask

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray) -> torch.Tensor:
        """
        Args:
            x_ng:
                Gene counts.
            feature_g:
                The list of the variable names in the input data.

        Returns:
            Filtered gene counts.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)

        filter_mask = self.filter(tuple(feature_g.tolist()))
        return x_ng[:, filter_mask]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filter_list={self.filter_list})"


class Log1p(nn.Module):
    """
    Log1p transform gene counts.

    .. math::

        y_{ng} = \\log(1 + x_{ng})
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


class NormalizeTotal(nn.Module):
    """
    Normalize total gene counts per cell to target count.

    .. math::

        \\mathrm{total\\_mrna\\_umis}_n = \\sum_{g=1}^G x_{ng}

        y_{ng} = \\frac{\\mathrm{target\\_count} \\times x_{ng}}{\\mathrm{total\\_mrna\\_umis}_n + \\mathrm{eps}}

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
        assert_positive("target_count", target_count)
        self.target_count = target_count
        assert_nonnegative("eps", eps)
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
        feature_schema:
            The variable names schema for the input data validation.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        mean_g: torch.Tensor | float,
        std_g: torch.Tensor | float,
        feature_schema: ArrayLike,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mean_g = mean_g
        self.std_g = std_g
        self.feature_schema = np.array(feature_schema)
        assert_nonnegative("eps", eps)
        self.eps = eps

    def forward(
        self,
        x_ng: torch.Tensor,
        feature_g: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_ng:
                Gene counts.
            feature_g:
                The list of the variable names in the input data. If ``None``, no validation is performed.

        Returns:
            Z-scored gene counts.
        """
        if feature_g is not None:
            assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
            assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        return (x_ng - self.mean_g) / (self.std_g + self.eps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean_g={self.mean_g}, std_g={self.std_g}, "
            f"feature_schema={self.feature_schema}), eps={self.eps}"
        )
