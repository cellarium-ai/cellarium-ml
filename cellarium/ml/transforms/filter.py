# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from functools import cache
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from cellarium.ml.utilities.testing import (
    assert_columns_and_array_lengths_equal,
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
