# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
import torch
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

    def __init__(self, filter_list: Sequence[str]) -> None:
        super().__init__()
        self.filter_list = np.array(filter_list)
        if len(self.filter_list) == 0:
            raise ValueError(f"`filter_list` must not be empty. Got {self.filter_list}")

    @cache
    def filter(self, var_names_g: tuple) -> np.ndarray[Any, np.dtype[np.int_]]:
        """
        Args:
            var_names_g: The list of the variable names in the input data.

        Returns:
            An array of indices of the features in ``var_names_g`` that are in :attr:`filter_list`.
        """
        mask = np.isin(var_names_g, self.filter_list)
        if not np.any(mask):
            raise AssertionError("No features in `var_names_g` matched the `filter_list`")
        mask_indices = np.where(mask)[0]
        return mask_indices

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor | np.ndarray]:
        """
        .. note::

            When used with :class:`~cellarium.ml.core.CellariumModule` or :class:`~cellarium.ml.core.CellariumPipeline`,
            ``x_ng`` and ``var_names_g`` keys in the input dictionary will be overwritten with the filtered values.

        Args:
            x_ng:
                Gene counts.
            var_names_g:
                The list of the variable names in the input data.

        Returns:
            A dictionary with the following keys:

            - ``x_ng``: Gene counts filtered by :attr:`filter_list`.
            - ``var_names_g``: The list of the variable names in the input data filtered by :attr:`filter_list`.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)

        filter_indices = self.filter(tuple(var_names_g.tolist()))
        ndx = torch.arange(x_ng.shape[0])
        x_ng = x_ng[ndx[:, None], filter_indices]
        var_names_g = var_names_g[filter_indices]

        return {"x_ng": x_ng, "var_names_g": var_names_g}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filter_list={self.filter_list})"
