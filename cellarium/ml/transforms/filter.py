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

    When ``ordering=False``, the output columns follow the order genes appear in the input
    ``var_names_g``:

    .. math::

        \\mathrm{mask}_g = \\mathrm{feature}_g \\in \\mathrm{filter\\_list}

        y_{ng} = x_{ng}[:, \\mathrm{mask}_g]

    When ``ordering=True`` (default), the output columns follow the order of :attr:`filter_list`:

    .. math::

        y_{ng} = x_{ng}[:, \\sigma(\\mathrm{filter\\_list})]

    where :math:`\\sigma` maps each entry in :attr:`filter_list` to its column index in the input.

    Args:
        filter_list:
            A list of features to filter by.
        ordering:
            If ``True`` (default), output columns are ordered to match :attr:`filter_list`.
            If ``False``, output columns follow the order genes appear in the input ``var_names_g``.
            Use ``ordering=True`` when running inference on data with a different gene ordering
            than seen during training.
        allow_missing:
            If ``True``, genes in :attr:`filter_list` that are absent from the input are
            zero-filled in the output. Requires ``ordering=True``. If ``False`` (default),
            all genes in :attr:`filter_list` must be present in the input.
    """

    def __init__(self, filter_list: Sequence[str], ordering: bool = True, allow_missing: bool = False) -> None:
        super().__init__()
        self.filter_list = np.array(filter_list)
        if len(self.filter_list) == 0:
            raise ValueError(f"`filter_list` must not be empty. Got {self.filter_list}")
        if allow_missing and not ordering:
            raise ValueError("`allow_missing=True` requires `ordering=True`.")
        self.ordering = ordering
        self.allow_missing = allow_missing

    @cache
    def filter(
        self, var_names_g: tuple
    ) -> (
        np.ndarray[Any, np.dtype[np.intp]]
        | tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]
    ):
        """
        Args:
            var_names_g: The list of the variable names in the input data.

        Returns:
            When ``ordering=False``: a 1-D array of source indices in input order.

            When ``ordering=True`` and ``allow_missing=False``: a 1-D array of source indices
            ordered to match :attr:`filter_list`.

            When ``ordering=True`` and ``allow_missing=True``: a tuple
            ``(src_indices, out_indices)`` where ``src_indices`` indexes columns in
            ``var_names_g`` and ``out_indices`` gives the corresponding destination column in
            the output (which always has ``len(filter_list)`` columns).
        """
        if not self.ordering:
            mask = np.isin(var_names_g, self.filter_list)
            if not np.any(mask):
                raise AssertionError("No features in `var_names_g` matched the `filter_list`")
            return np.where(mask)[0]

        # ordering=True: iterate over filter_list to enforce its order in the output.
        var_names_index = {name: idx for idx, name in enumerate(var_names_g)}
        src_indices: list[int] = []
        out_indices: list[int] = []
        missing: list[str] = []
        for out_idx, gene in enumerate(self.filter_list):
            if gene in var_names_index:
                src_indices.append(var_names_index[gene])
                out_indices.append(out_idx)
            else:
                missing.append(gene)

        if not src_indices:
            raise AssertionError("No features in `var_names_g` matched the `filter_list`")
        if missing and not self.allow_missing:
            raise AssertionError(f"The following features in `filter_list` were not found in `var_names_g`: {missing}")

        src_indices_ordered = np.array(src_indices, dtype=np.intp)
        if self.allow_missing:
            return src_indices_ordered, np.array(out_indices, dtype=np.intp)
        return src_indices_ordered

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

            - ``x_ng``: Gene counts filtered (and reordered if ``ordering=True``) to match
              :attr:`filter_list`. Shape ``(n, len(filter_list))`` when ``ordering=True``,
              otherwise ``(n, num_matched)``.
            - ``var_names_g``: Gene names corresponding to the output columns.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)

        result = self.filter(tuple(var_names_g.tolist()))

        if self.allow_missing:
            src_indices, out_indices = result
            assert isinstance(src_indices, np.ndarray) and isinstance(out_indices, np.ndarray)
            x_out = torch.zeros(x_ng.shape[0], len(self.filter_list), dtype=x_ng.dtype, device=x_ng.device)
            x_out[:, out_indices] = x_ng[:, src_indices]
            x_ng = x_out
            var_names_g = self.filter_list.copy()
        elif self.ordering:
            assert isinstance(result, np.ndarray)
            x_ng = x_ng[:, result]
            var_names_g = self.filter_list.copy()
        else:
            assert isinstance(result, np.ndarray)
            ndx = torch.arange(x_ng.shape[0])
            x_ng = x_ng[ndx[:, None], result]
            var_names_g = var_names_g[result]

        return {"x_ng": x_ng, "var_names_g": var_names_g}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"filter_list={self.filter_list}, "
            f"ordering={self.ordering}, "
            f"allow_missing={self.allow_missing})"
        )
