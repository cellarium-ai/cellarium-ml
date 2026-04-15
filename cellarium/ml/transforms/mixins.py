# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Mixin classes for data transformations."""

from functools import cache

import numpy as np

from cellarium.ml.utilities.data import get_var_names_g_indices


class FilterCompatibilityMixin:
    var_names_g: np.ndarray

    @cache
    def _get_indices(self, var_names_g: tuple) -> np.ndarray:
        return get_var_names_g_indices(np.array(var_names_g), self.var_names_g)
