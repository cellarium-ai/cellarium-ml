# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable
from typing import Any

from torch import nn


def convert_glob_to_regex(f: str) -> re.Pattern:
    """
    Converts the given glob string f to a type of regex which can then be used with .match() to
    check if the string matches the regex returned
    """
    return re.compile(fnmatch.translate(f))


def make_param_filter(param_filter: str | list[str]) -> Callable[[str, nn.Parameter], bool]:
    """
    Returns the corresponding filter for parameters for the given `param_filter`.
    Args:
        param_filter: Either a string or a list of strings which are glob expressions or a
        callable which represents the filter itself
    Returns:
        A callable method that when given a parameter will return whether the filter matches it.
    """

    param_filters = list(
        map(
            convert_glob_to_regex,
            [param_filter] if isinstance(param_filter, str) else param_filter,
        )
    )

    def glob_expression_param_filter(name: str) -> bool:
        return any(filter.fullmatch(name) for filter in param_filters)

    return glob_expression_param_filter


class LRAdjustmentGroup:
    """
    Stores data for a group of params that share a learning rate scalar.
    Stores a callable that returns True if a given model param corresponds
    to the group. Additionally, it stores the scale that should be applied to
    the LR of the model params that correspond to the group.
    """

    def __init__(
        self,
        param_filter: str | list[str],
        scale: float | None = 1.0,
    ) -> None:
        """
        param_filter: A string or a list of strings that contains glob expressions
        used to match whether a given model param name belongs to the group.

        scale: The scale that should be applied to the LR of this group
        """
        # Convert the strings into a callable that returns True if a given param
        # name corresponds to the LR group
        self.param_filter = param_filter
        self.scale = scale

    def set_scale(self, scale):
        self.scale = scale
