# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Testing utilities
-----------------

This module contains helper functions for testing.
"""

import numpy as np
import torch


def assert_positive(name: str, number: float):
    """
    Assert that a number is positive.

    Args:
        name: The name of the number.
        number: The number to check.

    Raises:
        ValueError: If the number is not positive.
    """
    if number <= 0:
        raise ValueError(f"`{name}` must be positive. Got {number}")


def assert_nonnegative(name: str, number: float):
    """
    Assert that a number is non-negative.

    Args:
        name: The name of the number.
        number: The number to check.

    Raises:
        ValueError: If the number is negative.
    """
    if number < 0:
        raise ValueError(f"`{name}` must be non-negative. Got {number}")


def assert_columns_and_array_lengths_equal(
    matrix_name: str,
    matrix: np.ndarray | torch.Tensor,
    array_name: str,
    array: np.ndarray | torch.Tensor,
):
    """
    Assert that the number of columns in a matrix matches the length of an array.

    Args:
        matrix_name: The name of the matrix.
        matrix: The matrix.
        array_name: The name of the array.
        array: The array.

    Raises:
        ValueError: If the number of columns in the matrix does not match the length of the array.
    """
    if matrix.shape[1] != len(array):
        raise ValueError(
            f"The number of `{matrix_name}` columns must match the `{array_name}` length. "
            f"Got {matrix.shape[1]} != {len(array)}"
        )


def assert_arrays_equal(
    a1_name: str,
    a1: np.ndarray,
    a2_name: str,
    a2: np.ndarray,
):
    """
    Assert that two arrays are equal.

    Args:
        a1_name: The name of the first array.
        a1: The first array.
        a2_name: The name of the second array.
        a2: The second array.

    Raises:
        ValueError: If the arrays are not equal.
    """
    if not np.array_equal(a1, a2):
        raise ValueError(f"`{a1_name}` must match `{a2_name}`. " f"Got {a1} != {a2}")
