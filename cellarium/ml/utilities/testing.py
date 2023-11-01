# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch


def assert_positive(name: str, number: float):
    if number <= 0:
        raise ValueError(f"`{name}` must be positive. Got {number}")


def assert_nonnegative(name: str, number: float):
    if number < 0:
        raise ValueError(f"`{name}` must be non-negative. Got {number}")


def assert_columns_and_array_lengths_equal(
    matrix_name: str,
    matrix: np.ndarray | torch.Tensor,
    array_name: str,
    array: np.ndarray | torch.Tensor,
):
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
    if not np.array_equal(a1, a2):
        raise ValueError(f"`{a1_name}` must match `{a2_name}`. " f"Got {a1} != {a2}")
