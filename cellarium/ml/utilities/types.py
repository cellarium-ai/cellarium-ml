# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

ConvertType = dict[str, Callable | dict[str, Callable]]
