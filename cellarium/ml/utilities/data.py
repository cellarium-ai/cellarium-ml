# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Data utilities
--------------
This module contains helper functions for data loading and processing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from anndata import AnnData
from anndata.experimental import AnnCollection


@dataclass
class AnnDataField:
    attr: str
    key: str | None = None
    convert_fn: Callable[[Any], np.ndarray] | None = None

    def __call__(self, adata: AnnData | AnnCollection) -> AnnDataField:
        self.adata = adata
        return self

    def __getitem__(self, idx: int | list[int] | slice) -> np.ndarray:
        if self.adata is None:
            raise ValueError("Must call AnnDataField with an AnnData or AnnCollection first")

        value = getattr(self.adata[idx], self.attr)
        if self.key is not None:
            value = value[self.key]

        if self.convert_fn is not None:
            value = self.convert_fn(value)

        return value

    @property
    def obs_column(self) -> str | None:
        result = None
        if self.attr == "obs":
            result = self.key
        return result
