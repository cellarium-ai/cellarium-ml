# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""deltacells: fast, compact, cloud-friendly tiles of single-cell count matrices."""

from deltacells.backends import (
    Backend,
    CountingBackend,
    GCSBackend,
    LocalBackend,
    RangeFile,
    ThrottledBackend,
    open_backend,
)
from deltacells.dataset import DeltaCellsDataset, open_dataset
from deltacells.format import FormatError, TileInfo
from deltacells.manifest import Manifest
from deltacells.reader import DecodedTile, SparseBatch, Tile, read_tile
from deltacells.writer import DatasetWriter, encode_tile, permute_columns, write_tile

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "CountingBackend",
    "DatasetWriter",
    "DecodedTile",
    "DeltaCellsDataset",
    "FormatError",
    "GCSBackend",
    "LocalBackend",
    "Manifest",
    "RangeFile",
    "SparseBatch",
    "ThrottledBackend",
    "Tile",
    "TileInfo",
    "encode_tile",
    "open_backend",
    "open_dataset",
    "permute_columns",
    "read_tile",
    "write_tile",
]
