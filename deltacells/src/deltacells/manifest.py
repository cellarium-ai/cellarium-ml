# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""The dataset manifest: a small JSON file describing a directory (or bucket prefix) of tiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

FORMAT_NAME = "deltacells-dataset"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
GENE_ORDER_NAME = "gene_order.npy"
VAR_NAMES_NAME = "var_names.txt"
TILE_PATTERN = "tiles/tile_{:06d}.dct"
OBS_SHARD_PATTERN = "obs/obs_{:04d}.parquet"
OBS_SCHEMA_NAME = "obs/schema.json"
VAR_TABLE_NAME = "var.parquet"
DEFAULT_OBS_TILES_PER_FILE = 1000


@dataclass
class Manifest:
    """Dataset-level metadata.

    Cells are numbered globally in tile order: tile ``i`` holds cells ``limits[i - 1] : limits[i]``. Every tile except the last
    holds exactly ``tile_size`` cells.
    """

    n_cells: int
    n_genes: int
    tile_size: int
    tile_cells: list[int]
    tile_nnz: list[int]
    tile_bytes: list[int]
    n_chunks: int
    zstd_level: int
    has_gene_order: bool = False
    has_var_names: bool = False
    metadata: dict = field(default_factory=dict)
    # optional per-cell metadata ("obs"), stored as parquet shards of obs_tiles_per_file tiles with one row group per tile
    has_obs: bool = False
    obs_tiles_per_file: int = 0
    obs_fingerprint: str = ""
    obs_shard_bytes: list[int] = field(default_factory=list)
    obs_shard_sha256: list[str] = field(default_factory=list)
    # optional gene table (var), one small parquet file in output column order
    has_var_table: bool = False
    format: str = FORMAT_NAME
    version: int = FORMAT_VERSION

    @property
    def n_tiles(self) -> int:
        return len(self.tile_cells)

    @property
    def limits(self) -> np.ndarray:
        """Cumulative cell counts: ``limits[i]`` is one past the last global cell index of tile ``i``."""
        return np.cumsum(np.asarray(self.tile_cells, dtype=np.int64))

    @staticmethod
    def tile_name(i: int) -> str:
        return TILE_PATTERN.format(i)

    @property
    def n_obs_shards(self) -> int:
        return -(-self.n_tiles // self.obs_tiles_per_file) if self.has_obs else 0

    @staticmethod
    def obs_shard_name(shard: int) -> str:
        return OBS_SHARD_PATTERN.format(shard)

    def obs_shard_tiles(self, shard: int) -> range:
        """The tiles whose obs rows are stored (one row group each) in obs shard ``shard``."""
        lo = shard * self.obs_tiles_per_file
        return range(lo, min(lo + self.obs_tiles_per_file, self.n_tiles))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str | bytes) -> Manifest:
        d = json.loads(text)
        if d.get("format") != FORMAT_NAME:
            raise ValueError("not a deltacells manifest")
        if d.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported manifest version {d.get('version')} (this build reads version {FORMAT_VERSION})"
            )
        m = cls(**d)
        if m.n_cells != sum(m.tile_cells) or not (len(m.tile_cells) == len(m.tile_nnz) == len(m.tile_bytes)):
            raise ValueError("inconsistent manifest")
        if any(c != m.tile_size for c in m.tile_cells[:-1]) or (
            m.tile_cells and not 0 < m.tile_cells[-1] <= m.tile_size
        ):
            raise ValueError("inconsistent manifest: every tile but the last must hold exactly tile_size cells")
        if m.has_obs and (
            m.obs_tiles_per_file < 1
            or not m.obs_fingerprint
            or len(m.obs_shard_bytes) != m.n_obs_shards
            or len(m.obs_shard_sha256) != m.n_obs_shards
        ):
            raise ValueError("inconsistent manifest: obs fields")
        return m
