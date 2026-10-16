# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Size / speed trade-offs of the encoder on one tile of an existing dataset: zstd level, number of chunks, and the effect of gene order.

    python benchmarks/bench_compression.py DATASET [--tile 0] [--levels 3,12,19] [--chunks 1,8]

The tile is decoded, then re-encoded with each setting (with the stored gene order, and with a random gene order to show what
sorting genes by expression buys). Decode times are single-threaded.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from deltacells import LocalBackend, Tile, encode_tile, permute_columns
from deltacells.manifest import MANIFEST_NAME, Manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--levels", default="3,12,19")
    parser.add_argument("--chunks", default="1,8")
    parser.add_argument("--threads", type=int, default=8, help="Encoder threads.")
    args = parser.parse_args(argv)

    backend = LocalBackend(args.dataset)
    Manifest.from_json(bytes(backend.read(MANIFEST_NAME)))  # validates the manifest
    matrix = Tile(backend.read(Manifest.tile_name(args.tile))).to_scipy()
    shuffled = permute_columns(matrix, np.random.default_rng(0).permutation(matrix.shape[1]))
    n = matrix.shape[0]
    print(f"tile {args.tile}: {n} cells, {matrix.nnz} nonzeros ({matrix.nnz / n:.0f}/cell)")
    print(
        f"{'gene order':<10}{'level':>6}{'chunks':>8}{'size MB':>9}{'kB/cell':>9}{'B/nnz':>7}{'encode s':>10}{'decode ms (1 thr)':>19}"
    )
    for label, m in (("sorted", matrix), ("random", shuffled)):
        for level in [int(x) for x in args.levels.split(",")]:
            for chunks in [int(x) for x in args.chunks.split(",")]:
                t0 = time.perf_counter()
                blob = encode_tile(m, n_chunks=chunks, level=level, threads=args.threads)
                t_enc = time.perf_counter() - t0
                tile = Tile(blob)
                out_i, out_v = np.empty(tile.nnz, np.int32), np.empty(tile.nnz, np.int32)
                scratch = np.empty(tile.info.scratch_nbytes, dtype=np.uint8)
                tile.decode(out_indices=out_i, out_values=out_v, scratch=scratch)
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    tile.decode(out_indices=out_i, out_values=out_v, scratch=scratch)
                    times.append(time.perf_counter() - t0)
                print(
                    f"{label:<10}{level:>6}{chunks:>8}{len(blob) / 1e6:>9.1f}{len(blob) / n / 1e3:>9.2f}{len(blob) / m.nnz:>7.3f}{t_enc:>10.1f}{statistics.median(times) * 1e3:>19.0f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
