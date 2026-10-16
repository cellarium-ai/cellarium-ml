# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Decode and gather speed of single tiles.

    python benchmarks/bench_decode.py DATASET [--tiles 3] [--threads 1,2,4,8] [--batch-size 5000]

Reports, per tile (medians over repeats, buffers reused so page-fault costs are excluded -- as ``DeltaCellsDataset`` does):
compressed size, decode time and throughput per thread count, and the time to gather a shuffled batch from a decoded tile.
Tiles are read from disk once first, so the numbers are for warm page cache / local disk, with no network.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from deltacells import LocalBackend, Tile
from deltacells.manifest import MANIFEST_NAME, Manifest


def median_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--tiles", type=int, default=3, help="How many tiles (from the start) to measure.")
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)

    backend = LocalBackend(args.dataset)
    manifest = Manifest.from_json(bytes(backend.read(MANIFEST_NAME)))
    thread_counts = [int(t) for t in args.threads.split(",")]
    rng = np.random.default_rng(0)
    print(
        f"{manifest.n_cells} cells x {manifest.n_genes} genes, {manifest.n_tiles} tiles; {manifest.n_chunks} chunks per tile, zstd {manifest.zstd_level}"
    )
    for i in range(min(args.tiles, manifest.n_tiles)):
        t0 = time.perf_counter()
        tile = Tile(backend.read(Manifest.tile_name(i)))
        t_read = time.perf_counter() - t0
        n = tile.n_cells
        print(
            f"\ntile {i}: {n} cells, {tile.nnz} nonzeros ({tile.nnz / n:.0f}/cell), {tile.nbytes / 1e6:.1f} MB = {tile.nbytes / n / 1e3:.2f} kB/cell "
            f"= {tile.nbytes / max(tile.nnz, 1):.3f} B/nnz   [file read + parse: {t_read * 1e3:.0f} ms]"
        )
        out_i, out_v = np.empty(tile.nnz, np.int32), np.empty(tile.nnz, np.int32)
        for threads in thread_counts:
            scratch = np.empty(min(threads, tile.n_chunks) * tile.info.scratch_nbytes, dtype=np.uint8)
            tile.decode(out_indices=out_i, out_values=out_v, scratch=scratch, threads=threads)  # warm up
            t = median_time(
                lambda: tile.decode(out_indices=out_i, out_values=out_v, scratch=scratch, threads=threads), args.repeats
            )
            print(
                f"  decode, {threads:>2} thread(s): {t * 1e3:7.1f} ms   {tile.nnz / t / 1e6:6.0f} M nonzeros/s   {n / t / 1e3:6.1f}k cells/s"
            )
        dec = tile.decode(values=np.int32)
        rows = rng.permutation(n)[: min(args.batch_size, n)]
        dec.gather(rows)
        t = median_time(lambda: dec.gather(rows), args.repeats)
        print(f"  gather {len(rows)} shuffled cells (int32 -> float32, fresh output arrays): {t * 1e3:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
