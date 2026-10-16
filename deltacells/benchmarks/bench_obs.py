# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Costs of the obs (cell metadata) path: what localizing columns reads from the dataset, how long it takes, and what lookups cost
afterwards.

    python benchmarks/bench_obs.py DATASET [--columns cell_type,donor,n_counts] [--bandwidth-mbps 150 --latency-ms 50]

* lists the obs columns by the number of bytes needed to fetch them (and per 100M cells);
* localizes the chosen columns into a fresh temporary cache through a counting (optionally throttled) backend and reports the
  bytes, ranged reads and time -- the bytes are the columns' bytes plus the parquet footers;
* times lookups of a shuffled batch of cells from one tile (steady state, after a warm-up call), and the one-off cost a process
  pays to open the cached columns (time and extra resident memory).
"""

from __future__ import annotations

import argparse
import resource
import statistics
import tempfile
import time

import numpy as np

from deltacells import CountingBackend, DeltaCellsDataset, LocalBackend, ThrottledBackend


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if peak > 1e8 else peak / 1e3  # macOS reports bytes, Linux kilobytes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument(
        "--columns", default="", help="Comma separated columns to localize (default: the three smallest)."
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bandwidth-mbps", type=float, default=None, help="Emulated per-connection bandwidth (MB/s).")
    parser.add_argument("--latency-ms", type=float, default=0.0, help="Emulated per-read latency.")
    args = parser.parse_args(argv)

    counting = CountingBackend(LocalBackend(args.dataset))
    backend = counting
    if args.bandwidth_mbps or args.latency_ms:
        backend = ThrottledBackend(
            counting, latency_s=args.latency_ms / 1e3, bandwidth_bytes_per_s=(args.bandwidth_mbps or 0) * 1e6 or None
        )
        counting = CountingBackend(backend)  # count what the dataset asks for, above the throttle
        backend = counting
    with tempfile.TemporaryDirectory() as cache:
        ds = DeltaCellsDataset(backend, cache_dir=cache)
        store = ds.obs
        if store is None:
            print("this dataset has no obs")
            return 1
        m = ds.manifest
        sizes = store.column_nbytes()
        total = sum(sizes.values())
        print(
            f"{ds.n_cells} cells in {ds.n_tiles} tiles; obs: {len(sizes)} columns, {m.n_obs_shards} parquet shard(s), {sum(m.obs_shard_bytes) / 1e6:.1f} MB"
        )
        print(
            f"all columns: {total / 1e6:.1f} MB to fetch = {total / ds.n_tiles / 1e3:.0f} kB per tile = {total / m.n_cells * 1e8 / 1e9:.1f} GB per 100M cells\n"
        )
        ordered = sorted(sizes, key=sizes.get)
        print(f"{'column':<32}{'kind':<10}{'MB':>9}{'per 100M cells':>16}")
        for c in ordered[:3] + ["..."] + ordered[-3:]:
            if c == "...":
                print("  ...")
                continue
            print(f"{c:<32}{store.kind(c):<10}{sizes[c] / 1e6:>9.2f}{sizes[c] / m.n_cells * 1e8 / 1e6:>13.0f} MB")

        columns = [c for c in args.columns.split(",") if c] or ordered[:3]
        counting.reset()
        t0 = time.perf_counter()
        fetched = store.localize(columns, threads=args.threads)
        dt = time.perf_counter() - t0
        want = sum(sizes[c] for c in columns)
        local_bytes = sum(store._path(c).stat().st_size for c in columns)
        print(
            f"\nlocalize {fetched}: {dt:.2f}s, read {counting.bytes_read / 1e6:.2f} MB in {counting.n_range_reads} ranged reads "
            f"({want / 1e6:.2f} MB of column data + footers; {100 * counting.bytes_read / sum(m.obs_shard_bytes):.1f}% of the obs shards); "
            f"cache on disk: {local_bytes / 1e6:.2f} MB (Arrow IPC, uncompressed)"
        )

        # steady-state lookups of a shuffled batch from one tile, after a warm-up call
        rng = np.random.default_rng(0)
        lo, hi = ds.tile_bounds(ds.n_tiles // 2)
        idx = rng.permutation(np.arange(lo, hi))[: min(args.batch_size, hi - lo)]
        r0 = rss_mb()
        t0 = time.perf_counter()
        store.take(idx, columns)
        first = time.perf_counter() - t0
        times = []
        for _ in range(50):
            idx = rng.permutation(np.arange(lo, hi))[: min(args.batch_size, hi - lo)]
            t0 = time.perf_counter()
            store.take(idx, columns)
            times.append(time.perf_counter() - t0)
        print(
            f"take({len(idx)} cells of one tile, {len(columns)} columns): first call {first * 1e3:.1f} ms (open + one-off initialization, "
            f"+{rss_mb() - r0:.0f} MB peak RSS), then median {statistics.median(times) * 1e3:.3f} ms"
        )
        t0 = time.perf_counter()
        df = store.to_pandas(columns)
        print(f"to_pandas({len(columns)} columns, {len(df)} cells): {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
