# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
End-to-end DataLoader throughput: tile fetch (optionally throttled to emulate a remote store) -> decode -> shuffled batches in shared
memory -> main process. Uses the reference loader in ``examples/``.

    python benchmarks/bench_pipeline.py DATASET --workers 0,1,2,4
    python benchmarks/bench_pipeline.py DATASET --workers 2 --bandwidth-mbps 150 --latency-ms 50   # emulate ~150 MB/s per connection

Reports cells per second after a warm-up period and, for the main process, the time spent waiting for the next batch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
from reference_loader import TileShuffleDataset, to_torch_csr  # noqa: E402

from deltacells import DeltaCellsDataset, LocalBackend, ThrottledBackend  # noqa: E402


def run(args, workers: int) -> None:
    backend = LocalBackend(args.dataset)
    if args.bandwidth_mbps or args.latency_ms:
        backend = ThrottledBackend(
            backend,
            latency_s=args.latency_ms / 1e3,
            bandwidth_bytes_per_s=args.bandwidth_mbps * 1e6 if args.bandwidth_mbps else None,
        )
    ds = DeltaCellsDataset(
        backend,
        max_cached_tiles=args.cached_tiles,
        max_prefetch_tiles=args.prefetch_tiles + 2,
        io_threads=args.io_threads,
        decode_threads=args.decode_threads,
    )
    loader = torch.utils.data.DataLoader(
        TileShuffleDataset(
            ds, args.batch_size, shuffle=True, seed=0, prefetch_tiles=args.prefetch_tiles, drop_last=True
        ),
        batch_size=None,
        num_workers=workers,
        prefetch_factor=2 if workers else None,
    )
    it = iter(loader)
    warm = max(2, 2 * max(workers, 1))
    for _ in range(warm):
        next(it)
    n_cells = n_batches = 0
    wait = 0.0
    t_start = t_wait = t_end = time.perf_counter()
    for batch in it:
        wait += time.perf_counter() - t_wait
        x = to_torch_csr(batch, ds.n_genes)  # what the training process does (then moves to the GPU and densifies)
        n_cells += x.shape[0]
        n_batches += 1
        t_end = t_wait = (
            time.perf_counter()
        )  # timing stops at the last received batch: DataLoader worker shutdown is excluded
        if n_batches >= args.max_batches:
            break
    dt = t_end - t_start
    print(
        f"workers={workers}: {n_cells / dt / 1e3:7.1f}k cells/s  ({dt / max(n_batches, 1) * 1e3:6.0f} ms per batch of {args.batch_size}; "
        f"main process waited {wait / max(n_batches, 1) * 1e3:.0f} ms per batch; {n_batches} batches timed)  [{backend.describe()}]",
        flush=True,
    )
    del it, loader


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--workers", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--cached-tiles", type=int, default=1, help="Decoded tiles kept per worker.")
    parser.add_argument("--prefetch-tiles", type=int, default=2, help="Tiles fetched ahead per worker.")
    parser.add_argument("--io-threads", type=int, default=4)
    parser.add_argument("--decode-threads", type=int, default=1)
    parser.add_argument("--bandwidth-mbps", type=float, default=None, help="Emulated per-connection bandwidth (MB/s).")
    parser.add_argument("--latency-ms", type=float, default=0.0, help="Emulated per-read latency.")
    args = parser.parse_args(argv)
    for w in [int(x) for x in args.workers.split(",")]:
        run(args, w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
