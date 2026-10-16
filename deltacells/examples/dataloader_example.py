# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
End-to-end example: build a small synthetic dataset, read it through a ``DataLoader`` and densify batches "on the device".

    python examples/dataloader_example.py                  # synthetic data in a temporary directory
    python examples/dataloader_example.py /path/to/dataset  # an existing dataset (e.g. made with `deltacells convert`)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_loader import TileShuffleDataset, to_torch_csr  # noqa: E402

from deltacells import DatasetWriter, open_dataset  # noqa: E402


def make_demo_dataset(root: str, n_cells: int = 2_000, n_genes: int = 500, tile_size: int = 500, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    with DatasetWriter(root, n_genes=n_genes, tile_size=tile_size, level=3, n_chunks=2) as writer:
        for lo in range(0, n_cells, tile_size):
            n = min(tile_size, n_cells - lo)
            m = sp.random(n, n_genes, density=0.1, format="csr", random_state=rng)
            m.data = rng.integers(1, 20, m.nnz).astype(np.float32)
            writer.add_tile(m)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        path = args.path
        if path is None:
            path = os.path.join(tmp, "demo")
            make_demo_dataset(path)
        ds = open_dataset(path, max_cached_tiles=1, decode_threads=1)
        print(f"{ds.n_cells} cells x {ds.n_genes} genes in {ds.n_tiles} tiles of {ds.tile_size}")
        loader = torch.utils.data.DataLoader(
            TileShuffleDataset(ds, args.batch_size, shuffle=True, seed=0, drop_last=True),
            batch_size=None,  # the dataset already yields whole batches
            num_workers=args.workers,
            prefetch_factor=2 if args.workers else None,
        )
        t0 = time.perf_counter()
        n_seen = 0
        for i, batch in enumerate(loader):
            x_sparse = to_torch_csr(batch, ds.n_genes)
            x_ng = (
                x_sparse.to_dense()
            )  # in training, move to the GPU first, then densify (e.g. cellarium's Densify transform)
            n_seen += x_ng.shape[0]
            if i == 0:
                print(
                    f"first batch: {tuple(x_ng.shape)}, {x_sparse._nnz()} nonzeros, total counts {x_ng.sum().item():.0f}"
                )
            if i + 1 >= args.max_batches:
                break
        print(f"{n_seen} cells in {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
