# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate a synthetic count dataset with realistic structure (skewed gene popularity, variable depth, mostly small counts), written
as a deltacells dataset with genes sorted by decreasing total counts. For benchmarking without real data.

    python benchmarks/make_synthetic.py OUTPUT_DIR --n-cells 100000 --n-genes 30000 --mean-nnz 3000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp

from deltacells import DatasetWriter


def synthetic_block(
    rng: np.random.Generator, n_cells: int, weights: np.ndarray, mean_nnz: float, block: int = 500
) -> sp.csr_matrix:
    """Cells whose gene detection probability follows ``weights`` and whose depth varies log-normally."""
    n_genes = len(weights)
    w = weights / weights.sum()
    parts = []
    for lo in range(0, n_cells, block):
        n = min(block, n_cells - lo)
        depth = rng.lognormal(mean=np.log(mean_nnz), sigma=0.5, size=(n, 1))
        p = 1.0 - np.exp(-depth * w[None, :] * 1.6)  # expected detected genes ~ depth for moderate depth
        present = rng.random((n, n_genes)) < p
        counts = 1 + rng.poisson(w[None, :] * depth * 8.0, size=(n, n_genes))
        parts.append(sp.csr_matrix(np.where(present, np.minimum(counts, 65535), 0).astype(np.float32)))
    return sp.vstack(parts, format="csr")


def synthetic_obs(rng: np.random.Generator, n: int, n_float_columns: int):
    """A synthetic obs frame: three categoricals (fixed vocabularies), counts, ``n_float_columns`` QC-like floats and a barcode string."""
    import pandas as pd

    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                rng.choice([f"type{i}" for i in range(20)], n), categories=[f"type{i}" for i in range(20)]
            ),
            "donor": pd.Categorical(
                rng.choice([f"donor{i}" for i in range(200)], n), categories=[f"donor{i}" for i in range(200)]
            ),
            "batch": pd.Categorical(
                rng.choice([f"batch{i}" for i in range(8)], n), categories=[f"batch{i}" for i in range(8)]
            ),
            "n_counts": rng.integers(500, 100_000, n).astype(np.int64),
        }
    )
    for i in range(n_float_columns):
        obs[f"qc_{i:02d}"] = rng.normal(size=n)
    obs["barcode"] = [f"{x:016X}" for x in rng.integers(0, 1 << 62, n)]
    return obs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--n-cells", type=int, default=50_000)
    parser.add_argument("--n-genes", type=int, default=30_000)
    parser.add_argument("--mean-nnz", type=float, default=3_000)
    parser.add_argument("--tile-size", type=int, default=10_000)
    parser.add_argument("--level", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--obs-columns",
        type=int,
        default=0,
        help="Also write synthetic obs with this many QC-like float columns (0: no obs).",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    weights = 1.0 / (np.arange(args.n_genes) + 10.0) ** 0.9
    rng.shuffle(weights)  # popularity is unrelated to the source gene order, as in real data
    t0 = time.time()
    schema = None
    if args.obs_columns:
        from deltacells.obs import ObsSchema

        schema = ObsSchema.infer([synthetic_obs(np.random.default_rng(args.seed + 1), 10, args.obs_columns)])
    writer = DatasetWriter(
        args.output,
        n_genes=args.n_genes,
        tile_size=args.tile_size,
        level=args.level,
        overwrite=args.overwrite,
        obs_schema=schema,
    )
    order = None
    for lo in range(0, args.n_cells, args.tile_size):
        block = synthetic_block(rng, min(args.tile_size, args.n_cells - lo), weights, args.mean_nnz)
        if order is None:  # sort genes by the first tile's totals, like `deltacells convert`
            order = np.argsort(-np.asarray(block.sum(0)).ravel(), kind="stable")
            writer.gene_order = order
        obs = None if schema is None else synthetic_obs(rng, block.shape[0], args.obs_columns)
        writer.add_tile(block, obs=obs)
        print(f"tile {lo // args.tile_size}: {block.nnz / block.shape[0]:.0f} nonzeros per cell", flush=True)
    m = writer.close()
    total = sum(m.tile_bytes)
    print(
        f"wrote {m.n_cells} cells in {time.time() - t0:.0f}s: {total / 1e6:.1f} MB = {total / m.n_cells / 1e3:.2f} kB per cell"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
