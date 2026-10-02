# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Compare storing and reading ONE shard of a count matrix (X only, no obs/var) in several formats:

* h5ad                  the shard as it is (original gene order, gzip), X read with ``anndata.io.read_elem``
* TileDB-SOMA           a SOMA SparseNDArray laid out for tile reads (tile extent = the shard, row-major, delta + zstd 3 on the
                        gene coordinate, byte-shuffle + zstd 3 on the values)   [needs ``tiledbsoma``]
* BPCells               BP-128 bit-packed matrix, read two ways   [needs ``bpcells``]:
                          - through its Python bindings (``DirMatrix[:, :]``), and
                          - with BPCells' own C++ decoders in a small harness that bypasses the bindings
                            (build it once with ``benchmarks/bpcells_decode/build.sh``)
* deltacells            this package, with reused output buffers (what ``DeltaCellsDataset`` does) and with fresh arrays per call

    python benchmarks/compare_formats.py data/shard_001.h5ad [--threads 8] [--level 19] [--repeats 5] [--only h5ad,deltacells]

What is timed is getting the shard's X from a file on local disk (warm page cache, no network) into RAM, as a scipy CSR matrix
(h5ad, TileDB-SOMA, BPCells bindings, deltacells) or as flat CSR arrays (the BPCells C++ decoder: values left as uint32, output
preallocated). Sizes are the bytes stored for X. All formats except h5ad use the same gene order (genes sorted by decreasing total
counts *of this shard*, which is optimistic compared to deriving the order from another shard); h5ad is left as it is, because that
is what users have. "Fresh arrays" rows allocate (and page-fault) their output on every call, like the readers that cannot reuse
buffers; "reused buffers" rows do not. TileDB-SOMA returns Arrow coordinates; "-> CSR" adds the row-pointer search and the
int64 -> int32 index cast.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from deltacells import Tile, encode_tile, permute_columns

HERE = Path(__file__).resolve().parent
DEFAULT_HARNESS = HERE / "bpcells_decode" / "build" / "bpcells_decode_harness"


def dir_bytes(path: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def median_seconds(fn, repeats: int) -> float:
    fn()  # warm up (page cache, imports, thread pools)
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
        del out
    return statistics.median(times)


def parse_harness_output(text: str) -> dict[str, float]:
    """Parse the ``RESULT key=value ...`` line printed by the BPCells decode harness."""
    for line in text.splitlines():
        if line.startswith("RESULT "):
            return {k: float(v) for k, v in (kv.split("=") for kv in line.split()[1:])}
    raise ValueError(f"no RESULT line in harness output: {text!r}")


def run_h5ad(path: str, repeats: int, threads: int):
    import anndata
    import h5py

    def read():
        with h5py.File(path, "r") as f:
            return anndata.io.read_elem(f["X"])

    with h5py.File(path, "r") as f:
        size = sum(f["X"][k].id.get_storage_size() for k in ("data", "indices", "indptr"))
    return [("h5ad (as is: original gene order, gzip)", size, {1: median_seconds(read, repeats)})]


def run_deltacells(x: sp.csr_matrix, workdir: str, level: int, repeats: int, threads: int):
    blob = encode_tile(x, level=level, n_chunks=8)
    path = os.path.join(workdir, "tile.dct")
    Path(path).write_bytes(blob)
    nnz = x.nnz
    reused, fresh = {}, {}
    out_i, out_v = np.empty(nnz, np.int32), np.empty(nnz, np.float32)
    for t in sorted({1, threads}):
        scratch = np.empty(t * Tile(blob).info.scratch_nbytes, dtype=np.uint8)

        def read_reused(t=t):
            tile = Tile(np.fromfile(path, dtype=np.uint8))
            return tile.decode(
                values=np.float32, threads=t, out_indices=out_i, out_values=out_v, scratch=scratch
            ).to_scipy()

        def read_fresh(t=t):
            return Tile(np.fromfile(path, dtype=np.uint8)).decode(values=np.float32, threads=t).to_scipy()

        reused[t], fresh[t] = median_seconds(read_reused, repeats), median_seconds(read_fresh, repeats)
    return [
        (f"deltacells (zstd {level}), reused buffers", len(blob), reused),
        (f"deltacells (zstd {level}), fresh arrays per call", len(blob), fresh),
    ]


def run_tiledbsoma(x: sp.csr_matrix, workdir: str, repeats: int, threads: int):
    import pandas as pd
    import tiledbsoma
    import tiledbsoma.io
    from anndata import AnnData

    n, g = x.shape
    uri = os.path.join(workdir, "soma")
    adata = AnnData(
        X=x.astype(np.float32),
        obs=pd.DataFrame(index=pd.Index([str(i) for i in range(n)], name="obs_id")),
        var=pd.DataFrame(index=pd.Index([str(j) for j in range(g)], name="var_id")),
    )
    zstd3 = {"_type": "ZstdFilter", "level": 3}
    create = {
        "dims": {
            "soma_joinid": {"tile": n},
            "soma_dim_0": {"tile": n},
            "soma_dim_1": {"tile": g, "filters": ["DeltaFilter", zstd3]},
        },
        "attrs": {"soma_data": {"filters": ["ByteShuffleFilter", zstd3]}},
        "cell_order": "row-major",
        "tile_order": "row-major",
        "goal_chunk_nnz": 2_000_000_000,
    }
    tiledbsoma.io.from_anndata(
        uri,
        adata,
        "RNA",
        X_layer_name="data",
        obs_id_name="obs_id",
        var_id_name="var_id",
        platform_config={"tiledb": {"create": create}},
    )
    size = dir_bytes(os.path.join(uri, "ms", "RNA", "X", "data"))
    times = {}
    for t in sorted({1, threads}):
        ctx = tiledbsoma.SOMATileDBContext(
            tiledb_config={"sm.compute_concurrency_level": str(t), "sm.io_concurrency_level": str(t)}
        )
        exp = tiledbsoma.Experiment.open(
            uri, context=ctx
        )  # keep a reference: the arrays close when it is garbage collected
        arr = exp.ms["RNA"].X["data"]

        def read(arr=arr):
            tb = arr.read((slice(0, n - 1), slice(None))).tables().concat()
            r, c, v = (tb[k].to_numpy() for k in ("soma_dim_0", "soma_dim_1", "soma_data"))
            indptr = np.searchsorted(r, np.arange(n + 1)).astype(np.int64)  # rows come back sorted
            return sp.csr_matrix((v, c.astype(np.int32), indptr), shape=(n, g))

        times[t] = median_seconds(read, repeats)
        exp.close()
    return [("TileDB-SOMA (tuned layout), -> CSR", size, times)]


def run_bpcells(x: sp.csr_matrix, workdir: str, repeats: int, threads: int, harness: Path | None):
    import bpcells.experimental as bp

    uri = os.path.join(workdir, "bpcells")
    xu = sp.csr_matrix((x.data.astype(np.uint32), x.indices, x.indptr), shape=x.shape)
    bp.DirMatrix.from_scipy_sparse(xu, uri)  # CSR input is stored cell-major (row-major)
    size = dir_bytes(uri)
    times = {}
    for t in sorted({1, threads}):
        mat = bp.DirMatrix(uri)
        mat.threads = 0 if t == 1 else t
        times[t] = median_seconds(lambda mat=mat: mat[:, :], repeats)
    rows = [("BPCells, Python bindings", size, times)]

    if harness is not None and harness.exists():
        expected = os.path.join(workdir, "bpcells_expected")
        os.makedirs(expected, exist_ok=True)
        xu.data.tofile(os.path.join(expected, "expected_val.bin"))
        xu.indices.astype(np.uint32).tofile(os.path.join(expected, "expected_idx.bin"))
        ranges = max(1, threads // 2)  # the two streams are decoded by `ranges` threads each
        proc = subprocess.run(
            [str(harness), uri, expected, str(max(5, repeats * 3)), str(ranges)],
            capture_output=True,
            text=True,
            check=False,
        )
        r = parse_harness_output(proc.stdout)
        if r["correct"] != 1:
            raise RuntimeError("the BPCells harness decoded different values than expected")
        read, d1, dn = r["read_ms"] / 1e3, r["decode_ms_1"] / 1e3, r["decode_ms_N"] / 1e3
        times_h = {1: read + d1}
        if threads != 1:
            times_h[int(r["threads_N"])] = read + dn
        rows.append(("BPCells, C++ decoder (uint32 values, preallocated)", size, times_h))
    else:
        print(
            f"note: BPCells C++ harness not found at {harness}; run benchmarks/bpcells_decode/build.sh to include it\n"
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5ad", help="One h5ad shard (integer counts; all of it is loaded).")
    parser.add_argument("--threads", type=int, default=8, help="Thread count for the multi-threaded column.")
    parser.add_argument("--level", type=int, default=19, help="zstd level for deltacells.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--only", default="h5ad,tiledbsoma,bpcells,deltacells", help="Comma separated subset of formats."
    )
    parser.add_argument(
        "--bpcells-harness", type=Path, default=DEFAULT_HARNESS, help="Path of the built BPCells decode harness."
    )
    args = parser.parse_args(argv)

    import anndata

    x = sp.csr_matrix(anndata.read_h5ad(args.h5ad).X)
    n = x.shape[0]
    totals = np.asarray(x.sum(0)).ravel()
    xs = permute_columns(x, np.argsort(-totals, kind="stable"))  # the shared gene order for the sorted formats
    print(f"{args.h5ad}: {n} cells x {x.shape[1]} genes, {x.nnz} nonzeros ({x.nnz / n:.0f} per cell)\n")

    rows = []
    with tempfile.TemporaryDirectory() as workdir:
        for name in args.only.split(","):
            try:
                if name == "h5ad":
                    rows += run_h5ad(args.h5ad, args.repeats, args.threads)
                elif name == "tiledbsoma":
                    rows += run_tiledbsoma(xs, workdir, args.repeats, args.threads)
                elif name == "bpcells":
                    rows += run_bpcells(xs, workdir, args.repeats, args.threads, args.bpcells_harness)
                elif name == "deltacells":
                    rows += run_deltacells(xs, workdir, args.level, args.repeats, args.threads)
                else:
                    raise ValueError(f"unknown format {name!r}")
            except ImportError as e:
                print(f"skipping {name}: {e}")

    multi = args.threads
    print(f"| format | stored X | kB/cell | file -> RAM, 1 thread | {multi} threads |")
    print("|---|---|---|---|---|")
    for label, size, times in rows:
        t1 = f"{times[1] * 1e3:.0f} ms"
        tn = next((f"{v * 1e3:.0f} ms" for k, v in times.items() if k != 1), "n/a (single-threaded reader)")
        print(f"| {label} | {size / 1e6:.1f} MB | {size / n / 1e3:.2f} | {t1} | {tn} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
