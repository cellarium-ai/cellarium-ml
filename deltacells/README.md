# deltacells

Fast, compact, cloud-friendly **tiles of single-cell count matrices**, built for feeding a training `DataLoader`.

A dataset is a directory (or bucket prefix) of *tiles*, each one self-contained object holding ~10k cells x all genes. Within a
tile the counts are stored as per-cell **gene gaps** (delta coding of the sorted gene indices) and **counts**, as byte-planar
uint16 streams compressed with **zstd**; a small C core decompresses a tile into CSR arrays in ~0.1 s per core (0.025 s on 8 cores)
without any intermediate AnnData / Arrow objects. Genes are sorted by decreasing total counts, which is what makes the
compression work.

> **Status: alpha (0.1.0).** Developed and tested on macOS arm64 / Python 3.10. The GCS backend has not been exercised against
> a real bucket. See [Limitations](#limitations).

## At a glance

One real shard (10,000 cells x 38,094 genes, snRNA-seq, 5,171 nonzeros per cell), **X only** (no obs / var), written in each format
and read from a local file back into RAM on an M2 Pro laptop (warm page cache, no network). Reproduce with
`python benchmarks/compare_formats.py shard.h5ad` (the BPCells C++ row needs `benchmarks/bpcells_decode/build.sh` once):

| format | stored X | kB/cell | file -> RAM, 1 thread | 8 threads |
|---|---|---|---|---|
| h5ad (as is: original gene order, gzip) | 124.6 MB | 12.46 | 1170 ms | n/a (single-threaded reader) |
| TileDB-SOMA (tuned layout), -> CSR | 56.7 MB | 5.67 | 1361 ms | 394 ms |
| BPCells, Python bindings | 67.4 MB | 6.74 | 1234 ms | 275 ms |
| BPCells, C++ decoder (uint32 values, preallocated) | 67.4 MB | 6.74 | **48 ms** | **30 ms** |
| **deltacells** (zstd 19), reused buffers | **38.4 MB** | **3.84** | 125 ms | 39 ms |
| deltacells (zstd 19), fresh arrays per call | 38.4 MB | 3.84 | 148 ms | 56 ms |

How to read it:

* **BPCells' decoder is the fastest reader here, by a wide margin over its own bindings.** Calling BPCells' BP-128 decoders directly
  (the harness in `benchmarks/bpcells_decode/`, which downloads and builds against a pinned BPCells commit) takes 48 ms for a tile,
  including reading its files; the Python bindings take 1.2 s because they build the result through an Eigen sparse matrix. The
  harness is a measurement tool, not a supported reader: it leaves values as uint32 and decodes into preallocated arrays.
* **deltacells trades decode CPU for size**: its files are 43% smaller than BPCells' (38.4 vs 67.4 MB) and it needs ~2.6x the decode
  time. Which wins end to end depends on where the tiles live. If fetching a tile is slower than decoding it, the smaller file wins:
  with fetch and decode overlapped, deltacells' tile time is lower below roughly 500 MB/s of per-worker bandwidth (one decode thread
  each), and the break-even is lower (~400 MB/s) when they are not overlapped. These crossovers are estimates computed from the
  numbers above, not measurements. With fast local storage and few cores to spare, BPCells' decoder is the cheaper reader; deltacells
  also needs about 2.6x the CPU time per tile.
* The sorted formats (TileDB-SOMA, BPCells, deltacells) share the same gene order (genes sorted by decreasing total counts of this
  shard); the TileDB-SOMA layout is the best I found (tile extent = the shard, row-major, delta + zstd 3 on the gene coordinate,
  byte-shuffle + zstd 3 on the values). h5ad is left as users have it. Every row includes reading its file from disk.
* The time is for X alone; obs is a separate concern. The h5ad file as a whole is 148 MB, of which X is 124.6 MB.

Through a `DataLoader` (the reference loader in `examples/`: shuffled tiles and cells, 5000-cell batches in shared memory, CSR
tensor rebuilt in the main process), same machine, 10 tiles:

| workers | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| cells / s | 50k | 45k | 78k | 95k |

and with the object store emulated by `ThrottledBackend` (50 ms latency, per-connection bandwidth, 2 workers), where prefetching the
next tiles matters:

| emulated bandwidth per connection | no prefetch | `prefetch_tiles=2` |
|---|---|---|
| 150 MB/s | 35k cells/s | 62k cells/s |
| 60 MB/s | 21k cells/s | 48k cells/s |

In the design experiments, the same pipeline structure gave 6-11k cells/s over TileDB-SOMA tiles and 70-130k cells/s around BPCells'
C++ decoder (1.3-1.8x faster than deltacells on this laptop) at 1-3 workers; those two are one-off scripts, not in this repository. These are laptop numbers with a warm cache and no real network; `benchmarks/` reproduces the deltacells ones on your
own data and hardware.

## Install

Needs a C compiler and **libzstd** (headers + library):

```bash
conda install zstd                      # or: brew install zstd / apt install libzstd-dev
pip install -e ".[test]"                # from this directory; extras: torch, h5ad (anndata), obs (pyarrow, pandas), gcs (google-cloud-storage)
```

`ZSTD_ROOT` points the build at a custom zstd prefix; `DELTACELLS_STATIC_ZSTD=1` links it statically. Details in
[DEVELOPING.md](DEVELOPING.md).

## Quick start

Convert h5ad shards of any sizes into a dataset of exact-size tiles (this sorts the genes, re-chunks the cells and compresses):

```bash
deltacells convert --h5ad-glob "shards/*.h5ad" --output /data/my_dataset --tile-size 10000
deltacells info /data/my_dataset
deltacells verify /data/my_dataset          # CRC-checks and decodes every tile
```

Read it:

```python
import numpy as np
from deltacells import open_dataset

ds = open_dataset("/data/my_dataset")             # or "gs://bucket/prefix"
ds.n_cells, ds.n_genes, ds.limits                 # limits: cumulative cells per tile
batch = ds.get_batch(np.array([5, 99_000, 12]))   # any global cell indices, any order; returns a CSR SparseBatch
batch.indptr, batch.indices, batch.values         # int64, int32, float32 numpy arrays
x = batch.to_torch_csr()                          # torch.sparse_csr tensor sharing memory (needs torch)
batch.to_scipy()
ds.var_names, ds.gene_order                       # gene names / original column of each stored gene
```

Cell metadata (obs) and the gene table (var), if the dataset has them (needs `pyarrow` and `pandas`; see
[Cell metadata](#cell-metadata-obs)):

```python
ds = open_dataset("gs://bucket/prefix", obs_columns=["cell_type", "donor_id"])  # copy just these columns to the local cache, once
ds.obs.columns                                    # every column and its kind; ds.obs.categories("cell_type") is the vocabulary
codes = ds.obs.take(batch_cell_indices, ["cell_type", "donor_id"])  # {"cell_type": int codes, ...}: local, ~0.2 ms per 5000 cells
ds.obs.localize(["age"])                          # add more columns whenever you like (only their bytes are read)
df = ds.obs.to_pandas(["cell_type", "age"])       # whole-dataset DataFrame, e.g. for plotting
ds.var                                            # gene table (pandas), in column order
```

Write your own data:

```python
from deltacells import DatasetWriter

with DatasetWriter("/data/out", n_genes=G, tile_size=10_000, gene_order=order, var_names=names) as w:
    for block in blocks:                  # scipy CSR / anything scipy can convert; integer counts in [0, 65535]
        w.add_tile(block)                 # every tile but the last must have exactly tile_size cells
```

Lower level: `encode_tile(matrix) -> bytes`, `write_tile(path, matrix)`, `Tile(bytes).decode(threads=4)`, `read_tile(path)`.

## Using it in a DataLoader (and from cellarium-ml)

`DeltaCellsDataset` is meant to sit behind a worker. Its contract is deliberately the one a source needs for an
`IterableDistributedAnnDataCollectionDataset`-style orchestrator:

| orchestrator needs | deltacells |
|---|---|
| number of cells / genes | `n_cells`, `n_genes` |
| shard boundaries for shuffling and splitting | `limits`, `tile_bounds(i)`, `tile_of(indices)` |
| fetch a batch by global index | `get_batch(indices, values_dtype=..., out_indices=..., out_values=...)` |
| size the output buffers first (e.g. shared memory) | `batch_nnz(indices)` |
| "I will need these shards next" | `prefetch(tile_ids)`, `prefetch_cells(indices)` |
| gene names / gene table | `var_names`, `var` |
| per-cell metadata for a batch | `obs.take(indices, columns)` after `obs.localize(columns)` (or `obs_columns=` when opening) |

Things to know:

* **One tile = one object read.** A batch that spans two tiles reads and decodes both, exactly like two h5ad shards. Iterate
  tile by tile (shuffle tiles, then cells within a tile) so that consecutive batches share a tile.
* **Prefetch.** Call `prefetch` with the tiles you will need soon (the reference loader asks for the next two). Fetching and
  parsing run on background threads and the compressed tiles are cached (`max_prefetch_tiles`).
* **Memory.** A decoded tile costs ~8 bytes per nonzero (~0.4 GB for 10k deep cells); `max_cached_tiles` (default 2) bounds that,
  and decode buffers are recycled instead of reallocated (fresh allocations page-fault, costing tens of milliseconds per tile).
* **Threads.** `decode_threads=1` is right when each worker has about one core; the C core releases the GIL, so more threads
  decode the tile's chunks in parallel. The dataset object is picklable (caches and threads are rebuilt in each worker).
* **Obs / metadata** are never fetched with the tiles; see the next section. cellarium-ml would hold the choice of columns
  (e.g. an `obs_columns=` argument on the datamodule, passed to `localize`) and the semantics of each column.
* **Gene order** is part of the data: columns are in the stored (sorted) order. Models that need the original order can use
  `ds.gene_order`; a fixed order across all tiles is what lets batches from different tiles be concatenated.

`examples/reference_loader.py` is a complete `IterableDataset` showing the pattern (tile shuffling, replica and worker splits,
prefetch hints, shared-memory output) and `examples/dataloader_example.py` runs it end to end on synthetic data.

## Cell metadata (obs)

Obs lives *beside* the tiles, not in them, so a loader that needs no metadata pays nothing for it, and a loader that needs a
few columns pays only for those. In the dataset it is stored as parquet **shards** (`obs/obs_0000.parquet`, ...) of up to 1000
tiles each with **one row group per tile**, so obs row group `i` is always tile `i`. Categorical columns are stored as integer
codes with one **global vocabulary** per column (shards from different files can have different category sets; the converter
unifies them), and columns with more than 20,000 distinct values (barcodes) are stored as strings.

Reading never touches the shards during training. Columns are **localized** once, on request:

* only the requested columns' bytes are read (ranged reads of the footer and of each row group's column chunk, in parallel), and
  written to a node-local cache (`$DELTACELLS_CACHE`, else `~/.cache/deltacells`) as one memory-mapped Arrow IPC file per column;
* the cache is **additive**: asking for another column later reads only that column; the columns you already have are untouched;
* it is shared by every process on the node (a lock makes concurrent processes fetch each column once) and keyed by the dataset's
  content fingerprint, so a changed dataset never reuses stale files;
* afterwards `obs.take(indices, columns)` is a zero-copy lookup in memory-mapped files: workers share the pages, opening a column
  costs milliseconds and about a megabyte, and a 5000-cell batch takes ~0.2 ms.

Pre-select the columns your models use with `obs_columns=[...]` (or `deltacells obs localize DATASET --columns a,b,c`) so nothing
stalls a training run; a column that was not localized is fetched on first use, with a warning. Localizing everything is possible
(`--all`) but not the point:

| 100,000 real cells x 50 obs columns (+ cell names) | stored size | per 100M cells |
|---|---|---|
| all 51 columns | 6.1 MB | 6.1 GB |
| `cell_type` (30 categories) | 0.1 MB | 55 MB |
| `donor_id` (178 categories) | 0.1 MB | 96 MB |
| `age` (float) | 0.1 MB | 75 MB |
| `frac_contamination` (float) | 0.6 MB | 556 MB |
| `cell_barcode` (string) | 0.7 MB | 745 MB |

`deltacells obs info DATASET` prints this table for any dataset. Localizing `cell_type`, `donor_id` and `age` read 0.29 MB in 31
ranged reads (the three columns plus the parquet footers; 4.8% of the obs bytes) and took 0.39 s against a store emulated at 150
MB/s with 50 ms latency (`benchmarks/bench_obs.py`). The requests are one per column per tile, so at 100M cells (10,000 tiles) the
time is dominated by request latency: roughly `columns x tiles x latency / threads` (an extrapolation, not a measurement; raise
`threads` for high-latency stores).

Missing values: categorical codes use `-1`; numeric columns with missing values come back as float64 with NaN; booleans and
strings with missing values come back as object arrays with `None`. `to_pandas` gives pandas categoricals for categorical columns.

## How it works

`docs/FORMAT.md` is the specification. In short:

1. Genes are sorted by decreasing total counts (first shard as a proxy): frequently detected genes get small, adjacent column
   indices, so the gaps between a cell's nonzero genes are small and repetitive.
2. Per cell, the gene indices are delta coded (`g0, g1-g0, g2-g1, ...`); with the counts that gives two uint16 streams.
3. Each stream is split into byte planes (all low bytes, then all high bytes), compressed with zstd, per chunk of ~1/8 of the
   tile's nonzeros. Chunks are independent, so a tile decodes in parallel, at ~0.5% size cost.
4. A tile is a header, a chunk table, `indptr`, and the compressed streams: one object, decodable on its own, CRC-protected.

Measured trade-offs on one real tile (`benchmarks/bench_compression.py`):

| gene order | zstd level | chunks | size | decode (1 thread) | encode (8 threads) |
|---|---|---|---|---|---|
| sorted | 3 | 8 | 43.1 MB | 138 ms | 0.4 s |
| sorted | 12 | 8 | 41.2 MB | 129 ms | 1.1 s |
| sorted | **19** | 8 | **38.4 MB** | **114 ms** | 7.2 s |
| sorted | 19 | 1 | 38.2 MB | 110 ms | 24.2 s |
| random | 19 | 8 | 49.3 MB | 119 ms | 8.2 s |

Sorting the genes saves 21-22% at every level; higher zstd levels are smaller *and* decode faster (they only cost compression time).

## Limitations

* **Counts are uint16**: integers in [0, 65535] and at most 65,536 genes. The writer raises rather than clipping. No normalized or
  float data.
* **GCS is untested against a real bucket**; the backend is a thin wrapper over `google-cloud-storage` and the pipeline logic is
  tested with a fake client and the throttled backend. Reads fetch whole tiles with `download_as_bytes` (one extra copy).
* **Platforms**: tested on macOS arm64 only (Linux x86-64 should work but has not been run); little-endian hosts only.
* **Obs needs `pyarrow` and `pandas`** and supports categorical, numeric, boolean and string columns (exclude others, e.g.
  datetimes). The obs localization and shard layout are only exercised against local files and the throttled backend; like the
  rest of the GCS path, ranged reads from a real bucket (`GCSBackend.read_range`) are untested. The local cache is uncompressed
  Arrow (several times the size of the parquet for small-integer columns: 4.8x for the three columns above); strings and floats cost their full size on disk.
* **No orchestrator**: no resume-from-checkpoint arithmetic, DDP padding or `drop_last_indices`; the reference loader is a sketch of
  how to call the dataset, not a replacement for cellarium's.
* **The dataset is not thread-safe for concurrent `get_batch`** calls (they are serialized by a lock).
* **Gene order depends on the first shard** passed to `convert`; two conversions of the same cells with different first shards (for
  example differently sized files) sort genes slightly differently. With `--no-sort-genes` the result depends only on cell order
  and tile size.
* Compression at level 19 takes ~7 s per 10k-cell tile with 8 threads (24 s with one chunk); lower levels trade size for speed.

## Tests and benchmarks

```bash
pytest tests                       # 245 tests: format validation, round trips and edge cases, gather, dataset (prefetch, caching,
                                   # pickling, corruption), obs (schema, localization byte counts, cache races), backends, h5ad
                                   # conversion, CLI, torch / DataLoader, benchmark smoke tests
python benchmarks/bench_decode.py DATASET
python benchmarks/bench_pipeline.py DATASET --workers 0,1,2,4 [--bandwidth-mbps 150 --latency-ms 50]
python benchmarks/bench_compression.py DATASET
python benchmarks/bench_obs.py DATASET --columns cell_type,donor_id
python benchmarks/check_against_h5ad.py DATASET "shards/*.h5ad"        # X and every obs column, cell by cell
python benchmarks/make_synthetic.py OUT_DIR       # realistic synthetic data if you have no real shards
```

## License

BSD-3-Clause (see `LICENSE`). Links libzstd (BSD-3-Clause / GPL-2.0 dual licensed; used under BSD).
