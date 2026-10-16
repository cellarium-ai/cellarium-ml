# Developing deltacells

## Build

The package has one C extension (`src/deltacells/_core.c`) that links against **libzstd** (headers and library). The build
looks for it in `$ZSTD_ROOT`, `$CONDA_PREFIX`, `sys.prefix`, Homebrew, `/usr/local` and `/usr`.

```bash
conda install zstd            # or: brew install zstd / apt install libzstd-dev
pip install -e ".[test]"      # editable install (builds the extension); the obs features need pyarrow and pandas (extra: obs)
# or, without installing, build the extension next to the sources:
python setup.py build_ext --inplace
```

`DELTACELLS_STATIC_ZSTD=1` links `libzstd.a` statically (self-contained wheels). Supported: Linux and macOS, little-endian,
Python >= 3.10.

## Test

```bash
pytest tests                  # from this directory
pytest deltacells/tests       # from a parent repository, after an in-place build or an install
```

`tests/conftest.py` makes the tests importable straight after `build_ext --inplace`, and skips collecting them when the package
is not available, so a parent repository's `pytest` run does not fail on a machine without the extension. Tests that need
`torch` or `anndata` are skipped when those are missing.

## Benchmarks

```bash
python benchmarks/make_synthetic.py /tmp/syn --n-cells 50000 --n-genes 30000     # synthetic data, no h5ad needed
deltacells convert --h5ad-glob "shards/*.h5ad" --output /tmp/real                # or convert real shards
python benchmarks/bench_decode.py /tmp/real
python benchmarks/make_synthetic.py /tmp/syn --obs-columns 20                    # synthetic data with obs
python benchmarks/bench_obs.py /tmp/real --columns cell_type,donor_id             # obs: bytes read, localization time, lookup latency
python benchmarks/compare_formats.py shard.h5ad                                  # h5ad vs TileDB-SOMA vs BPCells vs deltacells, one shard
benchmarks/bpcells_decode/build.sh                                               # optional: BPCells' own C++ decoder for that comparison
python benchmarks/bench_pipeline.py /tmp/real --workers 0,1,2,4
python benchmarks/bench_pipeline.py /tmp/real --workers 2 --bandwidth-mbps 150 --latency-ms 50   # emulate a remote store
python benchmarks/bench_compression.py /tmp/real --levels 3,12,19 --chunks 1,8
python benchmarks/check_against_h5ad.py /tmp/real "shards/*.h5ad"                # cell-by-cell comparison with the source
```

Benchmarking notes: run on an otherwise idle machine (the pipeline benchmark needs a few GB of free RAM per worker); the
`bench_pipeline` timing stops at the last received batch so that DataLoader worker shutdown is not counted, and very small
datasets give very short timed windows.

## Layout of the code

| file | role |
|---|---|
| `src/deltacells/_core.c` | decode a chunk, gather rows, compress a stream (all release the GIL) |
| `format.py` | binary layout constants, header / chunk-table parsing and validation |
| `writer.py` | `encode_tile`, `write_tile`, `DatasetWriter`, `permute_columns` |
| `reader.py` | `Tile` (parse, verify, decode), `DecodedTile`, `SparseBatch`, row gather |
| `manifest.py` | `Manifest` (dataset JSON) |
| `backends.py` | `LocalBackend`, `GCSBackend`, `ThrottledBackend` |
| `dataset.py` | `DeltaCellsDataset`: tile prefetch, decoded-tile cache with buffer reuse, `get_batch`, `obs`, `var` |
| `obs.py` | per-cell metadata: `ObsSchema`, tile-aligned parquet shards, `ObsStore` (localization to a per-column Arrow cache, `take`, `to_pandas`) |
| `convert.py`, `cli.py` | h5ad conversion and the `deltacells` command |
| `examples/` | a reference `IterableDataset` and a runnable example |
| `benchmarks/bpcells_decode/` | C++ harness timing BPCells' decoders directly (downloads a pinned BPCells at build time; nothing vendored) |

## Changing the format

Bump `VERSION` in `format.py` and `manifest.py`, keep readers rejecting versions they do not know, and update `docs/FORMAT.md`
and the tests in `tests/test_format.py`.
