# deltacells format, version 1

A *dataset* is a directory (or bucket prefix) of *tiles* plus a small manifest. A *tile* is one self-contained object holding
a block of consecutive cells (rows) of a count matrix in CSR form. All integers are **little-endian**; readers on big-endian
hosts must refuse to load tiles (the reference implementation does).

## Dataset layout

```
<root>/
  manifest.json            required; written last, so a directory without it is an incomplete dataset
  tiles/tile_000000.dct    tile i = cells limits[i-1] : limits[i]   (zero-padded 6-digit index)
  tiles/tile_000001.dct
  ...
  gene_order.npy           optional; int32 array, output column j was column gene_order[j] of the source
  var_names.txt            optional; one gene name per line, in output (column) order
  var.parquet              optional; the gene table (one row per output column, in order; the pandas index is kept)
  obs/schema.json          optional; the obs schema (see "Obs" below)
  obs/obs_0000.parquet     optional; obs shards: up to obs_tiles_per_file tiles each, one row group per tile
  obs/obs_0001.parquet
```

Every tile except the last holds exactly `tile_size` cells; the last holds between 1 and `tile_size`. A global cell index
`c` lives in tile `searchsorted(limits, c, side="right")` at local row `c - (limits[tile - 1] or 0)`, where `limits` is the cumulative
sum of `tile_cells`.

### manifest.json

JSON object with at least:

| key | meaning |
|---|---|
| `format` | `"deltacells-dataset"` |
| `version` | `1` |
| `n_cells`, `n_genes`, `tile_size` | integers |
| `tile_cells`, `tile_nnz`, `tile_bytes` | per-tile lists (cells, stored nonzeros, file size in bytes) |
| `n_chunks`, `zstd_level` | encoder settings (informational) |
| `has_gene_order`, `has_var_names`, `has_var_table` | whether the optional files exist |
| `has_obs` | whether the dataset has obs; if so the next four fields are required |
| `obs_tiles_per_file` | tiles (row groups) per obs shard; shard `s` holds tiles `s * K .. (s + 1) * K - 1` |
| `obs_shard_bytes`, `obs_shard_sha256` | per-shard size and SHA-256 (hex) |
| `obs_fingerprint` | 24 hex characters identifying the obs content (see below) |
| `metadata` | free-form JSON object |

Readers must reject manifests whose `format`/`version` they do not understand or whose counts are inconsistent.

## Tile layout

```
offset 0                 header                64 bytes
offset 64                chunk table           n_chunks * 64 bytes
offset indptr_off        indptr                (n_cells + 1) * int64, 8-byte aligned
...                      streams               each compressed stream starts on an 8-byte boundary
```

### Header (64 bytes)

| offset | size | field | notes |
|---|---|---|---|
| 0 | 8 | magic | `DELTACEL` |
| 8 | 4 | version | `1` |
| 12 | 4 | flags | must be `0` (readers reject unknown flags) |
| 16 | 8 | n_cells | |
| 24 | 8 | nnz | stored nonzeros in the tile |
| 32 | 4 | n_genes | `1 .. 65536` |
| 36 | 4 | n_chunks | |
| 40 | 8 | indptr_off | multiple of 8, at or after the end of the chunk table |
| 48 | 8 | total_size | size of the whole blob; must equal the object size |
| 56 | 4 | crc32 | CRC-32 (zlib / IEEE) of bytes `[64, total_size)` |
| 60 | 4 | reserved | `0` |

### Chunk table entry (64 bytes, `n_chunks` of them, starting at offset 64)

`cell_lo, cell_hi, nnz_lo, nnz_hi, gaps_off, gaps_len, vals_off, vals_len`, each a `uint64`.

Chunks cover the tile contiguously: the first has `cell_lo = nnz_lo = 0`, every chunk starts where the previous one ended, and
the last ends at `n_cells` / `nnz`. Chunk boundaries are chosen so that chunks hold roughly equal numbers of nonzeros, which
makes them a convenient unit of parallel decoding. A tile with zero cells has zero chunks.

### indptr

`int64[n_cells + 1]`, non-decreasing, `indptr[0] = 0`, `indptr[n_cells] = nnz`, and `indptr[cell_lo] = nnz_lo`,
`indptr[cell_hi] = nnz_hi` for every chunk. The nonzeros of cell `r` are the entries `indptr[r] : indptr[r+1]` of the streams
below. The indptr is stored uncompressed: it is 8 bytes per cell (0.2% of a typical tile).

### Streams

For each chunk with `n = nnz_hi - nnz_lo` nonzeros there are two streams, `gaps` and `vals`. Each is a single **zstd frame**
(including its content size) that decompresses to exactly `2 * n` bytes: the `n` uint16 values of the stream split into byte
planes, i.e. all `n` low bytes followed by all `n` high bytes.

* **gaps**: for each cell, the genes with nonzero counts in increasing order `g_0 < g_1 < ...`; the stored gaps are
  `g_0, g_1 - g_0, g_2 - g_1, ...` (the first gap of a cell is its first gene index, so cells decode independently). Every gene
  index is `< n_genes <= 65536`, so every gap fits in a uint16.
* **vals**: the counts, in the same order. Counts are integers in `[1, 65535]`; zeros are never stored.

A chunk with `n = 0` still has two (tiny) valid frames.

Decoding a chunk: decompress both streams; for every cell in `[cell_lo, cell_hi)` take its slice
`[indptr[cell] - nnz_lo, indptr[cell + 1] - nnz_lo)` and prefix-sum the gaps to get the gene indices; recombine the byte planes of
the values. A decoder must verify that each frame decompresses to exactly `2 * n` bytes and that every gene index it produces is
`< n_genes`.

## Obs (optional per-cell metadata)

Obs is deliberately **not** part of the tiles: a worker that needs no metadata never pays for it, and tiles stay small and quick to
decode. The obs table is stored beside them so that obs row group `i` always describes the cells of tile `i`.

**Shards.** Parquet files `obs/obs_NNNN.parquet` (4-digit index). Shard `s` holds the tiles `s * K .. min((s + 1) * K, n_tiles) - 1`
(`K = obs_tiles_per_file`, default 1000) with **exactly one row group per tile**, in tile order, and each row group has exactly
`tile_cells[t]` rows. zstd compression; column statistics are not needed (readers never push predicates down) and are best left
out, since they dominate the footer. Row order within a tile is the cell order of the tile.

**Schema** (`obs/schema.json`): `{"format": "deltacells-obs", "version": 1, "columns": [...]}`, each column
`{"name", "kind", "arrow_type", "categories"}`:

| kind | stored as | notes |
|---|---|---|
| `category` | integer codes (`arrow_type` int8 / int16 / int32 by vocabulary size) | code `i` means `categories[i]`; code `-1` means missing. The vocabulary is **global**: the union over all cells, so codes mean the same thing in every tile. |
| `numeric` | the Arrow type named in `arrow_type` (e.g. `int64`, `double`, `float` = float32) | missing values are nulls |
| `bool` | Arrow bool | missing values are nulls |
| `string` | Arrow string | also used for categorical columns whose vocabulary exceeds the writer's `max_categories` (default 20,000), e.g. cell barcodes |

The parquet files contain exactly the schema's columns, in order, with the Arrow types named in the schema.

**Fingerprint.** `obs_fingerprint` is the first 24 hex characters of the SHA-256 of the canonical JSON (sorted keys) of
`{"schema": <schema.json text>, "tile_cells": [...], "shards": [<sha256 of each shard>]}`. It changes whenever the obs content
does, which is what keys the local cache.

**Local cache (non-normative; what the reference implementation does).** Readers never read obs during training. The first time
columns are requested they are *localized*: for each requested column only its column chunks are read from the shards (ranged
reads of the footer and of the column chunks of each row group) and written to
`<cache>/<fingerprint>/obs/<url-quoted column name>.arrow`, an uncompressed Arrow IPC file with one record batch per tile, which
is then memory-mapped. Columns are added independently and atomically (temporary file, then rename) under a lock file, so
concurrent processes on a node fetch each column once. `<cache>` is, in order, the `cache_dir` argument, `$DELTACELLS_CACHE`,
`$XDG_CACHE_HOME/deltacells`, `~/.cache/deltacells`. A cached file whose record-batch count or row counts disagree with the
manifest is deleted and fetched again.

**var.parquet.** The gene table in output column order (the writer applies `gene_order`), with the original pandas index.

## Why this layout

* *One object per tile* means one GET per tile on object storage, and no global index to cache: everything needed to decode a
  tile is inside it.
* *Genes sorted by decreasing total counts* (done by the converter, recorded in `gene_order.npy`) makes the gaps small and
  repetitive; *delta coding* exploits that; *byte planes* put the almost-always-zero high bytes together; *zstd at a high level*
  squeezes the result (higher levels are smaller and, here, decode faster -- they only cost compression time).
* *Independent chunks* let one tile be decoded by several threads at ~1% size cost for 8 chunks.

## Limits (version 1)

* at most 65,536 genes; counts in `[0, 65535]` (the writer refuses anything else rather than clipping);
* a tile's `nnz` is bounded only by memory (offsets are 64-bit);
* no per-cell metadata in the tile itself (see "Obs" above for the sidecar);
* obs columns are limited to categorical, numeric, boolean and string; other pandas dtypes (e.g. datetimes) must be excluded or converted.
