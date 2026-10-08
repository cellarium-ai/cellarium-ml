# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Command line interface: ``deltacells convert | info | verify``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from deltacells.dataset import DeltaCellsDataset


def _info(args: argparse.Namespace) -> int:
    ds = DeltaCellsDataset(args.path, cache_dir=args.cache_dir)
    m = ds.manifest
    total = sum(m.tile_bytes)
    print(f"dataset:        {ds.backend.describe()}")
    print(f"cells x genes:  {m.n_cells} x {m.n_genes}")
    print(f"tiles:          {m.n_tiles} of {m.tile_size} cells (last: {m.tile_cells[-1]})")
    print(f"nonzeros:       {sum(m.tile_nnz)} ({sum(m.tile_nnz) / m.n_cells:.0f} per cell)")
    print(
        f"size:           {total / 1e6:.1f} MB = {total / m.n_cells / 1e3:.2f} kB per cell, {total / sum(m.tile_nnz):.3f} bytes per nonzero"
    )
    print(f"layout:         {m.n_chunks} chunks per tile, zstd level {m.zstd_level}")
    print(
        f"gene order:     {'stored (gene_order.npy)' if m.has_gene_order else 'as in the source'}; var names: {'yes' if m.has_var_names else 'no'}"
    )
    if m.has_obs:
        obs = ds.obs
        kinds = [obs.kind(c) for c in obs.columns]
        print(
            f"obs:            {len(kinds)} columns ({', '.join(f'{k} {kinds.count(k)}' for k in sorted(set(kinds)))}) in "
            f"{m.n_obs_shards} parquet shard(s) of up to {m.obs_tiles_per_file} tiles, {sum(m.obs_shard_bytes) / 1e6:.1f} MB; "
            f"{len(obs.localized_columns())} localized in {obs.cache_root}"
        )
    else:
        print("obs:            none")
    print(f"var table:      {'yes' if m.has_var_table else 'no'}")
    if m.metadata:
        print(f"metadata:       {m.metadata}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    ds = DeltaCellsDataset(args.path, verify=True, io_threads=max(1, args.threads))
    t0 = time.perf_counter()
    for i in range(ds.n_tiles):
        tile = ds.get_tile(i)  # parses and checks the CRC32
        dec = tile.decode(threads=args.threads)
        if dec.nnz != ds.manifest.tile_nnz[i]:
            print(f"tile {i}: nonzero count differs from the manifest", file=sys.stderr)
            return 1
        if i == 0 or (i + 1) % 50 == 0 or i == ds.n_tiles - 1:
            print(f"verified {i + 1}/{ds.n_tiles} tiles")
    if ds.manifest.has_obs:
        try:
            ds.obs.verify()
        except ValueError as e:
            print(f"obs: {e}", file=sys.stderr)
            return 1
        print(f"verified obs ({ds.manifest.n_obs_shards} shard(s))")
    print(f"OK: {ds.n_tiles} tiles, {ds.n_cells} cells in {time.perf_counter() - t0:.1f}s")
    return 0


def _convert(args: argparse.Namespace) -> int:
    from deltacells.convert import convert_h5ad

    convert_h5ad(
        args.h5ad_glob, args.output, tile_size=args.tile_size, sort_genes=not args.no_sort_genes, n_chunks=args.chunks,
        level=args.level, threads=args.threads, workers=args.workers, sort_genes_max_files=args.sort_genes_max_files or None,
        overwrite=args.overwrite, obs=not args.no_obs, var=not args.no_var,
        obs_names=not args.no_obs_names, obs_exclude=[c for c in args.obs_exclude.split(",") if c],
        max_categories=args.max_categories, obs_tiles_per_file=args.obs_tiles_per_file,
    )  # fmt: skip
    return 0


def _obs_info(args: argparse.Namespace) -> int:
    ds = DeltaCellsDataset(args.path, cache_dir=args.cache_dir)
    if ds.obs is None:
        print("this dataset has no obs")
        return 1
    obs = ds.obs
    sizes = obs.column_nbytes()
    local = set(obs.localized_columns())
    per_tile = {c: sizes[c] / ds.n_tiles for c in obs.columns}
    print(f"cache: {obs.cache_dir}")
    print(
        f"{'column':<40}{'kind':<10}{'stored as':<10}{'categories':>11}{'MB to fetch':>13}{'per 100M cells':>16}  local"
    )
    for c in obs.columns:
        spec = obs.schema[c]
        n_cat = "" if spec.categories is None else str(len(spec.categories))
        print(
            f"{c:<40}{spec.kind:<10}{spec.arrow_type:<10}{n_cat:>11}{sizes[c] / 1e6:>13.1f}"
            f"{per_tile[c] * 10_000 / 1e6:>13.0f} MB  {'yes' if c in local else ''}"
        )
    total = sum(sizes.values())
    print(
        f"{'(all columns)':<40}{'':<10}{'':<10}{'':>11}{total / 1e6:>13.1f}{total / ds.n_tiles * 10_000 / 1e6:>13.0f} MB"
    )
    return 0


def _obs_localize(args: argparse.Namespace) -> int:
    ds = DeltaCellsDataset(args.path, cache_dir=args.cache_dir)
    if ds.obs is None:
        print("this dataset has no obs", file=sys.stderr)
        return 1
    if bool(args.columns) == bool(args.all):
        print("give exactly one of --columns a,b,c or --all", file=sys.stderr)
        return 2
    columns = None if args.all else [c for c in args.columns.split(",") if c]
    t0 = time.perf_counter()
    fetched = ds.obs.localize(columns, threads=args.threads)
    print(
        f"{'fetched ' + str(fetched) if fetched else 'already local'} in {time.perf_counter() - t0:.1f}s -> {ds.obs.cache_dir}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deltacells", description="Work with deltacells datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="Print a summary of a dataset.")
    p.add_argument("path")
    p.add_argument("--cache-dir", default=None, help="Obs cache to report on (default: the usual cache).")
    p.set_defaults(func=_info)

    p = sub.add_parser("verify", help="Check every tile's CRC32 and decode it.")
    p.add_argument("path")
    p.add_argument("--threads", type=int, default=1)
    p.set_defaults(func=_verify)

    p = sub.add_parser("convert", help="Convert h5ad shards (any sizes) into a dataset of exact-size tiles.")
    p.add_argument("--h5ad-glob", nargs="+", required=True, help="One or more (quoted) glob patterns or paths.")
    p.add_argument("--output", required=True)
    p.add_argument("--tile-size", type=int, default=10_000)
    p.add_argument("--no-sort-genes", action="store_true", help="Keep the source gene order.")
    p.add_argument("--chunks", type=int, default=8, help="Independently compressed chunks per tile.")
    p.add_argument("--level", type=int, default=19, help="zstd level (higher: smaller, slower to write).")
    p.add_argument(
        "--threads", type=int, default=None, help="Compression threads per worker (default: cores / workers)."
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Processes that write tiles (default: the cores, limited so that the tiles in flight fit in memory).",
    )  # fmt: skip
    p.add_argument(
        "--sort-genes-max-files", type=int, default=10,
        help="Files (spread evenly over the inputs) whose counts decide the gene order (default: 10; 0: all of them).",
    )  # fmt: skip
    p.add_argument("--overwrite", action="store_true", help="Delete the output directory first if it exists.")
    p.add_argument("--no-obs", action="store_true", help="Do not store the cell metadata.")
    p.add_argument("--no-var", action="store_true", help="Do not store the gene table.")
    p.add_argument("--no-obs-names", action="store_true", help="Do not store the cell names as the 'obs_names' column.")
    p.add_argument("--obs-exclude", default="", help="Comma separated obs columns to leave out.")
    p.add_argument(
        "--max-categories", type=int, default=20_000, help="Categoricals with more categories are stored as strings."
    )
    p.add_argument("--obs-tiles-per-file", type=int, default=1000, help="Tiles (row groups) per obs parquet shard.")
    p.set_defaults(func=_convert)

    obs_parser = sub.add_parser("obs", help="Inspect and localize the cell metadata of a dataset.")
    obs_sub = obs_parser.add_subparsers(dest="obs_command", required=True)
    p = obs_sub.add_parser("info", help="List the obs columns, what fetching each costs, and which are local.")
    p.add_argument("path")
    p.add_argument("--cache-dir", default=None)
    p.set_defaults(func=_obs_info)
    p = obs_sub.add_parser(
        "localize", help="Copy obs columns into the node-local cache (only those columns' bytes are read)."
    )
    p.add_argument("path")
    p.add_argument("--columns", default="", help="Comma separated column names.")
    p.add_argument("--all", action="store_true", help="Localize every column.")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=_obs_localize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
