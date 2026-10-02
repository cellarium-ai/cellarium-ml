# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Ingest a set of h5ad shards into a single TileDB-SOMA experiment laid out for fast "read one tile" access.

* Cells keep the order of the (naturally sorted) input files and get ``soma_joinid`` = global row number.
* The h5ad files may have any number of cells each. They are re-chunked into tiles of exactly ``--tile-size``
  cells (only the last tile may be smaller), and each tile is written as one unit (one X fragment per tile).
  The resulting datastore therefore depends only on the cell order and ``--tile-size``, not on how the cells
  were split across input files.
* The X array has a tile extent of ``--tile-size`` cells along obs and the whole gene axis along var, stored
  row-major, so the cells of a tile are contiguous and come back sorted for building a CSR matrix.
* Categorical obs columns are harmonized across shards (the union of the categories is used for every tile).
  Columns with more than ``--max-categories`` distinct values are stored as plain strings.
* By default, genes are reordered by decreasing total counts in the first shard (``--no-sort-genes`` disables
  this). Frequently expressed genes then get small, repetitive column coordinates, which compresses better.
  The var rows (and so ``soma_dim_1``) follow the new order; the position in the input files is kept in the
  var column ``original_var_position``.
* Compression defaults are oriented to remote (network-bound) reads: delta coding for the gene coordinate and
  byte-shuffling for the values (see ``--dim1-filters`` and ``--data-filters``). Together with the gene sorting
  they roughly halve the size of X compared to SOMA's defaults.
* A pre-existing ``soma_joinid`` in an h5ad (as index name or column), which collides with SOMA's own row id,
  is kept as the column ``original_soma_joinid`` (in both obs and var).
* The obs index must be globally unique and is stored as ``obs_id``. With ``--allow-duplicate-obs-names`` the
  obs index is instead stored in the column ``original_index`` and ``obs_id`` is the global row number.

Example::

    python scripts/ingest_h5ad_to_tiledbsoma.py \\
        --h5ad-glob "data/*.h5ad" \\
        --output /data/soma/my_experiment \\
        --tile-size 10000
"""

import argparse
import glob
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tiledbsoma
import tiledbsoma.io

OBS_ID = "obs_id"
VAR_ID = "var_id"
ORIGINAL_SOMA_JOINID = "original_soma_joinid"
ORIGINAL_INDEX = "original_index"
ORIGINAL_VAR_POSITION = "original_var_position"
META_MODES = ["fragment_meta", "commits"]
ALL_MODES = ["fragments", "fragment_meta", "commits"]


def natural_sort_key(path: str) -> list[int | str]:
    """Sort ``adata_2.h5ad`` before ``adata_10.h5ad``."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", path)]


def resolve_files(patterns: list[str]) -> list[str]:
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"No files matched: {patterns}")
    return sorted(files, key=natural_sort_key)


def preserve_original_joinid(df: pd.DataFrame) -> pd.DataFrame:
    """
    SOMA reserves the name ``soma_joinid`` for its own row ids. Keep a pre-existing one (as the name of the
    index or as a column) under the name ``original_soma_joinid``. Returns a copy with an unnamed index.
    """
    df = df.copy()
    if "soma_joinid" in df.columns:
        df = df.rename(columns={"soma_joinid": ORIGINAL_SOMA_JOINID})
    if df.index.name == "soma_joinid":
        if ORIGINAL_SOMA_JOINID in df.columns:
            raise SystemExit(f"Found both a 'soma_joinid' index and an '{ORIGINAL_SOMA_JOINID}' column.")
        numeric = pd.to_numeric(pd.Series(df.index), errors="coerce")
        if numeric.notna().all() and (numeric % 1 == 0).all():
            df[ORIGINAL_SOMA_JOINID] = numeric.to_numpy().astype("int64")
        else:
            df[ORIGINAL_SOMA_JOINID] = np.asarray(df.index, dtype=str)
    df.index.name = None
    return df


class ShardScan:
    """What we need to know about all shards before ingesting any of them."""

    def __init__(
        self, files: list[str], max_categories: int, allow_duplicate_obs_names: bool, sort_genes: bool
    ) -> None:
        self.n_obs_per_file: list[int] = []
        obs_names: list[np.ndarray] = []
        categories: dict[str, dict] = {}  # column -> ordered union of categories (dict used as ordered set)
        self.var: pd.DataFrame | None = None
        columns: list[str] | None = None
        for path in files:
            adata = anndata.read_h5ad(path, backed="r")
            try:
                self.n_obs_per_file.append(adata.n_obs)
                obs = adata.obs
                if columns is None:
                    columns = list(obs.columns)
                    categories = {c: {} for c in columns if isinstance(obs[c].dtype, pd.CategoricalDtype)}
                    self.var = preserve_original_joinid(adata.var)
                    self.var.index = pd.Index(np.asarray(adata.var_names, dtype=str), name=VAR_ID)
                    if not self.var.index.is_unique:
                        raise SystemExit(f"var_names of {path} are not unique.")
                elif list(obs.columns) != columns:
                    raise SystemExit(f"obs columns of {path} differ from those of {files[0]}.")
                elif not np.array_equal(np.asarray(adata.var_names, dtype=str), self.var.index.to_numpy()):
                    raise SystemExit(f"var_names of {path} differ from those of {files[0]}.")
                for col, cats in categories.items():
                    if not isinstance(obs[col].dtype, pd.CategoricalDtype):
                        raise SystemExit(f"Column '{col}' is categorical in {files[0]} but not in {path}.")
                    cats.update(dict.fromkeys(obs[col].cat.categories))
                obs_names.append(np.asarray(adata.obs_names, dtype=str))
            finally:
                adata.file.close()
        assert self.var is not None
        self.n_obs = sum(self.n_obs_per_file)
        self.n_vars = len(self.var)

        # Genes in decreasing order of total counts in the first shard (proxy for the whole dataset).
        self.gene_order = np.arange(self.n_vars)
        if sort_genes:
            totals = np.asarray(anndata.read_h5ad(files[0]).X.sum(axis=0)).ravel()
            self.gene_order = np.argsort(-totals, kind="stable")
        self.var = self.var.iloc[self.gene_order].copy()
        self.var[ORIGINAL_VAR_POSITION] = self.gene_order
        # column j of the input maps to column new_column[j] of the output
        self.new_column = np.empty(self.n_vars, dtype=np.int32)
        self.new_column[self.gene_order] = np.arange(self.n_vars, dtype=np.int32)

        self.categories = {c: list(cats) for c, cats in categories.items() if len(cats) <= max_categories}
        self.string_columns = [c for c, cats in categories.items() if len(cats) > max_categories]

        self.original_obs_names: np.ndarray | None = None
        if allow_duplicate_obs_names:
            self.original_obs_names = np.concatenate(obs_names)
            self.obs_ids = np.arange(self.n_obs).astype(str)
        else:
            self.obs_ids = np.concatenate(obs_names)
            if not pd.Index(self.obs_ids).is_unique:
                raise SystemExit(
                    "obs_names are not globally unique across the shards. "
                    "Pass --allow-duplicate-obs-names to store them as 'original_index' instead."
                )

    def tile_bounds(self, tile_size: int) -> list[tuple[int, int]]:
        return [(lo, min(lo + tile_size, self.n_obs)) for lo in range(0, self.n_obs, tile_size)]

    def reorder_genes(self, x: sp.csr_matrix) -> sp.csr_matrix:
        """Permute the columns of ``x`` into the output gene order."""
        if np.array_equal(self.gene_order, np.arange(self.n_vars)):
            return x
        x = sp.csr_matrix((x.data, self.new_column[x.indices], x.indptr), shape=x.shape)
        x.sort_indices()
        return x

    def prepare_obs(self, obs: pd.DataFrame, start: int) -> pd.DataFrame:
        """Prepare the obs of a shard whose first cell has global row number ``start``."""
        n = len(obs)
        original_index = np.asarray(obs.index, dtype=str)
        obs = preserve_original_joinid(obs)
        if self.original_obs_names is not None:
            obs[ORIGINAL_INDEX] = original_index
        for col, cats in self.categories.items():
            obs[col] = pd.Categorical(obs[col], categories=cats)
        for col in self.string_columns:
            obs[col] = obs[col].astype("object")
        obs.index = pd.Index(self.obs_ids[start : start + n], name=OBS_ID)
        return obs


def iter_tiles(files: list[str], scan: ShardScan, tile_size: int) -> Iterator[tuple[int, anndata.AnnData]]:
    """Read the shards one by one and yield ``(start_row, tile)`` with tiles of exactly ``tile_size`` cells."""
    x_buf: sp.csr_matrix | None = None
    obs_buf: pd.DataFrame | None = None
    buf_start = 0  # global row number of the first buffered cell
    read = 0
    for path in files:
        adata = anndata.read_h5ad(path)
        x = scan.reorder_genes(sp.csr_matrix(adata.X))
        obs = scan.prepare_obs(adata.obs, start=read)
        read += adata.n_obs
        x_buf = x if x_buf is None else sp.vstack([x_buf, x], format="csr")
        obs_buf = obs if obs_buf is None else pd.concat([obs_buf, obs])
        while x_buf.shape[0] >= tile_size:
            tile = anndata.AnnData(X=x_buf[:tile_size], obs=obs_buf.iloc[:tile_size], var=scan.var)
            yield buf_start, tile
            x_buf, obs_buf = x_buf[tile_size:], obs_buf.iloc[tile_size:]
            buf_start += tile_size
    if x_buf is not None and x_buf.shape[0] > 0:
        yield buf_start, anndata.AnnData(X=x_buf, obs=obs_buf, var=scan.var)


def parse_filters(spec: str | None) -> list[str | dict] | None:
    """Parse e.g. ``"DeltaFilter,ZstdFilter:3"`` into a SOMA filter list (``None`` keeps the SOMA default)."""
    if spec is None or spec == "soma-default":
        return None
    filters: list[str | dict] = []
    for item in spec.split(","):
        name, _, level = item.strip().partition(":")
        filters.append({"_type": name, "level": int(level)} if level else name)
    return filters


def build_platform_config(
    tile_size: int,
    n_vars: int,
    capacity: int | None,
    goal_chunk_nnz: int,
    dim1_filters: list[str | dict] | None = None,
    data_filters: list[str | dict] | None = None,
) -> dict:
    """
    Tile layout: ``tile_size`` cells x all genes per space tile. Spanning the whole gene axis means that within
    a tile, cells are stored row-major (cell by cell), so a tile read comes back already sorted for CSR.
    ``goal_chunk_nnz`` is set high so that a whole tile is written as one X fragment.
    """
    create: dict = {
        "dims": {
            "soma_joinid": {"tile": tile_size},
            "soma_dim_0": {"tile": tile_size},
            "soma_dim_1": {"tile": n_vars},
        },
        "cell_order": "row-major",
        "tile_order": "row-major",
        "goal_chunk_nnz": goal_chunk_nnz,
    }
    if capacity is not None:
        create["capacity"] = capacity
    if dim1_filters is not None:
        create["dims"]["soma_dim_1"]["filters"] = dim1_filters
    if data_filters is not None:
        create["attrs"] = {"soma_data": {"filters": data_filters}}
    return {"tiledb": {"create": create}}


def consolidate(output: str, measurement_name: str, x_layer_name: str, context, x_fragments: bool) -> None:
    """
    Merge fragment metadata and commits for all arrays, so that opening an array loads one metadata file instead
    of one per fragment. var is rewritten with every tile and is small, so its fragments are fully merged.
    X (and obs) fragments are only merged if ``x_fragments`` is set.
    """
    with tiledbsoma.Experiment.open(output, "w", context=context) as exp:
        data_modes = ALL_MODES if x_fragments else META_MODES
        exp.ms[measurement_name].X[x_layer_name]._handle.consolidate_and_vacuum(data_modes)
        exp.obs._handle.consolidate_and_vacuum(data_modes)
        exp.ms[measurement_name].var._handle.consolidate_and_vacuum(ALL_MODES)


def count_fragments(array_uri: str) -> int:
    frag_dir = Path(array_uri) / "__fragments"
    return len(list(frag_dir.iterdir())) if frag_dir.exists() else -1


def report_fragments(output: str, measurement_name: str, x_layer_name: str, label: str) -> None:
    if "://" in output:
        return
    uris = {
        "X": f"{output}/ms/{measurement_name}/X/{x_layer_name}",
        "obs": f"{output}/obs",
        "var": f"{output}/ms/{measurement_name}/var",
    }
    counts = ", ".join(f"{name}: {count_fragments(uri)}" for name, uri in uris.items())
    print(f"Fragments {label}: {counts}")


def dir_size_gb(path: str) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad-glob", nargs="+", required=True, help="One or more (quoted) glob patterns.")
    parser.add_argument("--output", required=True, help="URI of the experiment to create (local path for now).")
    parser.add_argument("--tile-size", type=int, default=10_000, help="Cells per tile along the obs axis.")
    parser.add_argument("--measurement-name", default="RNA")
    parser.add_argument("--x-layer-name", default="data")
    parser.add_argument("--capacity", type=int, default=None, help="Sparse array capacity (nnz per data tile).")
    parser.add_argument("--goal-chunk-nnz", type=int, default=2_000_000_000, help="Max nnz per X write chunk.")
    parser.add_argument("--max-categories", type=int, default=10_000, help="Store larger categoricals as strings.")
    parser.add_argument(
        "--allow-duplicate-obs-names",
        action="store_true",
        help="Do not require unique obs_names; store them as 'original_index' and use the row number as obs_id.",
    )
    parser.add_argument(
        "--dim1-filters",
        default="DeltaFilter,ZstdFilter:3",
        help="Filters for the gene coordinate of X ('soma-default' for SOMA's plain ZSTD 3). Delta coding "
        "exploits the small gaps between consecutive nonzero genes, which sorting the genes makes even smaller. "
        "It makes files ~30%% smaller at ~30%% more decode CPU, which pays off for remote (network-bound) reads.",
    )
    parser.add_argument(
        "--data-filters",
        default="ByteShuffleFilter,ZstdFilter:3",
        help="Filters for the X values ('soma-default' for SOMA's).",
    )
    parser.add_argument(
        "--no-sort-genes",
        action="store_true",
        help="Keep the gene order of the h5ad files instead of sorting by decreasing total counts.",
    )
    parser.add_argument(
        "--consolidate-fragments",
        action="store_true",
        help="Also merge the X and obs data fragments (by default there is one fragment per tile).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory if it exists.")
    args = parser.parse_args()

    files = resolve_files(args.h5ad_glob)
    print(f"Found {len(files)} h5ad files: {files[0]} ... {files[-1]}")
    scan = ShardScan(files, args.max_categories, args.allow_duplicate_obs_names, not args.no_sort_genes)
    sizes = scan.n_obs_per_file
    bounds = scan.tile_bounds(args.tile_size)
    print(f"Total: {scan.n_obs} cells x {scan.n_vars} genes; cells per file: min {min(sizes)}, max {max(sizes)}")
    print(f"Writing {len(bounds)} tiles of {args.tile_size} cells")
    print(f"Categorical obs columns: {len(scan.categories)}; stored as strings: {scan.string_columns}")

    output = args.output
    if "://" not in output and Path(output).exists():
        if not args.overwrite:
            raise SystemExit(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    context = tiledbsoma.SOMATileDBContext()
    kwargs = {
        "measurement_name": args.measurement_name,
        "X_layer_name": args.x_layer_name,
        "obs_id_name": OBS_ID,
        "var_id_name": VAR_ID,
        "context": context,
        "platform_config": build_platform_config(
            args.tile_size,
            scan.n_vars,
            args.capacity,
            args.goal_chunk_nnz,
            parse_filters(args.dim1_filters),
            parse_filters(args.data_filters),
        ),
    }

    t0 = time.time()
    # Register the ids of all cells up front: soma_joinid is assigned in registration order, so it equals the
    # global row number. Registration only needs ids, so use X-less stand-ins for the tiles.
    var_ids = pd.DataFrame(index=scan.var.index)
    stand_ins = (
        anndata.AnnData(obs=pd.DataFrame(index=pd.Index(scan.obs_ids[lo:hi], name=OBS_ID)), var=var_ids)
        for lo, hi in bounds
    )
    registration = tiledbsoma.io.register_anndatas(
        None,
        stand_ins,
        measurement_name=args.measurement_name,
        obs_field_name=OBS_ID,
        var_field_name=VAR_ID,
        context=context,
    )
    print(f"Registered {scan.n_obs} cells in {time.time() - t0:.1f}s")

    schema_created = False
    for i, (start, tile) in enumerate(iter_tiles(files, scan, args.tile_size)):
        t1 = time.time()
        if not schema_created:
            # Create the (empty) schema from the first tile, then size it for the whole collection.
            tiledbsoma.io.from_anndata(
                output, tile, ingest_mode="schema_only", registration_mapping=registration, **kwargs
            )
            registration.prepare_experiment(output, context=context)
            schema_created = True
        tiledbsoma.io.from_anndata(output, tile, ingest_mode="write", registration_mapping=registration, **kwargs)
        print(f"[{i + 1}/{len(bounds)}] wrote cells {start}-{start + tile.n_obs} in {time.time() - t1:.1f}s")
    print(f"Ingest done in {time.time() - t0:.1f}s")

    report_fragments(output, args.measurement_name, args.x_layer_name, "after ingest")
    t1 = time.time()
    consolidate(output, args.measurement_name, args.x_layer_name, context, args.consolidate_fragments)
    print(f"Consolidated in {time.time() - t1:.1f}s")
    report_fragments(output, args.measurement_name, args.x_layer_name, "after consolidation")

    with tiledbsoma.Experiment.open(output, context=context) as exp:
        x = exp.ms[args.measurement_name].X[args.x_layer_name]
        print(f"obs rows: {exp.obs.count}, X shape: {x.shape}, X nnz: {x.nnz}")
    if "://" not in output:
        x_gb = dir_size_gb(f"{output}/ms/{args.measurement_name}/X/{args.x_layer_name}")
        print(f"Size on disk: {dir_size_gb(output):.3f} GB (X: {x_gb:.3f} GB, obs: {dir_size_gb(f'{output}/obs'):.3f} GB)")


if __name__ == "__main__":
    main()
