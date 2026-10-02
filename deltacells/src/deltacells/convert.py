# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert h5ad shards (any cell counts per file) into a deltacells dataset of exact-size tiles."""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import scipy.sparse as sp

from deltacells.manifest import DEFAULT_OBS_TILES_PER_FILE, Manifest
from deltacells.writer import DEFAULT_CHUNKS, DEFAULT_LEVEL, DatasetWriter

try:  # optional: needed only to store obs / var
    from deltacells.obs import DEFAULT_MAX_CATEGORIES, ObsSchema
except ImportError:  # pragma: no cover
    DEFAULT_MAX_CATEGORIES, ObsSchema = 20_000, None  # type: ignore[assignment,misc]


def natural_sort_key(path: str) -> list[int | str]:
    """Sort ``shard_2.h5ad`` before ``shard_10.h5ad``."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", path)]


def resolve_files(patterns: Sequence[str]) -> list[str]:
    """Expand glob patterns / paths into a naturally sorted, de-duplicated list of files."""
    files: set[str] = set()
    for pattern in [patterns] if isinstance(patterns, str) else patterns:
        files.update(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no files matched {list(patterns)}")
    return sorted(files, key=natural_sort_key)


def convert_h5ad(
    paths: Sequence[str] | str,
    output: str,
    *,
    tile_size: int = 10_000,
    sort_genes: bool = True,
    n_chunks: int = DEFAULT_CHUNKS,
    level: int = DEFAULT_LEVEL,
    threads: int | None = None,
    overwrite: bool = False,
    obs: bool = True,
    obs_names: bool = True,
    obs_exclude: Sequence[str] = (),
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    obs_tiles_per_file: int = DEFAULT_OBS_TILES_PER_FILE,
    var: bool = True,
    log: Callable[[str], None] | None = print,
) -> Manifest:
    """Write ``output`` from h5ad files (glob patterns or paths), preserving cell order.

    Files are read one at a time and re-chunked, so they may hold any number of cells; the result depends only on the cell order
    and ``tile_size``. All files must have the same ``var_names``. ``X`` must hold integer counts up to 65535.

    With ``sort_genes`` the genes are reordered by decreasing total counts in the first file (a proxy for the whole dataset);
    the original column of each output gene is stored as ``gene_order``.

    With ``obs`` (needs pyarrow and pandas) the cell metadata is stored too, as tile-aligned parquet shards (see
    :mod:`deltacells.obs`): one pass over all files builds a global vocabulary for each categorical column (columns with more than
    ``max_categories`` distinct values are stored as strings), ``obs_names`` adds the cell names as a string column, and
    ``obs_exclude`` drops columns. With ``var`` the gene table is stored as ``var.parquet``.
    """
    import anndata

    say = log or (lambda _: None)
    if os.path.exists(output) and os.listdir(output) and not overwrite:
        raise FileExistsError(f"{output} is not empty; pass overwrite=True to replace it")
    files = resolve_files(paths)
    say(f"{len(files)} files: {files[0]} ... {files[-1]}")
    if obs or var:
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "storing obs / var needs pandas and pyarrow (pip install deltacells[obs]); or pass obs=False, var=False"
            ) from e

    state: dict = {"var_names": None, "var": None, "n_obs": []}

    def scan() -> Iterator:
        """One pass over the files: checks var_names, counts cells and yields each file's obs frame (or None)."""
        for path in files:
            a = anndata.read_h5ad(path, backed="r")
            try:
                names = np.asarray(a.var_names, dtype=str)
                if state["var_names"] is None:
                    state["var_names"], state["var"] = names, a.var.copy()
                elif not np.array_equal(state["var_names"], names):
                    raise ValueError(f"var_names of {path} differ from those of {files[0]}")
                state["n_obs"].append(a.n_obs)
                yield _obs_frame(a, obs_names, obs_exclude) if obs else None
            finally:
                a.file.close()

    schema = None
    if obs:
        schema = ObsSchema.infer(scan(), max_categories=max_categories)
        as_strings = [c.name for c in schema.columns if c.kind == "string"]
        say(
            f"obs: {len(schema.columns)} columns ({sum(c.kind == 'category' for c in schema.columns)} categorical, strings: {as_strings})"
        )
    else:
        for _ in scan():
            pass
    n_obs = state["n_obs"]
    var_names = state["var_names"]
    n_genes = len(var_names)
    say(f"{sum(n_obs)} cells x {n_genes} genes (cells per file: min {min(n_obs)}, max {max(n_obs)})")

    if sort_genes:
        first = sp.csr_matrix(anndata.read_h5ad(files[0]).X)
        gene_order = np.argsort(-np.asarray(first.sum(axis=0)).ravel(), kind="stable")
        del first
    else:
        gene_order = None

    metadata = {
        "source": "h5ad",
        "n_source_files": len(files),
        "first_file": files[0],
        "last_file": files[-1],
        "sort_genes": sort_genes,
    }
    writer = DatasetWriter(
        output, n_genes=n_genes, tile_size=tile_size, gene_order=gene_order, var_names=list(var_names),
        var=state["var"] if var else None, obs_schema=schema, obs_tiles_per_file=obs_tiles_per_file,
        n_chunks=n_chunks, level=level, threads=threads, metadata=metadata, overwrite=overwrite,
    )  # fmt: skip
    buf: sp.csr_matrix | None = None
    obs_buf = None  # an Arrow table encoded with the global schema, so shards with different category sets combine
    for k, path in enumerate(files):
        adata = anndata.read_h5ad(path)
        x = sp.csr_matrix(adata.X)
        buf = x if buf is None else sp.vstack([buf, x], format="csr")
        if schema is not None:
            import pyarrow as pa

            table = schema.encode(_obs_frame(adata, obs_names, obs_exclude))
            obs_buf = table if obs_buf is None else pa.concat_tables([obs_buf, table])
        while buf.shape[0] >= tile_size:
            i = writer.add_tile(buf[:tile_size], obs=None if obs_buf is None else obs_buf.slice(0, tile_size))
            say(f"[file {k + 1}/{len(files)}] wrote tile {i} ({writer._bytes[i] / 1e6:.1f} MB)")
            buf = buf[tile_size:]
            obs_buf = None if obs_buf is None else obs_buf.slice(tile_size)
    if buf is not None and buf.shape[0] > 0:
        i = writer.add_tile(buf, obs=obs_buf)
        say(f"wrote final tile {i} with {buf.shape[0]} cells ({writer._bytes[i] / 1e6:.1f} MB)")
    manifest = writer.close()
    total = sum(manifest.tile_bytes)
    say(f"done: {manifest.n_tiles} tiles, {total / 1e6:.1f} MB = {total / manifest.n_cells / 1e3:.2f} kB per cell")
    if manifest.has_obs:
        say(f"obs: {sum(manifest.obs_shard_bytes) / 1e6:.1f} MB in {manifest.n_obs_shards} parquet shard(s)")
    return manifest


def _obs_frame(adata, include_names: bool, exclude: Sequence[str]):
    """The obs frame of an AnnData object, with the cell names as an ``obs_names`` column and ``exclude`` dropped."""
    df = adata.obs.drop(columns=[c for c in exclude if c in adata.obs.columns]).copy()
    if include_names:
        if "obs_names" in df.columns:
            raise ValueError("the obs already has an 'obs_names' column; pass obs_names=False or rename it")
        df["obs_names"] = np.asarray(adata.obs_names, dtype=str)
    return df
