# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Check a converted dataset against its source h5ad files: every cell, every count, in the stored gene order -- and, if the dataset
has obs, every obs column (categoricals compared by label, strings, numbers with NaN, booleans, the cell names).

    python benchmarks/check_against_h5ad.py DATASET "data/*.h5ad" [--max-files N]
"""

from __future__ import annotations

import argparse
import sys

import anndata
import numpy as np
import scipy.sparse as sp

from deltacells import open_dataset
from deltacells.convert import resolve_files


def check_obs(ds, adata, start: int) -> list[str]:
    """Compare the obs of one source file with the dataset's obs for the same cells; returns the names of mismatching columns."""
    import pandas as pd

    n = adata.n_obs
    idx = np.arange(start, start + n)
    store = ds.obs
    cols = store.columns
    got = store.take(idx, cols)
    bad = []
    for c in cols:
        if c == "obs_names":
            want = np.asarray(adata.obs_names, dtype=str)
            ok = list(got[c]) == list(want)
        else:
            s = adata.obs[c]
            if store.kind(c) == "category":
                vocab = np.asarray(store.categories(c) + [None], dtype=object)
                labels = vocab[got[c]]  # code -1 indexes the appended None
                want = s.astype(object).where(s.notna(), None).map(lambda v: v if v is None else str(v))
                ok = list(labels) == list(want)
            elif store.kind(c) == "string":
                want = s.astype(object).where(s.notna(), None).map(lambda v: v if v is None else str(v))
                ok = list(got[c]) == list(want)
            elif store.kind(c) == "bool":
                ok = [None if pd.isna(a) else bool(a) for a in got[c]] == [None if pd.isna(b) else bool(b) for b in s]
            else:
                ok = np.allclose(
                    got[c].astype(np.float64), s.to_numpy(dtype=np.float64, na_value=np.nan), equal_nan=True
                )
        if not ok:
            bad.append(c)
    return bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("h5ad_glob", nargs="+")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--cache-dir", default=None, help="Where to localize obs columns (default: the usual cache).")
    args = parser.parse_args(argv)

    ds = open_dataset(args.dataset, max_cached_tiles=1, cache_dir=args.cache_dir)
    order = ds.gene_order if ds.gene_order is not None else np.arange(ds.n_genes)
    files = resolve_files(args.h5ad_glob)[: args.max_files]
    start, n_bad = 0, 0
    for path in files:
        adata = anndata.read_h5ad(path)
        x = sp.csr_matrix(adata.X)[:, order]
        got = ds.get_batch(np.arange(start, start + x.shape[0])).to_scipy()
        ok = got.shape == x.shape and (got != x.astype(np.float32)).nnz == 0
        message = ""
        if ds.obs is not None:
            bad_cols = check_obs(ds, adata, start)
            ok = ok and not bad_cols
            message = (
                f" (obs: {len(ds.obs.columns)} columns {'OK' if not bad_cols else 'MISMATCH in ' + str(bad_cols)})"
            )
        n_bad += not ok
        print(f"{path}: cells {start}..{start + x.shape[0]} {'OK' if ok else 'MISMATCH'}{message}")
        start += x.shape[0]
    if args.max_files is None and start != ds.n_cells:
        print(f"cell count differs: files have {start}, dataset has {ds.n_cells}")
        return 1
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
