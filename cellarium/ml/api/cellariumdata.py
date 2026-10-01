# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import tempfile
from typing import Callable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from cellarium.ml import CellariumAnnDataDataModule
from cellarium.ml.api.utils import get_h5ad_files_limits, write_obs_parquet
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.data import AnnDataField, to_float_tensor, to_torch_sparse_coo, to_torch_sparse_csr


def get_datamodule(
    h5ad_paths: list[str],
    obs_columns: dict[str, tuple[str, Callable]] = {},
    var_key: str | None = None,
    batch_size: int = 4096,
    shuffle: bool = True,
    train_size: float = 1.0,
    stage: Literal["fit", "validate", "predict", "test"] = "fit",
    nexus_extract_uniform_sizes: bool | None = None,
    num_workers: int = 0,
    accelerator: Literal["cpu", "cuda", "mps"] = "cpu",
):
    if nexus_extract_uniform_sizes is None:
        nexus_extract_uniform_sizes = all(["extract_files" in path for path in h5ad_paths])  # a guess

    datamodule = CellariumAnnDataDataModule(
        dadc=DistributedAnnDataCollection(
            filenames=h5ad_paths,
            limits=get_h5ad_files_limits(
                h5ad_paths,
                nexus_extract_uniform_sizes=nexus_extract_uniform_sizes,
            ),
            obs_columns_to_validate=[c[0] for c in obs_columns.values()],
            max_cache_size=2,
        ),
        batch_keys={
            "x_ng": AnnDataField(
                attr="X",
                # mps has no kernel to move a sparse CSR tensor onto device, so use sparse COO there instead
                convert_fn=to_torch_sparse_coo if accelerator == "mps" else to_torch_sparse_csr,  # type: ignore[arg-type]
            ),
            "var_names_g": AnnDataField(attr="var_names") if var_key is None else AnnDataField(attr="var", key=var_key),
            "obs_names_n": AnnDataField(attr="obs_names"),
            **{
                batch_key: AnnDataField(attr="obs", key=col, convert_fn=fn)
                for batch_key, (col, fn) in obs_columns.items()
            },
        },
        batch_size=batch_size,
        shuffle=shuffle,
        train_size=train_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    datamodule.setup(stage=stage)
    return datamodule


class ObsmMapping(dict):
    def __init__(self, n_obs, *args, **kwargs):
        # n_obs is anything supporting len() (e.g. a DataFrame or a LazyObs) -- evaluated lazily,
        # on assignment, rather than up front, so constructing this doesn't force a LazyObs open.
        self._n_obs_source = n_obs
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        n_obs = len(self._n_obs_source)
        if len(value) != n_obs:
            raise ValueError(f"obsm['{key}'] has length {len(value)}, expected {n_obs} (n_obs)")
        super().__setitem__(key, value)


class LazyObs:
    """
    A lazily-opened, queryable view over an `obs` parquet database written by
    :func:`cellarium.ml.api.utils.write_obs_parquet`. The parquet file/dataset is opened (reading
    only schema and row-group metadata, not data) on first use and cached; individual queries only
    materialize the rows/columns they ask for, so the full `obs` is never required to fit in memory.

    If `parquet_path` is None, `h5ad_paths` must be given: the obs database is built (via
    `write_obs_parquet`, into an auto-generated file in the system temp directory) the first time
    it's actually needed, not at construction time.
    """

    def __init__(self, parquet_path: str | None, h5ad_paths: list[str] | None = None):
        if parquet_path is None and h5ad_paths is None:
            raise ValueError("Must provide either parquet_path or h5ad_paths")
        self._parquet_path = parquet_path
        self._h5ad_paths = h5ad_paths
        self._dataset: ds.Dataset | None = None
        self._index_col: str | None = None

    def _get_dataset(self) -> ds.Dataset:
        if self._dataset is None:
            if self._parquet_path is None:
                tmpdir = tempfile.mkdtemp(prefix="cellarium_obs_parquet_")
                parquet_path = os.path.join(tmpdir, "obs.parquet")
                assert self._h5ad_paths is not None
                write_obs_parquet(self._h5ad_paths, parquet_path)
                self._parquet_path = parquet_path
            dataset = ds.dataset(self._parquet_path, format="parquet")
            metadata = dataset.schema.metadata or {}
            pandas_metadata = metadata.get(b"pandas")
            if pandas_metadata is None:
                raise ValueError(
                    f"{self._parquet_path} has no pandas index metadata -- was it written by write_obs_parquet?"
                )
            index_columns = json.loads(pandas_metadata)["index_columns"]
            if len(index_columns) != 1 or not isinstance(index_columns[0], str):
                raise ValueError(f"Expected a single named obs index column, got {index_columns}")
            self._index_col = index_columns[0]
            self._dataset = dataset
        return self._dataset

    @property
    def _index_column_name(self) -> str:
        self._get_dataset()
        assert self._index_col is not None
        return self._index_col

    def __len__(self) -> int:
        return self._get_dataset().count_rows()

    @property
    def columns(self) -> list[str]:
        index_col = self._index_column_name
        return [name for name in self._get_dataset().schema.names if name != index_col]

    def _read(self, columns: list[str] | None, filter: ds.Expression | None = None) -> pd.DataFrame:
        index_col = self._index_column_name
        arrow_columns = None if columns is None else [index_col, *columns]
        table = self._get_dataset().to_table(columns=arrow_columns, filter=filter)
        return table.to_pandas()

    def __getitem__(self, key: str | list[str]) -> pd.DataFrame | pd.Series:
        columns = [key] if isinstance(key, str) else list(key)
        df = self._read(columns=columns)
        return df[key] if isinstance(key, str) else df

    @property
    def loc(self) -> "_LazyObsLoc":
        return _LazyObsLoc(self)

    def iter_batches(self, batch_size: int = 100_000, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        """Stream `obs` in chunks of `pd.DataFrame`, for full-corpus processing without loading it all at once."""
        index_col = self._index_column_name
        arrow_columns = None if columns is None else [index_col, *columns]
        for batch in self._get_dataset().to_batches(batch_size=batch_size, columns=arrow_columns):
            yield batch.to_pandas()

    def to_frame(self) -> pd.DataFrame:
        """Materialize the entire `obs` dataframe into memory."""
        return self._read(columns=None)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(parquet_path={self._parquet_path!r}, n_obs={len(self)}, columns={self.columns})"
        )


class _LazyObsLoc:
    """Implements `LazyObs.loc[row_labels]` / `LazyObs.loc[row_labels, columns]`."""

    def __init__(self, lazy_obs: LazyObs):
        self._lazy_obs = lazy_obs

    def __getitem__(self, key) -> pd.DataFrame:
        row_key, col_key = key if isinstance(key, tuple) else (key, None)
        row_labels = [row_key] if isinstance(row_key, str) else list(row_key)
        columns = None if col_key is None else ([col_key] if isinstance(col_key, str) else list(col_key))

        index_col = self._lazy_obs._index_column_name
        filter_expr = ds.field(index_col).isin(row_labels)
        df = self._lazy_obs._read(columns=columns, filter=filter_expr)

        missing = set(row_labels) - set(df.index)
        if missing:
            raise KeyError(f"Labels not found in obs index: {sorted(missing)[:5]}")

        return df.loc[row_labels]


class CellariumData:
    def __init__(
        self,
        h5ad_paths: list[str],
        total_mrna_umis_column: str | None = None,
        var_key: str | None = None,
        obs_columns: dict[str, tuple[str, Callable]] = {},
        batch_size: int = 4096,
        shuffle: bool = True,
        train_size: float = 1.0,
        stage: Literal["fit", "validate", "predict", "test"] = "fit",
        nexus_extract_uniform_sizes: bool | None = None,
        datamodule_num_workers: int = 0,
        accelerator: Literal["cpu", "cuda", "mps"] = "cuda",
        obs_parquet_path: str | None = None,
    ):
        if total_mrna_umis_column is not None:
            obs_columns["total_mrna_umis_n"] = (total_mrna_umis_column, to_float_tensor)
        self._datamodule = get_datamodule(
            h5ad_paths=h5ad_paths,
            obs_columns=obs_columns,
            var_key=var_key,
            batch_size=batch_size,
            shuffle=shuffle,
            train_size=train_size,
            stage=stage,
            nexus_extract_uniform_sizes=nexus_extract_uniform_sizes,
            num_workers=datamodule_num_workers,
            accelerator=accelerator,
        )
        # lazy: nothing is read, and no obs parquet database is built, until `.obs` is queried
        self._obs = LazyObs(obs_parquet_path, h5ad_paths=h5ad_paths)
        self._obsm = ObsmMapping(n_obs=self._datamodule.dadc)
        self._var = self._datamodule.dadc.adatas[0].var.copy()
        self._hvg: pd.Series | None = None
        if "var_names_g" in self._datamodule.batch_keys:
            anndatafield: AnnDataField = self._datamodule.batch_keys["var_names_g"]
            if anndatafield.key is not None:
                self._var.set_index(anndatafield.key, inplace=True)

    @property
    def datamodule(self) -> CellariumAnnDataDataModule:
        return self._datamodule

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def obs(self) -> LazyObs:
        return self._obs

    @property
    def obsm(self) -> ObsmMapping:
        return self._obsm

    @property
    def hvg(self) -> pd.Series | None:
        """Boolean HVG mask aligned to var_names_g, or None if not set."""
        return self._hvg

    @hvg.setter
    def hvg(self, value: pd.Series | np.ndarray | Sequence[str] | None) -> None:
        if value is None:
            self._hvg = None
            return
        var_names_g = self.datamodule.var_names_g
        if isinstance(value, pd.Series):
            mask = value.reindex(var_names_g)
            if mask.isna().any():
                raise ValueError(f"hvg is missing entries for: {mask[mask.isna()].index.tolist()[:5]}")
            self._hvg = mask.astype(bool)
        else:
            arr = np.asarray(value)
            if arr.dtype == bool:
                if len(arr) != len(var_names_g):
                    raise ValueError(f"hvg length {len(arr)} != n_vars {len(var_names_g)}")
                self._hvg = pd.Series(arr, index=var_names_g)
            else:
                unknown = set(arr) - set(var_names_g)
                if unknown:
                    raise ValueError(f"Unknown gene names in hvg: {sorted(unknown)[:5]}")
                self._hvg = pd.Series(np.isin(var_names_g, arr), index=var_names_g)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape [{len(self._datamodule.dadc)}, {len(self._var)}], "
            f"obsm keys: {list(self._obsm.keys())}, "
            f"hvg_set={self._hvg is not None})"
        )
