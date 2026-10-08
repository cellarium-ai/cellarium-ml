# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterator, Literal, Sequence

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._datamodule_context import temporary_batch_keys
from cellarium.ml.api.utils import LossHistory, get_h5ad_files_limits, write_obs_parquet
from cellarium.ml.data import DistributedAnnDataCollection, DistributedCollection, DistributedDeltaCellsCollection
from cellarium.ml.utilities.data import AnnDataField, to_float_tensor, to_torch_sparse_coo, to_torch_sparse_csr


def _build_datamodule(
    dadc: DistributedCollection,
    obs_columns: dict[str, tuple[str, Callable]],
    var_key: str | None,
    batch_size: int,
    shuffle: bool,
    train_size: float,
    stage: Literal["fit", "validate", "predict", "test"],
    num_workers: int,
) -> CellariumAnnDataDataModule:
    datamodule = CellariumAnnDataDataModule(
        dadc=dadc,
        batch_keys={
            "x_ng": AnnDataField(
                attr="X",
                # mps has no kernel to move a sparse CSR tensor onto device, so use sparse COO there instead
                convert_fn=to_torch_sparse_coo if torch.mps.is_available() else to_torch_sparse_csr,  # type: ignore[arg-type]
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

    # makes metadata local for collections that need it (a no-op for h5ad files); Lightning does this itself when
    # training, but the api calls `setup` directly
    datamodule.prepare_data()
    datamodule.setup(stage=stage)
    return datamodule


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
):
    if nexus_extract_uniform_sizes is None:
        nexus_extract_uniform_sizes = all(["extract_files" in path for path in h5ad_paths])  # a guess

    dadc = DistributedAnnDataCollection(
        filenames=h5ad_paths,
        limits=get_h5ad_files_limits(
            h5ad_paths,
            nexus_extract_uniform_sizes=nexus_extract_uniform_sizes,
        ),
        obs_columns_to_validate=[c[0] for c in obs_columns.values()],
        max_cache_size=2,
    )
    return _build_datamodule(dadc, obs_columns, var_key, batch_size, shuffle, train_size, stage, num_workers)


def get_deltacells_datamodule(
    uri: str,
    obs_columns: dict[str, tuple[str, Callable]] = {},
    var_key: str | None = None,
    batch_size: int = 4096,
    shuffle: bool = True,
    train_size: float = 1.0,
    stage: Literal["fit", "validate", "predict", "test"] = "fit",
    num_workers: int = 0,
    **collection_kwargs: Any,
):
    """
    Like :func:`get_datamodule` but reading a deltacells dataset (see
    :func:`~cellarium.ml.api.create_deltacells_dataset`) at ``uri``. ``collection_kwargs`` are passed to
    :class:`~cellarium.ml.data.DistributedDeltaCellsCollection`.
    """
    dadc = DistributedDeltaCellsCollection(uri, **collection_kwargs)
    return _build_datamodule(dadc, obs_columns, var_key, batch_size, shuffle, train_size, stage, num_workers)


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


@dataclass(frozen=True)
class TrainedModule:
    """
    A module trained by an api function, with the record of its training.

    Attributes:
        module: The trained :class:`~cellarium.ml.core.CellariumModule`.
        history: The metrics logged during training as a long-format table with columns ``step``, ``epoch``,
            ``metric`` and ``value`` (see :class:`~cellarium.ml.api.utils.LossHistory`). Empty if the api function
            does not record any.
        config: The main arguments the api function was called with.
        n_epochs: The number of epochs completed.
        complete: ``False`` if training was interrupted.
        batch_keys: The data fields, beyond those in ``cdata.datamodule.batch_keys``, that the module needs in order
            to run (e.g. scVI's ``"batch_index_n"``). See :meth:`CellariumData.using`.
    """

    module: CellariumModule
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict[str, Any] = field(default_factory=dict)
    n_epochs: int = 0
    complete: bool = True
    batch_keys: dict[str, AnnDataField] = field(default_factory=dict)

    def metric(self, name: str) -> pd.Series:
        """The values of the logged metric ``name`` (e.g. ``"val_loss"``), indexed by step."""
        logged = sorted(self.history["metric"].unique()) if "metric" in self.history else []
        if name not in logged:
            raise KeyError(f"No metric '{name}' in the history. Logged metrics: {logged}")
        rows = self.history[self.history["metric"] == name]
        return rows.set_index("step")["value"].rename(name)

    def __repr__(self) -> str:
        config = ", ".join(f"{k}={v}" for k, v in self.config.items())
        status = "" if self.complete else " (interrupted)"
        return f"{type(self.module.model).__name__}({config}), {self.n_epochs} epochs{status}"


class TrainedModulesMapping(dict[str, TrainedModule]):
    """The latest trained module of each kind (e.g. ``"scvi"``), keyed by name."""

    def __setitem__(self, key: str, value: TrainedModule) -> None:
        if not isinstance(value, TrainedModule):
            raise TypeError(f"trained_modules['{key}'] must be a TrainedModule, got {type(value).__name__}")
        super().__setitem__(key, value)


def fit_and_register(
    cdata: "CellariumData",
    trainer: pl.Trainer,
    module: CellariumModule,
    key: str | None,
    config: dict[str, Any] | None = None,
    loss_history: LossHistory | None = None,
    batch_keys: dict[str, AnnDataField] | None = None,
) -> TrainedModule:
    """
    Fit ``module`` on ``cdata.datamodule`` and store it in ``cdata.trained_modules[key]``, along with the metrics
    recorded by ``loss_history`` (if the trainer's logger is one) and the extra ``batch_keys`` the module needs to
    run. If training is interrupted, the partially trained module is stored the same way, marked ``complete=False``,
    before the interruption propagates. With ``key=None`` nothing is stored in ``cdata.trained_modules``.

    Returns:
        The :class:`TrainedModule` of the completed fit.
    """

    def register(complete: bool) -> TrainedModule:
        trained = TrainedModule(
            module=module,
            history=pd.DataFrame() if loss_history is None else loss_history.history,
            config={} if config is None else config,
            n_epochs=trainer.current_epoch,
            complete=complete,
            batch_keys={} if batch_keys is None else dict(batch_keys),
        )
        if key is not None:
            cdata.trained_modules[key] = trained
        return trained

    try:
        trainer.fit(module, cdata.datamodule)
    except (KeyboardInterrupt, SystemExit, NameError):
        # Lightning turns a KeyboardInterrupt into a SystemExit after shutting down gracefully
        # and sometimes throws a NameError if something goes wrong during shutdown
        if trainer.global_step > 0:
            register(complete=False)
        raise
    return register(complete=True)


class FitCache(dict[Hashable, TrainedModule]):
    """
    The trained modules whose fitted statistics api functions can reuse, keyed by the recipe that produced each (see
    :mod:`cellarium.ml.api._data_transforms`). Private to a :class:`CellariumData`, and separate from
    ``cdata.trained_modules``, which holds the modules for users to look at.
    """

    def find(self, recipe: Any) -> TrainedModule | None:
        """The trained module for ``recipe`` (preferring an exact match), or one that ``recipe.is_served_by``."""
        if recipe in self:
            return self[recipe]
        for other, trained in self.items():
            if recipe.is_served_by(other):
                return trained
        return None


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


class DeltaCellsLazyObs:
    """
    A queryable view over the ``obs`` of a :class:`~cellarium.ml.data.DistributedDeltaCellsCollection`, with the same
    interface as :class:`LazyObs`. Only the columns (and cells) a query asks for are read; categorical columns are
    pandas categoricals with the dataset's global categories. The ``obs_names`` of the dataset are the index of the
    returned frames, and ``.loc`` selects by them.
    """

    _NAMES = "obs_names"

    def __init__(self, dadc: DistributedDeltaCellsCollection):
        self._dadc = dadc

    @property
    def _store(self) -> Any:
        store = self._dadc.dataset.obs
        if store is None:
            raise ValueError(f"The deltacells dataset at {self._dadc.uri!r} has no obs.")
        return store

    def __len__(self) -> int:
        return self._dadc.n_obs

    @property
    def columns(self) -> list[str]:
        return [c for c in self._store.columns if c != self._NAMES]

    def _index(self, indices: np.ndarray | None = None) -> pd.Index:
        if indices is None:
            names = self._store.to_pandas(self._NAMES)[self._NAMES]
        else:
            names = self._store.take_pandas(indices, [self._NAMES], warn=False)[self._NAMES]
        return pd.Index(np.asarray(names, dtype=object), name=self._NAMES)

    def _take(self, indices: np.ndarray | None, columns: list[str]) -> pd.DataFrame:
        if indices is None:
            df = self._store.to_pandas(columns) if columns else pd.DataFrame(index=pd.RangeIndex(len(self)))
        else:
            df = self._store.take_pandas(indices, columns, warn=False)
        df.index = self._index(indices)
        return df

    def __getitem__(self, key: str | list[str]) -> pd.DataFrame | pd.Series:
        columns = [key] if isinstance(key, str) else list(key)
        df = self._take(None, columns)
        return df[key] if isinstance(key, str) else df

    @property
    def loc(self) -> "_DeltaCellsLazyObsLoc":
        return _DeltaCellsLazyObsLoc(self)

    def iter_batches(self, batch_size: int = 100_000, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        """Stream `obs` in chunks of `pd.DataFrame`, for full-corpus processing without loading it all at once."""
        columns = self.columns if columns is None else list(columns)
        for start in range(0, len(self), batch_size):
            yield self._take(np.arange(start, min(start + batch_size, len(self)), dtype=np.int64), columns)

    def to_frame(self) -> pd.DataFrame:
        """Materialize the entire `obs` dataframe into memory."""
        return self._take(None, self.columns)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(uri={self._dadc.uri!r}, n_obs={len(self)}, columns={self.columns})"


class _DeltaCellsLazyObsLoc:
    """Implements `DeltaCellsLazyObs.loc[row_labels]` / `DeltaCellsLazyObs.loc[row_labels, columns]`."""

    def __init__(self, lazy_obs: DeltaCellsLazyObs):
        self._lazy_obs = lazy_obs

    def __getitem__(self, key) -> pd.DataFrame:
        lazy_obs = self._lazy_obs
        row_key, col_key = key if isinstance(key, tuple) else (key, None)
        row_labels = [row_key] if isinstance(row_key, str) else list(row_key)
        columns = lazy_obs.columns if col_key is None else ([col_key] if isinstance(col_key, str) else list(col_key))

        names = pd.Index(np.asarray(lazy_obs._store.to_pandas(lazy_obs._NAMES)[lazy_obs._NAMES], dtype=object))
        missing = pd.Index(row_labels)[~pd.Index(row_labels).isin(names)]
        if len(missing):
            raise KeyError(f"Labels not found in obs index: {sorted(missing)[:5]}")
        positions = names.get_indexer_for(row_labels).astype(np.int64)
        return lazy_obs._take(positions, columns)


class CellariumData:
    """
    The data for the api functions: a datamodule over the cells, ``var``, a lazily queried ``obs``, ``obsm``, the
    highly variable genes, ``obs_computed`` (per-cell values computed by api functions) and ``trained_modules`` (the
    latest module trained by each api function, with its training history).

    The cells are read from sharded h5ad files (``h5ad_paths``) or from a deltacells dataset (``deltacells_uri``, or
    :meth:`from_deltacells`; make one with :func:`~cellarium.ml.api.create_deltacells_dataset`). With deltacells the
    genes come in the dataset's stored order and the categories of categorical ``obs`` columns are global strings; see
    :class:`~cellarium.ml.data.DistributedDeltaCellsCollection`.
    """

    def __init__(
        self,
        h5ad_paths: list[str] | None = None,
        total_mrna_umis_column: str | None = None,
        var_key: str | None = None,
        obs_columns: dict[str, tuple[str, Callable]] = {},
        batch_size: int = 4096,
        shuffle: bool = True,
        train_size: float = 1.0,
        stage: Literal["fit", "validate", "predict", "test"] = "fit",
        nexus_extract_uniform_sizes: bool | None = None,
        datamodule_num_workers: int = 0,
        obs_parquet_path: str | None = None,
        *,
        deltacells_uri: str | None = None,
        deltacells_kwargs: dict[str, Any] | None = None,
    ):
        if (h5ad_paths is None) == (deltacells_uri is None):
            raise ValueError("Provide exactly one of h5ad_paths and deltacells_uri.")
        obs_columns = dict(obs_columns)
        if total_mrna_umis_column is not None:
            obs_columns["total_mrna_umis_n"] = (total_mrna_umis_column, to_float_tensor)
        self._obs: LazyObs | DeltaCellsLazyObs
        if deltacells_uri is None:
            assert h5ad_paths is not None
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
            )
            # lazy: nothing is read, and no obs parquet database is built, until `.obs` is queried
            self._obs = LazyObs(obs_parquet_path, h5ad_paths=h5ad_paths)
        else:
            self._datamodule = get_deltacells_datamodule(
                deltacells_uri,
                obs_columns=obs_columns,
                var_key=var_key,
                batch_size=batch_size,
                shuffle=shuffle,
                train_size=train_size,
                stage=stage,
                num_workers=datamodule_num_workers,
                **(deltacells_kwargs or {}),
            )
            self._obs = DeltaCellsLazyObs(self._datamodule.dadc)
        self._obsm = ObsmMapping(n_obs=self._datamodule.dadc)
        self._obs_computed = ObsmMapping(n_obs=self._datamodule.dadc)
        self._var = self._datamodule.dadc.var.copy()
        self._trained_modules = TrainedModulesMapping()
        self._fit_cache = FitCache()
        self._hvg: pd.Series | None = None
        if "var_names_g" in self._datamodule.batch_keys:
            anndatafield: AnnDataField = self._datamodule.batch_keys["var_names_g"]
            if anndatafield.key is not None:
                self._var.set_index(anndatafield.key, inplace=True)

    @classmethod
    def from_deltacells(cls, uri: str, **kwargs: Any) -> "CellariumData":
        """
        Read the cells from the deltacells dataset at ``uri`` (a local path or ``gs://bucket/prefix``). The keyword
        arguments are those of the constructor, except ``h5ad_paths``, ``nexus_extract_uniform_sizes`` and
        ``obs_parquet_path``, which only apply to h5ad files.
        """
        return cls(deltacells_uri=uri, **kwargs)

    @property
    def datamodule(self) -> CellariumAnnDataDataModule:
        return self._datamodule

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def obs(self) -> LazyObs | DeltaCellsLazyObs:
        return self._obs

    @property
    def obsm(self) -> ObsmMapping:
        return self._obsm

    @property
    def obs_computed(self) -> ObsmMapping:
        """
        Per-cell values computed by api functions (e.g. ``"in_sketch"`` from geometric sketching), one
        entry of length n_obs each.
        """
        return self._obs_computed

    @property
    def trained_modules(self) -> TrainedModulesMapping:
        """The latest module trained by each api function (e.g. ``"scvi"``), with its training history."""
        return self._trained_modules

    @contextmanager
    def using(self, key: str) -> Iterator[TrainedModule]:
        """
        Make the data fields that the module ``cdata.trained_modules[key]`` needs in order to run available in
        ``cdata.datamodule`` for the duration of the ``with`` block (for example, scVI needs the batch column it was
        trained with), restoring the original state on exit, even if the block raises. Yields the
        :class:`TrainedModule`.

        Example:
            >>> with cdata.using("scvi") as trained:
            ...     ...  # code that runs trained.module over cdata.datamodule

        Raises:
            ValueError: If there is no trained module under ``key``.
        """
        if key not in self._trained_modules:
            available = list(self._trained_modules)
            raise ValueError(
                f"No trained module '{key}' in cdata.trained_modules. "
                + (f"Available: {available}. " if available else "None have been trained yet. ")
                + "Train modules with the api functions (e.g. cml.tl.scvi, cml.tl.pca, cml.pp.highly_variable_genes), "
                "which store them here."
            )
        trained = self._trained_modules[key]
        if not trained.batch_keys:
            yield trained  # nothing extra needed (e.g. PCA), so leave the datamodule untouched
            return
        with temporary_batch_keys(self._datamodule, trained.batch_keys):
            yield trained

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
        lines = [
            f"{self.__class__.__name__}("
            f"shape [{len(self._datamodule.dadc)}, {len(self._var)}], "
            f"obsm keys: {list(self._obsm.keys())}, "
            f"obs_computed keys: {list(self._obs_computed.keys())}, "
            f"hvg_set={self._hvg is not None})"
        ]
        if self._trained_modules:
            lines.append("  trained_modules:")
            lines.extend(f"    {key}: {trained!r}" for key, trained in self._trained_modules.items())
        return "\n".join(lines)
