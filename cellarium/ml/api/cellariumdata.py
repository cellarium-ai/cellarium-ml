# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterator, Literal, Sequence

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import scipy.sparse
import torch

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._datamodule_context import temporary_batch_keys
from cellarium.ml.api._view_collections import (
    RamCollection,
    ViewCollection,
    available_memory_bytes,
    load_into_ram,
    unwrap_collection,
)
from cellarium.ml.api.deltacells_store import write_collection_to_deltacells
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


def _kind(dtype: Any) -> str:
    """The kind of values of a pandas dtype, for deciding whether values of two dtypes can share a column."""
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype):
        return "string"
    return "other"


def _nullable(values: pd.Series) -> pd.Series:
    """``values`` in a dtype that can hold missing values: pandas' nullable bool, integer and string types."""
    kind = _kind(values.dtype)
    if kind == "bool":
        return values.astype("boolean")
    if kind == "int":
        return values.astype("Int64")
    if kind == "string":
        return values.astype("string")
    if kind == "other":
        raise TypeError(f"Values of dtype {values.dtype} cannot be copied between datasets.")
    return values  # categorical, float and datetime values can hold missing values already


def _per_cell_series(value: Any, n: int, what: str) -> pd.Series:
    """``value`` (one entry per cell) as a Series with a default index, in a dtype that can hold missing values."""
    if isinstance(value, pd.Series):
        series = value.reset_index(drop=True)
    elif isinstance(value, pd.api.extensions.ExtensionArray):  # for example a Categorical: keep its dtype
        series = pd.Series(value)
    else:
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError(f"{what} must have one entry per cell, but has shape {array.shape}.")
        series = pd.Series(array)
    if len(series) != n:
        raise ValueError(f"{what} has {len(series)} entries but there are {n} cells.")
    return _nullable(series)


class ObsComputedMapping(ObsmMapping):
    """
    The values that api functions computed for each cell (see :attr:`CellariumData.obs_computed`): an
    :class:`ObsmMapping` that can also take values computed on a view of the cells with :meth:`update_from`.
    """

    def __init__(self, n_obs: Any, owner: "CellariumData", *args: Any, **kwargs: Any):
        super().__init__(n_obs, *args, **kwargs)
        self._owner = owner

    def update_from(self, view: "CellariumData", key: str, as_key: str | None = None, overwrite: bool = True) -> None:
        """
        Copy the values ``view.obs_computed[key]`` (for example the subtypes found by clustering a subset of the
        cells) to the cells they belong to in this data, as ``obs_computed[as_key]`` (``key`` by default).

        ``view`` must descend from the same root as this data, and its cells must be among this data's. The entry is
        a :class:`pandas.Series` indexed by the obs names, with pandas' nullable dtypes (``string`` or ``category``
        for labels, ``Int64``, ``boolean``, floats). Cells that the view does not hold are missing (NA). If the entry
        already exists, its values for those cells are kept, so that several views can each fill in their cells of one
        entry; the values it already has for cells of the view are replaced, except where the view's value is missing.
        New labels are added to a categorical entry's categories.

        Args:
            view: The view (or the data itself, or any other data from the same root) that computed the values.
            key: The key of the values in ``view.obs_computed``. If it is a Series, its index must be exactly
                ``view.obs.index``.
            as_key: The key to store them under here, if not ``key``.
            overwrite: If ``False``, raise an error instead of replacing a value that is already present.

        Raises:
            ValueError: If ``view`` is from a different root or has cells that this data does not, if the dtypes of
                the new and existing values do not fit together, or if ``overwrite`` is ``False`` and a value would
                be replaced.
            KeyError: If ``view.obs_computed`` has no ``key``.
        """
        owner = self._owner
        positions = owner._positions_of(view)
        if key not in view.obs_computed:
            raise KeyError(f"No '{key}' in the obs_computed of the view. Available: {list(view.obs_computed)}")
        n = owner.n_obs
        raw = view.obs_computed[key]
        if isinstance(raw, pd.Series) and not raw.index.equals(view.obs.index):
            raise ValueError(f"The index of obs_computed['{key}'] of the view must be exactly the index of view.obs.")
        values = _per_cell_series(raw, view.n_obs, f"obs_computed['{key}'] of the view")
        target = key if as_key is None else as_key

        if target not in self:
            merged = values.set_axis(positions).reindex(np.arange(n))
        else:
            existing = _per_cell_series(self[target], n, f"obs_computed['{target}']")
            merged = self._merge(existing, values, positions, key, target, overwrite)
        merged.index = owner.obs.index
        self[target] = merged

    @staticmethod
    def _merge(
        existing: pd.Series, values: pd.Series, positions: np.ndarray, key: str, target: str, overwrite: bool
    ) -> pd.Series:
        """``existing`` with the present ``values`` (for the cells at ``positions``) written into it."""
        existing_kind, kind = _kind(existing.dtype), _kind(values.dtype)
        fits = (
            existing_kind == kind
            or {existing_kind, kind} == {"category", "string"}
            or (existing_kind == "float" and kind == "int")
        )
        if not fits:
            raise ValueError(
                f"The values '{key}' of the view (dtype {values.dtype}) do not fit the values of "
                f"obs_computed['{target}'] here (dtype {existing.dtype})."
            )
        present = values.notna().to_numpy()
        where, new = positions[present], values[present]
        if not overwrite and existing.iloc[where].notna().any():
            raise ValueError(
                f"{int(existing.iloc[where].notna().sum())} cells already have a value in obs_computed['{target}']; "
                "pass overwrite=True to replace them."
            )
        if existing_kind == "category":
            labels = new.astype(object)
            missing = pd.Index(labels.unique()).difference(existing.cat.categories, sort=False)
            merged = existing.cat.add_categories(missing)
            new = labels.astype(merged.dtype)
        elif existing_kind == "string":
            new = new.astype("string")
            merged = existing.copy()
        elif existing_kind == "float":
            merged = existing.astype("float64")
            new = new.astype("float64")
        else:
            merged = existing.copy()
        merged.iloc[where] = new.array
        return merged


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

    @property
    def index(self) -> pd.Index:
        """The obs index (the cell names), in cell order."""
        return self._read(columns=[]).index

    def _take(self, positions: np.ndarray | None, columns: list[str]) -> pd.DataFrame:
        """The ``columns`` of the cells at ``positions`` (all cells if ``None``), in the order of ``positions``.

        Assumes that row ``i`` of the parquet database is cell ``i`` of the data, as it is when the database is built
        from the h5ad files.
        """
        index_col = self._index_column_name
        table = self._get_dataset().to_table(columns=[index_col, *columns])
        if positions is not None:
            table = table.take(pa.array(positions, type=pa.int64()))
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

    @property
    def index(self) -> pd.Index:
        """The obs index (the cell names), in cell order."""
        return self._index()

    def _take(self, indices: np.ndarray | None, columns: list[str]) -> pd.DataFrame:
        if not columns:
            df = pd.DataFrame(index=pd.RangeIndex(len(self) if indices is None else len(indices)))
        elif indices is None:
            df = self._store.to_pandas(columns)
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


class ViewLazyObs:
    """
    The ``obs`` of a :class:`CellariumDataView`: the rows ``positions`` (sorted) of the ``obs`` of the root data
    (a :class:`LazyObs` or :class:`DeltaCellsLazyObs`), with the same interface. Nothing is copied; the columns (and
    cells) a query asks for are read from the root.
    """

    def __init__(self, source: "LazyObs | DeltaCellsLazyObs", positions: np.ndarray):
        self._source = source
        self._positions = positions
        self._names: pd.Index | None = None

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def columns(self) -> list[str]:
        return self._source.columns

    @property
    def index(self) -> pd.Index:
        """The obs index (the cell names) of the view, in cell order."""
        if self._names is None:
            self._names = self._take(None, []).index
        return self._names

    def _take(self, indices: np.ndarray | None, columns: list[str]) -> pd.DataFrame:
        positions = self._positions if indices is None else self._positions[indices]
        return self._source._take(positions, columns)

    def __getitem__(self, key: str | list[str]) -> pd.DataFrame | pd.Series:
        columns = [key] if isinstance(key, str) else list(key)
        df = self._take(None, columns)
        return df[key] if isinstance(key, str) else df

    @property
    def loc(self) -> "_ViewLazyObsLoc":
        return _ViewLazyObsLoc(self)

    def iter_batches(self, batch_size: int = 100_000, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        """Stream `obs` in chunks of `pd.DataFrame`, for full-corpus processing without loading it all at once."""
        columns = self.columns if columns is None else list(columns)
        for start in range(0, len(self), batch_size):
            yield self._take(np.arange(start, min(start + batch_size, len(self)), dtype=np.int64), columns)

    def to_frame(self) -> pd.DataFrame:
        """Materialize the entire `obs` dataframe of the view into memory."""
        return self._take(None, self.columns)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_obs={len(self)}, columns={self.columns})"


class _ViewLazyObsLoc:
    """Implements `ViewLazyObs.loc[row_labels]` / `ViewLazyObs.loc[row_labels, columns]`."""

    def __init__(self, lazy_obs: ViewLazyObs):
        self._lazy_obs = lazy_obs

    def __getitem__(self, key) -> pd.DataFrame:
        lazy_obs = self._lazy_obs
        row_key, col_key = key if isinstance(key, tuple) else (key, None)
        row_labels = [row_key] if isinstance(row_key, str) else list(row_key)
        columns = lazy_obs.columns if col_key is None else ([col_key] if isinstance(col_key, str) else list(col_key))

        names = lazy_obs.index
        missing = pd.Index(row_labels)[~pd.Index(row_labels).isin(names)]
        if len(missing):
            raise KeyError(f"Labels not found in obs index: {sorted(missing)[:5]}")
        return lazy_obs._take(names.get_indexer_for(row_labels).astype(np.int64), columns)


def _clone_datamodule(
    datamodule: CellariumAnnDataDataModule,
    dadc: DistributedCollection,
    stage: Literal["fit", "validate", "predict", "test"],
    num_workers: int | None = None,
) -> CellariumAnnDataDataModule:
    """A datamodule over ``dadc`` with the settings of ``datamodule`` (and the data fields it currently reads)."""
    sizes = {name: datamodule.hparams[name] for name in ("train_size", "val_size", "pred_size")}
    clone = CellariumAnnDataDataModule(
        dadc=dadc,
        batch_keys=dict(datamodule.batch_keys),
        batch_size=datamodule.batch_size,
        iteration_strategy=datamodule.iteration_strategy,
        shuffle=datamodule.shuffle,
        shuffle_seed=datamodule.shuffle_seed,
        drop_last_indices=datamodule.drop_last_indices,
        drop_incomplete_batch=datamodule.drop_incomplete_batch,
        worker_seed=datamodule.worker_seed,
        test_mode=datamodule.test_mode,
        prefetch_lookahead=datamodule.prefetch_lookahead,
        num_workers=datamodule.num_workers if num_workers is None else num_workers,
        prefetch_factor=datamodule.prefetch_factor,
        persistent_workers=datamodule.persistent_workers,
        pin_memory=datamodule.pin_memory,
        **sizes,
    )
    # as in `_build_datamodule`: makes the metadata local and sets up the datasets
    clone.prepare_data()
    clone.setup(stage=stage)
    return clone


def _subset_rows(value: Any, positions: np.ndarray) -> Any:
    """The rows ``positions`` of a per-cell value (array, sparse matrix, tensor, Series or DataFrame), as a copy."""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.iloc[positions].copy()
    if scipy.sparse.issparse(value):
        return value.tocsr()[positions]
    if isinstance(value, torch.Tensor):
        return value[torch.from_numpy(positions)].clone()
    return np.asarray(value)[positions]


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
        self._obs: LazyObs | DeltaCellsLazyObs | ViewLazyObs
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
        self._stage = stage
        self._deltacells_uri = deltacells_uri
        self._deltacells_kwargs = dict(deltacells_kwargs or {})
        self._var = self._datamodule.dadc.var.copy()
        if "var_names_g" in self._datamodule.batch_keys:
            anndatafield: AnnDataField = self._datamodule.batch_keys["var_names_g"]
            if anndatafield.key is not None:
                self._var.set_index(anndatafield.key, inplace=True)
        self._init_state()
        # lineage: a root is the data that views are taken from (see `CellariumDataView`)
        self._parent: CellariumData | None = None
        self._root: CellariumData = self
        self._root_indices: np.ndarray | None = None  # None: all cells, in order

    def _init_state(self) -> None:
        """Set up the state that belongs to these cells: ``obsm``, ``obs_computed``, trained modules, ..."""
        self._obsm = ObsmMapping(n_obs=self._datamodule.dadc)
        self._obs_computed = ObsComputedMapping(n_obs=self._datamodule.dadc, owner=self)
        self._trained_modules = TrainedModulesMapping()
        self._fit_cache = FitCache()
        self._hvg: pd.Series | None = None

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
    def n_obs(self) -> int:
        """The number of cells."""
        return len(self._datamodule.dadc)

    @property
    def deltacells_uri(self) -> str | None:
        """The location of the deltacells dataset that these cells are read from, or ``None`` if they are not."""
        return self._deltacells_uri

    @property
    def parent(self) -> "CellariumData | None":
        """The data this view was taken from, or ``None`` for data that is not a view."""
        return self._parent

    @property
    def root(self) -> "CellariumData":
        """The data that views descend from: the data that is not a view (``self`` unless this is a view)."""
        return self._root

    @property
    def root_indices(self) -> np.ndarray:
        """The positions of these cells in :attr:`root`, in increasing order (all of them, for the root itself)."""
        if self._root_indices is None:
            return np.arange(self.n_obs, dtype=np.int64)
        return self._root_indices.copy()

    def to_ram(self, max_gb: float | None = None) -> "CellariumDataView":
        """
        Read these cells into memory, and return a :class:`CellariumDataView` of them that reads from there. This is
        **memory-heavy**: the counts take about 8 bytes per nonzero (several GB for a few hundred thousand cells), plus
        their ``obs``. It is for training repeatedly on a small view, where reading every shard of the data on every
        epoch would dominate the time.

        It reads the cells once, in a pass over the shards that hold them (for a view of cells scattered over the
        data that is a pass over all of it). The memory needed is estimated from a sample first; nothing is loaded if
        it would exceed ``max_gb``, and the memory in use is checked while loading.

        The returned view has the same cells (and root indices) as these, so ``update_from`` works between them, and
        starts with copies of this data's ``obsm`` and ``obs_computed`` but with no trained modules, as any view does.
        Its datamodule uses no dataloader workers (``num_workers=0``), and cannot: the cells live in this process.
        Views of it read from the same memory. Calling this on data that is already in memory returns it as it is.

        Args:
            max_gb: The most memory (in GiB) to use. By default half of the memory that is available, taking any
                container limit into account.

        Raises:
            MemoryError: If the cells do not fit.
        """
        if isinstance(unwrap_collection(self._datamodule.dadc), RamCollection):
            return self  # type: ignore[return-value]
        max_bytes = int(max_gb * 2**30) if max_gb is not None else available_memory_bytes() // 2
        ram = load_into_ram(self._datamodule.dadc, max_bytes)
        return CellariumDataView(self, np.arange(self.n_obs, dtype=np.int64), collection=ram, num_workers=0)

    def to_deltacells(
        self,
        uri: str,
        tile_size: int = 10_000,
        sort_genes: bool = True,
        overwrite: bool = False,
        staging_dir: str | None = None,
        deltacells_kwargs: dict[str, Any] | None = None,
        filesystem: Any = None,
        **writer_kwargs: Any,
    ) -> "CellariumDataView":
        """
        Write these cells to a new deltacells dataset at ``uri``, and return a :class:`CellariumDataView` of them that
        reads from there. This is for training repeatedly on a view of cells that are scattered over a large dataset: an
        epoch over the view reads every shard that holds one of its cells, but an epoch over the new, small dataset only
        reads what it needs.

        It is one pass over the shards that hold the cells (for a view of cells scattered over the data, a pass over all
        of it), plus a look at a few tiles to sort the genes by total counts. The cells are written in their order in
        the data (which is shuffled, if the data is) with all of their ``obs`` and the ``var``; the cells of the
        original data are the same ones, so ``obs`` names are unchanged. ``X`` must hold integer counts from 0 to 65535
        (as deltacells requires), and if it does not nothing is written.

        The returned view has the same cells (and root indices) as these, so ``update_from`` works between them, and
        starts with copies of this data's ``obsm`` and ``obs_computed`` but with no trained modules, as any view does.
        To use the dataset later, on its own, open it with :meth:`CellariumData.from_deltacells`; that is a new root,
        unrelated to the data it was made from.

        Args:
            uri: A local directory or a ``gs://bucket/prefix`` (staged locally, then uploaded, which needs ``gcsfs``).
                It must be empty or not exist unless ``overwrite``.
            tile_size: Cells per tile (the unit of reading and of shuffling).
            sort_genes: Store the genes by decreasing total counts (better compression).
            overwrite: Replace an existing dataset at ``uri``.
            staging_dir: Where to write the dataset before uploading it to ``gs://`` (default: the system temp
                directory).
            deltacells_kwargs: Arguments for opening the new dataset, as for :class:`CellariumData` (by default those
                this data was opened with, if it was read from a deltacells dataset).
            filesystem: An fsspec filesystem to upload with instead of ``gcsfs`` (for testing).
            **writer_kwargs: Further arguments of :class:`deltacells.DatasetWriter`, for example ``level``.
        """
        write_collection_to_deltacells(
            self._datamodule.dadc,
            uri,
            tile_size=tile_size,
            sort_genes=sort_genes,
            overwrite=overwrite,
            staging_dir=staging_dir,
            filesystem=filesystem,
            **writer_kwargs,
        )
        collection = DistributedDeltaCellsCollection(uri, **{**self._deltacells_kwargs, **(deltacells_kwargs or {})})
        view = CellariumDataView(self, np.arange(self.n_obs, dtype=np.int64), collection=collection)
        view._deltacells_uri = uri
        view._deltacells_kwargs = {**self._deltacells_kwargs, **(deltacells_kwargs or {})}
        return view

    def _positions_of(self, other: "CellariumData") -> np.ndarray:
        """The positions in these cells of the cells of ``other``, a view of the same root with a subset of them."""
        if other.root is not self.root:
            raise ValueError(
                "The view does not descend from the same root as this data (for example, it was made from a "
                "different CellariumData), so its cells cannot be matched to these."
            )
        mine, theirs = self.root_indices, other.root_indices
        positions = np.searchsorted(mine, theirs)
        if positions[-1] >= len(mine) or not np.array_equal(mine[positions], theirs):
            raise ValueError("The view has cells that are not among the cells of this data.")
        return positions.astype(np.int64)

    # without this, `iter(cdata)` would walk `__getitem__` and yield single-cell views
    __iter__ = None  # type: ignore[assignment]

    def __getitem__(self, selection: Any) -> "CellariumDataView":
        """
        A :class:`CellariumDataView` of the selected cells; no data is copied. The selection can be

        * a boolean mask (a numpy array of length ``n_obs``, or a :class:`pandas.Series` whose index is exactly
          ``cdata.obs.index``),
        * integer positions (an array, a list or a single integer; negative positions count from the end), or
        * a slice.

        The view holds its cells in the order of the data, whatever the order of the selection, without repeats.

        Raises:
            ValueError: If the selection is empty or a Series mask has a different index.
        """
        return CellariumDataView(self, self._selected_positions(selection))

    def _selected_positions(self, selection: Any) -> np.ndarray:
        """The sorted, unique positions of the cells ``selection`` selects (see :meth:`__getitem__`)."""
        n = self.n_obs
        if isinstance(selection, slice):
            positions = np.arange(*selection.indices(n), dtype=np.int64)
        else:
            if isinstance(selection, pd.Series):
                if selection.dtype != bool:
                    raise TypeError(
                        f"A Series selection must be a boolean mask, got dtype {selection.dtype}; "
                        "use .to_numpy() to select by position."
                    )
                if len(selection) != n or not selection.index.equals(self.obs.index):
                    raise ValueError("The index of a Series mask must be exactly the index of cdata.obs.")
                selection = selection.to_numpy()
            array = np.asarray(selection)
            if array.size == 0:
                positions = np.empty(0, dtype=np.int64)
            elif array.dtype == bool:
                if array.shape != (n,):
                    raise IndexError(f"A boolean mask must have shape ({n},), got {array.shape}")
                positions = np.flatnonzero(array)
            elif np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.int64).ravel()
                if array.min() < -n or array.max() >= n:
                    raise IndexError(f"Positions must lie in [-{n}, {n}), got {array.min()} to {array.max()}")
                positions = np.where(array < 0, array + n, array)
            else:
                raise TypeError(
                    f"Select cells with a boolean mask, integer positions or a slice, not {array.dtype} values."
                )
        positions = np.unique(positions)
        if len(positions) == 0:
            raise ValueError("The selection is empty: a view must contain at least one cell.")
        return positions

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def obs(self) -> LazyObs | DeltaCellsLazyObs | ViewLazyObs:
        return self._obs

    @property
    def obsm(self) -> ObsmMapping:
        return self._obsm

    @property
    def obs_computed(self) -> ObsComputedMapping:
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
        if self._parent is not None:
            lines.append(f"  view of {self._parent.n_obs} cells (the root has {self._root.n_obs})")
        if self._trained_modules:
            lines.append("  trained_modules:")
            lines.extend(f"    {key}: {trained!r}" for key, trained in self._trained_modules.items())
        return "\n".join(lines)


class CellariumDataView(CellariumData):
    """
    A subset of the cells of a :class:`CellariumData` (or of another view), made with ``cdata[selection]`` (see
    :meth:`CellariumData.__getitem__`). It works wherever a :class:`CellariumData` does, for example with every api
    function, and no cell data is copied: its datamodule reads the cells from the data it was taken from, so an
    epoch over a view still reads every shard that holds one of its cells, but only the cells of the view are
    converted and passed on to the model.

    A view has its own, view-sized ``obs`` (reading from the root's), ``obsm``, ``obs_computed``, ``hvg``,
    ``trained_modules`` and fitted-statistics cache, so models trained on a view do not touch the parent's. ``obsm``
    and ``obs_computed`` start as copies of the parent's, restricted to the cells of the view; the rest starts empty
    (the parent's HVGs and trained modules were fit to other cells). To use a module trained on the parent anyway,
    assign it explicitly: ``view.trained_modules["scvi"] = parent.trained_modules["scvi"]``. The datamodule has the
    settings that the parent's had when the view was made.

    Attributes:
        parent: The data this view was taken from.
        root: The data that is not a view, which every view descends from.
        root_indices: The positions of the cells of the view in :attr:`root`, in increasing order.

    Args:
        parent: The data to take the cells from.
        positions: The positions in ``parent`` of the cells of the view; strictly increasing.
        collection: The collection that the view's datamodule reads, if not a view of the parent's. It must hold the
            cells ``positions`` of ``parent``, in order (used when the cells have been copied somewhere faster).
        num_workers: Number of dataloader workers, if not the parent's.
    """

    def __init__(
        self,
        parent: CellariumData,
        positions: np.ndarray,
        *,
        collection: DistributedCollection | None = None,
        num_workers: int | None = None,
    ):
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            raise ValueError("A view must contain at least one cell.")
        if np.any(positions[1:] <= positions[:-1]):
            raise ValueError("positions must be strictly increasing (sorted and unique).")
        if positions[0] < 0 or positions[-1] >= parent.n_obs:
            raise IndexError(f"positions must lie in [0, {parent.n_obs}), got {positions[0]} to {positions[-1]}")
        if collection is None:
            collection = ViewCollection(parent.datamodule.dadc, positions)
        elif len(collection) != len(positions):
            raise ValueError(f"The collection has {len(collection)} cells but the view has {len(positions)}.")
        self._parent = parent
        self._root = parent.root
        self._root_indices = parent.root_indices[positions]
        self._stage = parent._stage
        self._deltacells_uri = parent._deltacells_uri
        self._deltacells_kwargs = dict(parent._deltacells_kwargs)
        self._datamodule = _clone_datamodule(parent.datamodule, collection, self._stage, num_workers)
        self._obs = ViewLazyObs(self._root.obs, self._root_indices)  # type: ignore[arg-type]
        self._var = parent.var.copy()
        self._init_state()
        for source, target in ((parent.obsm, self._obsm), (parent.obs_computed, self._obs_computed)):
            for key, value in source.items():
                try:
                    target[key] = _subset_rows(value, positions)
                except (TypeError, ValueError, IndexError) as error:
                    warnings.warn(f"'{key}' of the parent could not be restricted to the view and is left out: {error}")
