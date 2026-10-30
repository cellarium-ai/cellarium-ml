# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import json
import math
import os
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterator, Literal, Sequence

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import scipy.sparse
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from tqdm.auto import tqdm

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._cell_ids import CELL_ID_COLUMN
from cellarium.ml.api._datamodule_context import temporary_batch_keys
from cellarium.ml.api._obsm_storage import RowSubset, subset_rows
from cellarium.ml.api._view_collections import (
    RamCollection,
    ViewCollection,
    available_memory_bytes,
    load_into_ram,
    unwrap_collection,
)
from cellarium.ml.api.deltacells_store import write_collection_to_deltacells
from cellarium.ml.api.utils import LossHistory, get_h5ad_files_limits, write_obs_parquet
from cellarium.ml.data import (
    DistributedAnnDataCollection,
    DistributedCollection,
    DistributedDeltaCellsCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.utilities.core import FunctionComposer
from cellarium.ml.utilities.data import AnnDataField, to_float_tensor, to_torch_sparse_coo, to_torch_sparse_csr

# a pass over a view reads this many times as much of the data as the view's cells need (or more) ...
_READ_AMPLIFICATION_WARNING = 5.0
# ... and it is only worth a warning if the view's cells are in at least this many shards
_MIN_SHARDS_FOR_READ_AMPLIFICATION_WARNING = 10


def _make_names_unique(names: Any) -> np.ndarray:
    """
    The gene ``names`` with the later copies of a repeated name suffixed (``"TBCE"``, ``"TBCE-1"``, ``"TBCE-2"``), as
    :func:`anndata.utils.make_index_unique` does. A module-level function so that data workers can pickle it.
    """
    return np.asarray(anndata.utils.make_index_unique(pd.Index(np.asarray(names).astype(str))), dtype=object)


def _build_datamodule(
    dadc: DistributedCollection,
    obs_columns: dict[str, tuple[str, Callable]],
    var_key: str | None,
    batch_size: int,
    shuffle: bool,
    train_size: float,
    stage: Literal["fit", "validate", "predict", "test"],
    num_workers: int,
    var_names_make_unique: bool = False,
) -> CellariumAnnDataDataModule:
    convert_fn = _make_names_unique if var_names_make_unique else None
    datamodule = CellariumAnnDataDataModule(
        dadc=dadc,
        batch_keys={
            "x_ng": AnnDataField(
                attr="X",
                # mps has no kernel to move a sparse CSR tensor onto device, so use sparse COO there instead
                convert_fn=to_torch_sparse_coo if torch.mps.is_available() else to_torch_sparse_csr,  # type: ignore[arg-type]
            ),
            "var_names_g": AnnDataField(attr="var_names", convert_fn=convert_fn)
            if var_key is None
            else AnnDataField(attr="var", key=var_key, convert_fn=convert_fn),
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
    var_names_make_unique: bool = False,
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
    return _build_datamodule(
        dadc, obs_columns, var_key, batch_size, shuffle, train_size, stage, num_workers, var_names_make_unique
    )


def get_deltacells_datamodule(
    uri: str,
    obs_columns: dict[str, tuple[str, Callable]] = {},
    var_key: str | None = None,
    batch_size: int = 4096,
    shuffle: bool = True,
    train_size: float = 1.0,
    stage: Literal["fit", "validate", "predict", "test"] = "fit",
    num_workers: int = 0,
    var_names_make_unique: bool = False,
    **collection_kwargs: Any,
):
    """
    Like :func:`get_datamodule` but reading a deltacells dataset (see
    :func:`~cellarium.ml.api.create_deltacells_dataset`) at ``uri``. ``collection_kwargs`` are passed to
    :class:`~cellarium.ml.data.DistributedDeltaCellsCollection`.
    """
    dadc = DistributedDeltaCellsCollection(uri, **collection_kwargs)
    return _build_datamodule(
        dadc, obs_columns, var_key, batch_size, shuffle, train_size, stage, num_workers, var_names_make_unique
    )


def _check_var_names_unique(var: pd.DataFrame, names: pd.Index, var_key: str | None) -> None:
    """
    Raise a ``ValueError`` if the gene ``names`` (``var[var_key]``, or the ``var_names`` if ``var_key`` is ``None``)
    are not unique, listing the ``var_key`` values that would give unique names.
    """
    if names.is_unique:
        return
    where = "var_names" if var_key is None else f"var_key={var_key!r}"
    repeated = names[names.duplicated()].unique()

    def is_name_like(column: pd.Series) -> bool:
        dtype = column.cat.categories.dtype if isinstance(column.dtype, pd.CategoricalDtype) else column.dtype
        return bool(pd.api.types.is_string_dtype(dtype)) and column.is_unique and not column.isna().any()

    options: list[str | None] = [column for column in var.columns if column != var_key and is_name_like(var[column])]
    if var_key is not None and var.index.is_unique:
        options.append(None)
    suggestion = (
        f"Pass a var_key with unique names, one of {options[:10]}"
        f"{' (and more)' if len(options) > 10 else ''} (None is the var_names)."
        if options
        else "None of the var columns has unique names."
    )
    raise ValueError(
        f"The gene names ({where}) must be unique, but {len(repeated)} of them are repeated, "
        f"for example {repeated[:5].tolist()}. {suggestion} "
        "Or pass var_names_make_unique=True to keep these names and suffix the repeated ones "
        "('name', 'name-1', ...); the original names are kept in var."
    )


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

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, pd.Series):
            value = self._owner._series_by_cell(value, f"obs_computed['{key}']")
        super().__setitem__(key, value)

    def update_from(self, view: "CellariumData", key: str, as_key: str | None = None, overwrite: bool = True) -> None:
        """
        Copy the values ``view.obs_computed[key]`` (for example the subtypes found by clustering a subset of the
        cells) to the cells they belong to in this data, as ``obs_computed[as_key]`` (``key`` by default).

        ``view`` must descend from the same root as this data, and its cells must be among this data's. The entry is
        a :class:`pandas.Series` indexed by the cell IDs, with pandas' nullable dtypes (``string`` or ``category``
        for labels, ``Int64``, ``boolean``, floats). Cells that the view does not hold are missing (NA). If the entry
        already exists, its values for those cells are kept, so that several views can each fill in their cells of one
        entry; the values it already has for cells of the view are replaced, except where the view's value is missing.
        New labels are added to a categorical entry's categories.

        Args:
            view: The view (or the data itself, or any other data from the same root) that computed the values.
            key: The key of the values in ``view.obs_computed``.
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
        values = _per_cell_series(view.obs_computed[key], view.n_obs, f"obs_computed['{key}'] of the view")
        self._put(key if as_key is None else as_key, values, positions, key, overwrite)

    def _put(self, target: str, values: pd.Series, positions: np.ndarray, source: str, overwrite: bool) -> None:
        """
        Write ``values`` (nullable, one per position) into ``obs_computed[target]`` for the cells at ``positions`` of
        the owner, keeping what the entry has for its other cells (they are missing if it did not exist). ``source`` is
        what the values are called in error messages.
        """
        owner = self._owner
        n = owner.n_obs
        if target not in self:
            merged = values.set_axis(positions).reindex(np.arange(n))
        else:
            existing = _per_cell_series(self[target], n, f"obs_computed['{target}']")
            merged = self._merge(existing, values, positions, source, target, overwrite)
        merged.index = owner.cell_ids
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
        # everything needed from the trainer has been recorded above. The module keeps running without it, and a new
        # fit attaches it to its trainer again; but a module that keeps its trainer keeps the trainer's datamodule,
        # loops and callbacks alive for as long as the module is in `trained_modules`
        module.trainer = None
        return trained

    cdata._warn_if_read_amplified("Training the model")
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

    @property
    def index(self) -> pd.Index:
        """The cell IDs, in cell order: the positions ``0, 1, ...`` of the cells
        (see :mod:`~cellarium.ml.api._cell_ids`).
        The frames and Series that are read from here are indexed by them, so no cell name is read along."""
        return pd.RangeIndex(len(self), name=CELL_ID_COLUMN)

    @property
    def names(self) -> pd.Index:
        """The obs names (the cell names), in cell order. These are read from the database when asked for."""
        return self._names_at(None)

    def _names_at(self, positions: np.ndarray | None) -> pd.Index:
        """The obs names of the cells at ``positions`` (all cells if ``None``)."""
        dataset, index_col = self._get_dataset(), self._index_column_name
        if positions is None:
            table = dataset.to_table(columns=[index_col])
        else:  # only these rows are kept in memory, not the names of all the cells
            table = dataset.take(pa.array(positions, type=pa.int64()), columns=[index_col])
        return table.to_pandas().index

    def _take(self, positions: np.ndarray | None, columns: list[str]) -> pd.DataFrame:
        """The ``columns`` of the cells at ``positions`` (all cells if ``None``), in the order of ``positions``, indexed
        by the cell IDs.

        Assumes that row ``i`` of the parquet database is cell ``i`` of the data, as it is when the database is built
        from the h5ad files.
        """
        ids = self.index if positions is None else pd.Index(positions, dtype=np.int64, name=CELL_ID_COLUMN)
        if not columns:
            return pd.DataFrame(index=ids)
        table = self._get_dataset().to_table(columns=list(columns))
        if positions is not None:
            table = table.take(pa.array(positions, type=pa.int64()))
        df = table.to_pandas()
        df.index = ids
        return df

    def __getitem__(self, key: str | list[str]) -> pd.DataFrame | pd.Series:
        columns = [key] if isinstance(key, str) else list(key)
        df = self._take(None, columns)
        return df[key] if isinstance(key, str) else df

    @property
    def loc(self) -> "_LazyObsLoc":
        return _LazyObsLoc(self)

    def iter_batches(self, batch_size: int = 100_000, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        """Stream `obs` in chunks of `pd.DataFrame`, for full-corpus processing without loading it all at once."""
        columns = self.columns if columns is None else list(columns)
        start = 0
        for batch in self._get_dataset().to_batches(batch_size=batch_size, columns=columns):
            df = batch.to_pandas()
            df.index = pd.RangeIndex(start, start + len(df), name=CELL_ID_COLUMN)
            start += len(df)
            yield df

    def to_frame(self) -> pd.DataFrame:
        """Materialize the entire `obs` dataframe into memory."""
        return self._take(None, self.columns)

    def __repr__(self) -> str:
        # build the database first if need be, so that `parquet_path` is set by the time it is formatted
        self._get_dataset()
        return (
            f"{self.__class__.__name__}(parquet_path={self._parquet_path!r}, n_obs={len(self)}, columns={self.columns})"
        )


class _LazyObsLoc:
    """Implements `LazyObs.loc[row_labels]` / `LazyObs.loc[row_labels, columns]`: the cells are selected by their obs
    names, and the frame that is returned is indexed by their cell IDs."""

    def __init__(self, lazy_obs: LazyObs):
        self._lazy_obs = lazy_obs

    def __getitem__(self, key) -> pd.DataFrame:
        lazy_obs = self._lazy_obs
        row_key, col_key = key if isinstance(key, tuple) else (key, None)
        row_labels = [row_key] if isinstance(row_key, str) else list(row_key)
        columns = lazy_obs.columns if col_key is None else ([col_key] if isinstance(col_key, str) else list(col_key))

        # find the rows of the cells in arrow, without turning the names of all the cells into python strings
        index_col = lazy_obs._index_column_name
        names = lazy_obs._get_dataset().to_table(columns=[index_col]).column(index_col)
        matches = pc.is_in(names, value_set=pa.array(row_labels).cast(names.type))
        found = np.flatnonzero(matches.to_pandas().to_numpy())
        found_names = pd.Index(names.take(pa.array(found, type=pa.int64())).to_pandas())
        order = found_names.get_indexer(row_labels)
        if (order < 0).any():
            missing = [label for label, o in zip(row_labels, order) if o < 0]
            raise KeyError(f"Labels not found in obs index: {sorted(missing)[:5]}")
        return lazy_obs._take(found[order].astype(np.int64), columns)


class DeltaCellsLazyObs:
    """
    A queryable view over the ``obs`` of a :class:`~cellarium.ml.data.DistributedDeltaCellsCollection`, with the same
    interface as :class:`LazyObs`. Only the columns (and cells) a query asks for are read; categorical columns are
    pandas categoricals with the dataset's global categories. The returned frames are indexed by the cell IDs: those
    in the dataset's ``cellarium_cell_id`` column if it has one (see :mod:`~cellarium.ml.api._cell_ids`), and the
    positions of the cells if not. The ``obs_names`` of the dataset are :attr:`names`, and ``.loc`` selects by them.
    """

    _NAMES = "obs_names"

    def __init__(self, dadc: DistributedDeltaCellsCollection):
        self._dadc = dadc
        self._cell_ids: pd.Index | None = None

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

    def _names_at(self, indices: np.ndarray | None = None) -> pd.Index:
        """The obs names of the cells at ``indices`` (all cells if ``None``)."""
        if indices is None:
            names = self._store.to_pandas(self._NAMES)[self._NAMES]
        else:
            names = self._store.take_pandas(indices, [self._NAMES], warn=False)[self._NAMES]
        return pd.Index(np.asarray(names, dtype=object), name=self._NAMES)

    @property
    def names(self) -> pd.Index:
        """The obs names (the cell names), in cell order."""
        return self._names_at()

    @property
    def index(self) -> pd.Index:
        """The cell IDs, in cell order (see :mod:`~cellarium.ml.api._cell_ids`)."""
        if self._cell_ids is None:
            if CELL_ID_COLUMN in self._store.columns:
                ids = np.asarray(self._store.to_pandas([CELL_ID_COLUMN])[CELL_ID_COLUMN], dtype=np.int64)
                self._cell_ids = pd.Index(ids, name=CELL_ID_COLUMN)
            else:
                self._cell_ids = pd.RangeIndex(len(self), name=CELL_ID_COLUMN)
        return self._cell_ids

    def _take(self, indices: np.ndarray | None, columns: list[str]) -> pd.DataFrame:
        if not columns:
            df = pd.DataFrame(index=pd.RangeIndex(len(self) if indices is None else len(indices)))
        elif indices is None:
            df = self._store.to_pandas(columns)
        else:
            df = self._store.take_pandas(indices, columns, warn=False)
        df.index = self.index if indices is None else self.index[indices]
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

        names = lazy_obs.names
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
        self._cell_ids: pd.Index | None = None

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def columns(self) -> list[str]:
        return self._source.columns

    @property
    def index(self) -> pd.Index:
        """The cell IDs of the view, in cell order (those of the root; see :mod:`~cellarium.ml.api._cell_ids`)."""
        if self._cell_ids is None:
            self._cell_ids = self._take(None, []).index
        return self._cell_ids

    @property
    def names(self) -> pd.Index:
        """The obs names (the cell names) of the view, in cell order. Read from the root when first asked for."""
        if self._names is None:
            self._names = self._source._names_at(self._positions)
        return self._names

    def _names_at(self, indices: np.ndarray | None) -> pd.Index:
        """The obs names of the cells of the view at ``indices`` (all of them if ``None``)."""
        if indices is None:
            return self.names
        if self._names is not None:
            return self._names[indices]
        return self._source._names_at(self._positions[indices])

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

        names = lazy_obs.names
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


class _PositionedDataset(IterableDistributedAnnDataCollectionDataset):
    """
    The dataset of :meth:`CellariumData.predict_batches`: it adds the positions of the cells of each batch (in the
    collection that it reads, which is the order of the cells of the data) to the batch, as ``"row_indices_n"``.
    """

    def __getitem__(self, idx: Any) -> Any:
        data = super().__getitem__(idx)
        if isinstance(idx, slice):
            positions = np.arange(*idx.indices(len(self.dadc)), dtype=np.int64)
        else:
            positions = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        data["row_indices_n"] = positions
        return data


def _device_of(module: torch.nn.Module) -> torch.device | None:
    """The device of the first parameter or buffer of ``module``, or ``None`` if it has none."""
    for tensor in (*module.parameters(), *module.buffers()):
        return tensor.device
    return None


def _subset_rows(value: Any, positions: np.ndarray, max_bytes: int | None = None) -> Any:
    """
    The rows ``positions`` of a per-cell value (array, sparse matrix, tensor, Series or DataFrame), as a copy.

    A value that lives on disk (a :class:`numpy.memmap`, or a :class:`~cellarium.ml.api._obsm_storage.RowSubset` of
    one) is read into memory, unless ``max_bytes`` is given and the rows take more than that: then they stay on disk
    and are returned as a lazy :class:`~cellarium.ml.api._obsm_storage.RowSubset`, which copies nothing.
    """
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.iloc[positions].copy()
    if scipy.sparse.issparse(value):
        return value.tocsr()[positions]
    if isinstance(value, torch.Tensor):
        return value[torch.from_numpy(positions)].clone()
    if isinstance(value, (np.memmap, RowSubset)):
        return subset_rows(value, positions, max_bytes)
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

    The genes are identified by their names, so these must be unique: the names are the ``var_names`` of the data, or
    the ``var`` column ``var_key`` if it is given. If they are not, construction raises a ``ValueError`` that lists
    the ``var`` columns that are unique. With ``var_names_make_unique=True`` the later copies of a repeated name are
    instead suffixed (``"TBCE"``, ``"TBCE-1"``, ...), as in :func:`anndata.utils.make_index_unique`, and the names as
    they were stay in ``var`` as the column ``"<var_key>_original"`` (``"var_names_original"`` without a ``var_key``).

    ``max_gb`` is the memory budget (in GiB) for what api functions hold in memory: the cells read by :meth:`to_ram` and
    :meth:`to_anndata`, and per-cell results stored in ``obsm`` (which go to a temporary file when they do not fit,
    and which a view reads into memory only if its rows fit). Views inherit it. By default it is half of the memory that
    is available, taking any container limit into account. It can be changed at any time with the :attr:`max_gb`
    attribute.
    """

    def __init__(
        self,
        h5ad_paths: list[str] | None = None,
        deltacells_uri: str | None = None,
        deltacells_kwargs: dict[str, Any] | None = None,
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
        var_names_make_unique: bool = False,
        max_gb: float | None = None,
    ):
        if (h5ad_paths is None) == (deltacells_uri is None):
            raise ValueError("Provide exactly one of h5ad_paths and deltacells_uri.")
        # the arguments as they were given, for `save`
        self._init_kwargs = copy.deepcopy({name: value for name, value in locals().items() if name != "self"})
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
                var_names_make_unique=var_names_make_unique,
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
                var_names_make_unique=var_names_make_unique,
                **(deltacells_kwargs or {}),
            )
            self._obs = DeltaCellsLazyObs(self._datamodule.dadc)
        self._stage = stage
        self.max_gb = max_gb
        self._deltacells_uri = deltacells_uri
        self._deltacells_kwargs = dict(deltacells_kwargs or {})
        self._var = self._datamodule.dadc.var.copy()
        if "var_names_g" in self._datamodule.batch_keys:
            anndatafield: AnnDataField = self._datamodule.batch_keys["var_names_g"]
            if anndatafield.key is not None:
                self._var.set_index(anndatafield.key, inplace=True)
            if var_names_make_unique:
                # the original names stay available, in a column named after where they came from
                original_names = self._var.index.to_numpy()
                self._var.index = pd.Index(self._datamodule.var_names_g, name=self._var.index.name)
                self._var[f"{anndatafield.key or 'var_names'}_original"] = original_names
            else:
                assert anndatafield.key is None or isinstance(anndatafield.key, str)  # None: the var_names
                _check_var_names_unique(self._datamodule.dadc.var, self._var.index, anndatafield.key)
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
        # the obsm entries that cml.tl.embed made, with the trained module and the array it made each from
        self._embedding_sources: dict[str, tuple[TrainedModule, Any]] = {}
        self._read_amplification_warned = False

    def read_amplification(self) -> float | None:
        """
        How much more of the data an epoch over a view reads than the view's cells need: the fraction of the shards
        (files, or tiles of a deltacells dataset) of the data that hold a cell of the view, divided by the fraction of
        its cells that the view has. It is 1 for a view of whole shards and is large for a view of cells that are
        scattered over many shards (for example the T cells of a large atlas), as every pass over such a view reads
        nearly all of the data. ``None`` if the cells are not a view of another dataset's (data that is not a view, and
        views that :meth:`to_deltacells` and :meth:`to_ram` made, which read a dataset of their own).
        """
        collection = self._datamodule.dadc
        if not isinstance(collection, ViewCollection):
            return None
        shard_ends = np.asarray(collection.source.limits)
        positions = collection.positions  # sorted
        shard_of_cell = np.searchsorted(shard_ends, positions, side="right")
        n_touched = 1 + int(np.count_nonzero(np.diff(shard_of_cell)))
        return (n_touched / len(shard_ends)) / (len(positions) / len(collection.source))

    def _warn_if_read_amplified(self, operation: str) -> None:
        """Warn (once for this data) that ``operation`` reads far more than the cells of this view if they are scattered
        over many shards, and say how to avoid it."""
        if self._read_amplification_warned:
            return
        amplification = self.read_amplification()
        if amplification is None or amplification < _READ_AMPLIFICATION_WARNING:
            return
        collection = self._datamodule.dadc
        n_shards = len(collection.source.limits)
        n_touched = int(round(amplification * self.n_obs / len(collection.source) * n_shards))
        if n_touched < _MIN_SHARDS_FOR_READ_AMPLIFICATION_WARNING:
            return
        self._read_amplification_warned = True
        warnings.warn(
            f"{operation} reads {n_touched} of the {n_shards} shards of the data ({n_touched / n_shards:.0%}) to get "
            f"the {self.n_obs} cells of this view ({self.n_obs / len(collection.source):.1%} of the data): about "
            f"{amplification:.0f}x as much as the cells need, and every further pass over the view does the same. "
            "To make later passes cost only what the view's cells need, write it as a dataset of its own first (one "
            "pass), and use the view that this returns: view = view.to_deltacells(uri).",
            stacklevel=3,
        )

    def _cached_embedding_state(self, module_key: str) -> Literal["current", "stale"] | None:
        """
        Whether ``obsm["X_<module_key>"]`` is an embedding that :func:`~cellarium.ml.api.tools.embed` made: ``None`` if
        it is not (or there is no such entry), ``"current"`` if it was made by the module that is now in
        ``trained_modules[module_key]``, and ``"stale"`` if by another one (for example by a module of the parent of a
        view, which the view has since replaced).
        """
        name = f"X_{module_key}"
        source = self._embedding_sources.get(name)
        if source is None or self._obsm.get(name) is not source[1]:
            return None
        return "current" if self._trained_modules.get(module_key) is source[0] else "stale"

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
    def max_gb(self) -> float | None:
        """The memory budget (in GiB) for what api functions hold in memory, or ``None``
        for half of what is available."""
        return self._max_gb

    @max_gb.setter
    def max_gb(self, value: float | None) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"max_gb must be positive, got {value}.")
        self._max_gb = value

    def _max_bytes(self, max_gb: float | None = None) -> int:
        """The memory budget in bytes: ``max_gb`` if given, else :attr:`max_gb`, else half of the available memory."""
        max_gb = self._max_gb if max_gb is None else max_gb
        return int(max_gb * 2**30) if max_gb is not None else available_memory_bytes() // 2

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
            max_gb: The most memory (in GiB) to use. By default the :attr:`max_gb` of this data, which by default is
                half of the memory that is available, taking any container limit into account.

        Raises:
            MemoryError: If the cells do not fit.
        """
        if isinstance(unwrap_collection(self._datamodule.dadc), RamCollection):
            return self  # type: ignore[return-value]
        ram = load_into_ram(self._datamodule.dadc, self._max_bytes(max_gb))
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
        original data are the same ones, so ``obs`` names are unchanged. ``X`` must hold integer counts from 0 to
        2**31 - 1 (as deltacells requires), and if it does not nothing is written.

        The dataset also stores the :attr:`cell_ids` of its cells, as the ``obs`` column ``cellarium_cell_id``. The
        returned view has the same cells (and root indices) as these, so ``update_from`` works between them, and
        starts with copies of this data's ``obsm`` and ``obs_computed`` but with no trained modules, as any view does.
        To use the dataset later, on its own, open it with :meth:`CellariumData.from_deltacells`: that is a new root
        (so views of it and of the data it was made from cannot be matched with ``update_from``), but its cells have
        the same cell IDs as in the original data, so labels computed on it can be brought back to the original data
        with :meth:`save_labels` and :meth:`load_labels`.

        Args:
            uri: A local directory or a ``gs://bucket/prefix`` (each tile is uploaded as it is written, which needs
                ``gcsfs``). It must be empty or not exist unless ``overwrite``.
            tile_size: Cells per tile (the unit of reading and of shuffling).
            sort_genes: Store the genes by decreasing total counts (better compression).
            overwrite: Replace an existing dataset at ``uri``: the files of a deltacells dataset that are there are
                deleted first; if there is anything else, this raises instead.
            staging_dir: Scratch directory for the tiles waiting to be uploaded to ``gs://`` (default: the system temp
                directory). Only a few tiles are ever there, not the dataset.
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
            cell_ids=self.cell_ids.to_numpy(),
            **writer_kwargs,
        )
        collection = DistributedDeltaCellsCollection(uri, **{**self._deltacells_kwargs, **(deltacells_kwargs or {})})
        view = CellariumDataView(self, np.arange(self.n_obs, dtype=np.int64), collection=collection)
        view._deltacells_uri = uri
        view._deltacells_kwargs = {**self._deltacells_kwargs, **(deltacells_kwargs or {})}
        return view

    def to_anndata(
        self,
        downsample_method: Literal["random", "geometric_sketch"] | None = None,
        target_n_cells: int | None = None,
        *,
        geometric_sketch_kwargs: dict[str, Any] | None = None,
        collect_embeddings: bool | str | Sequence[str] = True,
        seed: int = 0,
        max_gb: float | None = None,
    ) -> anndata.AnnData:
        """
        Read these cells into memory as an :class:`~anndata.AnnData`, to continue the analysis with scanpy. This is
        **memory-heavy** (see :meth:`to_ram`), so it is meant for a view of a few cells; to take a manageable number of
        cells from many, downsample them first, at random or with a geometric sketch.

        The AnnData has the raw counts as ``X`` (a float32 CSR matrix), the cells in the order of the data, and

        * ``obs``: all the ``obs`` columns, and the entries of ``obs_computed`` as columns ``computed_<key>``,
        * ``var``: the ``var`` and a boolean ``highly_variable`` column (``False`` if ``hvg`` has not been set),
        * ``obsm``: the entries of ``obsm``, and the embeddings by trained modules (see ``collect_embeddings``) as
          ``X_<key>``, where ``key`` is the key of the module in ``trained_modules`` (for example ``X_pca``).

        The embeddings are computed from the cells in memory once they have been selected, so they do not need another
        pass over the data.

        Example:
            >>> is_t_cell = (cdata.obs["cell_type"] == "T cell").to_numpy()
            >>> adata = cdata[is_t_cell].to_anndata(downsample_method="geometric_sketch", target_n_cells=10_000)

        Args:
            downsample_method: How to choose the cells that are kept, if not all of them: ``"random"`` (uniformly,
                without replacement), ``"geometric_sketch"`` (see :func:`~cellarium.ml.api.tools.geometric_sketch`,
                which makes a pass over the data) or ``None`` (all the cells). Random sampling reads cells that are
                scattered over the data, which for a large dataset means reading nearly all of its files.
            target_n_cells: The number of cells to keep. It is required with a ``downsample_method`` and an error
                without one. If the data has no more cells than this, all of them are kept (and nothing is sampled).
            geometric_sketch_kwargs: Further arguments of :func:`~cellarium.ml.api.tools.geometric_sketch` (for
                example ``embedding_module_key``, ``n_pcs`` or ``accelerator``). By default the sketch is made in
                the embedding of the PCA in ``trained_modules`` (the one under ``"pca"``, if there is one), and if
                no PCA has been trained a warning says so and a random projection of the normalized, log-transformed
                counts is used. ``target_n_cells``, ``seed`` and ``return_new_adata`` are set by this method and
                cannot be given.
            collect_embeddings: The embeddings to put in ``obsm``: ``True`` for those of all trained modules that
                are embedding models (PCA and scVI; a module whose embedding cannot be computed is skipped with a
                warning, as is one whose ``X_<key>`` is already in ``obsm``), ``False`` for none, or the key (or
                keys) of the modules to embed with, which raises an error if one cannot be computed. A module that
                needs more data to run (such as scVI's batch column) gets it from ``obs``, as in :meth:`using`.
                A view only has the modules that were trained on it (or assigned to it).
            seed: The seed of the random sample and of the geometric sketch.
            max_gb: The most memory (in GiB) to read the counts into, as in :meth:`to_ram`: by default the
                :attr:`max_gb` of this data (half of the memory that is available, unless set). Nothing is read if the
                cells would not fit. It does not apply to the cells of a geometric sketch, which the sketch has read
                already.

        Raises:
            ValueError: If the arguments do not fit together (for example a ``target_n_cells`` without a
                ``downsample_method``), or a key of ``collect_embeddings`` is not a trained embedding model.
            MemoryError: If the cells do not fit in ``max_gb``.
        """
        from cellarium.ml.api._to_anndata import to_anndata

        return to_anndata(
            self, downsample_method, target_n_cells, geometric_sketch_kwargs, collect_embeddings, seed, max_gb
        )

    def update_labels_from_anndata(
        self, adata: anndata.AnnData, column: str, as_key: str | None = None, overwrite: bool = True
    ) -> None:
        """
        Copy a column of ``adata.obs`` (for example the clusters that scanpy found in the AnnData of
        :meth:`to_anndata`) to the cells it belongs to, as ``obs_computed[as_key]`` (``column`` by default), creating
        the entry if it is not there. The cells are matched by the cell IDs in ``adata.obs["cellarium_cell_id"]``
        (which :meth:`to_anndata` puts there), so ``adata`` can have had cells removed, or be of any subset of
        these cells, and its obs names do not matter.

        The entry is a :class:`pandas.Series` indexed by :attr:`cell_ids`, with pandas' nullable dtypes (``string``
        or ``category`` for labels, ``Int64``, ``boolean``, floats). The cells that ``adata`` does not hold are
        missing (NA), unless the entry already exists: then its values for them are kept, so that several AnnDatas
        (of different subsets of the cells) can fill in one entry. The values it has for cells of ``adata`` are
        replaced, except where ``adata``'s value is missing. New labels are added to a categorical entry's categories.

        Use one key for each clustering (for example ``"leiden_1.0"``, and ``"t_cell_subtype"`` for the clusters of a
        view of T cells): labels that are put under the same key are the labels of one entry.

        Args:
            adata: The AnnData with the labels, which must have ``obs["cellarium_cell_id"]``.
            column: The column of ``adata.obs`` with the labels.
            as_key: The key to store them under in ``obs_computed``, if not ``column``.
            overwrite: If ``False``, raise an error instead of replacing a value that is already present.

        Raises:
            KeyError: If ``adata.obs`` has no ``cellarium_cell_id`` or ``column``, or has cells that are not in this
                data.
            ValueError: If the cell IDs of ``adata`` are not unique, if the dtypes of the new and existing values do
                not fit together, or if ``overwrite`` is ``False`` and a value would be replaced.
        """
        for name in (CELL_ID_COLUMN, column):
            if name not in adata.obs:
                raise KeyError(
                    f"adata.obs has no '{name}'. "
                    + (
                        "It comes with the AnnData of to_anndata()."
                        if name == CELL_ID_COLUMN
                        else f"Columns: {list(adata.obs.columns)}"
                    )
                )
        cell_ids = adata.obs[CELL_ID_COLUMN].to_numpy()
        if len(np.unique(cell_ids)) != len(cell_ids):
            raise ValueError("The cell IDs in adata.obs['cellarium_cell_id'] are not unique.")
        positions = self._positions_of_ids(cell_ids)
        what = f"adata.obs['{column}']"
        values = _per_cell_series(adata.obs[column], adata.n_obs, what)
        self._obs_computed._put(column if as_key is None else as_key, values, positions, what, overwrite)

    def save_labels(self, keys: str | Sequence[str], path: str) -> None:
        """
        Write the ``obs_computed`` entries ``keys`` (labels, or any per-cell values) of the cells that have a value
        in at least one of them to the parquet file ``path``, with their :attr:`cell_ids`. Labels that were computed
        on a dataset made by :meth:`to_deltacells` (opened on its own, in another session) can be brought back to the
        data it was made from with :meth:`load_labels`, since the cells have the same IDs in both.
        """
        keys = [keys] if isinstance(keys, str) else list(keys)
        for key in keys:
            if key not in self._obs_computed:
                raise KeyError(f"No '{key}' in obs_computed. Available: {list(self._obs_computed)}")
        n = self.n_obs
        frame = pd.DataFrame(
            {key: _per_cell_series(self._obs_computed[key], n, f"obs_computed['{key}']") for key in keys}
        )
        frame.insert(0, CELL_ID_COLUMN, self.cell_ids.to_numpy())
        frame[frame[keys].notna().any(axis=1).to_numpy()].to_parquet(path, index=False)

    def load_labels(self, path: str, keys: str | Sequence[str] | None = None, overwrite: bool = True) -> None:
        """
        Read the values that :meth:`save_labels` wrote to ``path`` into ``obs_computed``, for the cells with those cell
        IDs, under the keys they were saved with. The cells of the file must all be among the cells of this data (the
        data that the labels were computed on, or one that it is a subset of: for example the original data, for
        labels computed on a compacted dataset). Entries that exist keep their values for the other cells, as in
        :meth:`update_labels_from_anndata`.

        Args:
            path: The parquet file written by :meth:`save_labels`.
            keys: The entries to read (all of those in the file by default).
            overwrite: If ``False``, raise an error instead of replacing a value that is already present.
        """
        frame = pd.read_parquet(path)
        if CELL_ID_COLUMN not in frame:
            raise ValueError(f"{path} has no '{CELL_ID_COLUMN}' column: it was not written by save_labels.")
        cell_ids = frame[CELL_ID_COLUMN].to_numpy()
        if len(np.unique(cell_ids)) != len(cell_ids):
            raise ValueError(f"The cell IDs in {path} are not unique.")
        available = [column for column in frame.columns if column != CELL_ID_COLUMN]
        keys = available if keys is None else ([keys] if isinstance(keys, str) else list(keys))
        for key in keys:
            if key not in available:
                raise KeyError(f"No '{key}' in {path}. Available: {available}")
        positions = self._positions_of_ids(cell_ids)
        for key in keys:
            values = _nullable(frame[key].reset_index(drop=True))
            self._obs_computed._put(key, values, positions, f"'{key}' of {path}", overwrite)

    def save(self, path: str, overwrite: bool = False) -> None:
        """
        Save this data and what has been computed from it to the directory ``path``, to be opened again with
        :meth:`load`, possibly in another session.

        The cell counts are not copied: the directory holds the arguments that this data was made with (the paths of
        the files, or the deltacells dataset), what identifies its cells, ``obs_computed``, ``obsm``, ``hvg`` and
        ``trained_modules`` (each module as a Lightning checkpoint, with its training history). A view is saved with
        the data it was taken from (its :attr:`root`) and the positions of its cells in it; the ``obsm``,
        ``obs_computed``, ``hvg`` and ``trained_modules`` of the root are not saved. The fitted statistics that api
        functions reuse are not saved either, so the first call that needs them computes them again.

        The functions in ``obs_columns`` must be importable by name (defined at the top level of a module) to be saved.

        Args:
            path: A local directory, which must not exist, unless ``overwrite`` is ``True``.
            overwrite: Replace the directory if it holds a saved session. Entries of ``obsm`` that were read from it
                by :meth:`load` stay on disk, so save somewhere else for those (an error says so); this also applies
                to other objects that read from it.

        Raises:
            ValueError: If something cannot be saved: ``obs_columns`` functions or the configuration of a trained
                module that cannot be pickled, or ``obsm`` entries that are not numeric arrays.
        """
        from cellarium.ml.api._session import save_session

        save_session(self, path, overwrite)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        h5ad_paths: list[str] | None = None,
        deltacells_uri: str | None = None,
        deltacells_kwargs: dict[str, Any] | None = None,
        obs_parquet_path: str | None = None,
        obs_columns: dict[str, tuple[str, Callable]] | None = None,
        batch_size: int | None = None,
        datamodule_num_workers: int | None = None,
        max_gb: float | None = None,
    ) -> "CellariumData":
        """
        Open the data that :meth:`save` saved to ``path``, with its ``obs_computed``, ``obsm`` (read from the files,
        not into memory), ``hvg`` and ``trained_modules`` (on the CPU). A saved view is loaded as a view of the data it
        was taken from, which is :attr:`root`; views that were saved separately do not share a root.

        The data is opened from where it was before. If it is not there, or is not the same data (a different number
        of cells, or other cells at the start and end of its files), a ``ValueError`` says what is wrong. The
        arguments below override what was saved, to load onto the same cells from somewhere else (for example the
        files of the data in another bucket, or a deltacells dataset that :meth:`to_deltacells` made from it).
        ``None`` keeps what was saved.

        .. warning::
            The directory holds pickled objects, so load only directories that you trust.

        Args:
            path: The directory that :meth:`save` wrote.
            h5ad_paths: The paths of the h5ad files to read the cells from, instead of those that were saved.
            deltacells_uri: The deltacells dataset to read the cells from, instead of the saved data.
            deltacells_kwargs: Arguments for the deltacells dataset (for example ``cache_dir``), instead of those that
                were saved.
            obs_parquet_path: The obs parquet database of h5ad files (see the constructor).
            obs_columns: The ``obs_columns`` of the constructor, instead of the saved ones.
            batch_size: The batch size of the datamodule.
            datamodule_num_workers: The number of dataloader workers.
            max_gb: The memory budget (see :attr:`max_gb`).
        """
        from cellarium.ml.api._session import load_session

        return load_session(
            path,
            h5ad_paths,
            deltacells_uri,
            deltacells_kwargs,
            obs_parquet_path,
            obs_columns,
            batch_size,
            datamodule_num_workers,
            max_gb,
        )

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

        * a boolean mask: a numpy array (or list) with one entry per cell, or a boolean :class:`pandas.Series`. A
          Series is matched to the cells by its index: one with a range index of one entry per cell, or an index equal
          to ``cdata.cell_ids`` (as the ``obs`` columns and ``obs_computed`` entries have), has one entry per cell, in
          the order of the cells; any other integer index (a shorter range included) holds cell IDs, and selects the
          cells of this data whose ID is in the index with a ``True`` value (so a mask computed on a view, or on the
          data this one was taken from, works here); and a string index holds obs names (which reads the names of all
          the cells, so is slow for large data);
        * integer positions (an array, a list or a single integer; negative positions count from the end), as numpy
          and scanpy take them (use :meth:`select_cell_ids` for cell IDs, which are not the positions in a dataset
          that was made by :meth:`to_deltacells`);
        * obs names (an array or list of strings), or
        * a slice.

        The view holds its cells in the order of the data, whatever the order of the selection, without repeats.

        Raises:
            ValueError: If the selection is empty or has missing values.
            KeyError: If cells are selected by names that are not in the data.
        """
        return CellariumDataView(self, self._selected_positions(selection))

    def select_cell_ids(self, cell_ids: Any) -> "CellariumDataView":
        """
        A :class:`CellariumDataView` of the cells with the given cell IDs (see :attr:`cell_ids`), which must all be
        among the cells of this data.
        """
        positions = np.unique(self._positions_of_ids(cell_ids))
        if len(positions) == 0:
            raise ValueError("The selection is empty: a view must contain at least one cell.")
        return CellariumDataView(self, positions)

    @property
    def cell_ids(self) -> pd.Index:
        """
        The integer IDs of the cells, in cell order, which identify them for all indexing (``obs`` and ``obs_computed``
        are indexed by them, and ``adata.obs["cellarium_cell_id"]`` holds them). The ID of a cell is its position in
        the dataset that was opened, and a view has the IDs of its cells in the data it was taken from. The one
        exception is a deltacells dataset made by :meth:`to_deltacells`, which stores the IDs the cells had in the
        data it was made from: opened later, on its own, it has those same IDs, so that labels computed on it can be
        brought back to the original data.
        """
        return self._obs.index

    @property
    def obs_names(self) -> pd.Index:
        """
        The names of the cells (as in an :class:`~anndata.AnnData`), in cell order. The cells are identified by
        :attr:`cell_ids` everywhere else, so the names are only read when asked for, which for a large dataset takes
        time and memory.
        """
        return self._obs.names

    def _positions_of_ids(self, cell_ids: Any) -> np.ndarray:
        """The positions in these cells of the cells with ``cell_ids``, in that order; ``KeyError``
        for an unknown ID."""
        cell_ids = np.asarray(cell_ids).ravel()
        if cell_ids.size and not np.issubdtype(cell_ids.dtype, np.integer):
            raise TypeError(f"Cell IDs must be integers, got {cell_ids.dtype} values.")
        positions = self.cell_ids.get_indexer(cell_ids.astype(np.int64))
        if (positions < 0).any():
            unknown = cell_ids[positions < 0]
            raise KeyError(f"{len(unknown)} cell IDs are not among the cells of this data, for example {unknown[:5]}.")
        return positions.astype(np.int64)

    def _positions_of_names(self, names: Any) -> np.ndarray:
        """The positions in these cells of the cells with the obs ``names``, in that order."""
        names = pd.Index(names)
        positions = self.obs_names.get_indexer_for(names)
        if (positions < 0).any():
            raise KeyError(f"Cells not found by name: {names[positions < 0][:5].tolist()}")
        return positions.astype(np.int64)

    def _check_names_match(self, names: pd.Index, what: str) -> None:
        """Check, on a sample of cells, that ``names`` (one per cell) are the obs names of these cells."""
        n = self.n_obs
        if len(names) != n:
            raise ValueError(f"{what} has {len(names)} entries but there are {n} cells.")
        sample = np.unique(np.concatenate(([0, n - 1], np.random.default_rng(0).integers(0, n, size=62))))
        expected = self._obs._names_at(sample)
        if not np.array_equal(names[sample].astype(str).to_numpy(), expected.astype(str).to_numpy()):
            raise ValueError(
                f"The index of {what} is not the obs names of the cells of this data, in order. Align it first "
                "(for example with series.reindex(cdata.obs_names)), index it by cdata.cell_ids, or pass "
                "series.to_numpy() to use the values as they are, one per cell."
            )

    def _series_by_cell(self, series: pd.Series, what: str) -> pd.Series:
        """
        ``series`` indexed by :attr:`cell_ids`, as per-cell values are kept. A Series that has one value per cell
        (a range index, the cell IDs, or the obs names, checked on a sample of the cells) is taken as it is, in cell
        order; one indexed by other integers holds cell IDs, and is aligned to the cells, with missing values for
        the cells it does not have.
        """
        ids = self.cell_ids
        index = series.index
        if index.equals(ids):
            result = series.copy(deep=False)
        elif pd.api.types.is_integer_dtype(index.dtype) and not (
            isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1 and len(index) == len(ids)
        ):
            if not index.is_unique:
                raise ValueError(f"The cell IDs in the index of {what} are not unique.")
            result = _nullable(series).reindex(ids)
        elif isinstance(index, pd.RangeIndex):
            result = series.copy(deep=False)
        elif pd.api.types.is_string_dtype(index.dtype) or index.dtype == object:
            self._check_names_match(index, what)
            result = series.copy(deep=False)
        else:
            raise ValueError(f"The index of {what} ({index.dtype}) is neither cell IDs nor obs names.")
        result.index = ids
        return result

    def _selected_positions(self, selection: Any) -> np.ndarray:
        """The sorted, unique positions of the cells ``selection`` selects (see :meth:`__getitem__`)."""
        n = self.n_obs
        if isinstance(selection, slice):
            positions = np.arange(*selection.indices(n), dtype=np.int64)
        elif isinstance(selection, pd.Series):
            positions = self._positions_from_mask_series(selection)
        else:
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
            elif array.dtype.kind in "OUS":
                positions = self._positions_of_names(array.ravel())
            else:
                raise TypeError(
                    f"Select cells with a boolean mask, integer positions, obs names or a slice, not {array.dtype} "
                    "values."
                )
        positions = np.unique(positions)
        if len(positions) == 0:
            raise ValueError("The selection is empty: a view must contain at least one cell.")
        return positions

    def _positions_from_mask_series(self, mask: pd.Series) -> np.ndarray:
        """The positions of the cells that a boolean Series selects, by its index (see :meth:`__getitem__`)."""
        if not pd.api.types.is_bool_dtype(mask.dtype):
            raise TypeError(
                f"A Series selection must be a boolean mask, got dtype {mask.dtype}; use .to_numpy() to select by "
                "position."
            )
        if mask.isna().any():
            raise ValueError("A boolean mask cannot have missing values.")
        values = mask.to_numpy(dtype=bool)
        index = mask.index
        ids = self.cell_ids
        if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1 and len(index) == len(ids):
            return np.flatnonzero(values)  # one value per cell
        if pd.api.types.is_integer_dtype(index.dtype):
            if index.equals(ids):
                return np.flatnonzero(values)
            if not index.is_unique:
                raise ValueError("The cell IDs in the index of the mask are not unique.")
            found = ids.get_indexer(index[values])
            return found[found >= 0]  # the cells that this data does not have are not selected
        if pd.api.types.is_string_dtype(index.dtype) or index.dtype == object:
            return self._positions_of_names(index[values])
        raise TypeError(f"The index of the mask ({index.dtype}) is neither cell IDs nor obs names.")

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

    def predict_batches(
        self,
        module: CellariumModule | None = None,
        device: str | torch.device | None = None,
        batch_size: int | None = None,
        desc: str = "Reading cells",
    ) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches of all the cells (and no others, once each), for passes that compute something from
        every cell. Works for views as for the data they are taken from: it reads the cells of this data.

        Unlike the dataloaders of the datamodule, this does not shuffle, drop cells or depend on a trainer: the batches
        cover exactly the cells of this data, in an order that is not the order of the cells (the data is read
        efficiently, by dataloader workers each reading a contiguous stretch), so every batch carries the positions of
        its cells as ``batch["row_indices_n"]`` (a tensor of positions in this data, i.e. in ``obs``) for putting
        per-cell results in their place. The batch has the fields of ``datamodule.batch_keys``, which include
        ``"x_ng"``, ``"var_names_g"`` and ``"obs_names_n"``, and the iteration raises a ``RuntimeError`` at the end if
        it did not cover all the cells.

        With a ``module``, each batch is what the model of the module would see: the module's ``cpu_transforms`` are
        applied by the dataloader workers (as in training), and the rest of its ``transforms`` (those of
        ``module.transforms``) are applied here, on ``device``, under ``torch.no_grad()``. The model itself is not
        applied. The module is put in ``eval`` mode, and its transforms are moved to ``device`` for the iteration, and
        restored when it ends. Without a ``module`` the batches are as the datamodule reads them, with no transforms
        at all (so for example no gene filter), for the caller to transform.

        Example:
            >>> for batch in cdata.predict_batches(trained.module, device="cuda"):
            ...     x_ng = batch["x_ng"]  # transformed as the model of trained.module sees it, on the GPU
            ...     positions = batch["row_indices_n"]

        Args:
            module: A configured (or configurable) :class:`~cellarium.ml.core.CellariumModule` whose transforms to
                apply.
            device: The device to move the batches to (and the module's transforms), or ``None`` to leave them where
                they are (the cpu).
            batch_size: The number of cells per batch (up to), by default that of the datamodule.
            desc: The label of the progress bar.

        Yields:
            The batches, as dictionaries.
        """
        self._warn_if_read_amplified("This pass over the cells")
        datamodule = self._datamodule
        datamodule._check_num_workers()
        batch_size = datamodule.batch_size if batch_size is None else batch_size
        n_obs = self.n_obs

        collate_fn: Callable = datamodule.collate_fn
        if module is not None:
            module.configure_model()  # idempotent; makes the transforms accessible
            if len(module.cpu_transforms) > 0:
                collate_fn = FunctionComposer(first_applied=collate_fn, second_applied=module.cpu_transforms)
        # the settings that matter for completeness are fixed here, not taken from the datamodule (which is set up for
        # training, e.g. shuffled). `cache_efficient` reads each worker's contiguous stretch of cells, which is much
        # faster than `same_order` but yields batches out of order: hence the positions
        dataset = _PositionedDataset(
            dadc=datamodule.dadc,
            batch_keys=dict(datamodule.batch_keys),
            batch_size=batch_size,
            iteration_strategy="cache_efficient",
            shuffle=False,
            drop_last_indices=False,
            drop_incomplete_batch=False,
            worker_seed=datamodule.worker_seed,
            prefetch_lookahead=datamodule.prefetch_lookahead,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=datamodule.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=datamodule.prefetch_factor,
            persistent_workers=False,
            pin_memory=datamodule.pin_memory,
        )

        target = None if device is None else torch.device(device)
        was_training = module.training if module is not None else False
        original_device = _device_of(module.transforms) if module is not None else None
        if module is not None:
            module.eval()
            if target is not None:
                module.transforms.to(target)
        try:
            seen = 0
            for batch in tqdm(loader, total=math.ceil(n_obs / batch_size), desc=desc, unit="batch"):
                positions = batch.pop("row_indices_n")  # stays on the cpu
                if target is not None:
                    batch = move_data_to_device(batch, target)
                if module is not None:
                    with torch.no_grad():
                        batch = module.transforms(batch)
                batch["row_indices_n"] = positions
                seen += len(positions)
                yield batch
            if seen != n_obs:
                raise RuntimeError(f"The batches covered {seen} cells, but there are {n_obs}.")
        finally:
            if module is not None:
                if target is not None and original_device is not None:
                    module.transforms.to(original_device)
                module.train(was_training)

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
        self.max_gb = parent.max_gb
        self._init_state()
        # entries on disk stay there, as a lazy view of the parent's file, if the view's rows do not fit in memory
        on_disk = any(isinstance(value, (np.memmap, RowSubset)) for value in parent.obsm.values())
        max_bytes = parent._max_bytes() if on_disk else None
        for source, target in ((parent.obsm, self._obsm), (parent.obs_computed, self._obs_computed)):
            for key, value in source.items():
                try:
                    target[key] = _subset_rows(value, positions, max_bytes)
                except (TypeError, ValueError, IndexError) as error:
                    warnings.warn(f"'{key}' of the parent could not be restricted to the view and is left out: {error}")
        # an embedding of the parent stays the embedding of its module, in the rows of the view
        for key, (trained, array) in parent._embedding_sources.items():
            if key in self._obsm and parent.obsm.get(key) is array:
                self._embedding_sources[key] = (trained, self._obsm[key])
