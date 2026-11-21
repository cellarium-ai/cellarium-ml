# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Context managers for temporarily mutating a :class:`CellariumAnnDataDataModule`, used by
``api/tools`` and ``api/preprocessing`` functions that need to inject batch information or
restrict validation/train-val splits for the duration of a single training run.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from cellarium.ml import CellariumAnnDataDataModule
from cellarium.ml.api._view_collections import unwrap_collection
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.core import train_val_split
from cellarium.ml.utilities.data import AnnDataField, batch_categories_to_codes


def check_batch_key(datamodule: CellariumAnnDataDataModule, batch_key: str, arg_name: str = "batch_key") -> None:
    """
    Check that ``batch_key`` is a categorical ``obs`` column of the data, from the schema of the data (deltacells) or
    the first h5ad file, without reading its values, so that a wrong column fails at once with a clear error rather
    than in a worker of the dataloader. (Missing values are found as the cells are read, see
    :func:`~cellarium.ml.utilities.data.batch_categories_to_codes`.)

    Raises:
        ValueError: If ``batch_key`` is not an ``obs`` column of the data, or is not categorical.
    """
    if not datamodule.obs_is_categorical(batch_key):
        raise ValueError(
            f"{arg_name} {batch_key!r} is not a categorical obs column, but a batch column must be categorical in the "
            f"data: e.g. adata.obs[{batch_key!r}] = adata.obs[{batch_key!r}].astype('category') before writing the "
            "h5ad files (with the same categories in every file), or before making the deltacells dataset."
        )


def batch_index_field(batch_key: str) -> AnnDataField:
    """The field ``batch_index_n`` of the codes of the categorical batch column ``batch_key`` (see
    :func:`check_batch_key`)."""
    return AnnDataField(attr="obs", key=batch_key, convert_fn=batch_categories_to_codes)


@contextmanager
def temporary_batch_keys(
    datamodule: CellariumAnnDataDataModule,
    extra_batch_keys: dict[str, AnnDataField],
) -> Iterator[None]:
    """
    Temporarily merge ``extra_batch_keys`` into ``datamodule.batch_keys`` for the duration of the
    ``with`` block, and restrict ``obs_columns_to_validate`` to the ``obs`` columns referenced by
    those fields. Restores the original ``batch_keys`` entries and ``obs_columns_to_validate`` on
    exit, even if the block raises.

    Args:
        datamodule: The datamodule to mutate.
        extra_batch_keys: Mapping of batch key name to :class:`AnnDataField`, merged into
            ``datamodule.batch_keys``. Names already present in ``datamodule.batch_keys`` are
            restored to their original field afterward rather than removed.
    """
    dadc = unwrap_collection(datamodule.dadc)  # the cells of a view are read, and validated, by its source
    # only h5ad collections validate the obs columns of each file they read
    h5ad_dadc = dadc if isinstance(dadc, DistributedAnnDataCollection) else None

    original_fields = {name: datamodule.batch_keys.get(name) for name in extra_batch_keys}
    if h5ad_dadc is not None:
        original_obs_columns_to_validate = h5ad_dadc.obs_columns_to_validate
        original_schema_obs_columns_to_validate = h5ad_dadc.schema.obs_columns_to_validate

    obs_columns_to_validate: list[str] = []
    for field in extra_batch_keys.values():
        if field.attr != "obs" or field.key is None:
            continue
        keys = field.key if isinstance(field.key, list) else [field.key]
        for key in keys:
            if key not in obs_columns_to_validate:
                obs_columns_to_validate.append(key)

    try:
        datamodule.batch_keys.update(extra_batch_keys)
        if h5ad_dadc is not None:
            # mutate the schema object in place: LazyAnnData instances hold a reference to this exact
            # object, so reassigning `dadc.schema` (a new object) would not affect already-constructed shards
            h5ad_dadc.obs_columns_to_validate = obs_columns_to_validate
            h5ad_dadc.schema.obs_columns_to_validate = obs_columns_to_validate
        else:
            # e.g. deltacells: make the new obs columns local before any worker needs them
            datamodule.prepare_data()
        yield
    finally:
        for name, original_field in original_fields.items():
            if original_field is None:
                datamodule.batch_keys.pop(name, None)
            else:
                datamodule.batch_keys[name] = original_field
        if h5ad_dadc is not None:
            h5ad_dadc.obs_columns_to_validate = original_obs_columns_to_validate
            h5ad_dadc.schema.obs_columns_to_validate = original_schema_obs_columns_to_validate


@contextmanager
def temporary_val_split(
    datamodule: CellariumAnnDataDataModule,
    val_size: float | int | None,
    train_size: float | int | None = None,
) -> Iterator[None]:
    """
    Temporarily set ``datamodule.n_train``/``datamodule.n_val`` via :func:`train_val_split` for the
    duration of the ``with`` block, restoring the originals on exit, even if the block raises.
    """
    original_n_train = datamodule.n_train
    original_n_val = datamodule.n_val
    try:
        datamodule.n_train, datamodule.n_val = train_val_split(
            len(datamodule.dadc), train_size=train_size, val_size=val_size
        )
        yield
    finally:
        datamodule.n_train = original_n_train
        datamodule.n_val = original_n_val
