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
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.core import train_val_split
from cellarium.ml.utilities.data import AnnDataField


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
    dadc = datamodule.dadc
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
