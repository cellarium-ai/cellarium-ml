# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable, Sequence

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml.data import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
from cellarium.ml.utilities.data import AnnDataField, collate_fn


class CellariumAnnDataDataModule(pl.LightningDataModule):
    """
    DataModule for :class:`~cellarium.ml.data.IterableDistributedAnnDataCollectionDataset`.

    Example::

        >>> from cellarium.ml import CellariumAnnDataDataModule
        >>> from cellarium.ml.utilities.data import AnnDataField, densify

        >>> dm = CellariumAnnDataDataModule(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     shard_size=10_000,
        ...     max_cache_size=2,
        ...     batch_keys={
        ...         "x_ng": AnnDataField(attr="X", convert_fn=densify),
        ...         "feature_g": AnnDataField(attr="var_names"),
        ...     },
        ...     batch_size=5000,
        ...     shuffle=True,
        ...     seed=0,
        ...     drop_last=True,
        ...     num_workers=4,
        ... )
        >>> dm.setup()
        >>> for batch in dm.train_dataloader():
        ...     print(batch.keys())  # x_ng, feature_g

    Example::

        >>> from cellarium.ml import CellariumAnnDataDataModule
        >>> from cellarium.ml.utilities.data import AnnDataField, densify

        >>> dm = CellariumAnnDataDataModule(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     shard_size=10_000,
        ...     max_cache_size=2,
        ...     batch_keys={
        ...         "x_ng": AnnDataField(attr="X", convert_fn=densify),
        ...         "feature_g": AnnDataField(attr="var_names"),
        ...     },
        ...     batch_size=5000,
        ...     shuffle=True,
        ...     seed=0,
        ...     drop_last=True,
        ...     num_workers=4,
        ... )
        >>> dm.setup()
        >>> for batch in dm.train_dataloader():
        ...     print(batch.keys())  # x_ng, feature_g

    Args:
        filenames:
            Names of anndata files.
        limits:
            List of global cell indices (limits) for the last cells in each shard.
            If ``None``, the limits are inferred from ``shard_size`` and ``last_shard_size``.
        shard_size:
            The number of cells in each anndata file (shard).
            Must be specified if the ``limits`` is not provided.
        last_shard_size:
            Last shard size. If not ``None``, the last shard will have this size possibly
            different from ``shard_size``.
        max_cache_size:
            Max size of the cache.
        cache_size_strictly_enforced:
            Assert that the number of retrieved anndatas is not more than maxsize.
        label:
            Column in :attr:`obs` to place batch information in. If it's ``None``, no column is added.
        keys:
            Names for each object being added. These values are used for column values for
            ``label`` or appended to the index if ``index_unique`` is not ``None``.
            If ``None``, ``keys`` are set to ``filenames``.
        index_unique:
            Whether to make the index unique by using the keys. If provided, this
            is the delimeter between ``{orig_idx}{index_unique}{key}``. When ``None``,
            the original indices are kept.
        indices_strict:
            If  ``True``, arrays from the subset objects will always have the same order
            of indices as in selection used to subset.
            This parameter can be set to ``False`` if the order in the returned arrays
            is not important, for example, when using them for stochastic gradient descent.
            In this case the performance of subsetting can be a bit better.
        batch_keys:
            Dictionary that specifies which attributes and keys of the :attr:`dadc` to return
            in the batch data and how to convert them. Keys must correspond to
            the input keys of the transforms or the model. Values must be instances of
            :class:`cellarium.ml.utilities.data.AnnDataField`.
        batch_size:
            How many samples per batch to load.
        shuffle:
            If ``True``, the data is reshuffled at every epoch.
        seed:
            Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        drop_last:
            If ``True``, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas. If ``False``,
            the sampler will add extra indices to make the data evenly divisible across
            the replicas.
        test_mode:
            If ``True`` enables tracking of cache and worker informations.
        num_workers:
            How many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
    """

    def __init__(
        self,
        # DistributedAnnDataCollection args
        filenames: Sequence[str] | str,
        limits: Iterable[int] | None = None,
        shard_size: int | None = None,
        last_shard_size: int | None = None,
        max_cache_size: int = 1,
        cache_size_strictly_enforced: bool = True,
        label: str | None = None,
        keys: Sequence[str] | None = None,
        index_unique: str | None = None,
        indices_strict: bool = True,
        # IterableDistributedAnnDataCollectionDataset args
        batch_keys: dict[str, AnnDataField] | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        test_mode: bool = False,
        # DataLoader args
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        # DistributedAnnDataCollection args
        self.filenames = filenames
        self.limits = limits
        self.shard_size = shard_size
        self.last_shard_size = last_shard_size
        self.max_cache_size = max_cache_size
        self.cache_size_strictly_enforced = cache_size_strictly_enforced
        self.label = label
        self.keys = keys
        self.index_unique = index_unique
        self.indices_strict = indices_strict
        # IterableDistributedAnnDataCollectionDataset args
        self.batch_keys = batch_keys or {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.test_mode = test_mode
        # DataLoader args
        self.num_workers = num_workers
        # DistributedAnnDataCollection
        obs_columns = [field.obs_column for field in self.batch_keys.values() if field.obs_column is not None]
        self.dadc = DistributedAnnDataCollection(
            filenames=self.filenames,
            limits=self.limits,
            shard_size=self.shard_size,
            last_shard_size=self.last_shard_size,
            max_cache_size=self.max_cache_size,
            cache_size_strictly_enforced=self.cache_size_strictly_enforced,
            label=self.label,
            keys=self.keys,
            index_unique=self.index_unique,
            indices_strict=self.indices_strict,
            obs_columns=obs_columns,
        )

    @property
    def n_obs(self) -> int:
        return self.dadc.n_obs

    @property
    def n_vars(self) -> int:
        return self.dadc.n_vars

    @property
    def var_names(self) -> np.ndarray:
        return self.dadc.var_names.values

    def setup(self, stage: str | None = None) -> None:
        """
        .. note::
           setup is called from every process across all the nodes. Setting state here is recommended.

        """
        self.dataset = IterableDistributedAnnDataCollectionDataset(
            dadc=self.dadc,
            batch_keys=self.batch_keys,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
            test_mode=self.test_mode,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Training dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Prediction dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
