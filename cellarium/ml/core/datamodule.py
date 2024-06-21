# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from typing import Literal

import lightning.pytorch as pl
import torch
from anndata import AnnData

from cellarium.ml.data import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
from cellarium.ml.utilities.core import train_val_split
from cellarium.ml.utilities.data import AnnDataField, collate_fn


class CellariumAnnDataDataModule(pl.LightningDataModule):
    """
    DataModule for :class:`~cellarium.ml.data.IterableDistributedAnnDataCollectionDataset`.

    Example::

        >>> from cellarium.ml import CellariumAnnDataDataModule
        >>> from cellarium.ml.data import DistributedAnnDataCollection
        >>> from cellarium.ml.utilities.data import AnnDataField, densify

        >>> dm = CellariumAnnDataDataModule(
        ...     DistributedAnnDataCollection(
        ...         "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...         shard_size=10_000,
        ...     ),
        ...     max_cache_size=2,
        ...     batch_keys={
        ...         "x_ng": AnnDataField(attr="X", convert_fn=densify),
        ...         "var_names_g": AnnDataField(attr="var_names"),
        ...     },
        ...     batch_size=5000,
        ...     iteration_strategy="cache_efficient",
        ...     shuffle=True,
        ...     seed=0,
        ...     drop_last=True,
        ...     num_workers=4,
        ... )
        >>> dm.setup()
        >>> for batch in dm.train_dataloader():
        ...     print(batch.keys())  # x_ng, var_names_g

    Args:
        dadc:
            An instance of :class:`~cellarium.ml.data.DistributedAnnDataCollection` or :class:`AnnData`.
        batch_keys:
            Dictionary that specifies which attributes and keys of the :attr:`dadc` to return
            in the batch data and how to convert them. Keys must correspond to
            the input keys of the transforms or the model. Values must be instances of
            :class:`cellarium.ml.utilities.data.AnnDataField`.
        batch_size:
            How many samples per batch to load.
        iteration_strategy:
            Strategy to use for iterating through the dataset. Options are ``same_order`` and ``cache_efficient``.
            ``same_order`` will iterate through the dataset in the same order independent of the number of replicas
            and workers. ``cache_efficient`` will try to minimize the amount of anndata files fetched by each worker.
        shuffle:
            If ``True``, the data is reshuffled at every epoch.
        seed:
            Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        drop_last:
            If ``True``, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas. If ``False``,
            the sampler will add extra indices to make the data evenly divisible across
            the replicas.
        train_size:
            Size of the train split. If :class:`float`, should be between ``0.0`` and ``1.0`` and represent
            the proportion of the dataset to include in the train split. If :class:`int`, represents
            the absolute number of train samples. If ``None``, the value is automatically set to the complement
            of the ``val_size``.
        val_size:
            Size of the validation split. If :class:`float`, should be between ``0.0`` and ``1.0`` and represent
            the proportion of the dataset to include in the validation split. If :class:`int`, represents
            the absolute number of validation samples. If ``None``, the value is set to the complement of
            the ``train_size``. If ``train_size`` is also ``None``, it will be set to ``0``.
        test_mode:
            If ``True`` enables tracking of cache and worker informations.
        num_workers:
            How many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
    """

    def __init__(
        self,
        dadc: DistributedAnnDataCollection | AnnData,
        # IterableDistributedAnnDataCollectionDataset args
        batch_keys: dict[str, AnnDataField] | None = None,
        batch_size: int = 1,
        iteration_strategy: Literal["same_order", "cache_efficient"] = "cache_efficient",
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        train_size: float | int | None = None,
        val_size: float | int | None = None,
        test_mode: bool = False,
        # DataLoader args
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        # Don't save dadc to the checkpoint
        self.hparams["dadc"] = None

        self.dadc = dadc
        # IterableDistributedAnnDataCollectionDataset args
        self.batch_keys = batch_keys or {}
        self.batch_size = batch_size
        self.iteration_strategy = iteration_strategy
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.n_train, self.n_val = train_val_split(len(dadc), train_size, val_size)
        self.test_mode = test_mode
        # DataLoader args
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """
        .. note::
           setup is called from every process across all the nodes. Setting state here is recommended.

        .. note::
            :attr:`val_dataset` is not shuffled and uses the ``same_order`` iteration strategy.

        """
        if stage == "fit":
            self.train_dataset = IterableDistributedAnnDataCollectionDataset(
                dadc=self.dadc,
                batch_keys=self.batch_keys,
                batch_size=self.batch_size,
                iteration_strategy=self.iteration_strategy,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
                test_mode=self.test_mode,
                start_idx=0,
                end_idx=self.n_train,
            )
            self.val_dataset = IterableDistributedAnnDataCollectionDataset(
                dadc=self.dadc,
                batch_keys=self.batch_keys,
                batch_size=self.batch_size,
                iteration_strategy="same_order",
                shuffle=False,
                seed=self.seed,
                drop_last=False,
                test_mode=self.test_mode,
                start_idx=self.n_train,
                end_idx=self.n_train + self.n_val,
            )

        if stage == "predict":
            self.predict_dataset = IterableDistributedAnnDataCollectionDataset(
                dadc=self.dadc,
                batch_keys=self.batch_keys,
                batch_size=self.batch_size,
                iteration_strategy=self.iteration_strategy,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
                test_mode=self.test_mode,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Prediction dataloader."""
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
