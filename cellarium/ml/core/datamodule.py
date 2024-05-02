# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import lightning.pytorch as pl
import torch
from anndata import AnnData

from cellarium.ml.data import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
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
        dadc: DistributedAnnDataCollection | AnnData,
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
        self.save_hyperparameters(logger=False)
        # Don't save dadc to the checkpoint
        self.hparams["dadc"] = None

        self.dadc = dadc
        # IterableDistributedAnnDataCollectionDataset args
        self.batch_keys = batch_keys or {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.test_mode = test_mode
        # DataLoader args
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """
        .. note::
           setup is called from every process across all the nodes. Setting state here is recommended.

        """
        dadc_train = DistributedAnnDataCollection(
            self.dadc.filenames[:-1],
            limits=self.dadc.limits[:-1],
            max_cache_size=self.dadc.max_cache_size,
            obs_columns_to_validate=self.dadc.obs_columns_to_validate,
        )
        dadc_val = DistributedAnnDataCollection(
            self.dadc.filenames[-1:],
            limits=[10_000],
            max_cache_size=self.dadc.max_cache_size,
            obs_columns_to_validate=self.dadc.obs_columns_to_validate,
        )
        self.dataset_train = IterableDistributedAnnDataCollectionDataset(
            dadc=dadc_train,
            batch_keys=self.batch_keys,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
            test_mode=self.test_mode,
        )
        self.dataset_val = IterableDistributedAnnDataCollectionDataset(
            dadc=dadc_val,
            batch_keys=self.batch_keys,
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed,
            drop_last=self.drop_last,
            test_mode=self.test_mode,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Training dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Validation dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Prediction dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
