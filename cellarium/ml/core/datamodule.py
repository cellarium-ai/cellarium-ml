# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import warnings
from typing import Any, Literal

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
        ...     shuffle_seed=0,
        ...     drop_last_indices=True,
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
        shuffle_seed:
            Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        drop_last_indices:
            If ``True``, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas. If ``False``,
            the sampler will add extra indices to make the data evenly divisible across
            the replicas.
        drop_incomplete_batch:
            If ``True``, the dataloader will drop the incomplete batch if the dataset size is not divisible by
            the batch size.
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
        worker_seed:
            Random seed used to seed the workers. If ``None``, then the workers will not be seeded.
            The seed of the individual worker is computed based on the ``worker_seed``, global worker id,
            and the epoch. Note that the this seed affects ``cpu_transforms`` when they are used.
            When resuming training, the seed should be set to a different value to ensure that the
            workers are not seeded with the same seed as the previous run.
        test_mode:
            If ``True`` enables tracking of cache and worker informations.
        num_workers:
            How many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
        prefetch_factor:
            Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches
            prefetched across all workers. (default value depends on the set value for num_workers. If value of
            ``num_workers=0`` default is ``None``. Otherwise, if value of ``num_workers > 0`` default is ``2``)
        persistent_workers:
            If ``True``, the data loader will not shut down the worker processes after a dataset has been consumed once.
            This allows to maintain the workers ``Dataset`` instances alive.
    """

    def __init__(
        self,
        dadc: DistributedAnnDataCollection | AnnData,
        # IterableDistributedAnnDataCollectionDataset args
        batch_keys: dict[str, dict[str, AnnDataField] | AnnDataField] | None = None,
        batch_size: int = 1,
        iteration_strategy: Literal["same_order", "cache_efficient"] = "cache_efficient",
        shuffle: bool = False,
        shuffle_seed: int = 0,
        drop_last_indices: bool = False,
        drop_incomplete_batch: bool = False,
        train_size: float | int | None = None,
        val_size: float | int | None = None,
        worker_seed: int | None = None,
        test_mode: bool = False,
        # DataLoader args
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
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
        self.shuffle_seed = shuffle_seed
        self.drop_last_indices = drop_last_indices
        self.n_train, self.n_val = train_val_split(len(dadc), train_size, val_size)
        self.worker_seed = worker_seed
        self.test_mode = test_mode
        # DataLoader args
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_incomplete_batch = drop_incomplete_batch
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

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
                shuffle_seed=self.shuffle_seed,
                drop_last_indices=self.drop_last_indices,
                drop_incomplete_batch=self.drop_incomplete_batch,
                worker_seed=self.worker_seed,
                test_mode=self.test_mode,
                start_idx=0,
                end_idx=self.n_train,
            )

        if stage in {"fit", "validate"}:
            self.val_dataset = IterableDistributedAnnDataCollectionDataset(
                dadc=self.dadc,
                batch_keys=self.batch_keys,
                batch_size=self.batch_size,
                iteration_strategy="same_order",
                shuffle=False,
                shuffle_seed=self.shuffle_seed,
                drop_last_indices=self.drop_last_indices,
                drop_incomplete_batch=self.drop_incomplete_batch,
                worker_seed=self.worker_seed,
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
                shuffle_seed=self.shuffle_seed,
                drop_last_indices=self.drop_last_indices,
                drop_incomplete_batch=self.drop_incomplete_batch,
                worker_seed=self.worker_seed,
                test_mode=self.test_mode,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Prediction dataloader."""
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def state_dict(self) -> dict[str, Any]:
        assert self.trainer is not None
        state = {
            "iteration_strategy": self.iteration_strategy,
            "num_workers": self.num_workers,
            "num_replicas": self.trainer.num_devices,
            "num_nodes": self.trainer.num_nodes,
            "batch_size": self.batch_size,
            "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "drop_last_indices": self.drop_last_indices,
            "drop_incomplete_batch": self.drop_incomplete_batch,
            "n_train": self.n_train,
            "worker_seed": self.worker_seed,
            "epoch": self.trainer.current_epoch,
            "resume_step": self.trainer.global_step,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if hasattr(self, "train_dataset"):
            assert self.trainer is not None
            if state_dict["iteration_strategy"] != self.iteration_strategy:
                raise ValueError(
                    "Cannot resume training with a different iteration strategy. "
                    f"Expected {self.iteration_strategy}, got {state_dict['iteration_strategy']}."
                )
            if state_dict["num_workers"] != self.num_workers:
                raise ValueError(
                    "Cannot resume training with a different number of workers. "
                    f"Expected {self.num_workers}, got {state_dict['num_workers']}."
                )
            if state_dict["num_replicas"] != self.trainer.num_devices:
                raise ValueError(
                    "Cannot resume training with a different number of replicas. "
                    f"Expected {self.trainer.num_devices}, got {state_dict['num_replicas']}."
                )
            if state_dict["num_nodes"] != self.trainer.num_nodes:
                raise ValueError(
                    "Cannot resume training with a different number of nodes. "
                    f"Expected {self.trainer.num_nodes}, got {state_dict['num_nodes']}."
                )
            if state_dict["batch_size"] != self.batch_size:
                raise ValueError(
                    "Cannot resume training with a different batch size. "
                    f"Expected {self.batch_size}, got {state_dict['batch_size']}."
                )
            if state_dict["accumulate_grad_batches"] != 1:
                raise ValueError("Training with gradient accumulation is not supported when resuming training.")
            if state_dict["shuffle"] != self.shuffle:
                raise ValueError(
                    "Cannot resume training with a different shuffle value. "
                    f"Expected {self.shuffle}, got {state_dict['shuffle']}."
                )
            if state_dict["shuffle_seed"] != self.shuffle_seed:
                raise ValueError(
                    "Cannot resume training with a different shuffle seed. "
                    f"Expected {self.shuffle_seed}, got {state_dict['shuffle_seed']}."
                )
            if state_dict["drop_last_indices"] != self.drop_last_indices:
                raise ValueError(
                    "Cannot resume training with a different drop_last_indices value. "
                    f"Expected {self.drop_last_indices}, got {state_dict['drop_last_indices']}."
                )
            if state_dict["drop_incomplete_batch"] != self.drop_incomplete_batch:
                raise ValueError(
                    "Cannot resume training with a different drop_incomplete_batch value. "
                    f"Expected {self.drop_incomplete_batch}, got {state_dict['drop_incomplete_batch']}."
                )
            if state_dict["n_train"] != self.n_train:
                raise ValueError(
                    "Cannot resume training with a different train size. "
                    f"Expected {self.n_train}, got {state_dict['n_train']}."
                )
            if (self.worker_seed is not None) and (state_dict["worker_seed"] == self.worker_seed):
                warnings.warn(
                    "Resuming training with the same worker seed as the previous run. "
                    "This may lead to repeated behavior in the workers upon resuming training."
                )

            self.train_dataset.load_state_dict(state_dict)
