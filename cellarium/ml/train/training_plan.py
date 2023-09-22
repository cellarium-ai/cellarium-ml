# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from importlib import import_module

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from cellarium.ml.module import BaseModule, PredictMixin


class TrainingPlan(pl.LightningModule):
    """
    Lightning module task to train ``cellarium.ml`` modules.

    Args:
        module:
            A ``cellarium.ml`` module to train.
        optim_fn:
            A Pytorch optimizer class, e.g., :class:`~torch.optim.Adam`. If ``None``,
            defaults to :class:`torch.optim.Adam`.
        optim_kwargs:
            Keyword arguments for optimiser. If ``None``, defaults to ``default_lr``.
        scheduler_fn:
            A Pytorch lr scheduler class, e.g., :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
        scheduler_kwargs:
            Keyword arguments for lr scheduler.
        default_lr:
            Default learning rate to use if ``optim_kwargs`` does not contain ``lr``.
    """

    def __init__(
        self,
        module: BaseModule,
        optim_fn: type[torch.optim.Optimizer] | str | None = None,
        optim_kwargs: dict | None = None,
        scheduler_fn: type[torch.optim.lr_scheduler.LRScheduler] | str | None = None,
        scheduler_kwargs: dict | None = None,
        default_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.module = module

        # set up optimizer and scheduler
        if isinstance(optim_fn, str):
            class_module, class_name = optim_fn.rsplit(".", 1)
            optim_module = import_module(class_module)
            self.optim_fn = getattr(optim_module, class_name)
        elif optim_fn is None:
            self.optim_fn = torch.optim.Adam
        else:
            self.optim_fn = optim_fn

        if isinstance(scheduler_fn, str):
            class_module, class_name = scheduler_fn.rsplit(".", 1)
            scheduler_module = import_module(class_module)
            self.scheduler_fn = getattr(scheduler_module, class_name)
        else:
            self.scheduler_fn = scheduler_fn

        optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        if "lr" not in optim_kwargs:
            optim_kwargs["lr"] = default_lr
        self.optim_kwargs = optim_kwargs
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs

    def training_step(  # type: ignore[override]
        self, batch: dict[str, np.ndarray | torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.module(*args, **kwargs)
        if loss is not None:
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
        return loss

    def forward(self, batch: dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass of the model."""
        assert isinstance(self.module, PredictMixin)
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        return self.module.predict(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizers for the model."""
        optim_config: OptimizerLRSchedulerConfig = {
            "optimizer": self.optim_fn(self.module.parameters(), **self.optim_kwargs)
        }
        if self.scheduler_fn is not None:
            assert self.scheduler_kwargs is not None
            scheduler = self.scheduler_fn(optim_config["optimizer"], **self.scheduler_kwargs)
            optim_config["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        return optim_config

    def on_train_epoch_start(self) -> None:
        """
        Calls the ``set_epoch`` method on the iterable dataset of the given dataloader.

        If the dataset is ``IterableDataset`` and has ``set_epoch`` method defined, then
        ``set_epoch`` must be called at the beginning of every epoch to ensure shuffling
        applies a new ordering. This has no effect if shuffling is off.
        """
        # dataloader is wrapped in a combined loader and can be accessed via
        # flattened property which returns a list of dataloaders
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html
        combined_loader = self.trainer.fit_loop._combined_loader
        assert combined_loader is not None
        dataloaders = combined_loader.flattened
        for dataloader in dataloaders:
            dataset = dataloader.dataset
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.current_epoch)

    def on_train_start(self) -> None:
        """
        Calls the ``on_train_start`` method on the module.
        If the module has ``on_train_start`` method defined, then
        ``on_train_start`` must be called at the beginning of training.
        """
        on_train_start = getattr(self.module, "on_train_start", None)
        if callable(on_train_start):
            on_train_start(self.trainer)

    def on_train_epoch_end(self) -> None:
        """
        Calls the ``on_epoch_end`` method on the module.
        If the module has ``on_epoch_end`` method defined, then
        ``on_epoch_end`` must be called at the end of every epoch.
        """
        on_epoch_end = getattr(self.module, "on_epoch_end", None)
        if callable(on_epoch_end):
            on_epoch_end(self.trainer)
