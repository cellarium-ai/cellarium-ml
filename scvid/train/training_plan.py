# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from importlib import import_module
from typing import Any

import lightning.pytorch as pl
import torch


class TrainingPlan(pl.LightningModule):
    """
    Lightning module task to train scvi-distributed modules.

    Args:
        module: torch.nn.Module to train.
        optim_fn: A Pytorch optimizer class, e.g., :class:`~torch.optim.Adam`. If `None`,
            defaults to :class:`torch.optim.Adam`.
        optim_kwargs: Keyword arguments for optimiser. If `None`, defaults to `dict(lr=1e-3)`.
        scheduler_fn: A Pytorch lr scheduler class, e.g., :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
        scheduler_kwargs: Keyword arguments for lr scheduler.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        optim_fn: type[torch.optim.Optimizer] | str | None = None,
        optim_kwargs: dict | None = None,
        scheduler_fn: type[torch.optim.lr_scheduler.LRScheduler] | str | None = None,
        scheduler_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.module = module

        # import optimizer and scheduler if they are passed as strings
        if isinstance(optim_fn, str):
            class_module, class_name = optim_fn.rsplit(".", 1)
            module = import_module(class_module)
            optim_fn = getattr(module, class_name)
        if isinstance(scheduler_fn, str):
            class_module, class_name = scheduler_fn.rsplit(".", 1)
            module = import_module(class_module)
            scheduler_fn = getattr(module, class_name)

        # set up optimizer and scheduler
        self.optim_fn = torch.optim.Adam if optim_fn is None else optim_fn
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        if "lr" not in optim_kwargs:
            optim_kwargs["lr"] = 1e-3
        self.optim_kwargs = optim_kwargs
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for Pyro training."""
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.module(*args, **kwargs)
        if loss is not None:
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers for the model."""
        optim_config = {}
        optim_config["optimizer"] = self.optim_fn(
            self.module.parameters(), **self.optim_kwargs
        )
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(
                optim_config["optimizer"], **self.scheduler_kwargs
            )
            optim_config["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        return optim_config

    def on_train_epoch_start(self):
        """
        Calls the ``set_epoch`` method on the iterable dataset of the given dataloader.

        If the dataset is ``IterableDataset`` and has ``set_epoch`` method defined, then
        ``set_epoch`` must be called at the beginning of every epoch to ensure shuffling
        applies a new ordering. This has no effect if shuffling is off.
        """
        dataloaders = self.trainer.fit_loop._combined_loader.flattened
        for dataloader in dataloaders:
            dataset = dataloader.dataset
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.current_epoch)


class DummyTrainingPlan(pl.LightningModule):
    """
    Lightning module task that does not perform any actual optimization (no gradient updates).
    It can be used for cases where only the forward pass is required (e.g., for calculating
    sufficient statistics, EM algorithms, etc.).
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))
