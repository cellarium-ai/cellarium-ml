# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import pyro
import pytorch_lightning as pl
import torch


class PyroTrainingPlan(pl.LightningModule):
    """
    Lightning module task to train Pyro scvi-tools modules.

    .. note:: This is a stripped down version of the :class:`~scvi.train.LowLevelPyroTrainingPlan`.
        https://github.com/scverse/scvi-tools/blob/bf2121975bdfc31bfb1f6feb6446c331188b47dd/scvi/train/_trainingplans.py#L745

    Args:
        pyro_module: A Pyro module. This object should have callable `model` and `guide` attributes or methods.
        loss_fn: A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
            If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
        optim: A Pytorch optimizer class, e.g., :class:`~torch.optim.Adam`. If `None`,
            defaults to :class:`torch.optim.Adam`.
        optim_kwargs: Keyword arguments for optimiser. If `None`, defaults to `dict(lr=1e-3)`.
        scheduler: A Pytorch lr scheduler class, e.g., :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
        scheduler_kwargs: Keyword arguments for lr scheduler.
    """

    def __init__(
        self,
        pyro_module: pyro.nn.PyroModule,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim_fn: Optional[callable] = None,
        optim_kwargs: Optional[dict] = None,
        scheduler_fn: Optional[callable] = None,
        scheduler_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.module = pyro_module

        self.optim_fn = torch.optim.Adam if optim_fn is None else optim_fn
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        if "lr" not in optim_kwargs:
            optim_kwargs["lr"] = 1e-3
        self.optim_kwargs = optim_kwargs
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs

        self.loss_fn = (
            pyro.infer.Trace_ELBO().differentiable_loss if loss_fn is None else loss_fn
        )

    def training_step(self, batch, batch_idx):
        """Training step for Pyro training."""
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.loss_fn(self.module.model, self.module.guide, *args, **kwargs)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
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

    def training_step(self, batch, batch_idx):
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        self.module(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.SGD([self._dummy_param], lr=1.0)

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
