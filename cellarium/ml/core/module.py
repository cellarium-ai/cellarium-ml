# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Iterable
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig

from cellarium.ml.core.pipeline import CellariumPipeline
from cellarium.ml.models import CellariumModel
from cellarium.ml.utilities.core import initialize_object, uninitialize_object


class CellariumModule(pl.LightningModule):
    """
    ``CellariumModule`` organizes code into following sections:

        * :attr:`model`: A :class:`cellarium.ml.models.CellariumModel` to train with
          :meth:`training_step` method and epoch end hooks.
        * :attr:`transforms`: A :class:`cellarium.ml.core.pipeline.CellariumPipeline` of transforms to apply to the
          input data before passing it to the model.
        * :attr:`optim_fn` and :attr:`optim_kwargs`: A Pytorch optimizer class and its keyword arguments.
        * :attr:`scheduler_fn` and :attr:`scheduler_kwargs`: A Pytorch lr scheduler class and its
          keyword arguments.

    Args:
        transforms:
            A list of transforms to apply to the input data before passing it to the model.
            If ``None``, no transforms are applied.
        model:
            A :class:`cellarium.ml.models.CellariumModel` to train.
        optim_fn:
            A Pytorch optimizer class, e.g., :class:`~torch.optim.Adam`. If ``None``,
            no optimizer is used.
        optim_kwargs:
            Keyword arguments for optimiser.
        scheduler_fn:
            A Pytorch lr scheduler class, e.g., :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
        scheduler_kwargs:
            Keyword arguments for lr scheduler.
    """

    def __init__(
        self,
        transforms: Iterable[torch.nn.Module] | None = None,
        model: CellariumModel | None = None,
        optim_fn: type[torch.optim.Optimizer] | None = None,
        optim_kwargs: dict[str, Any] | None = None,
        scheduler_fn: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # We avoid saving the `nn.Module` objects to the hparams because that will save a copy
        # of model weights which already are being saved in the model checkpoint's `state_dict`.
        # Instead, we save the class name and the init args which then can be used to
        # re-initialize the model and transforms.
        # In order to achieve this, we temporarily re-assign `model` and `transforms` to their un-initialized states
        # and then call `save_hyperparameters` which will save these values as hparams.
        # Then, we re-assign `model` and `transforms` back to their initialized states.
        # `initialize_object` handles the case when the object was passed as a dictionary of class path and init args.
        _transforms, _model = transforms, model
        transforms = [uninitialize_object(transform) for transform in _transforms] if _transforms is not None else None
        model = uninitialize_object(_model)
        self.save_hyperparameters(logger=False)
        transforms = [initialize_object(transform) for transform in _transforms] if _transforms is not None else None
        model = initialize_object(_model)

        self.pipeline = CellariumPipeline(transforms)
        if model is None:
            raise ValueError(f"`model` must be an instance of {CellariumModel}. Got {model}")
        self.pipeline.append(model)

        # set up optimizer and scheduler
        self.optim_fn = optim_fn
        self.optim_kwargs = optim_kwargs or {}
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs or {}

    @property
    def model(self) -> CellariumModel:
        """The model"""
        return self.pipeline[-1]

    @property
    def transforms(self) -> CellariumPipeline:
        """The transforms pipeline"""
        return self.pipeline[:-1]

    def training_step(  # type: ignore[override]
        self, batch: dict[str, np.ndarray | torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:
        """
        Forward pass for training step.

        Args:
            batch:
                A dictionary containing the batch data.
            batch_idx:
                The index of the batch.

        Returns:
            Loss tensor or ``None`` if no loss.
        """
        output = self.pipeline(batch)
        loss = output.get("loss")
        if loss is not None:
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
        return loss

    def forward(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Forward pass for inference step.

        Args:
            batch: A dictionary containing the batch data.

        Returns:
            A dictionary containing the batch data and inference outputs.
        """
        return self.pipeline.predict(batch)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig | None:
        """Configure optimizers for the model."""
        if self.optim_fn is None:
            if self.optim_kwargs:
                warnings.warn("Optimizer kwargs are provided but no optimizer is defined.", UserWarning)
            if self.scheduler_fn is not None:
                warnings.warn("Scheduler is defined but no optimizer is defined.", UserWarning)
            return None

        optim_config: OptimizerLRSchedulerConfig = {
            "optimizer": self.optim_fn(self.model.parameters(), **self.optim_kwargs)
        }
        if self.scheduler_fn is not None:
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
        Calls the ``on_train_start`` method on the :attr:`model` attribute.
        If the :attr:`model` attribute has ``on_train_start`` method defined, then
        ``on_train_start`` must be called at the beginning of training.
        """
        on_train_start = getattr(self.model, "on_train_start", None)
        if callable(on_train_start):
            on_train_start(self.trainer)

    def on_train_epoch_end(self) -> None:
        """
        Calls the ``on_epoch_end`` method on the :attr:`model` attribute.
        If the :attr:`model` attribute has ``on_epoch_end`` method defined, then
        ``on_epoch_end`` must be called at the end of every epoch.
        """
        on_epoch_end = getattr(self.model, "on_epoch_end", None)
        if callable(on_epoch_end):
            on_epoch_end(self.trainer)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """
        Calls the ``on_batch_end`` method on the module.
        """
        on_batch_end = getattr(self.model, "on_batch_end", None)
        if callable(on_batch_end):
            on_batch_end(self.trainer)
