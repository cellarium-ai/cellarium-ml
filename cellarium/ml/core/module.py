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
from cellarium.ml.utilities.core import copy_module


class CellariumModule(pl.LightningModule):
    """
    ``CellariumModule`` organizes code into following sections:

        * :attr:`transforms`: A list of transforms to apply to the input data before passing it to the model.
        * :attr:`model`: A :class:`cellarium.ml.models.CellariumModel` to train with
          :meth:`training_step` method and epoch end hooks.
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
        is_initialized:
            Whether the model has been initialized. This is set to ``False`` by default under the assumption that
            ``torch.device("meta")`` context was used and is set to ``True`` after
            the first call to :meth:`configure_model`.
    """

    def __init__(
        self,
        transforms: Iterable[torch.nn.Module] | None = None,
        model: CellariumModel | None = None,
        optim_fn: type[torch.optim.Optimizer] | None = None,
        optim_kwargs: dict[str, Any] | None = None,
        scheduler_fn: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        is_initialized: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.pipeline: CellariumPipeline | None = None

    def configure_model(self) -> None:
        # This hook is called during each of fit/val/test/predict stages in the same process, so ensure that
        # implementation of this hook is idempotent, i.e., after the first time the hook is called, subsequent
        # calls to it should be a no-op.
        if self.pipeline is not None:
            return

        # Steps involved in configuring the model:
        # 1. Make a copy of modules on the meta device and assign to hparams.
        # 2. Send the original modules to the host device and add to self.pipeline.
        # 3. Reset the model parameters if it has not been initialized before.
        # For more context, see discussions in
        # https://dev-discuss.pytorch.org/t/state-of-model-creation-initialization-seralization-in-pytorch-core/1240
        #
        # Benefits of this approach:
        # 1. The checkpoint stores modules on the meta device.
        # 2. Loading from a checkpoint skips a wasteful step of initializing module parameters
        #    before loading the state_dict.
        # 3. The module parameters are directly initialized on the host gpu device instead of being initialized
        #    on the cpu and then moved to the gpu device (given that modules were instantiated under
        #    the ``torch.device("meta")`` context).
        model, self.hparams["model"] = copy_module(
            self.hparams["model"], self_device=self.device, copy_device=torch.device("meta")
        )
        if self.hparams["transforms"]:
            transforms, self.hparams["transforms"] = zip(
                *(
                    copy_module(transform, self_device=self.device, copy_device=torch.device("meta"))
                    for transform in self.hparams["transforms"]
                )
            )
        else:
            transforms = None

        self.pipeline = CellariumPipeline(transforms)
        if model is None:
            raise ValueError(f"`model` must be an instance of {CellariumModel}. Got {model}")
        self.pipeline.append(model)

        if not self.hparams["is_initialized"]:
            model.reset_parameters()
            self.hparams["is_initialized"] = True

    @property
    def model(self) -> CellariumModel:
        """The model"""
        if self.pipeline is None:
            raise RuntimeError("The model is not configured. Call `configure_model` before accessing the model.")

        return self.pipeline[-1]

    @property
    def transforms(self) -> CellariumPipeline:
        """The transforms pipeline"""
        if self.pipeline is None:
            raise RuntimeError("The model is not configured. Call `configure_model` before accessing the model.")

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
        if self.pipeline is None:
            raise RuntimeError("The model is not configured. Call `configure_model` before accessing the model.")

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
        if self.pipeline is None:
            raise RuntimeError("The model is not configured. Call `configure_model` before accessing the model.")

        return self.pipeline.predict(batch)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """
        Forward pass for validation step.

        Args:
            batch:
                A dictionary containing the batch data.
            batch_idx:
                The index of the batch.

        Returns:
            None
        """
        if self.pipeline is None:
            raise RuntimeError("The model is not configured. Call `configure_model` before accessing the model.")

        self.pipeline.validate(batch)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig | None:
        """Configure optimizers for the model."""
        optim_fn = self.hparams["optim_fn"]
        optim_kwargs = self.hparams["optim_kwargs"] or {}
        scheduler_fn = self.hparams["scheduler_fn"]
        scheduler_kwargs = self.hparams["scheduler_kwargs"] or {}

        if optim_fn is None:
            if optim_kwargs:
                warnings.warn("Optimizer kwargs are provided but no optimizer is defined.", UserWarning)
            if scheduler_fn is not None:
                warnings.warn("Scheduler is defined but no optimizer is defined.", UserWarning)
            return None

        optim_config: OptimizerLRSchedulerConfig = {"optimizer": optim_fn(self.model.parameters(), **optim_kwargs)}
        if scheduler_fn is not None:
            scheduler = scheduler_fn(optim_config["optimizer"], **scheduler_kwargs)
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
