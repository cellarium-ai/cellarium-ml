# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from importlib import import_module
from typing import IO, Any
from unittest.mock import patch

import lightning.pytorch as pl
import numpy as np
import torch
from jsonargparse import Namespace
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig

from cellarium.ml.core.saving import _load_state
from cellarium.ml.models import CellariumModel, PredictMixin


class CellariumModule(pl.LightningModule):
    """
    ``CellariumModule`` organizes code into following sections:

        * :attr:`model`: A :class:`cellarium.ml.models.CellariumModel` to train with
          :meth:`training_step` method and epoch end hooks.
        * :attr:`optim_fn` and :attr:`optim_kwargs`: A Pytorch optimizer class and its keyword arguments.
        * :attr:`scheduler_fn` and :attr:`scheduler_kwargs`: A Pytorch lr scheduler class and its
          keyword arguments.

    Args:
        model:
            A :class:`cellarium.ml.models.CellariumModel` to train.
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
        config:
            A dictionary or :class:`jsonargparse.Namespace` containing the initialization hyperparameters.
            If not ``None``, the configuration will be saved as ``"hyper_parameters"`` in the checkpoint.
    """

    def __init__(
        self,
        model: CellariumModel,
        optim_fn: type[torch.optim.Optimizer] | str | None = None,
        optim_kwargs: dict | None = None,
        scheduler_fn: type[torch.optim.lr_scheduler.LRScheduler] | str | None = None,
        scheduler_kwargs: dict | None = None,
        default_lr: float = 1e-3,
        config: dict[str, Any] | Namespace | None = None,
    ) -> None:
        super().__init__()
        self.model = model

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

        if config is not None:
            self._set_hparams(config)

    @classmethod
    @patch("lightning.pytorch.core.saving._load_state", new=_load_state)
    def load_from_checkpoint(
        cls,
        checkpoint_path: _PATH | IO,
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: _PATH | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> CellariumModule:
        r"""
        Primary way of loading a model from a checkpoint. When Cellarium ML saves a checkpoint it stores the config
        argument passed to ``__init__``  in the checkpoint under ``"hyper_parameters"``.

        Any arguments specified through ``**kwargs`` will override args stored in ``"hyper_parameters"``.

        Args:
            checkpoint_path:
                Path to checkpoint. This can also be a URL, or file-like object
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file:
                Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure as in this example:

                .. code-block:: yaml

                    model:
                      model:
                        class_path: cellarium.ml.models.OnePassMeanVarStdFromCLI
                        init_args:
                          g_genes: 36350
                          target_count: 10000
                      optim_fn: null
                      optim_kwargs: null
                      scheduler_fn: null
                      scheduler_kwargs: null
                      default_lr: 0.001

                If you train a model using :mod:`cellarium.ml.cli` module you most likely won't need this
                since Cellarium ML CLI will always save the hyperparameters to the checkpoint.

                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                These will be converted into a :class:`dict` and passed into your
                :class:`CellariumModule` for use.
            strict:
                Whether to strictly enforce that the keys in :attr:`checkpoint_path` match the keys
                returned by this module's state dict.
            \**kwargs: Any extra keyword args needed to init the model. Can also be used to override saved
                hyperparameter values.

        Return:
            :class:`CellariumModule` instance with loaded weights and hyperparameters.

        Example::

            # load weights without mapping ...
            module = CellariumModule.load_from_checkpoint("path/to/checkpoint.ckpt")

            # or load weights mapping all weights from GPU 1 to GPU 0 ...
            map_location = {"cuda:1": "cuda:0"}
            module = CellariumModule.load_from_checkpoint(
                "path/to/checkpoint.ckpt",
                map_location=map_location
            )

            # or load weights and hyperparameters from separate files.
            module = CellariumModule.load_from_checkpoint(
                "path/to/checkpoint.ckpt",
                hparams_file="/path/to/config.yaml"
            )

            # override some of the params with new values
            module = CellariumModule.load_from_checkpoint(
                "path/to/checkpoint.ckpt",
                optim_fn=torch.optim.AdamW,
                default_lr=0.0001,
            )
        """
        return super().load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
        )

    def training_step(  # type: ignore[override]
        self, batch: dict[str, np.ndarray | torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:
        args, kwargs = self.model._get_fn_args_from_batch(batch)
        loss = self.model(*args, **kwargs)
        if loss is not None:
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
        return loss

    def forward(self, batch: dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass of the model."""
        assert isinstance(self.model, PredictMixin)
        args, kwargs = self.model._get_fn_args_from_batch(batch)
        return self.model.predict(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizers for the model."""
        optim_config: OptimizerLRSchedulerConfig = {
            "optimizer": self.optim_fn(self.model.parameters(), **self.optim_kwargs)
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
