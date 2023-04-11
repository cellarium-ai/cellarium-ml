# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import lightning.pytorch as pl
import torch


# Create PyTorch Lightning callback name ModuleCheckpoint
# This callback should save the .module attribute (which is a torch.nn.Module) of the lightning module
# The parameter should be the filepath to save the checkpoint
# And frequencies of saving the checkpoing with boolean values: on_train_batch_end, on_train_epoch_end, on_train_end
# Add type hints to the parameters
class ModuleCheckpoint(pl.Callback):
    """
    Saves the :attr:`module` of the lightning module at the specified time.

    Args:
        filepath: Path to save the module checkpoint.
        save_on_train_batch_end: Whether to save the module on train batch end.
        save_on_train_epoch_end: Whether to save the module on train epoch end.
        save_on_train_end: Whether to save the module on train end.
    """

    def __init__(
        self,
        filepath: str,
        save_on_train_batch_end: bool = False,
        save_on_train_epoch_end: bool = False,
        save_on_train_end: bool = True,
    ):
        self.filepath = filepath
        self.save_on_train_batch_end = save_on_train_batch_end
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self.save_on_train_end = save_on_train_end

    # add on_train_start method that asserts that pl_module has a module attribute
    # and that it is a torch.nn.Module
    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        assert hasattr(pl_module, "module")
        assert isinstance(pl_module.module, torch.nn.Module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any
    ) -> None:
        if self.save_on_train_batch_end:
            torch.save(pl_module.module, self.filepath)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any
    ) -> None:
        if self.save_on_train_epoch_end:
            torch.save(pl_module.module, self.filepath)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any
    ) -> None:
        if self.save_on_train_end:
            torch.save(pl_module.module, self.filepath)
