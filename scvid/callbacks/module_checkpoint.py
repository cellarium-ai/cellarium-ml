# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch


class ModuleCheckpoint(pl.Callback):
    """
    Saves the :attr:`module` of the lightning module at the specified time.

    Args:
        dirpath: Directory to save the module checkpoint.
            By default, dirpath is ``None`` and will be set at runtime to the location
            specified by :class:`~lightning.pytorch.trainer.trainer.Trainer`'s
            :paramref:`~lightning.pytorch.trainer.trainer.Trainer.default_root_dir` argument,
            and if the Trainer uses a logger, the path will also contain logger name and version.
        filename: Filename to save the module checkpoint.
            By default, filename is ``None`` and will be set to ``module_checkpoint.pt``.
        save_on_train_batch_end: Whether to save the module on train batch end.
        save_on_train_epoch_end: Whether to save the module on train epoch end.
        save_on_train_end: Whether to save the module on train end.
    """

    def __init__(
        self,
        dirpath: Path | str | None = None,
        filename: str | None = None,
        save_on_train_batch_end: bool = False,
        save_on_train_epoch_end: bool = False,
        save_on_train_end: bool = True,
    ):
        self.dirpath = dirpath
        if filename is None:
            filename = "module_checkpoint.pt"
        self.filename = filename
        self.save_on_train_batch_end = save_on_train_batch_end
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self.save_on_train_end = save_on_train_end

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        assert hasattr(pl_module, "module")
        assert isinstance(pl_module.module, torch.nn.Module)
        # resolve dirpath at runtime
        dirpath = self._resolve_ckpt_dir(trainer)
        self.dirpath = dirpath

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.save_on_train_batch_end:
            torch.save(pl_module.module, self.filepath)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.save_on_train_epoch_end:
            torch.save(pl_module.module, self.filepath)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.save_on_train_end:
            torch.save(pl_module.module, self.filepath)

    @property
    def filepath(self) -> Path:
        """The full filepath of the module checkpoint."""
        assert self.dirpath is not None
        return Path(self.dirpath) / self.filename

    def _resolve_ckpt_dir(self, trainer: pl.Trainer) -> Path | str:
        """Determines module checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModuleCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".
        """
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModuleCheckpoint
            return self.dirpath

        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        return ckpt_path
