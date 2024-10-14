# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
import torch


def l1_norm(x: torch.Tensor) -> float:
    return x.detach().abs().mean().item()


def record_out_coords(
    trainer: list[dict], width: int, name: str, batch_idx: int
) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
    """
    Returns a hook to record layer output coordinate size.

    Args:
        records:
            The list of records to append to.
        width:
            The width of the model.
        name:
            The name of the layer.
        t:
            The time step.

    Returns:
        A hook to record layer output coordinate size.
    """

    def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        stats = {f"out.t={batch_idx}/{name}": math.log(l1_norm(output), 2)}
        for logger in trainer.loggers:
            logger.log_metrics(stats, step=width)

    return hook


def record_grad_coords(
    trainer: list[dict], width: int, name: str, batch_idx: int
) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
    """
    Returns a hook to record layer output coordinate size.

    Args:
        records:
            The list of records to append to.
        width:
            The width of the model.
        name:
            The name of the layer.
        t:
            The time step.

    Returns:
        A hook to record layer output coordinate size.
    """

    def hook(grad: torch.Tensor) -> None:
        if l1_norm(grad) == 0:
            stats = {f"grad.t={batch_idx}/{name}": l1_norm(grad)}
        else:
            stats = {f"grad.t={batch_idx}/{name}": math.log(l1_norm(grad), 2)}
        for logger in trainer.loggers:
            logger.log_metrics(stats, step=width)

    return hook


class CoordCheck(pl.Callback):
    """
    A callback that logs the loss scale during mixed-precision training.
    """

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.remove_hooks = []
        width = math.log(pl_module.model.d_model, 2)
        for name, module in pl_module.model.named_children():
            # record layer outputs
            if name == "transformer":
                for sub_name, sub_module in module.blocks.named_children():
                    self.remove_hooks.append(
                        sub_module.register_forward_hook(
                            record_out_coords(trainer, width, f"transformer.blocks.{sub_name}", batch_idx)
                        )
                    )
            else:
                self.remove_hooks.append(
                    module.register_forward_hook(record_out_coords(trainer, width, name, batch_idx))
                )  # type: ignore[arg-type]

        self.param_hooks = []
        for param_name, param in pl_module.model.named_parameters():
            self.param_hooks.append(param.register_hook(record_grad_coords(trainer, width, param_name, batch_idx)))
            if l1_norm(param) == 0:
                stats = {f"param.t={batch_idx}/{param_name}": l1_norm(param)}
            else:
                stats = {f"param.t={batch_idx}/{param_name}": math.log(l1_norm(param), 2)}
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=width)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: Any,
        batch_idx: int,
    ) -> None:
        for hook in self.remove_hooks:
            hook.remove()

        for hook in self.param_hooks:
            hook.remove()
