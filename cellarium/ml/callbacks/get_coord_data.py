# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.hooks import RemovableHandle


def l1_norm(x: torch.Tensor) -> float:
    return x.detach().abs().mean().item()


def record_out_coords_hook(
    trainer: pl.Trainer, name: str, batch_idx: int, multiplier: float
) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
    """
    Returns a hook to record layer output coordinate size.
    Args:
        records:
            The list of records to append to.
        name:
            The name of the layer.
        t:
            The time step.
        multiplier:
            The multiplier.
    Returns:
        A hook to record layer output coordinate size.
    """

    def hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        stats = {
            "l1": l1_norm(output) * multiplier,
            "module": f"{name}.out",
            "type": "out",
            "t": batch_idx,
        }
        for logger in trainer.loggers:
            logger.log_metrics(stats, step=batch_idx)  # type: ignore[arg-type]

    return hook


class GetCoordData(pl.Callback):
    """
    A callback that logs the loss scale during mixed-precision training.

    Args:
        layer_name_to_multiplier_name:
            A dictionary mapping layer names to their corresponding multipliers.
            If not provided, all layers will have a multiplier of 1.0.
    """

    def __init__(self, layer_name_to_multiplier_name: dict[str, str] | None = None) -> None:
        self.layer_name_to_multiplier_name = layer_name_to_multiplier_name or {}

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Store the hooks to remove them later
        self.out_hooks: list[RemovableHandle] = []
        # Create a mapping from modules to names
        module_to_name = {module: name for name, module in pl_module.named_modules()}
        # Store the initial parameter values before the optimizer step
        self.on_batch_start_param_values: dict[str, torch.Tensor] = {}

        def record_coords_hook(module: torch.nn.Module) -> None:
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module_name = module_to_name[module]
                multiplier_name = self.layer_name_to_multiplier_name.get(module_name, None)
                if multiplier_name is not None:
                    multiplier = getattr(pl_module, multiplier_name)
                else:
                    multiplier = 1.0

                # out coords
                self.out_hooks.append(
                    module.register_forward_hook(record_out_coords_hook(trainer, module_name, batch_idx, multiplier))
                )

                # param coords
                for param_name, param in module.named_parameters():
                    full_param_name = f"{module_name}.{param_name}"
                    self.on_batch_start_param_values[full_param_name] = param.clone().detach()
                    param_multiplier = multiplier if param_name == "weight" else 1.0

                    stats = {
                        "l1": l1_norm(param) * param_multiplier,
                        "module": full_param_name,
                        "type": "param",
                        "t": batch_idx,
                    }
                    for logger in trainer.loggers:
                        logger.log_metrics(stats, step=batch_idx)  # type: ignore[arg-type]

        pl_module.apply(record_coords_hook)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Remove the hooks
        for hook in self.out_hooks:
            hook.remove()

        # Create a mapping from modules to names
        module_to_name = {module: name for name, module in pl_module.named_modules()}

        def record_coords_hook(module: torch.nn.Module) -> None:
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module_name = module_to_name[module]
                multiplier_name = self.layer_name_to_multiplier_name.get(module_name, None)
                if multiplier_name is not None:
                    multiplier = getattr(pl_module, multiplier_name)
                else:
                    multiplier = 1.0

                # param delta coords
                for param_name, param in module.named_parameters():
                    full_param_name = f"{module_name}.{param_name}"
                    prev_param_value = self.on_batch_start_param_values[full_param_name]
                    param_delta = param.detach() - prev_param_value
                    param_multiplier = multiplier if param_name == "weight" else 1.0

                    stats = {
                        "l1": (l1_norm(param_delta)) * param_multiplier,
                        "module": f"{full_param_name}.delta",
                        "type": "delta",
                        "t": batch_idx,
                    }
                    for logger in trainer.loggers:
                        logger.log_metrics(stats, step=batch_idx)  # type: ignore[arg-type]

        pl_module.apply(record_coords_hook)
        self.on_batch_start_param_values = {}
