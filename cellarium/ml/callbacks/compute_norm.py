# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import re
from collections import defaultdict

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy


class ComputeNorm(pl.Callback):
    """
    A callback to compute the model wise and per layer l2 norm of the parameters and gradients.

    Args:
        layer_name:
            The name of the layer to compute the per layer norm.
            If ``None``, the callback will compute the model wise norm only.
    """

    def __init__(self, layer_name: str | None = None) -> None:
        self.layer_pattern: re.Pattern[str] | None
        if layer_name is not None:
            self.layer_pattern = re.compile(r".*(" + layer_name + r"\.)(\d+)(\.).*")
        else:
            self.layer_pattern = None

    def on_before_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: torch.Tensor) -> None:
        """Compute the model wise norm of the parameters."""
        param_norm_sq = torch.tensor(0.0, device=pl_module.device)

        for _, param in pl_module.named_parameters():
            if param.requires_grad:
                param_norm_sq += torch.pow(torch.norm(param.detach()), 2.0)

        if (
            isinstance(trainer.strategy, FSDPStrategy)
            and trainer.strategy.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            assert trainer.strategy.model is not None
            # Sum all local norms to get the total norm
            dist.all_reduce(param_norm_sq, op=dist.ReduceOp.SUM, group=trainer.strategy.model.process_group)

        pl_module.log("model_wise_param_norm", torch.sqrt(param_norm_sq).item(), rank_zero_only=True)

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        """Compute the model wise and per layer norm of the gradients."""
        model_wise_grad_norm_sq = torch.tensor(0.0, device=pl_module.device)
        per_layer_grad_norm_sq: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0, device=pl_module.device)
        )

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            param_grad_norm_sq = torch.pow(torch.norm(param.grad), 2.0)
            model_wise_grad_norm_sq += param_grad_norm_sq

            # get a match if module name contains `*.layer_name.i.*` where i is layer num
            if self.layer_pattern:
                match = self.layer_pattern.match(name)
                if match:
                    layer_id = match.group(2)
                    per_layer_grad_norm_sq[layer_id] += param_grad_norm_sq

        if (
            isinstance(trainer.strategy, FSDPStrategy)
            and trainer.strategy.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            assert trainer.strategy.model is not None
            # Sum all local norms to get the total norm
            dist.all_reduce(model_wise_grad_norm_sq, op=dist.ReduceOp.SUM, group=trainer.strategy.model.process_group)
            for layer_id in per_layer_grad_norm_sq:
                dist.all_reduce(
                    per_layer_grad_norm_sq[layer_id], op=dist.ReduceOp.SUM, group=trainer.strategy.model.process_group
                )

        pl_module.log("model_wise_grad_norm", torch.sqrt(model_wise_grad_norm_sq).item(), rank_zero_only=True)
        if per_layer_grad_norm_sq:
            pl_module.log_dict(
                {
                    f"per_layer_grad_norm/layer_{layer_id}": torch.sqrt(per_layer_grad_norm_sq[layer_id]).item()
                    for layer_id in per_layer_grad_norm_sq
                },
                rank_zero_only=True,
            )
