# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import re
from collections import defaultdict

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only


class ComputeNorm(pl.Callback):
    """
    Callback to compute the model wise and per layer norm of the parameters and gradients.
    """

    @rank_zero_only
    def on_before_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: torch.Tensor) -> None:
        """Compute the model wise norm of the parameters."""
        param_norm = torch.tensor(0.0, device=pl_module.device)
        for _, param in pl_module.named_parameters():
            if param.requires_grad:
                param_norm += torch.pow(torch.norm(param.detach()), 2.0)

        pl_module.log("model_wise_param_norm", torch.sqrt(param_norm).item())

    @rank_zero_only
    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        """Compute the model wise and per layer norm of the gradients."""
        model_wise_grad_norm = torch.tensor(0.0, device=pl_module.device)
        per_layer_grad_norm: dict[str, torch.Tensor] = defaultdict(lambda: torch.tensor(0.0, device=pl_module.device))
        layer_pattern = re.compile(r".*(blocks\.)(\d+)(\.).*")

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            param_grad_norm = torch.pow(torch.norm(param.grad), 2.0)
            model_wise_grad_norm += param_grad_norm

            # get a match if module name contains `*.blocks.i.*` where i is block num
            match = layer_pattern.match(name)
            if match:
                layer_id = match.group(2)
                per_layer_grad_norm[layer_id] += param_grad_norm

        pl_module.log("model_wise_grad_norm", torch.sqrt(model_wise_grad_norm).item())
        pl_module.log_dict(
            {
                f"per_layer_grad_norm/layer_{layer_id}": torch.sqrt(per_layer_grad_norm[layer_id]).item()
                for layer_id in per_layer_grad_norm
            }
        )
