# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import re

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only


class NormMonitor(pl.Callback):
    """
    A callback that logs the loss scale during mixed-precision training.
    """

    @rank_zero_only
    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        """Compute the model wise norm of the parameters."""
        model = pl_module.model
        # param_norm = torch.tensor(0.0).to(model.device)
        param_norm = 0.0
        for _, param in model.named_parameters():
            if param.requires_grad:
                # simply add if we want to include all params
                param_norm += torch.pow(torch.norm(param), 2.0)

        pl_module.log("model_wise_params_norm", torch.sqrt(param_norm).item())

    @rank_zero_only
    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: torch.optim.Optimizer
    ) -> None:
        """Compute the model wise and per layer norm of the gradients."""
        model = pl_module.model
        # params_grad_norm = torch.tensor(0.0).to(model.device)
        params_grad_norm = 0.0
        for _, param in model.named_parameters():
            if param.grad is not None:
                params_grad_norm += torch.pow(torch.norm(param.grad), 2.0)
        params_grad_norm = torch.sqrt(params_grad_norm)

        pl_module.log("model_wise_grad_norm", params_grad_norm.item())

        per_layer_grad_norm = {}
        layer_pattern = re.compile(r".*(blocks\.)(\d+)(\.).*")
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # get a match if module name contains `layers.i.0` where i is layer num
            match = layer_pattern.match(name)
            if match:
                layer_id = match.group(2)
                if layer_id not in per_layer_grad_norm:
                    per_layer_grad_norm[layer_id] = 0.0
                per_layer_grad_norm[layer_id] += torch.pow(
                    torch.norm(param.grad), 2.0
                )

        pl_module.log_dict(
            {
                f"per_layer_grad_norm/layer_{layer_id}": torch.sqrt(
                    per_layer_grad_norm[layer_id]
                ).item()
                for layer_id in per_layer_grad_norm
            }
        )
