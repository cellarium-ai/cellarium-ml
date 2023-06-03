# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import lightning.pytorch as pl
import pyro
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only


class GradientNormMonitor(pl.Callback):
    @rank_zero_only
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % trainer.log_every_n_steps == 0:
            for name, value in pl_module.module.named_parameters():
                #  self.logger.experiment.add_histogram(
                #      tag=k, values=v.grad, global_step=self.trainer.global_step
                #  )
                for logger in trainer.loggers:
                    logger.log_metrics({name: value.norm().item()}, step=trainer.global_step)
                    logger.log_metrics(
                        {f"{name}_grad_norm": value.grad.norm().item() / value.numel()},
                        step=trainer.global_step,
                    )
                    for i, g_i in enumerate(value.grad.reshape(-1)[:4]):
                        logger.experiment.add_scalar(
                            f"{name}_grad_{i}",
                            g_i.item(),
                            trainer.global_step,
                        )
