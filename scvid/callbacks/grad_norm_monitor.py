# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import pyro
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only


class GradientNormMonitor(pl.Callback):
    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx,
    ) -> None:
        step = trainer.fit_loop.epoch_loop._batches_that_stepped
        if step % trainer.log_every_n_steps == 0:
            for name, value in pl_module.module.named_parameters():
                for logger in trainer.loggers:
                    logger.log_metrics({name: value.norm().item()}, step=step)
                if step == 0:

                    def hook(g, name=name, trainer=trainer):
                        step = trainer.fit_loop.epoch_loop._batches_that_stepped
                        if step % trainer.log_every_n_steps == 0 and step != 0:
                            for logger in trainer.loggers:
                                logger.log_metrics(
                                    {f"{name}_grad_norm": g.norm().item() / g.numel()},
                                    step=step,
                                )
                                for i, g_i in enumerate(g.reshape(-1)[:3]):
                                    logger.experiment.add_scalar(
                                        f"{name}_grad_{i}",
                                        g_i.item(),
                                        trainer.fit_loop.epoch_loop._batches_that_stepped,
                                    )

                    value.register_hook(hook)
