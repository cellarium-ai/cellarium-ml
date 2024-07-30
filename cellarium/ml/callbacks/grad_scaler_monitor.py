# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import rank_zero_only


class GradScalerMonitor(pl.Callback):
    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if (trainer.global_step > 200) and ((trainer.global_step + 1) % trainer.log_every_n_steps != 0):  # type: ignore[attr-defined]
            return

        scale = {"GradScaler": trainer.precision_plugin.scaler._scale.item()}
        for logger in trainer.loggers:
            logger.log_metrics(scale, step=trainer.global_step)
