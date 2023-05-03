# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from datetime import timedelta

import lightning.pytorch as pl


class TimeMonitor(pl.callbacks.Timer):
    """
    A callback that monitors the time elapsed during training.

    At the end of training, the number of batches per second is calculated and
    logged to the logger(s).
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_batches = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().on_train_batch_end(trainer)
        self.num_batches += 1

        timer_stats = {}
        timer_stats["time_elapsed"] = self.time_elapsed(stage="train")

        for logger in trainer.loggers:
            logger.log_metrics(
                timer_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped
            )

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().on_train_end(trainer, pl_module)

        timer_stats = {}
        timer_stats["batches_per_second"] = (
            self.num_batches
            * trainer.num_devices
            * trainer.num_nodes
            / self.time_elapsed("train")
        )

        for logger in trainer.loggers:
            logger.log_metrics(timer_stats, step=trainer.num_devices)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["num_batches"] = self.num_batches
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.num_batches = state_dict["num_batches"]
