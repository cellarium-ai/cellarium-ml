# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import rank_zero_only


class LossScaleMonitor(pl.Callback):
    """
    A callback that logs the loss scale during mixed-precision training.
    """

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pl_module.log("loss_scale", trainer.precision_plugin.scaler._scale.item())  # type: ignore[attr-defined]
