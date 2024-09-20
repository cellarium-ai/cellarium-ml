# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import rank_zero_only


class GradScalerMonitor(pl.Callback):
    """
    Automatically monitors and logs the scale of the gradient scaler during training.
    """

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pl_module.log("loss_scale", trainer.precision_plugin.scaler._scale.item())
