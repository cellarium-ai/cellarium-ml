# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import pytorch_lightning as pl

from scvid.module import ProbabilisticPCAPyroModule


class VarianceMonitor(pl.Callback):
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        assert isinstance(
            pl_module.module, ProbabilisticPCAPyroModule
        ), "VarianceMonitor callback should only be used in conjunction with ProbabilisticPCAPyroModule"
        W_variance = pl_module.module.W_variance
        sigma_variance = pl_module.module.sigma_variance
        # total variance of the data
        total_variance = pl_module.module.total_variance
        pl_module.log("total_variance", W_variance + sigma_variance)
        pl_module.log("W_variance", W_variance)
        pl_module.log("sigma_variance", sigma_variance)
        if total_variance is not None:
            pl_module.log(
                "total_variance_ratio", (W_variance + sigma_variance) / total_variance
            )
            pl_module.log("W_variance_ratio", W_variance / total_variance)
            pl_module.log("sigma_variance_ratio", sigma_variance / total_variance)
