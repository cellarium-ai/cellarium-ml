# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional

import pytorch_lightning as pl

from scvid.module import ProbabilisticPCAPyroModule


class VarianceMonitor(pl.Callback):
    r"""
    Automatically monitors and logs explained variance by the model during training.

    Args:
        total_variance: Total variance of the data. Used to calculate the explained variance ratio.
    """

    def __init__(self, total_variance: Optional[float] = None):
        self.total_variance = total_variance

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        r"""
        Called when the train begins.

        Raises:
            AssertionError: If ``pl_module.module`` is not a ``ProbabilisticPCAPyroModule`` instance.
            MisconfigurationException: If ``Trainer`` has no ``logger``.
        """
        assert isinstance(
            pl_module.module, ProbabilisticPCAPyroModule
        ), "VarianceMonitor callback should only be used in conjunction with ProbabilisticPCAPyroModule"

        if not trainer.loggers:
            raise pl.utilities.exceptions.MisconfigurationException(
                "Cannot use `LearningRateMonitor` callback with `Trainer` that has no logger."
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Called when the train batch ends."""
        W_variance = pl_module.module.W_variance
        sigma_variance = pl_module.module.sigma_variance

        variance_stats = {}
        variance_stats["total_explained_variance"] = W_variance + sigma_variance
        variance_stats["W_variance"] = W_variance
        variance_stats["sigma_variance"] = sigma_variance
        if self.total_variance is not None:
            variance_stats["total_explained_variance_ratio"] = (
                W_variance + sigma_variance
            ) / self.total_variance
            variance_stats["W_variance_ratio"] = W_variance / self.total_variance
            variance_stats["sigma_variance_ratio"] = (
                sigma_variance / self.total_variance
            )

        for logger in trainer.loggers:
            logger.log_metrics(
                variance_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped
            )
