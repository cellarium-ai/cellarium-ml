# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only

from scvid.module import OnePassMeanVarStd, ProbabilisticPCA


class VarianceMonitor(pl.Callback):
    r"""
    Automatically monitors and logs explained variance by the model during training.

    Args:
        total_variance: Total variance of the data. Used to calculate the explained variance ratio.
        mean_var_std_ckpt_path: Path to checkpoint containing OnePassMeanVarStd.
    """

    def __init__(
        self,
        total_variance: float | None = None,
    ):
        self.total_variance = total_variance
        self.W_gk = torch.tensor(0.0)
        self.n_averaged = 0

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        r"""
        Called when the train begins.

        Raises:
            AssertionError: If ``pl_module.module`` is not a ``ProbabilisticPCA`` instance.
            MisconfigurationException: If ``Trainer`` has no ``logger``.
        """
        assert isinstance(
            pl_module.module, ProbabilisticPCA
        ), "VarianceMonitor callback should only be used in conjunction with ProbabilisticPCA"

        if not trainer.loggers:
            raise pl.utilities.exceptions.MisconfigurationException(
                "Cannot use `LearningRateMonitor` callback with `Trainer` that has no logger."
            )
        # attempt to get the total variance from the checkpoint
        if (
            self.total_variance is None
            and hasattr(pl_module.module, "mean_var_std_ckpt_path")
            and pl_module.module.mean_var_std_ckpt_path is not None
        ):
            mean_var_std_ckpt_path = pl_module.module.mean_var_std_ckpt_path
            state_dict = torch.load(mean_var_std_ckpt_path)
            onepass = OnePassMeanVarStd(10000)
            # onepass.load_state_dict(state_dict)
            onepass.x_sums = state_dict["x_sums"]
            onepass.x_squared_sums = state_dict["x_squared_sums"]
            onepass.x_size = state_dict["x_size"]
            self.total_variance = onepass.var_g.sum().item()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Called when the train batch ends."""
        step = trainer.fit_loop.epoch_loop._batches_that_stepped
        if step % trainer.log_every_n_steps == 0:
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
                    variance_stats,
                    step=trainer.fit_loop.epoch_loop._batches_that_stepped,
                )

            L_k = pl_module.module.L_k
            # log L_k as a histogram
            for logger in trainer.loggers:
                for i, l_i in enumerate(L_k[:3]):
                    logger.experiment.add_scalar(
                        f"L_k_{i}",
                        l_i.item(),
                        trainer.fit_loop.epoch_loop._batches_that_stepped,
                    )
                for i, g_i in enumerate(pl_module.module.W_kg.reshape(-1)[:3]):
                    logger.experiment.add_scalar(
                        f"W_kg_{i}",
                        g_i.item(),
                        trainer.fit_loop.epoch_loop._batches_that_stepped,
                    )

            #  # if hasattr(pl_module, "average_module"):
            #  W_variance = self.W_variance
            #
            #  variance_stats = {}
            #  variance_stats["total_explained_variance_average"] = (
            #      W_variance + sigma_variance
            #  )
            #  variance_stats["W_variance_average"] = W_variance
            #  if self.total_variance is not None:
            #      variance_stats["total_explained_variance_ratio_average"] = (
            #          W_variance + sigma_variance
            #      ) / self.total_variance
            #      variance_stats["W_variance_ratio_average"] = (
            #          W_variance / self.total_variance
            #      )
            #
            #  for logger in trainer.loggers:
            #      logger.log_metrics(
            #          variance_stats,
            #          step=trainer.fit_loop.epoch_loop._batches_that_stepped,
            #      )

    #  def on_train_batch_start(
    #      self,
    #      trainer: pl.Trainer,
    #      pl_module: pl.LightningModule,
    #      batch: Any,
    #      batch_idx: int,
    #  ) -> None:
    #      if self.n_averaged == 0:
    #          self.W_gk = pl_module.module.W_gk
    #      else:
    #          self.W_gk = self.W_gk + (pl_module.module.W_gk - self.W_gk) / (
    #              self.n_averaged + 1
    #          )
    #      self.n_averaged += 1
    #
    #  @property
    #  @torch.inference_mode()
    #  def W_variance(self) -> torch.Tensor:
    #      r"""
    #      .. note::
    #         Gradients are disabled, used for inference only.
    #      """
    #      return torch.trace(self.W_gk @ self.W_gk.T)
