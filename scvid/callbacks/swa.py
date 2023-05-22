# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import Any

import lightning.pytorch as pl
import torch


class StochasticWeightAveraging(pl.Callback):
    def __init__(self, swa_start: int, ckpt_path: str = None):
        r"""

        Implements the Stochastic Weight Averaging (SWA) Callback to average a model.

        Stochastic Weight Averaging was proposed in ``Averaging Weights Leads to
        Wider Optima and Better Generalization`` by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        Arguments:
            swa_epoch_start: If provided as int, the procedure will start from
                the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1,
                the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch
            device: if provided, the averaged model will be stored on the ``device``.
                When None is provided, it will infer the `device` from ``pl_module``.
                (default: ``"cpu"``)

        """

        self.n_averaged: int = 0
        self.swa_start = swa_start
        self.ckpt_path = ckpt_path

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        # copy the model before moving it to accelerator device.
        if self.ckpt_path is not None:
            pl_module.module.load_state_dict(torch.load(self.ckpt_path))
        pl_module.average_module = deepcopy(pl_module.module)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.swa_start <= trainer.global_step:
            self.update_parameters(pl_module.average_module, pl_module.module)

    def update_parameters(
        self, average_model: torch.nn.Module, model: torch.nn.Module
    ) -> None:
        state_dict = {}
        for name in average_model.state_dict():
            param_swa = average_model.state_dict()[name].detach()
            param_model = model.state_dict()[name].detach()
            if self.n_averaged == 0:
                state_dict[name] = param_model
            else:
                state_dict[name] = param_swa + (param_model - param_swa) / (
                    self.n_averaged + 1
                )
        average_model.load_state_dict(state_dict)
        self.n_averaged += 1

    #  def state_dict(self) -> dict[str, Any]:
    #      return {
    #          "n_averaged": self.n_averaged,
    #          "latest_update_epoch": self._latest_update_epoch,
    #          "scheduler_state": None
    #          if self._swa_scheduler is None
    #          else self._swa_scheduler.state_dict(),
    #          "average_model_state": None
    #          if self._average_model is None
    #          else self._average_model.state_dict(),
    #      }
    #
    #  def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #      self._init_n_averaged = state_dict["n_averaged"]
    #      self._latest_update_epoch = state_dict["latest_update_epoch"]
    #      self._scheduler_state = state_dict["scheduler_state"]
    #      self._load_average_model_state(state_dict["average_model_state"])
