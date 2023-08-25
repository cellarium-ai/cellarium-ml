# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.optim.lr_scheduler import LambdaLR


class LinearLR(LambdaLR):
    """
    Learning rate scheduler with a learning rate that decreases linearly from the initial lr
    set in the optimizer to 0, after a warmup period during which it increases linearly from 0
    to the initial lr set in the optimizer.

    Args:
        optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps:
            The number of steps for the warmup phase.
        num_training_steps:
            The total number of training steps.
        last_epoch:
            The index of the last epoch when resuming training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, self._lr_lambda, last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )
