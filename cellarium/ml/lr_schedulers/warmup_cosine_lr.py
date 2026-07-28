# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineLR(LambdaLR):
    """
    Learning rate scheduler with linear warmup followed by cosine decay to zero.

    Args:
        optimizer:
            The optimizer for which to schedule the learning rate.
        num_training_steps:
            The total number of optimizer steps in the schedule.
        num_warmup_steps:
            The number of linear warmup steps. If ``None``, it is computed from
            ``warmup_fraction``.
        warmup_fraction:
            Fraction of ``num_training_steps`` used for linear warmup when
            ``num_warmup_steps`` is ``None``.
        last_epoch:
            The index of the last epoch when resuming training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int | None = None,
        warmup_fraction: float = 0.2,
        last_epoch: int = -1,
    ) -> None:
        if num_training_steps <= 0:
            raise ValueError("`num_training_steps` must be positive.")
        if not 0 <= warmup_fraction <= 1:
            raise ValueError("`warmup_fraction` must be between 0 and 1.")
        if num_warmup_steps is None:
            num_warmup_steps = int(num_training_steps * warmup_fraction)
        if not 0 <= num_warmup_steps <= num_training_steps:
            raise ValueError("`num_warmup_steps` must be between 0 and `num_training_steps`.")

        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.warmup_fraction = warmup_fraction
        super().__init__(optimizer, self._lr_lambda, last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        if current_step >= self.num_training_steps:
            return 0.0
        if self.num_warmup_steps > 0 and current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))

        cosine_steps = self.num_training_steps - self.num_warmup_steps
        if cosine_steps <= 0:
            return 0.0
        progress = float(current_step - self.num_warmup_steps) / float(cosine_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
