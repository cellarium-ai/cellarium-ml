# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping


class MetaSafeEarlyStopping(EarlyStopping):
    def setup(self, trainer, pl_module, stage: str) -> None:
        super().setup(trainer, pl_module, stage)

        # override the empty meta tensor with a real CPU tensor
        if self.mode == "min":
            self.best_score = torch.tensor(np.inf, device="cpu")
        else:
            self.best_score = torch.tensor(-np.inf, device="cpu")
