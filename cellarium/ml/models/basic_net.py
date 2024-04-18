# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.models.nt_xent import NT_Xent
from cellarium.ml.transforms import Log1p, NormalizeTotal, Randomize, ZScore
from cellarium.ml.utilities.data import get_rank_and_num_replicas

from lightly.loss import NTXentLoss

import logging

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger()


class BasicNet(CellariumModel, PredictMixin):
    def __init__(self, in_channel, out_channel):
        super(BasicNet, self).__init__()
        self.net = torch.nn.Linear(in_channel, out_channel)

    def forward(self, x_ng: torch.Tensor):
        # rank, num_replicas = get_rank_and_num_replicas()

        out = self.net(x_ng)
        loss_n = torch.norm(out, dim=1)
        
        return {'loss': loss_n.mean()}

    def predict(self, x_ng: torch.Tensor, **kwargs: Any):
        with torch.no_grad():
            z = self.net(x_ng)

        return z
