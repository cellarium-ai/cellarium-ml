# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
import torch.nn as nn

from scvid.data.util import get_rank_and_num_replicas
from scvid.module import BaseModule, GatherLayer


class DistributedPCA(BaseModule):
    def __init__(self, k: int, transform: nn.Module | None = None) -> None:
        super().__init__()
        self.k = k
        self.transform = transform
        self.US_gk: torch.Tensor
        self.S_k: torch.Tensor
        self.x_mean_g1: torch.Tensor
        self.x_size: torch.Tensor
        self.register_buffer("US_gk", torch.tensor(0))
        self.register_buffer("S_k", torch.tensor(0))
        self.register_buffer("x_mean_g1", torch.tensor(0))
        self.register_buffer("x_size", torch.tensor(0))
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, torch.Tensor]
    ) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng: torch.Tensor) -> None:
        k = self.k
        if self.transform is not None:
            x_ng = self.transform(x_ng)
        # calcualte mini-batch mean
        x_mean_g1 = x_ng.mean(dim=0)[:, None]
        # perform SVD on centered data
        x_gn = x_ng.T
        U_gk, S_k, _ = torch.svd_lowrank(x_gn - x_mean_g1, q=k + 6)
        # merge results
        n = self.x_size
        m = x_ng.size(0)
        if n > 0:
            US_gk = U_gk[:, :k] @ torch.diag(S_k[:k])
            mean_correction = (self.x_mean_g1 - x_mean_g1) * math.sqrt(n * m / (n + m))
            C = torch.cat([self.US_gk, US_gk, mean_correction], dim=-1)
            # perform SVD on merged results
            U_gk, S_k, _ = torch.svd_lowrank(C, q=k + 6)
            print("full", S_k[:5] ** 2 / (n + m))
        # update buffers
        self.US_gk = U_gk[:, :k] @ torch.diag(S_k[:k])
        self.S_k = S_k[:k]
        self.x_mean_g1 = self.x_mean_g1 * n / (n + m) + x_mean_g1 * m / (n + m)
        self.x_size = n + m
