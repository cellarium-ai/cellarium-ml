# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import lightning.pytorch as pl
import torch
import torch.distributed as dist


class DistributedPCA(pl.Callback):
    """
    Distributed PCA.
    """

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # parameters
        mean_correct = pl_module.module.mean_correct
        k = pl_module.module.k_components
        p = pl_module.module.p_oversamples
        V_kg = pl_module.module.V_kg
        S_k = pl_module.module.S_k
        m = pl_module.module.x_size
        if mean_correct:
            x_mean_g = pl_module.module.x_mean_g
        # initialize i, rank, and world_size
        rank = trainer.global_rank
        world_size = trainer.world_size
        i = 0
        while world_size > 1:
            # level i
            # at most two local ranks (leading and trailing)
            if rank % 2 == 0:  # leading rank
                if rank < world_size:
                    # if there is a trailing rank
                    # then concatenate and merge SVDs
                    src = (rank + 1) * 2**i
                    A = torch.diag(S_k) @ V_kg
                    B = torch.zeros_like(A)
                    n = torch.zeros_like(m)
                    dist.recv(B, src=src)
                    dist.recv(n, src=src)
                    C = torch.cat([A, B], dim=0)
                    if mean_correct:
                        A_mean = x_mean_g
                        B_mean = torch.zeros_like(A_mean)
                        dist.recv(B_mean, src=src)
                        mean_correction = (
                            math.sqrt(m * n / (m + n)) * (A_mean - B_mean)[None, :]
                        )
                        C = torch.cat([C, mean_correction], dim=0)
                        x_mean_g = A_mean * m / (m + n) + B_mean * n / (m + n)
                    _, S_q, V_gq = torch.svd_lowrank(C, q=k + p)
                    # update parameters
                    S_k = S_q[:k]
                    V_kg = V_gq.T[:k]
                    m = m + n
            else:  # trailing rank
                # send to a leading rank and exit
                dst = (rank - 1) * 2**i
                dist.send(torch.diag(S_k) @ V_kg, dst=dst)
                dist.send(m, dst=dst)
                if mean_correct:
                    dist.send(x_mean_g, dst=dst)
                break
            # update rank, world_size, and level i
            rank = rank // 2
            world_size = math.ceil(world_size / 2)
            i += 1
        else:
            assert trainer.global_rank == 0
            pl_module.module.V_kg = V_kg
            pl_module.module.S_k = S_k
            pl_module.module.x_size = m
            if mean_correct:
                pl_module.module.x_mean_g = x_mean_g
