# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import lightning.pytorch as pl
import torch
import torch.distributed as dist


class DistributedPCA(pl.Callback):
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        k = pl_module.module.k
        i = 0
        rank = trainer.global_rank
        rank = rank
        world_size = trainer.world_size
        C = pl_module.module.US_gk
        C_ = pl_module.module.x_mean_g1
        n = pl_module.module.x_size
        while world_size > 1:
            if rank % 2 == 0:
                # if effective rank is even then receive U and S from rank+1
                if rank + 1 >= world_size:
                    pass
                else:
                    A = C
                    A_ = C_
                    n = n
                    B = torch.zeros_like(A)
                    B_ = torch.zeros_like(A_)
                    m = torch.zeros_like(n)
                    dist.recv(B, src=(rank + 1) * 2**i)
                    dist.recv(B_, src=(rank + 1) * 2**i)
                    dist.recv(m, src=(rank + 1) * 2**i)
                    C = torch.cat(
                        [A, B, (A_ - B_) * math.sqrt(n * m / (n + m))], dim=-1
                    )
                    U_gk, S_k, _ = torch.svd_lowrank(C, q=k + 6)
                    C = U_gk[:, :k] @ torch.diag(S_k[:k])
                    C_ = A_ * n / (n + m) + B_ * m / (n + m)
                    n = n + m
            else:
                # if rank is odd then send U and S to rank-1
                dist.send(C, dst=(rank - 1) * 2**i)
                dist.send(C_, dst=(rank - 1) * 2**i)
                dist.send(n, dst=(rank - 1) * 2**i)
                break
            i += 1
            rank = rank // 2
            world_size = math.ceil(world_size / 2)
        else:
            print("full", S_k[:5] ** 2 / n)
