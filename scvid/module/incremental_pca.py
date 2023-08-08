# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from lightning.pytorch.strategies import DDPStrategy

from scvid.module import BaseModule, PredictMixin


class IncrementalPCA(BaseModule, PredictMixin):
    """
    Distributed and Incremental PCA.

    **Reference:**

    1. *A Distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks*,
       M. A. Iwen, B. W. Ong
       (https://users.math.msu.edu/users/iwenmark/Papers/distrib_inc_svd.pdf)
    2. *Incremental Learning for Robust Visual Tracking*,
       D. Ross, J. Lim, R.-S. Lin, M.-H. Yang
       (https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)

    Args:
        g_genes: Number of genes.
        k_components: Number of principal components.
        svd_lowrank_niter: Number of iterations for the low-rank SVD algorithm. Default: ``2``.
        transform: If not ``None`` is used to transform the input data.
        perform_mean_correction: If ``True`` then the mean correction is applied to the update step.
            If ``False`` then the data is assumed to be centered and the mean correction
            is not applied to the update step.
    """

    def __init__(
        self,
        g_genes: int,
        k_components: int,
        svd_lowrank_niter: int = 2,
        perform_mean_correction: bool = False,
        transform: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.g_genes = g_genes
        self.k_components = k_components
        self.svd_lowrank_niter = svd_lowrank_niter
        self.perform_mean_correction = perform_mean_correction
        self.transform = transform
        self.V_kg: torch.Tensor
        self.S_k: torch.Tensor
        self.x_mean_g: torch.Tensor
        self.x_size: torch.Tensor
        self.register_buffer("V_kg", torch.zeros(k_components, g_genes))
        self.register_buffer("S_k", torch.zeros(k_components))
        self.register_buffer("x_mean_g", torch.zeros(g_genes))
        self.register_buffer("x_size", torch.tensor(0))
        self._dummy_param = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, np.ndarray | torch.Tensor]
    ) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng: torch.Tensor) -> None:
        if self.transform is not None:
            x_ng = self.transform(x_ng)

        g = self.g_genes
        k = self.k_components
        niter = self.svd_lowrank_niter
        m = self.x_size
        n = x_ng.size(0)
        assert k <= min(n, g), (
            f"Rank of svd_lowrank (k_components): {k}"
            f" must be less than min(n_cells, g_genes): {min(n, g)}"
        )

        # compute SVD of new data
        if self.perform_mean_correction:
            x_mean_g = x_ng.mean(dim=0)
            _, S_k, V_gk = torch.svd_lowrank(x_ng - x_mean_g, q=k, niter=niter)
        else:
            _, S_k, V_gk = torch.svd_lowrank(x_ng, q=k, niter=niter)

        # if not the first batch, merge results
        if m > 0:
            SV_kg = torch.einsum("k,gk->kg", S_k, V_gk)
            C = torch.cat([torch.einsum("k,kg->kg", self.S_k, self.V_kg), SV_kg], dim=0)
            if self.perform_mean_correction:
                mean_correction = (
                    math.sqrt(m * n / (m + n)) * (self.x_mean_g - x_mean_g)[None, :]
                )
                C = torch.cat([C, mean_correction], dim=0)
            # perform SVD on merged results
            _, S_k, V_gk = torch.svd_lowrank(C, q=k, niter=niter)

        # update buffers
        self.V_kg = V_gk.T
        self.S_k = S_k
        self.x_size = m + n
        if self.perform_mean_correction:
            self.x_mean_g = self.x_mean_g * m / (m + n) + x_mean_g * n / (m + n)

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "Distributed and Incremental PCA requires that "
                "the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is False, (
                "Distributed and Incremental PCA requires that "
                "broadcast_buffers is set to False."
            )

    def on_epoch_end(self, trainer: pl.Trainer) -> None:
        # no need to merge if only one process
        if trainer.world_size == 1:
            return

        # parameters
        k = self.k_components
        niter = self.svd_lowrank_niter
        self_X_size = self.x_size
        self_X = torch.einsum("k,kg->kg", self.S_k, self.V_kg).contiguous()
        if self.perform_mean_correction:
            self_X_mean = self.x_mean_g

        # initialize merging level i, rank, and world_size
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
                    other_X = torch.zeros_like(self_X)
                    other_X_size = torch.zeros_like(self_X_size)
                    dist.recv(other_X, src=src)
                    dist.recv(other_X_size, src=src)
                    total_X_size = self_X_size + other_X_size

                    # obtain joined_X
                    joined_X = torch.cat([self_X, other_X], dim=0)
                    if self.perform_mean_correction:
                        other_X_mean = torch.zeros_like(self_X_mean)
                        dist.recv(other_X_mean, src=src)
                        mean_correction = (
                            math.sqrt(self_X_size * other_X_size / total_X_size)
                            * (self_X_mean - other_X_mean)[None, :]
                        )
                        joined_X = torch.cat([joined_X, mean_correction], dim=0)

                    # perform SVD on joined_X
                    _, S_k, V_gk = torch.svd_lowrank(joined_X, q=k, niter=niter)

                    # update parameters
                    V_kg = V_gk.T
                    if self.perform_mean_correction:
                        self_X_mean = (
                            self_X_mean * self_X_size / total_X_size
                            + other_X_mean * other_X_size / total_X_size
                        )
                    self_X = torch.einsum("k,kg->kg", S_k, V_kg).contiguous()
                    self_X_size = total_X_size
            else:  # trailing rank
                # send to a leading rank and exit
                dst = (rank - 1) * 2**i
                dist.send(self_X, dst=dst)
                dist.send(self_X_size, dst=dst)
                if self.perform_mean_correction:
                    dist.send(self_X_mean, dst=dst)
                break
            # update rank, world_size, and level i
            rank = rank // 2
            world_size = math.ceil(world_size / 2)
            i += 1
        else:
            assert trainer.global_rank == 0
            self.V_kg = V_kg
            self.S_k = S_k
            self.x_size = self_X_size
            if self.perform_mean_correction:
                self.x_mean_g = self_X_mean

    @property
    def explained_variance_k(self) -> torch.Tensor:
        r"""
        The amount of variance explained by each of the selected components. The variance
        estimation uses ``x_size`` degrees of freedom.

        Equal to ``k_components`` largest eigenvalues of the covariance matrix of input data.
        """
        return self.S_k**2 / self.x_size

    @property
    def components_kg(self) -> torch.Tensor:
        r"""
        Principal axes in feature space, representing the directions of maximum variance
        in the data. Equivalently, the right singular vectors of the centered input data,
        parallel to its eigenvectors. The components are sorted by decreasing ``explained_variance_k``.
        """
        return self.V_kg

    def predict(self, x_ng: torch.Tensor) -> torch.Tensor:
        r"""
        Centering and embedding of the input data ``x_ng`` into the principal component space.
        """
        if self.transform is not None:
            x_ng = self.transform(x_ng)
        return (x_ng - self.x_mean_g) @ self.V_kg.T
