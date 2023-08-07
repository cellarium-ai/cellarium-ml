# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import torch
import torch.nn as nn

from scvid.module import BaseModule, PredictMixin


class IncrementalPCA(BaseModule, PredictMixin):
    """
    Incremental PCA.

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
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

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
            SV_kg = torch.einsum("k,kg->kg", S_k, V_gk.T)
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
