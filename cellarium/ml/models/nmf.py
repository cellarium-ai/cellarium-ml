# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

# from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def initialize_centroids(self, X):
        # KMeans++ initialization
        indices = [torch.randint(0, X.size(0), (1,)).item()]
        for _ in range(1, self.n_clusters):
            dist_sq = torch.min(torch.cdist(X, X[indices]) ** 2, dim=1)[0]
            probs = dist_sq / torch.sum(dist_sq)
            cumulative_probs = torch.cumsum(probs, dim=0)
            r = torch.rand(1).item()
            next_index = torch.searchsorted(cumulative_probs, r).item()
            indices.append(next_index)
        self.centroids = X[indices]

    def fit(self, X):

        if self.centroids is None:
            self.initialize_centroids(X)

        for i in range(self.max_iter):
            # Assignment Step: Assign each data point to the nearest centroid
            distances = torch.cdist(X, self.centroids)
            labels = torch.argmin(distances, dim=1)

            # Update Step: Calculate new centroids
            new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(self.n_clusters)])

            # Check for convergence
            if torch.all(torch.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)


def euclidean(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The `Euclidean distance
    .. math::
        \ell(x, y) = \frac{1}{2} \sum_{n = 0}^{N - 1} (x_n - y_n)^2

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    return F.mse_loss(input, target, reduction='sum') * 0.5


def consensus(D_rkg=None, k=10, density_threshold=0.25, local_neighborhood_size=0.30):
    R, num_component, g = D_rkg.shape
    D = F.normalize(D_rkg, dim=2, p=2)
    D = D.reshape(R * num_component, g)
    L = int(R * local_neighborhood_size)

    euc_dist = torch.cdist(D, D, p=2)
    L_nearest_neigh, _ = torch.topk(euc_dist, L + 1, largest=False)
    local_neigh_dist = L_nearest_neigh.sum(1) / L

    D = D[local_neigh_dist < density_threshold, :]
    # D = pd.DataFrame(D.cpu().numpy())

    D_mean = D.mean(0)
    D_norm = D - D_mean

    # kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
    # kmeans.fit(D_norm)
    # kmeans_cluster_labels = pd.Series(kmeans.labels_+1, index=D.index)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(D_norm)
    kmeans_cluster_labels = kmeans.predict(D)

    D = pd.DataFrame(D.cpu().numpy())
    kmeans_cluster_labels = kmeans_cluster_labels.cpu().numpy()

    median_D = D.groupby(kmeans_cluster_labels).median()
    median_D = torch.Tensor(median_D.values)
    median_D = F.normalize(median_D, dim=1, p=1)

    return median_D


def get_final_alpha_wKL(x_ng, D, n_iterations):
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
    """
    _alpha_tol = 0.005
    alpha_nk = torch.zeros((x_ng.shape[0], D.shape[0]), requires_grad=True, device=D.device)
    alpha_buffer = alpha_nk.exp().clone()

    optimizer = torch.optim.AdamW([alpha_nk], lr=0.2)

    for _ in range(n_iterations):

        optimizer.zero_grad()

        alpha_nk_exp = alpha_nk.exp()
        loss = euclidean(torch.matmul(alpha_nk_exp, D), x_ng)
        loss = loss.mul(2).sqrt()

        loss.backward()
        optimizer.step()

        alpha_diff = torch.linalg.norm(alpha_nk.exp() - alpha_buffer) / torch.linalg.norm(alpha_nk.exp())
        if alpha_diff <= _alpha_tol:
            break
        alpha_buffer = alpha_nk.exp().clone()

    alpha = F.normalize(alpha_nk.exp(), dim=1, p=1)

    return alpha.detach()


def get_full_D(x_ng, alpha_nk, A_kk, B_kg, factors_kg, n_iterations):
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
    """
    _D_tol = 0.005

    n, k_dimension = alpha_nk.shape

    A_kk = A_kk + torch.matmul(alpha_nk.T, alpha_nk) / n
    B_kg = B_kg + torch.matmul(alpha_nk.T, x_ng) / n

    D_buffer = factors_kg.clone()

    for _ in range(n_iterations):
        for k in range(k_dimension):
            scalar = A_kk[k, k]
            a_1k = A_kk[k, :]
            b_1g = B_kg[k, :]

            # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
            u_1g = torch.clamp(
                factors_kg[k, :] + (b_1g - torch.matmul(a_1k, factors_kg)) / scalar,
                min=0.0,
            )

            factors_1g = u_1g / torch.clamp(torch.linalg.norm(u_1g), min=1.0)
            factors_kg[k, :] = factors_1g

        D_diff = torch.linalg.norm(factors_kg - D_buffer) / torch.linalg.norm(factors_kg)
        if D_diff <= _D_tol:
            break
        D_buffer = factors_kg.clone()

    return A_kk, B_kg, factors_kg


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class NonNegativeMatrixFactorization(CellariumModel, PredictMixin):
    """
    Use the online NMF algorithm of Mairal et al. [1] to factorize the count matrix
    into a dictionary of gene expression programs and a matrix of cell program loadings.

    **References:**

    1. `Online learning for matrix factorization and sparse coding. Mairal, Bach, Ponce, Sapiro. JMLR 2009.

    Args:
        var_names_g: The variable names schema for the input data validation.
        k: The number of gene expression programs to infer.
        algorithm: The algorithm to use for the online NMF. Currently only "mairal" is supported.
    """

    def __init__(self, var_names_g: Sequence[str], k: int, r: int,
                 density_threshold: float, local_neighborhood_size: float,
                 log_variational: bool,
                 algorithm: Literal["mairal"] = "mairal") -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        g = len(self.var_names_g)
        self.n_vars = g
        self.algorithm = algorithm
        self.log_variational = log_variational

        self.A_rkk: torch.Tensor
        self.B_rkg: torch.Tensor
        self.D_rkg: torch.Tensor
        self.D_kg: torch.Tensor
        self.register_buffer("A_rkk", torch.empty(r, k, k))
        self.register_buffer("B_rkg", torch.empty(r, k, g))
        self.register_buffer("D_rkg", torch.empty(r, k, g))
        self.register_buffer("D_kg", torch.empty(k, g))
        self._dummy_param = torch.nn.Parameter(torch.empty(()))

        self.full_A_kk: torch.Tensor
        self.full_B_kg: torch.Tensor
        self.full_D_kg: torch.Tensor
        self.register_buffer("full_A_kk", torch.empty(k, k))
        self.register_buffer("full_B_kg", torch.empty(k, g))
        self.register_buffer("full_D_kg", torch.empty(k, g))

        self._D_tol = 0.005  # 0.01 #
        self._alpha_tol = 0.005  # 0.01 #

        self.k = k
        self.n_nmf = r
        self.density_threshold = density_threshold
        self.local_neighborhood_size = local_neighborhood_size

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.D_kg.zero_()
        self.A_rkk.zero_()
        self.B_rkg.zero_()
        self.D_rkg.uniform_(0.0, 2.0)  # TODO: figure out best initialization
        self._dummy_param.data.zero_()

        self.full_A_kk.zero_()
        self.full_B_kg.zero_()
        self.full_D_kg.uniform_(0.0, 2.0)  # TODO: figure out best initialization

    def online_dictionary_learning(self, x_ng: torch.Tensor, factors_kg: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        """

        n, g = x_ng.shape
        r = factors_kg.shape[0]
        k = factors_kg.shape[1]

        # updata alpha
        alpha_rnk = torch.zeros((r, n, k), requires_grad=True, device=factors_kg.device)
        alpha_rnk = self.solve_alpha_wKL(alpha_rnk, x_ng, 100)

        # update D
        self.A_rkk = self.A_rkk + torch.bmm(alpha_rnk.transpose(1, 2), alpha_rnk) / n
        self.B_rkg = self.B_rkg + torch.bmm(alpha_rnk.transpose(1, 2), x_ng.expand(r, n, g)) / n
        updated_factors_kg = self.dictionary_update_3d(factors_kg, 100)

        return updated_factors_kg

    def solve_alpha_wKL(self, alpha_rnk: torch.nn.Parameter,
                        x_ng: torch.Tensor,
                        n_iterations: int) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """

        r = alpha_rnk.shape[0]
        n, g = x_ng.shape

        alpha_buffer = alpha_rnk.exp().clone()

        optimizer = torch.optim.AdamW([alpha_rnk], lr=0.2)

        for _ in range(n_iterations):

            optimizer.zero_grad()

            alpha_rnk_exp = alpha_rnk.exp()

            loss = euclidean(torch.bmm(alpha_rnk_exp, self.D_rkg), x_ng.expand(r, n, g))
            loss = loss.mul(2).sqrt()

            loss.backward()
            optimizer.step()

            alpha_diff = torch.linalg.norm(
                alpha_rnk.exp().view(-1, self.k) - alpha_buffer.view(-1, self.k)
            ) / torch.linalg.norm(alpha_rnk.exp().view(-1, self.k))
            if alpha_diff <= self._alpha_tol:
                break
            alpha_buffer = alpha_rnk.exp().clone()

        return alpha_rnk.exp().detach()

    def dictionary_update_3d(self, factors_kg: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """
        D_buffer = factors_kg.clone()
        updated_factors_kg = factors_kg.clone()

        for _ in range(n_iterations):
            for k in range(self.k):
                scalar = self.A_rkk[:, k, k].view(self.n_nmf, 1, 1)
                a_1k = self.A_rkk[:, k, :].unsqueeze(1)
                b_1g = self.B_rkg[:, k, :].unsqueeze(1)

                # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
                u_1g = torch.clamp(
                    updated_factors_kg[:, k, :].unsqueeze(1)
                    + (b_1g - torch.bmm(a_1k, updated_factors_kg)) / scalar,
                    min=0.0,
                )

                u_1g_reshape = u_1g.squeeze(1)
                u_1g_reshape = torch.linalg.norm(u_1g_reshape, dim=1)

                updated_factors_1g = u_1g / torch.clamp(u_1g_reshape.view(self.n_nmf, 1, 1), min=1.0)
                updated_factors_kg[:, k, :] = updated_factors_1g.squeeze(1)

            D_diff = torch.linalg.norm(
                updated_factors_kg.view(-1, self.n_vars) - D_buffer.view(-1, self.n_vars)
            ) / torch.linalg.norm(updated_factors_kg.view(-1, self.n_vars))
            if D_diff <= self._D_tol:
                break
            D_buffer = updated_factors_kg.clone()

        return updated_factors_kg

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        if self.log_variational:
            x_ = torch.log1p(x_ng)
        else:
            std = torch.std(x_ng, dim=0) + 1e-4
            x_ = x_ng / std

        if self.algorithm == "mairal":
            self.D_rkg = self.online_dictionary_learning(x_ng=x_, factors_kg=self.D_rkg)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return {}

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(
                trainer.strategy, DDPStrategy
            ), "OnePassMeanVarStd requires that the trainer uses the DDP strategy."
            assert (
                    trainer.strategy._ddp_kwargs["broadcast_buffers"] is False
            ), "OnePassMeanVarStd requires that broadcast_buffers is set to False."

    def on_epoch_end(self, trainer: pl.Trainer) -> None:
        D_kg = consensus(D_rkg=self.D_rkg, k=self.k,
                         density_threshold=self.density_threshold,
                         local_neighborhood_size=self.local_neighborhood_size)

        # k, _ = D_kg.shape

        self.D_kg = D_kg  # [:k, :]

        trainer.save_checkpoint(trainer._default_root_dir / "consensus.ckpt")

    def predict(
            self,
            x_ng: torch.Tensor,
            var_names_g: np.ndarray,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Predict the gene expression programs for the given gene counts matrix.
        """

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        ## get the final alpha_nk
        if self.log_variational:
            x_ = torch.log1p(x_ng)
        else:
            std = torch.std(x_ng, dim=0) + 1e-4
            x_ = x_ng / std

        alpha_nk = get_final_alpha_wKL(x_ng=x_, D=self.D_kg, n_iterations=200)

        ## get the final D for full transcrptome
        x_ = torch.log1p(x_ng)
        A, B, D = get_full_D(x_, alpha_nk, self.full_A_kk, self.full_B_kg, self.full_D_kg, 200)

        self.full_A_kk = A
        self.full_B_kg = B
        self.full_D_kg = D

        return {"alpha_nk": alpha_nk}

    @property
    def factors_kg(self) -> torch.Tensor:
        """
        Inferred gene expression programs (i.e. "factors").
        """
        return self.D_rkg
