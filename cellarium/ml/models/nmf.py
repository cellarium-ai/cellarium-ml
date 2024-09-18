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

from cellarium.ml.transforms import Filter
from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

# from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings

warnings.filterwarnings("ignore")


class KMeans:
    def __init__(self, n_clusters, max_iter=200, tol=1e-4):
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
    kmeans_cluster_labels = kmeans.predict(D_norm)

    D = pd.DataFrame(D.cpu().numpy())
    kmeans_cluster_labels = kmeans_cluster_labels.cpu().numpy()

    silhouette = silhouette_score(D.values, kmeans_cluster_labels,
                                  metric='euclidean')
    print("silhouette score: " + str(round(silhouette, 4)))

    median_D = D.groupby(kmeans_cluster_labels).median()
    median_D = torch.Tensor(median_D.values)
    median_D = F.normalize(median_D, dim=1, p=1)

    return median_D

# @torch.enable_grad()
# def get_final_alpha_wKL(x_ng, D, n_iterations):
#     """
#     Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

#     Args:
#         factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
#         n_iterations: The number of iterations to perform.
#     """
#     _alpha_tol = 0.005
#     alpha_nk = torch.zeros((x_ng.shape[0], D.shape[0]), requires_grad=True, device=D.device)
#     alpha_buffer = alpha_nk.exp().clone()

#     optimizer = torch.optim.AdamW([alpha_nk], lr=0.01)

#     for _ in range(n_iterations):

#         optimizer.zero_grad()

#         alpha_nk_exp = alpha_nk.exp()
#         loss = euclidean(torch.matmul(alpha_nk_exp, D), x_ng)
#         loss = loss.mul(2).sqrt()

#         loss.backward()
#         optimizer.step()

#         alpha_diff = torch.linalg.norm(alpha_nk.exp() - alpha_buffer) / torch.linalg.norm(alpha_nk.exp())
#         if alpha_diff <= _alpha_tol:
#             break
#         alpha_buffer = alpha_nk.exp().clone()

#     alpha = F.normalize(alpha_nk.exp(), dim=1, p=1)

#     return alpha.detach()


@torch.enable_grad()
def solve_alpha_wKL( 
    alpha_rnk: torch.nn.Parameter,
    D_rkg: torch.Tensor,
    x_ng: torch.Tensor,
    n_iterations: int,
    alpha_tol: float,
) -> torch.Tensor:
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
    """

    r = alpha_rnk.shape[0]
    k = alpha_rnk.shape[-1]
    n, g = x_ng.shape

    alpha_buffer = alpha_rnk.exp().clone()

    optimizer = torch.optim.AdamW([alpha_rnk], lr=0.2)

    for _ in range(n_iterations):

        optimizer.zero_grad()

        alpha_rnk_exp = alpha_rnk.exp()

        loss = euclidean(torch.bmm(alpha_rnk_exp, D_rkg), x_ng.expand(r, n, g))
        loss = loss.mul(2).sqrt()

        loss.backward()
        optimizer.step()

        alpha_diff = torch.linalg.norm(
            alpha_rnk.exp().view(-1, k) - alpha_buffer.view(-1, k)
        ) / torch.linalg.norm(alpha_rnk.exp().view(-1, k))
        if alpha_diff <= alpha_tol:
            break
        alpha_buffer = alpha_rnk.exp().clone()

    return alpha_rnk.exp().detach()


def dictionary_update(
    D_rkg: torch.Tensor,
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    n_iterations: int = 1,
    D_tol: float = 0.005,
) -> torch.Tensor:
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
    """
    D_buffer = D_rkg.clone()
    updated_factors_rkg = D_rkg.clone()
    r, k, g = D_rkg.shape

    for _ in range(n_iterations):
        for k in range(k):
            scalar_r11 = A_rkk[:, k, k].view(r, 1, 1)
            a_r1k = A_rkk[:, k, :].unsqueeze(1)
            b_r1g = B_rkg[:, k, :].unsqueeze(1)

            # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
            u_r1g = torch.clamp(
                updated_factors_rkg[:, k, :].unsqueeze(1)
                + (b_r1g - torch.bmm(a_r1k, updated_factors_rkg)) / scalar_r11,
                min=0.0,
            )

            u_rg = u_r1g.squeeze(1)
            u_rg = torch.linalg.norm(u_rg, dim=1)

            updated_factors_r1g = u_r1g / torch.clamp(u_rg.view(r, 1, g), min=1.0)
            updated_factors_rkg[:, k, :] = updated_factors_r1g.squeeze(1)

        D_diff = torch.linalg.norm(
            updated_factors_rkg.view(-1, g) - D_buffer.view(-1, g)
        ) / torch.linalg.norm(updated_factors_rkg.view(-1, g))
        if D_diff <= D_tol:
            break
        D_buffer = updated_factors_rkg.clone()

    return updated_factors_rkg


# def get_final_alpha(x_ng, D, n_iterations):
#     """
#     Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

#     Args:
#         factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
#         n_iterations: The number of iterations to perform.
#     """

#     _alpha_tol = 0.005
#     alpha_nk = torch.zeros((x_ng.shape[0], D.shape[0]), device=D.device)
#     alpha_buffer = alpha_nk.clone()

#     k_dimension = alpha_nk.shape[1]

#     DDT_kk = torch.matmul(D, D.T)
#     DXT_kn = torch.matmul(D, x_ng.T)

#     for _ in range(n_iterations):
#         for k in range(k_dimension):
#             scalar = DDT_kk[k, k]
#             a_1k = DDT_kk[k, :]
#             b_1g = DXT_kn[k, :]

#             u_1g = torch.clamp(
#                 alpha_nk[:, k] + (b_1g - torch.matmul(a_1k, alpha_nk.T)) / scalar,
#                 min=0.0,
#             )

#             # u_1g = u_1g / torch.clamp(torch.linalg.norm(u_1g), min=1.0)
#             alpha_nk[:, k] = u_1g

#         alpha_diff = torch.linalg.norm(alpha_nk - alpha_buffer) / torch.linalg.norm(alpha_nk)
#         if alpha_diff <= _alpha_tol:
#             break
#         alpha_buffer = alpha_nk.clone()

#     alpha = F.normalize(alpha_nk, dim=1, p=1)

#     return alpha_nk.detach()


# def get_full_D(x_ng, alpha_nk, A_kk, B_kg, factors_kg, n_iterations):
#     """
#     Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

#     Args:
#         factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
#         n_iterations: The number of iterations to perform.
#     """
#     _D_tol = 0.005

#     n, k_dimension = alpha_nk.shape

#     A_kk = A_kk + torch.matmul(alpha_nk.T, alpha_nk) / n
#     B_kg = B_kg + torch.matmul(alpha_nk.T, x_ng) / n

#     D_buffer = factors_kg.clone()

#     for _ in range(n_iterations):
#         for k in range(k_dimension):
#             scalar = A_kk[k, k]
#             a_1k = A_kk[k, :]
#             b_1g = B_kg[k, :]

#             # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
#             u_1g = torch.clamp(
#                 factors_kg[k, :] + (b_1g - torch.matmul(a_1k, factors_kg)) / scalar,
#                 min=0.0,
#             )

#             factors_1g = u_1g / torch.clamp(torch.linalg.norm(u_1g), min=1.0)
#             factors_kg[k, :] = factors_1g

#         D_diff = torch.linalg.norm(factors_kg - D_buffer) / torch.linalg.norm(factors_kg)
#         if D_diff <= _D_tol:
#             break
#         D_buffer = factors_kg.clone()

#     return A_kk, B_kg, factors_kg


# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find("BatchNorm") != -1:
#         torch.nn.init.normal_(m.weight, 1.0, 0.02)
#         torch.nn.init.zeros_(m.bias)
#     elif classname.find("Linear") != -1:
#         torch.nn.init.xavier_normal_(m.weight)
#         torch.nn.init.zeros_(m.bias)


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

    def __init__(
        self, 
        var_names_schema: Sequence[str],  # all genes in the data: i.e. the "schema"
        var_names_g: Sequence[str],  # genes used for fitting
        k: int, 
        r: int, 
        # full_g: int,
        density_threshold: float, 
        local_neighborhood_size: float,
        log_variational: bool, 
        algorithm: Literal["mairal"] = "mairal",
    ) -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.var_names_schema = np.array(var_names_schema)
        g = len(self.var_names_g)
        self.n_vars = g
        self.algorithm = algorithm
        self.log_variational = log_variational
        # self.full_g = full_g

        self.A_rkk: torch.Tensor
        self.B_rkg: torch.Tensor
        self.D_rkg: torch.Tensor
        self.D_kg: torch.Tensor
        self.register_buffer("A_rkk", torch.empty(r, k, k))
        self.register_buffer("B_rkg", torch.empty(r, k, g))
        self.register_buffer("D_rkg", torch.empty(r, k, g))
        self.register_buffer("D_kg", torch.empty(k, g))
        self._dummy_param = torch.nn.Parameter(torch.empty(()))

        # self.full_A_kk: torch.Tensor
        # self.full_B_kg: torch.Tensor
        # self.full_D_kg: torch.Tensor
        # self.register_buffer("full_A_kk", torch.empty(k, k))
        # self.register_buffer("full_B_kg", torch.empty(k, full_g))
        # self.register_buffer("full_D_kg", torch.empty(k, full_g))

        self._D_tol = 0.005
        self._alpha_tol = 0.005

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

        # self.full_A_kk.zero_()
        # self.full_B_kg.zero_()
        # self.full_D_kg.uniform_(0.0, 2.0)  # TODO: figure out best initialization

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
        alpha_rnk = solve_alpha_wKL(
            alpha_rnk=alpha_rnk, 
            x_ng=x_ng, 
            D_rkg=self.D_rkg,
            n_iterations=100,
            alpha_tol=self._alpha_tol,
        )

        # update D
        with torch.no_grad():
            self.A_rkk = self.A_rkk + torch.bmm(alpha_rnk.transpose(1, 2), alpha_rnk) / n
            self.B_rkg = self.B_rkg + torch.bmm(alpha_rnk.transpose(1, 2), x_ng.expand(r, n, g)) / n
            updated_factors_kg = dictionary_update(
                D_rkg=factors_kg.unsqueeze(0),  # is this faster? .view(1, k, g)
                A_rkk=self.A_rkk,
                B_rkg=self.B_rkg,
                n_iterations=100,
                D_tol=self._D_tol,
            ).squeeze()

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

        # TODO: make variance scaling a real "Transform" class and move these to the config file transforms
        if self.log_variational:
            x_ = torch.log1p(x_ng)
        else:
            std = torch.std(x_ng, dim=0) + 1e-4
            x_ = x_ng / std
            x_ = torch.clamp(x_, min=0.0, max=100.0)
        # TODO ===========

        if self.algorithm == "mairal":
            self.D_rkg = self.online_dictionary_learning(x_ng=x_, factors_kg=self.D_rkg)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return {}

    # def on_train_start(self, trainer: pl.Trainer) -> None:
    #     if trainer.world_size > 1:
    #         assert isinstance(
    #             trainer.strategy, DDPStrategy
    #         ), "OnePassMeanVarStd requires that the trainer uses the DDP strategy."
    #         assert (
    #                 trainer.strategy._ddp_kwargs["broadcast_buffers"] is False
    #         ), "OnePassMeanVarStd requires that broadcast_buffers is set to False."

    def on_end(self, trainer: pl.Trainer) -> None:
        D_kg = consensus(D_rkg=self.D_rkg, k=self.k,
                         density_threshold=self.density_threshold,
                         local_neighborhood_size=self.local_neighborhood_size)

        self.D_kg = D_kg

        if self.mode == 'nmf':
            trainer.save_checkpoint(trainer._default_root_dir + "/NMF.ckpt")
        else:
            trainer.save_checkpoint(trainer._default_root_dir + "/consensusNMF.ckpt")

    def on_prediction_end(self, trainer: pl.Trainer) -> None:

        trainer.save_checkpoint(trainer._default_root_dir + "/consensusNMF_predict.ckpt")

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

        transform = Filter(self.var_names_hvg)
        x_filtered_ng = transform(x_ng, var_names_g)

        ## get the final alpha_nk
        if self.log_variational:
            x_ = torch.log1p(x_filtered_ng['x_ng'])
        else:
            std = torch.std(x_filtered_ng['x_ng'], dim=0) + 1e-4
            x_ = x_filtered_ng['x_ng'] / std
            x_ = torch.clamp(x_, min=0.0, max=100.0)

        alpha_nk = get_final_alpha_wKL(x_ng=x_, D=self.D_kg, n_iterations=1000)

        # Compute prediction error as a frobenius norm
        rf_pred = torch.matmul(alpha_nk, self.D_kg)

        return {"alpha_nk": alpha_nk, "pred_count": rf_pred}

    @property
    def factors_kg(self) -> torch.Tensor:
        """
        Inferred gene expression programs (i.e. "factors").
        """
        return self.D_rkg


class NonNegativeMatrixRefit(NonNegativeMatrixFactorization):

    def __init__(
        self, 
        trained_factors_kg: torch.Tensor,
        trained_var_names_g: Sequence[str],  # genes used for fitting trained_factors_kg
        var_names_g: Sequence[str],  # the genes to be refit
        log_variational: bool,  # TODO: factor out to config transforms same as above
        algorithm: Literal["mairal"] = "mairal",
    ) -> None:
        super().__init__()
        self.trained_var_names_g = np.array(trained_var_names_g)
        self.var_names_g = np.array(var_names_g)
        self.algorithm = algorithm
        self.log_variational = log_variational

        self.trained_factors_kg = trained_factors_kg

        g = len(self.var_names_g)
        k = trained_factors_kg.shape[0]

        self.A_kk: torch.Tensor
        self.B_kg: torch.Tensor
        self.D_kg: torch.Tensor
        self.register_buffer("A_kk", torch.empty(k, k))
        self.register_buffer("B_kg", torch.empty(k, g))
        self.register_buffer("D_kg", torch.empty(k, g))
        self._dummy_param = torch.nn.Parameter(torch.empty(()))

        self._D_tol = 0.005
        self._alpha_tol = 0.005

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.D_kg.uniform_(0.0, 2.0)  # TODO: figure out best initialization
        self.A_kk.zero_()
        self.B_kg.zero_()
        self._dummy_param.data.zero_()

    def forward(
        self,
        x_ng: torch.Tensor,  # already filtered to the desired output genes: must be a superset of trained_var_names_g
        var_names_g: np.ndarray,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Refit new gene expression programs for the given gene counts matrix, based on the trained factors.
        """

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert var_names_g == self.var_names_g, "var_names_g must match the genes to be refit"
        assert set(var_names_g).issuperset(self.trained_var_names_g), "var_names_g must be a superset of trained_var_names_g"

        n = x_ng.shape[0]
        filter_to_trained_genes = Filter(self.trained_var_names_g)
        x_filtered_ng = filter_to_trained_genes(x_ng, var_names_g)

        # TODO: factor this filtering out to the config file transforms
        ## get the final alpha_nk
        if self.log_variational:
            x_ = torch.log1p(x_filtered_ng['x_ng'])
        else:
            std = torch.std(x_filtered_ng['x_ng'], dim=0) + 1e-4
            x_ = x_filtered_ng['x_ng'] / std
            x_ = torch.clamp(x_, min=0.0, max=100.0)

        # with torch.set_grad_enabled(True):
        alpha_nk = torch.zeros((x_ng.shape[0], self.D_kg.shape[0]), requires_grad=True, device=self.D_kg.device)
        alpha_nk = solve_alpha_wKL(
            alpha_rnk=alpha_nk.unsqueeze(0), 
            x_ng=x_ng, 
            D_rkg=self.D_kg.unsqueeze(0),
            n_iterations=100,
            alpha_tol=self._alpha_tol,
        ).squeeze()
        # alpha_nk = get_final_alpha_wKL(x_ng=x_, D=self.trained_factors_kg, n_iterations=1000)

        # Compute prediction error as a frobenius norm
        rf_pred = torch.matmul(alpha_nk, self.trained_factors_kg)
        # prediction_error = ((x_ - rf_pred)**2).sum().sum()

        # TODO: preprocessing we want to move to config file transforms?
        x_ng = (x_ng.T / x_ng.sum(1)).T * 1e4
        x_ = torch.log1p(x_ng)

        # get the final D for full transcrptome
        self.A_kk = self.A_kk + torch.matmul(alpha_nk.T, alpha_nk) / n
        self.B_kg = self.B_kg + torch.matmul(alpha_nk.T, x_ng) / n
        self.D_kg = dictionary_update(
            D_rkg=self.D_kg.unsqueeze(0),
            A_rkk=self.A_kk.unsqueeze(0),
            B_rkg=self.B_kg.unsqueeze(0),
            n_iterations=100,
            D_tol=self._D_tol,
        ).squeeze()

        return {"alpha_nk": alpha_nk, "pred_count": rf_pred}
    
    def predict(
        self,
        x_ng: torch.Tensor,  # already filtered to the desired output genes: must be a superset of trained_var_names_g
        var_names_g: np.ndarray,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        
        return self.forward(x_ng, var_names_g)
