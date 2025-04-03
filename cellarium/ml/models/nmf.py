# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Sequence
from typing import Literal

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, IterableDataset

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.transforms import Filter
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

warnings.filterwarnings("ignore")


def calculate_rec_error(
    dataset: DataLoader | IterableDataset,
    pipeline,  #: "CellariumPipeline",
) -> anndata.AnnData:
    """
    Compute reconstruction error using the pipeline.

    Args:
        dataset: Dataset.
        pipeline: Pipeline.

    Returns:
        reconstruction error
    """

    k_values = pipeline[-1].k_values
    pipeline[-1].get_rec_error = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rec_error = torch.zeros((len(k_values)))
    for batch in tqdm.tqdm(dataset):
        batch["x_ng"] = torch.from_numpy(batch["x_ng"]).to(device)
        out = pipeline.predict(batch)
        rec_error += out["rec_error"]

    return rec_error


def get_embedding(
    dataset: DataLoader | IterableDataset,
    pipeline,  #: "CellariumPipeline",
    k: int,
    if_get_final_gene_loading: bool = True,
) -> pd.DataFrame:
    """
    Helper function to embed the dataset using the pipeline.

    Args:
        dataset: Dataset.
        pipeline: Pipeline.
        k: select the K to get cell embedding.

    Returns:
        pd.DataFrame with cell embeddings indexed by adata.obs_names from dataset.
    """

    pipeline[-1].if_get_full_D = if_get_final_gene_loading
    pipeline[-1].get_rec_error = False
    pipeline[-1].the_best_k = k
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding = []
    index = []
    for batch in tqdm.tqdm(dataset):
        if isinstance(batch["x_ng"], np.ndarray):
            batch["x_ng"] = torch.from_numpy(batch["x_ng"]).to(device)
        out = pipeline.predict(batch)
        z = out["alpha_nk"]
        embedding += [z.cpu()]
        index.extend(batch["obs_names_n"])

    return pd.DataFrame(torch.cat(embedding).numpy(), index=index)


class KMeans:
    def __init__(self, n_clusters, max_iter=200, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: torch.Tensor | None = None

    def initialize_centroids(self, x_np: torch.Tensor):
        # KMeans++ initialization
        first_random_index = int(torch.randint(0, x_np.shape[0], (1,)).item())
        indices: list[int] = [first_random_index]
        for _ in range(1, self.n_clusters):
            square_distance_n = torch.min(torch.cdist(x_np, x_np[indices], p=2) ** 2, dim=1)[0]
            probs_n = square_distance_n / torch.sum(square_distance_n)
            # cumulative_probs_n = torch.cumsum(probs_n, dim=0)
            # r = torch.rand(1).item()
            # next_index = torch.searchsorted(cumulative_probs_n, r).item()
            next_index = int(torch.multinomial(probs_n, 1).item())
            indices.append(next_index)

        self.centroids = x_np[indices]

    def fit(self, x_ng: torch.Tensor):
        if self.centroids is None:
            self.initialize_centroids(x_ng)
            assert isinstance(self.centroids, torch.Tensor)

        for i in range(self.max_iter):
            # Assignment Step: Assign each data point to the nearest centroid
            distances = torch.cdist(x_ng, self.centroids)
            labels = torch.argmin(distances, dim=1)

            # Update Step: Calculate new centroids
            new_centroids = torch.stack([x_ng[labels == k].mean(dim=0) for k in range(self.n_clusters)])

            # Check for convergence
            if torch.all(torch.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, x_ng: torch.Tensor):
        assert isinstance(self.centroids, torch.Tensor)
        distances = torch.cdist(x_ng, self.centroids)
        return torch.argmin(distances, dim=1)


def nmf_frobenius_loss(x: torch.Tensor, loadings_nk: torch.Tensor, factors_kg: torch.Tensor):
    # compute prediction error as the frobenius norm
    return F.mse_loss(torch.matmul(loadings_nk, factors_kg), x, reduction="sum")


def consensus(D_rkg: torch.Tensor, k: int, density_threshold: float, local_neighborhood_size: float):
    r, num_component, g = D_rkg.shape
    d_norm_rkg = F.normalize(D_rkg, dim=2, p=2)
    d_norm_mg = d_norm_rkg.reshape(r * num_component, g)

    if r > 1:
        L = int(r * local_neighborhood_size)
        euc_dist = torch.cdist(d_norm_mg, d_norm_mg, p=2)
        L_nearest_neigh, _ = torch.topk(euc_dist, L + 1, largest=False)
        local_neigh_dist = L_nearest_neigh.sum(1) / L

        topk_euc_dist = euc_dist[local_neigh_dist < density_threshold, :]
        topk_euc_dist = topk_euc_dist[:, local_neigh_dist < density_threshold]
        d_norm_mg = d_norm_mg[local_neigh_dist < density_threshold, :]

        # mean centering prevents underflow in distance computations in the kmeans algorithm
        d_norm_mg_mean = d_norm_mg.mean(0)
        d_norm_mg = d_norm_mg - d_norm_mg_mean

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(d_norm_mg)
        kmeans_cluster_labels = kmeans.predict(d_norm_mg)
        df_d_norm_mg = pd.DataFrame(d_norm_mg.cpu().numpy())
        kmeans_cluster_labels = kmeans_cluster_labels.cpu().numpy()

        silhouette = silhouette_score(df_d_norm_mg.values, kmeans_cluster_labels, metric="cosine")

        median_D = df_d_norm_mg.groupby(kmeans_cluster_labels).median()
        median_D = torch.Tensor(median_D.values)
        median_D = F.normalize(median_D, dim=1, p=1)

    else:
        topk_euc_dist = None
        local_neigh_dist = None
        median_D = F.normalize(d_norm_mg, dim=1, p=1)
        silhouette = 1.0

    return {
        "topk_euc_dist": topk_euc_dist,
        "local_neigh_dist": local_neigh_dist,
        "consensus_D": median_D,
        "stability": silhouette,
    }


def compute_loadings(
    x_ng: torch.Tensor,
    factors_rkg: torch.Tensor,
    n_iterations: int,
    learning_rate: float = 0.2,
    normalize: bool = False,
    alpha_tol: float = 5e-5,
) -> torch.Tensor:
    """
    Algorithm 1 step 4 from Mairal et al. [1] for computing the loadings.

    Args:
        x_ng: The data.
        factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The max number of iterations to perform.
        learning_rate: The learning rate for the optimizer.
        normalize: Whether to normalize the output.
        alpha_tol: The tolerance for the change in alpha for stopping.

    Returns:
        The computed loadings.
    """
    n, _ = x_ng.shape
    r, k, _ = factors_rkg.shape
    alpha_unconstrained_rnk = torch.nn.Parameter(
        torch.zeros((r, n, k), requires_grad=True, device=factors_rkg.device)  # TODO: try others
    )
    alpha_init_rnk = alpha_unconstrained_rnk.exp().clone()
    alpha_buffer_rnk = alpha_init_rnk.clone()

    optimizer = torch.optim.Adam([alpha_unconstrained_rnk], lr=learning_rate)

    for i in range(n_iterations):
        optimizer.zero_grad()

        # TODO: use the nmf_frobenius_loss function
        loss = (
            F.mse_loss(
                torch.bmm(alpha_unconstrained_rnk.exp(), factors_rkg),
                x_ng.unsqueeze(0),
                reduction="sum",
            )
            * 0.5
        )

        loss.backward()
        optimizer.step()
        alpha_rnk = alpha_unconstrained_rnk.exp()

        alpha_max_diff = (
            F.mse_loss(alpha_rnk, alpha_buffer_rnk, reduction="none").sum(dim=[-2, -1])
            / alpha_init_rnk.square().sum(dim=[-2, -1])
        ).max()
        if alpha_max_diff <= alpha_tol:
            break
        alpha_buffer_rnk = alpha_rnk.clone()

    if normalize:
        alpha_rnk = F.normalize(alpha_rnk, dim=-1, p=1)

    return alpha_rnk.detach()


def compute_factors(
    A_rkk: torch.Tensor,
    B_rkg: torch.Tensor,
    factors_rkg: torch.Tensor,
    n_iterations: int,
    D_tol: float = 2e-5,
) -> torch.Tensor:
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        A_rkk: Mairal's matrix A.
        B_rkg: Mairal's matrix B.
        factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
        D_tol: The tolerance for the change in D for stopping.

    Returns:
        The updated dictionary.
    """
    factors_buffer_rkg = factors_rkg.clone()
    updated_factors_rkg = factors_rkg.clone()
    r, n_factors, g = factors_rkg.shape

    for i in range(n_iterations):
        for k in range(n_factors):
            scalar = A_rkk[:, k, k].view(r, 1, 1)
            a_r1k = A_rkk[:, k, :].unsqueeze(1)
            b_r1g = B_rkg[:, k, :].unsqueeze(1)

            # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
            u_r1g = torch.clamp(
                updated_factors_rkg[:, k, :].unsqueeze(1) + (b_r1g - torch.bmm(a_r1k, updated_factors_rkg)) / scalar,
                min=0.0,
            )

            updated_factors_r1g = u_r1g / torch.clamp(
                torch.linalg.vector_norm(u_r1g, ord=2, dim=-1, keepdim=True), min=1.0
            )
            updated_factors_rkg[:, k, :] = updated_factors_r1g.squeeze(1)

        D_max_diff = (
            F.mse_loss(updated_factors_rkg, factors_buffer_rkg, reduction="none").sum(dim=[-2, -1])
            / updated_factors_rkg.square().sum(dim=[-2, -1])
        ).max()
        if D_max_diff <= D_tol:
            break
        factors_buffer_rkg = updated_factors_rkg.clone()

    return updated_factors_rkg


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class NMFInit:
    @staticmethod
    def __call__(x: torch.Tensor, k: int, transformed_data_mean: float | None = None) -> None:
        """Modify the values of x in place as a way to initialize a dictionary factor matrix for NMF."""
        pass


class NMFInitSklearnRandom(NMFInit):
    @staticmethod
    def __call__(x: torch.Tensor, k: int, transformed_data_mean: float | None = None) -> None:
        """Modify the values of x in place according to the sklearn NMF init random recipe."""
        # https://github.com/scikit-learn/scikit-learn/blob/
        # 99bf3d8e4eed5ba5db19a1869482a238b6223ffd/sklearn/decomposition/_nmf.py#L304-L315
        assert transformed_data_mean is not None
        factor = np.sqrt(transformed_data_mean / k)
        x.normal_(0.0, factor).abs_()


class NMFInitUniformRandom(NMFInit):
    @staticmethod
    def __call__(x: torch.Tensor, k: int, transformed_data_mean: float | None = None) -> None:
        """Modify the values of x in place according to a Joshua Welch NMF init random recipe."""
        # https://www.nature.com/articles/s41587-021-00867-x#Sec10
        # algorithm 1 step 2
        x.uniform_(0.0, 2.0)


class LoadingsEncoder(torch.nn.Module):
    """
    Encode count data x_ng to loadings alpha_rnk, where r is the number of replicates of NMF and
    k is the number of gene expression programs. Makes r into a batch dimension, reshaping the input
    to shape (r, n, g).
    The output alpha_rnk is of shape (r, n, k).
    The encoder is a simple feedforward neural network with two linear layers and a ReLU activation function.
    The one-hot-encoded r is concatenated to the input x_ng in the g dimension, making the input
    shape (r, n, r + g), with the (r, n) dimensions as batch dimensions. This is critical to allow
    the replicates to be trained independently.
    The output entries are all >= 0 but the output is not normalized in the k dimension.
    """

    def __init__(self, g: int, r: int, k: int, hidden_size: int = 32):
        super().__init__()
        self.g = g
        self.r = r
        self.k = k
        self.linear1 = torch.nn.Linear(r + g, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, k)
        self.relu = torch.nn.ReLU()
        self.one_hot_r1r = torch.eye(r).unsqueeze(1)

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: The count data.

        Returns:
            The loadings alpha_rnk, where r is the replicate batch dimension,
            n is cells, and k is the factors.
        """
        n, g = x_ng.shape
        r = self.r
        x_rng = x_ng.view(1, n, g).expand(r, n, g)
        one_hot_rnr = self.one_hot_r1r.expand(r, n, r)
        x_rnm = torch.cat((one_hot_rnr, x_rng), dim=-1)
        h_rnh = self.linear1(x_rnm)
        h_rnh = self.relu(h_rnh)
        alpha_rnk = self.linear2(h_rnh)
        alpha_rnk = self.relu(alpha_rnk)
        return alpha_rnk


class NonNegativeMatrixFactorization(CellariumModel, PredictMixin):
    """
    Use the online NMF algorithm of Mairal et al. [1] to factorize the count matrix
    into a dictionary of gene expression programs and a matrix of cell program loadings.

    **References:**

    1. `Online learning for matrix factorization and sparse coding. Mairal, Bach, Ponce, Sapiro. JMLR 2009.

    Args:
        var_names_g: The variable names schema for the input data validation.
        k_values: The number of gene expression programs to infer.
        algorithm: The algorithm to use for the online NMF. Currently only "mairal" is supported.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        var_names_hvg: Sequence[str],
        k_values: list[int],
        r: int,
        full_g: int,
        log_variational: bool,
        algorithm: Literal["mairal", "amortized_mairal"] = "mairal",
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
    ) -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.var_names_hvg = np.array(var_names_hvg)
        g = len(self.var_names_hvg)
        self.n_vars = g
        self.algorithm = algorithm
        self.log_variational = log_variational
        self.full_g = full_g
        self.k_values = k_values
        self.the_best_k = k_values[0]  # default and has to be reassigned
        self.get_rec_error = True
        self.if_get_full_D = False
        self.transformed_data_mean = transformed_data_mean
        self.init = init
        if init == "sklearn_random":
            if transformed_data_mean is None:
                raise ValueError("transformed_data_mean must be provided when using the sklearn_random initialization")

        for i in self.k_values:
            self.register_buffer(f"A_{i}_rkk", torch.empty(r, i, i))
            self.register_buffer(f"B_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_kg", torch.empty(i, g))
            self._dummy_param = torch.nn.Parameter(torch.empty(()))

            self.register_buffer(f"full_A_{i}_kk", torch.empty(i, i))
            self.register_buffer(f"full_B_{i}_kg", torch.empty(i, g))
            self.register_buffer(f"full_D_{i}_kg", torch.empty(i, g))

        self._D_tol = 1e-5
        self._alpha_tol = 1e-5

        self.n_nmf = r

        if self.algorithm == "amortized_mairal":
            self.loadings_encoders = torch.nn.ModuleDict({str(k): LoadingsEncoder(g=g, r=r, k=k) for k in k_values})

        self.reset_parameters()

    def reset_parameters(self) -> None:
        match self.init:
            case "sklearn_random":
                init_fn: NMFInit = NMFInitSklearnRandom()
            case "uniform_random":
                init_fn = NMFInitUniformRandom()
            case _:
                raise ValueError(f"Unknown initialization method: {self.init}")

        for i in self.k_values:
            getattr(self, f"D_{i}_kg").zero_()

            getattr(self, f"A_{i}_rkk").zero_()
            getattr(self, f"B_{i}_rkg").zero_()
            init_fn(getattr(self, f"D_{i}_rkg"), k=i, transformed_data_mean=self.transformed_data_mean)

            getattr(self, f"full_A_{i}_kk").zero_()
            getattr(self, f"full_B_{i}_kg").zero_()
            init_fn(getattr(self, f"full_D_{i}_kg"), k=i, transformed_data_mean=self.transformed_data_mean)

        if self.loadings_encoder is not None:
            # TODO
            pass

    def _compute_loadings(self, x_ng: torch.Tensor, factors_rkg: torch.Tensor, n_iterations: int) -> torch.Tensor:
        """
        Run compute_loadings.

        Args:
            x_ng: The data.
            factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The max number of iterations to perform.

        Returns:
            The computed loadings.
        """
        normalize = False

        if self.algorithm == "mairal":
            alpha_rnk = compute_loadings(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                n_iterations=n_iterations,
                learning_rate=0.05,
                normalize=normalize,
                alpha_tol=self._alpha_tol,
            )
        elif self.algorithm == "amortized_mairal":
            _, k, _ = factors_rkg.shape
            alpha_rnk = self.loadings_encoders[str(k)](x_ng)
            if normalize:
                alpha_rnk = F.normalize(alpha_rnk, dim=-1, p=1)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        return alpha_rnk

    def _compute_factors(
        self,
        factors_rkg: torch.Tensor,
        A_rkk: torch.Tensor,
        B_rkg: torch.Tensor,
        n_iterations: int = 1,
    ) -> torch.Tensor:
        """
        Run compute_factors.

        Args:
            factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
            A_rkk: Mairal's matrix A.
            B_rkg: Mairal's matrix B.
            n_iterations: The number of iterations to perform.

        Returns:
            The updated dictionary.
        """
        factors_rkg = compute_factors(
            A_rkk=A_rkk,
            B_rkg=B_rkg,
            factors_rkg=factors_rkg,
            n_iterations=n_iterations,
            D_tol=self._D_tol,
        )
        return factors_rkg

    def online_dictionary_learning(
        self, x_ng: torch.Tensor, factors_rkg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            x_ng: The data.
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).

        Returns:
            The updated dictionary.
        """

        n, g = x_ng.shape
        r, k, _ = factors_rkg.shape

        # update alpha, Mairal Algorithm 1 step 4
        alpha_rnk = self._compute_loadings(
            x_ng=x_ng,
            factors_rkg=factors_rkg,
            n_iterations=200,
        )

        with torch.no_grad():
            # update A and B, Mairal Algorithm 1 step 5 and 6
            A_rkk = getattr(self, f"A_{k}_rkk")
            B_rkg = getattr(self, f"B_{k}_rkg")
            A_rkk = A_rkk + torch.bmm(alpha_rnk.transpose(1, 2), alpha_rnk) / n
            B_rkg = B_rkg + torch.bmm(alpha_rnk.transpose(1, 2), x_ng.expand(r, n, g)) / n
            setattr(self, f"A_{k}_rkk", A_rkk)
            setattr(self, f"B_{k}_rkg", B_rkg)

            # update D, Mairal Algorithm 1 step 7
            updated_factors_rkg = self._compute_factors(
                factors_rkg=factors_rkg,
                A_rkk=A_rkk,
                B_rkg=B_rkg,
                n_iterations=200,
            )

        return updated_factors_rkg, alpha_rnk

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
            std_g = torch.std(x_ng, dim=0) + 1e-4
            x_ = x_ng / std_g
            x_ = torch.clamp(x_, min=0.0, max=100.0)

        if self.algorithm in "mairal":
            for i in self.k_values:
                D_rkg = getattr(self, f"D_{i}_rkg")
                D_rkg, _ = self.online_dictionary_learning(x_ng=x_, factors_rkg=D_rkg)
                setattr(self, f"D_{i}_rkg", D_rkg)
            output = {}

        elif self.algorithm == "amortized_mairal":
            # TODO
            for i in self.k_values:
                D_rkg = getattr(self, f"D_{i}_rkg")
                D_rkg, alpha_rnk = self.online_dictionary_learning(x_ng=x_, factors_rkg=D_rkg)
                setattr(self, f"D_{i}_rkg", D_rkg)
            nmf_loss = nmf_frobenius_loss(
                x=x_,
                loadings_nk=alpha_rnk,
                factors_kg=D_rkg,
            )
            output = {"loss": nmf_loss}

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return output

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "NonNegativeMatrixFactorization requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is True, (
                "NonNegativeMatrixFactorization requires that the `broadcast_buffers` parameter of "
            )
            "lightning.pytorch.strategies.DDPStrategy is set to True."

    def on_end(self, trainer: pl.Trainer) -> None:
        trainer.save_checkpoint(trainer.default_root_dir + "/NMF.ckpt")

    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Predict the gene expression programs for the given gene counts matrix.
        """

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        # assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        transform = Filter([str(s) for s in self.var_names_hvg])
        x_filtered_ng = transform(x_ng, var_names_g)["x_ng"]

        # get the final alpha_nk
        if self.log_variational:
            x_ = torch.log1p(x_filtered_ng)
        else:
            std_g = torch.std(x_filtered_ng, dim=0) + 1e-4
            x_ = x_filtered_ng / std_g
            x_ = torch.clamp(x_, min=0.0, max=100.0)

        if self.get_rec_error:
            rec_error = []
            for k in self.k_values:
                D_kg = getattr(self, f"D_{k}_kg")

                alpha_nk = compute_loadings(
                    x_ng=x_,
                    factors_rkg=D_kg.to(x_.device).unsqueeze(0),
                    n_iterations=1000,
                    learning_rate=0.01,
                    alpha_tol=self._alpha_tol,
                ).squeeze(0)
                rec_error.append(
                    np.sum(nmf_frobenius_loss(x_, alpha_nk.to(x_.device), D_kg.to(x_.device)).cpu().numpy())
                )

            return {"rec_error": torch.tensor(rec_error)}

        else:
            k = self.the_best_k
            D_kg = getattr(self, f"D_{k}_kg")
            if (D_kg == 0).all():
                raise ValueError("D_kg is all zeros, please run compute_consensus_factors() before calling predict()")

            # compute loadings, Mairal Algorithm 1 step 4
            alpha_nk = compute_loadings(
                x_ng=x_,
                factors_rkg=D_kg.to(x_.device).unsqueeze(0),
                n_iterations=1000,
                alpha_tol=self._alpha_tol,
                learning_rate=0.01,
                normalize=False,
            ).squeeze(0)

            # update A and B, Mairal Algorithm 1 step 5 and 6
            A_kk: torch.Tensor = getattr(self, f"full_A_{k}_kk")
            B_kg: torch.Tensor = getattr(self, f"full_B_{k}_kg")
            A = A_kk + torch.matmul(alpha_nk.T, alpha_nk) / x_ng.shape[0]
            B = B_kg + torch.matmul(alpha_nk.T, x_) / x_ng.shape[0]
            setattr(self, f"full_A_{k}_kk", A)
            setattr(self, f"full_B_{k}_kg", B)

            ## get the final D for full transcrptome
            if self.if_get_full_D:
                x_ng = (x_ng.T / x_ng.sum(1)).T * 1e4
                x_ = torch.log1p(x_ng)
                A_kk = getattr(self, f"full_A_{k}_kk")
                B_kg = getattr(self, f"full_B_{k}_kg")
                full_D_kg = getattr(self, f"full_D_{k}_kg")

                # compute factors, Mairal Algorithm 1 step 7
                D = compute_factors(
                    A_rkk=A_kk.unsqueeze(0),
                    B_rkg=B_kg.unsqueeze(0),
                    factors_rkg=full_D_kg.unsqueeze(0),
                    n_iterations=100,
                    D_tol=self._D_tol,
                ).squeeze(0)
                setattr(self, f"full_D_{k}_kg", D)

            return {"alpha_nk": alpha_nk}

    @property
    def factors_kg(self) -> dict[int, torch.Tensor]:
        """
        Inferred consensus gene expression programs (i.e. "factors") defined on the highly variable genes
        used to train the model.
        """
        return {k: getattr(self, f"D_{k}_kg") for k in self.k_values}


def compute_consensus_factors(
    nmf_model: NonNegativeMatrixFactorization,
    density_threshold=0.2,
    local_neighborhood_size=0.3,
):
    """
    Run the consensus step of consensus NMF, and store the consensus factors as attributes of `nmf_model`.

    Args:
        nmf_model: The trained NMF model.
        density_threshold: The threshold for the density of the local neighborhood.
        local_neighborhood_size: The size of the local neighborhood.

    Returns:
        A dictionary of the consensus factors_kg for each value of k.
    """
    torch.manual_seed(0)
    k_values = nmf_model.k_values

    consensus_stat = {}
    for k in k_values:
        D_rkg = getattr(nmf_model, f"D_{k}_rkg")
        consensus_output = consensus(
            D_rkg=D_rkg, k=k, density_threshold=density_threshold, local_neighborhood_size=local_neighborhood_size
        )
        setattr(nmf_model, f"D_{k}_kg", consensus_output["consensus_D"])

        consensus_stat[k] = consensus_output
        print("silhouette score of k=%d: %s" % (k, str(round(consensus_output["stability"], 4))))

    return consensus_stat
