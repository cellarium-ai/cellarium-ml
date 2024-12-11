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

# from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.transforms import Filter
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

warnings.filterwarnings("ignore")


def calculate_rec_error(
    dataset,  # : IterableDistributedAnnDataCollectionDataset,
    pipeline,  # : CellariumPipeline,
) -> anndata.AnnData:
    """
    Embed the dataset using the pipeline.

    Args:
        dataset: Dataset.
        pipeline: Pipeline.

    Returns:
        reconstruction error
    """

    k_range = pipeline[-1].k_range
    pipeline[-1].get_rec_error = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rec_error = np.zeros((len(k_range), 1)).astype("float64")
    for batch in tqdm.tqdm(dataset):
        batch["x_ng"] = torch.from_numpy(batch["x_ng"]).to(device)
        out = pipeline.predict(batch)
        rec_error += out["rec_error"]

    return rec_error


def get_embedding(
    dataset,  # : IterableDistributedAnnDataCollectionDataset,
    pipeline,  # : CellariumPipeline,
    k: int,
    if_get_final_gene_loading: bool = True,
) -> pd.DataFrame:
    """
    Embed the dataset using the pipeline.

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
        batch["x_ng"] = torch.from_numpy(batch["x_ng"]).to(device)
        out = pipeline.predict(batch)
        z = out["alpha_nk"]
        embedding += [z.cpu()]
        index.extend(batch["obs_names_n"])

    return pd.DataFrame(torch.cat(embedding).numpy(), index=index)


def update_consensusD(pipeline, density_threshold=0.2, local_neighborhood_size=0.3):
    torch.manual_seed(0)
    k_range = pipeline[-1].k_range

    consensus_stat = {}
    for k in k_range:
        D_rkg = getattr(pipeline[-1], f"D_{k}_rkg")
        consensus_output = consensus(
            D_rkg=D_rkg, k=k, density_threshold=density_threshold, local_neighborhood_size=local_neighborhood_size
        )
        setattr(pipeline[-1], f"D_{k}_kg", consensus_output["consensus_D"])

        consensus_stat[k] = consensus_output
        print("silhouette score of k=%d: %s" % (k, str(round(consensus_output["stability"], 4))))

    return consensus_stat


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
        for _ in self.n_clusters:
            square_distance_n = torch.min(torch.cdist(x_np, x_np[indices], p=2) ** 2, dim=1)[0]
            probs_n = square_distance_n / torch.sum(square_distance_n)
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


def cal_reconstruction_error(x, alpha_nk, D_kg):
    # Compute prediction error as a frobenius norm
    rf_pred = torch.matmul(alpha_nk, D_kg)
    prediction_error = ((x - rf_pred) ** 2).sum()  # .sum()

    return prediction_error


def euclidean(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The `Euclidean distance
    .. math::
        \\ell(x, y) = \frac{1}{2} \\sum_{n = 0}^{N - 1} (x_n - y_n)^2

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    return F.mse_loss(input, target, reduction="sum") * 0.5


def consensus(D_rkg=None, k=10, density_threshold=0.25, local_neighborhood_size=0.30):
    r, num_component, g = D_rkg.shape
    d_norm_mg = F.normalize(D_rkg, dim=2, p=2)
    d_norm_mg = d_norm_mg.reshape(r * num_component, g)
    L = int(r * local_neighborhood_size)

    euc_dist = torch.cdist(d_norm_mg, d_norm_mg, p=2)
    L_nearest_neigh, _ = torch.topk(euc_dist, L + 1, largest=False)
    local_neigh_dist = L_nearest_neigh.sum(1) / L

    topk_euc_dist = euc_dist[local_neigh_dist < density_threshold, :]
    topk_euc_dist = topk_euc_dist[:, local_neigh_dist < density_threshold]
    d_norm_mg = d_norm_mg[local_neigh_dist < density_threshold, :]
    # D = pd.DataFrame(D.cpu().numpy())

    D_mean = d_norm_mg.mean(0)
    D_norm = d_norm_mg - D_mean

    # kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
    # kmeans.fit(D_norm)
    # kmeans_cluster_labels = pd.Series(kmeans.labels_+1, index=D.index)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(D_norm)
    kmeans_cluster_labels = kmeans.predict(D_norm)
    df_d_norm_mg = pd.DataFrame(d_norm_mg.cpu().numpy())
    kmeans_cluster_labels = kmeans_cluster_labels.cpu().numpy()

    silhouette = silhouette_score(df_d_norm_mg.values, kmeans_cluster_labels, metric="cosine")

    median_D = df_d_norm_mg.groupby(kmeans_cluster_labels).median()
    median_D = torch.Tensor(median_D.values)
    median_D = F.normalize(median_D, dim=1, p=1)

    return {
        "topk_euc_dist": topk_euc_dist,
        "local_neigh_dist": local_neigh_dist,
        "consensus_D": median_D,
        "stability": silhouette,
    }


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

    optimizer = torch.optim.AdamW([alpha_nk], lr=0.01)

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


def get_final_alpha(x_ng, D, n_iterations):
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
    """

    _alpha_tol = 0.005
    alpha_nk = torch.zeros((x_ng.shape[0], D.shape[0]), device=D.device)
    alpha_buffer = alpha_nk.clone()

    k_dimension = alpha_nk.shape[1]

    DDT_kk = torch.matmul(D, D.T)
    DXT_kn = torch.matmul(D, x_ng.T)

    for _ in range(n_iterations):
        for k in range(k_dimension):
            scalar = DDT_kk[k, k]
            a_1k = DDT_kk[k, :]
            b_1g = DXT_kn[k, :]

            u_1g = torch.clamp(
                alpha_nk[:, k] + (b_1g - torch.matmul(a_1k, alpha_nk.T)) / scalar,
                min=0.0,
            )

            u_1g = u_1g / torch.clamp(torch.linalg.norm(u_1g), min=1.0)
            alpha_nk[:, k] = u_1g

        alpha_diff = torch.linalg.norm(alpha_nk - alpha_buffer) / torch.linalg.norm(alpha_nk)
        if alpha_diff <= _alpha_tol:
            break
        alpha_buffer = alpha_nk.clone()

    alpha_nk = F.normalize(alpha_nk, dim=1, p=1)

    return alpha_nk.detach()


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
        k_list: The number of gene expression programs to infer.
        algorithm: The algorithm to use for the online NMF. Currently only "mairal" is supported.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        var_names_hvg: Sequence[str],
        k_list: list[int],
        r: int,
        full_g: int,
        log_variational: bool,
        algorithm: Literal["mairal"] = "mairal",
    ) -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.var_names_hvg = np.array(var_names_hvg)
        g = len(self.var_names_hvg)
        self.n_vars = g
        self.algorithm = algorithm
        self.log_variational = log_variational
        self.full_g = full_g
        self.k_list = k_list
        self.the_best_k = k_list[0]  # default and has to be reassigned
        self.get_rec_error = True
        self.if_get_full_D = False

        # self.A_rkk: torch.Tensor
        # self.B_rkg: torch.Tensor
        # self.D_rkg: torch.Tensor
        # self.D_kg: torch.Tensor

        for i in self.k_list:
            self.register_buffer(f"A_{i}_rkk", torch.empty(r, i, i))
            self.register_buffer(f"B_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_kg", torch.empty(i, g))
            self._dummy_param = torch.nn.Parameter(torch.empty(()))

            # self.full_A_kk: torch.Tensor
            # self.full_B_kg: torch.Tensor
            # self.full_D_kg: torch.Tensor
            self.register_buffer(f"full_A_{i}_kk", torch.empty(i, i))
            self.register_buffer(f"full_B_{i}_kg", torch.empty(i, full_g))
            self.register_buffer(f"full_D_{i}_kg", torch.empty(i, full_g))

        self._D_tol = 0.005
        self._alpha_tol = 0.005

        self.n_nmf = r

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in self.k_list:
            getattr(self, f"D_{i}_kg").zero_()
            getattr(self, f"A_{i}_rkk").zero_()
            getattr(self, f"B_{i}_rkg").zero_()
            getattr(self, f"D_{i}_rkg").uniform_(0.0, 2.0)  # TODO: figure out best initialization

            getattr(self, f"full_A_{i}_kk").zero_()
            getattr(self, f"full_B_{i}_kg").zero_()
            getattr(self, f"full_D_{i}_kg").uniform_(0.0, 2.0)  # TODO: figure out best initialization

    def online_dictionary_learning(self, x_ng: torch.Tensor, factors_rkg: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        """

        n, g = x_ng.shape
        r = factors_rkg.shape[0]
        k = factors_rkg.shape[1]

        # updata alpha
        alpha_rnk = torch.zeros((r, n, k), requires_grad=True, device=factors_rkg.device)
        alpha_rnk = self.solve_alpha_wKL(alpha_rnk, x_ng, 100)

        # update D
        with torch.no_grad():
            A_rkk = getattr(self, f"A_{k}_rkk")
            B_rkg = getattr(self, f"B_{k}_rkg")

            A_rkk = A_rkk + torch.bmm(alpha_rnk.transpose(1, 2), alpha_rnk) / n
            B_rkg = B_rkg + torch.bmm(alpha_rnk.transpose(1, 2), x_ng.expand(r, n, g)) / n

            setattr(self, f"A_{k}_rkk", A_rkk)
            setattr(self, f"B_{k}_rkg", B_rkg)

            # compiled_dictionary_update_3d = torch.compile(self.dictionary_update_3d)
            # updated_factors_kg = compiled_dictionary_update_3d(factors_rkg, 100)
            updated_factors_kg = self.dictionary_update_3d(factors_rkg, 100)

        return updated_factors_kg

    # @cuda.jit
    def solve_alpha_wKL(self, alpha_rnk: torch.Tensor, x_ng: torch.Tensor, n_iterations: int) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """

        r, _, k = alpha_rnk.shape
        n, g = x_ng.shape

        alpha_buffer = alpha_rnk.exp().clone()

        optimizer = torch.optim.AdamW([alpha_rnk], lr=0.2)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            alpha_rnk_exp = alpha_rnk.exp()

            D_rkg = getattr(self, f"D_{k}_rkg")

            loss = euclidean(torch.bmm(alpha_rnk_exp, D_rkg), x_ng.expand(r, n, g))
            loss = loss.mul(2).sqrt()

            loss.backward()
            optimizer.step()

            alpha_diff = torch.linalg.norm(alpha_rnk.exp().view(-1, k) - alpha_buffer.view(-1, k)) / torch.linalg.norm(
                alpha_rnk.exp().view(-1, k)
            )
            if alpha_diff <= self._alpha_tol:
                break
            alpha_buffer = alpha_rnk.exp().clone()

        return alpha_rnk.exp().detach()

    # @cuda.jit
    def dictionary_update_3d(self, factors_rkg: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_rkg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """

        i = factors_rkg.shape[1]
        D_buffer = factors_rkg.clone()
        updated_factors_kg = factors_rkg.clone()

        for _ in range(n_iterations):
            for k in range(i):
                A_rkk = getattr(self, f"A_{i}_rkk")
                B_rkg = getattr(self, f"B_{i}_rkg")

                scalar = A_rkk[:, k, k].view(self.n_nmf, 1, 1)
                a_1k = A_rkk[:, k, :].unsqueeze(1)
                b_1g = B_rkg[:, k, :].unsqueeze(1)

                # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
                u_1g = torch.clamp(
                    updated_factors_kg[:, k, :].unsqueeze(1) + (b_1g - torch.bmm(a_1k, updated_factors_kg)) / scalar,
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
            x_ = torch.clamp(x_, min=0.0, max=100.0)

        if self.algorithm == "mairal":
            for i in self.k_list:
                D_rkg = getattr(self, f"D_{i}_rkg")
                D_rkg = self.online_dictionary_learning(x_ng=x_, factors_rkg=D_rkg)
                setattr(self, f"D_{i}_rkg", D_rkg)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return {}

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(
                trainer.strategy, DDPStrategy
            ), "NonNegativeMatrixFactorization requires that the trainer uses the DDP strategy."
            assert (
                trainer.strategy._ddp_kwargs["broadcast_buffers"] is True
            ), "NonNegativeMatrixFactorization requires that the `broadcast_buffers` parameter of "
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
        x_filtered_ng = transform(x_ng, var_names_g)

        ## get the final alpha_nk
        if self.log_variational:
            x_ = torch.log1p(x_filtered_ng["x_ng"])
        else:
            std = torch.std(x_filtered_ng["x_ng"], dim=0) + 1e-4
            x_ = x_filtered_ng["x_ng"] / std
            x_ = torch.clamp(x_, min=0.0, max=100.0)

        if self.get_rec_error:
            rec_error = []
            for k in self.k_list:
                D_kg = getattr(self, f"D_{k}_kg")

                alpha_nk = get_final_alpha_wKL(x_ng=x_, D=D_kg.to(x_.device), n_iterations=1000)
                rec_error.append(
                    np.sum(cal_reconstruction_error(x_, alpha_nk.to(x_.device), D_kg.to(x_.device)).cpu().numpy())
                )

            return {"rec_error": torch.tensor(rec_error)}

        else:
            k = self.the_best_k
            D_kg = getattr(self, f"D_{k}_kg")

            # with torch.set_grad_enabled(True):
            alpha_nk = get_final_alpha_wKL(x_ng=x_, D=D_kg.to(x_.device), n_iterations=1000)
            # alpha_nk = get_final_alpha(x_ng=x_, D=self.D_kg, n_iterations=1000)

            ## get the final D for full transcrptome
            if self.if_get_full_D:
                x_ng = (x_ng.T / x_ng.sum(1)).T * 1e4
                x_ = torch.log1p(x_ng)
                A_kk = getattr(self, f"full_A_{k}_kk")
                B_kg = getattr(self, f"full_B_{k}_kg")
                full_D_kg = getattr(self, f"full_D_{k}_kg")
                A, B, D = get_full_D(x_, alpha_nk, A_kk, B_kg, full_D_kg, 100)

                setattr(self, f"full_A_{k}_kk", A)
                setattr(self, f"full_B_{k}_kg", B)
                setattr(self, f"full_D_{k}_kg", D)

            return {"alpha_nk": alpha_nk}  # , "pred_count": rf_pred

    @property
    def factors_kg(self) -> torch.Tensor:
        """
        Inferred gene expression programs (i.e. "factors").
        """
        return self.D_rkg