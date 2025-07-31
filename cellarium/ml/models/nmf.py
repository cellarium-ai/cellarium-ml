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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, IterableDataset

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.transforms import Filter
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

warnings.filterwarnings("ignore")


def get_mean_var(X):
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X)
    return (scaler.mean_, scaler.var_)


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

        df_d_norm_mg = pd.DataFrame(d_norm_mg.cpu().numpy())
        kmeans = KMeans(n_clusters=k, random_state=1)  # n_init=10,
        kmeans.fit(df_d_norm_mg)
        kmeans_cluster_labels = pd.Series(kmeans.labels_ + 1, index=df_d_norm_mg.index)

        silhouette = silhouette_score(df_d_norm_mg.values, kmeans_cluster_labels, metric="cosine")

        median_D = df_d_norm_mg.groupby(kmeans_cluster_labels).median()  # mean() # quantile(0.9) #
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


def update_consensusD(pipeline, density_threshold=0.2, local_neighborhood_size=0.1):
    torch.manual_seed(0)
    k_range = pipeline[-1].k_values
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


def get_full_D(A_kk, B_kg, factors_kg, n_iterations):
    """
    Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

    Args:
        factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        n_iterations: The number of iterations to perform.
    """
    _D_tol = 1e-8

    _, k_dimension = A_kk.shape

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
            # u_1g = factors_kg[k, :] + (b_1g - torch.matmul(a_1k, factors_kg)) / scalar

            factors_1g = u_1g / torch.clamp(torch.linalg.norm(u_1g), min=1.0)
            factors_kg[k, :] = factors_1g

        D_diff = torch.linalg.norm(factors_kg - D_buffer) / torch.linalg.norm(factors_kg)
        if D_diff <= _D_tol:
            break
        D_buffer = factors_kg.clone()

    return factors_kg


def efficient_ols_all_cols(X, Y, XtX, XtY, normalize_y=True):
    """
    Solve OLS: Beta = (X^T X)^{-1} X^T Y,
    accumulating X^T X and X^T Y in row-batches.

    Optionally mean/variance-normalize each column of Y *globally*
    (using the entire dataset's mean/var), while still only converting
    each row-batch to dense on-the-fly.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_predictors)
        Predictor matrix.
    Y : np.ndarray or scipy.sparse.spmatrix, shape (n_samples, n_targets)
        Outcomes. Each column is one target variable.
    batch_size : int
        Number of rows to process per chunk.
    normalize_y : bool
        If True, compute global mean & var of Y columns, then subtract mean
        and divide by std for each batch.

    Returns
    -------
    Beta : np.ndarray, shape (n_predictors, n_targets)
        The OLS coefficients for each target.
    """

    # -- Optionally compute global mean & variance of Y columns
    if normalize_y:
        meanY, varY = get_mean_var(Y)

        # Avoid zero or near-zero std
        eps = 1e-12
        varY[varY < eps] = eps
        stdY = np.sqrt(varY)

    # -- Optionally apply normalization
    if normalize_y:
        Y = (Y - meanY) / stdY

    # -- Accumulate partial sums
    XtX += X.T @ X
    XtY += X.T @ Y

    # -- Solve the normal equations
    #    Beta = (X^T X)^(-1) X^T Y
    #    Using lstsq for stability.
    Beta, residuals, rank, s = np.linalg.lstsq(XtX, XtY, rcond=None)
    return Beta, XtX, XtY


def compute_loadings(
    x_ng: torch.Tensor,
    factors_rkg: torch.Tensor,
    n_iterations: int,
    normalize: bool = False,
    alpha_tol: float = 1e-5,
    transformed_data_mean: float = 0.28,
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
    r, n_factors, g = factors_rkg.shape

    alpha_rnk = torch.rand((r, n, n_factors), device=factors_rkg.device).abs()
    # factor = np.sqrt(transformed_data_mean / n_factors)
    # alpha_rnk = factor * torch.randn((r, n, n_factors), device=factors_rkg.device).abs()
    # alpha_rnk = factor * torch.zeros((r, n, n_factors), device=factors_rkg.device).abs()
    alpha_buffer_rnk = alpha_rnk.clone()

    DDT = torch.bmm(factors_rkg, factors_rkg.transpose(1, 2))
    xDT = torch.bmm(x_ng.expand(r, n, g), factors_rkg.transpose(1, 2))

    for i in range(n_iterations):
        for k in range(n_factors):
            scalar = DDT[:, k, k].view(r, 1, 1)
            a_rk1 = DDT[:, :, k].unsqueeze(2)
            b_rn1 = xDT[:, :, k].unsqueeze(2)

            # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
            u_r1g = torch.clamp(
                alpha_rnk[:, :, k].unsqueeze(2) + (b_rn1 - torch.bmm(alpha_rnk, a_rk1)) / scalar,
                min=0.0,
            )
            alpha_rnk[:, :, k] = u_r1g.squeeze(2)

        alpha_max_diff = (
            F.mse_loss(alpha_rnk, alpha_buffer_rnk, reduction="none").sum(dim=[-2, -1])
            / alpha_rnk.square().sum(dim=[-2, -1])
        ).max()
        if alpha_max_diff <= alpha_tol:
            break
        alpha_buffer_rnk = alpha_rnk.clone()

    return alpha_rnk


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
        algorithm: Literal["mairal"] = "mairal",
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
            self.register_buffer(f"full_B_{i}_kg", torch.empty(i, full_g))
            self.register_buffer(f"full_D_{i}_kg", torch.empty(i, full_g))

        self._D_tol = 1e-5
        self._alpha_tol = 1e-5

        self.n_nmf = r

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
        alpha_rnk = compute_loadings(
            x_ng=x_ng,
            factors_rkg=factors_rkg,
            n_iterations=n_iterations,
            normalize=False,
            alpha_tol=self._alpha_tol,
        )
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

    def online_dictionary_learning(self, x_ng: torch.Tensor, factors_rkg: torch.Tensor) -> torch.Tensor:
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
            n_iterations=100,
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
                n_iterations=100,
            )

        return updated_factors_rkg

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
            # x_ = torch.clamp(x_, min=0.0, max=100.0)

        if self.algorithm == "mairal":
            for i in self.k_values:
                D_rkg = getattr(self, f"D_{i}_rkg")
                D_rkg = self.online_dictionary_learning(x_ng=x_, factors_rkg=D_rkg)
                setattr(self, f"D_{i}_rkg", D_rkg)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return {}

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "NonNegativeMatrixFactorization requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is True, (
                "NonNegativeMatrixFactorization requires that the `broadcast_buffers` parameter of "
            )
            "lightning.pytorch.strategies.DDPStrategy is set to True."

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        for i in self.k_values:
            getattr(self, f"A_{i}_rkk").zero_()
            getattr(self, f"B_{i}_rkg").zero_()

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
            # x_ = torch.clamp(x_, min=0.0, max=100.0)

        if self.get_rec_error:
            rec_error = []
            for k in self.k_values:
                D_kg = getattr(self, f"D_{k}_kg")

                alpha_nk = compute_loadings(
                    x_ng=x_,
                    factors_rkg=D_kg.to(x_.device).unsqueeze(0),
                    n_iterations=1000,
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
                normalize=False,
            ).squeeze(0)

            ## get the final D for full transcrptome
            if self.if_get_full_D:
                x_ng = (x_ng.T / x_ng.sum(1)).T * 1e6

                ## update A and B, Mairal Algorithm 1 step 5 and 6
                A_kk = getattr(self, f"full_A_{k}_kk")
                B_kg = getattr(self, f"full_B_{k}_kg")
                # full_D_kg = getattr(self, f"full_D_{k}_kg")

                # A = A_kk + torch.matmul(alpha_nk.T, alpha_nk) / x_.shape[0]
                # B = B_kg + torch.matmul(alpha_nk.T, x_) / x_.shape[0]
                # setattr(self, f"full_A_{k}_kk", A)
                # setattr(self, f"full_B_{k}_kg", B)

                ## compute factors, Mairal Algorithm 1 step 7
                # D = get_full_D(
                #     A_kk=A,
                #     B_kg=B,
                #     factors_kg=full_D_kg,
                #     n_iterations=200,
                # )
                D, A, B = efficient_ols_all_cols(
                    alpha_nk.cpu().numpy(), x_ng.cpu().numpy(), A_kk.cpu().numpy(), B_kg.cpu().numpy()
                )
                setattr(self, f"full_D_{k}_kg", torch.tensor(D))
                setattr(self, f"full_A_{k}_kk", torch.tensor(A))
                setattr(self, f"full_B_{k}_kg", torch.tensor(B))
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
