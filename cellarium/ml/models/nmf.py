# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


def anls_solve(loss) -> torch.Tensor:
    """
    The ANLS solver for the dictionary update step in the online NMF algorithm.

    Args:
        loss: The loss function to minimize.
    """
    # to be implemented
    raise NotImplementedError


def kl_div(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    .. math::
        \\ell(x, y) = \\sum_{n = 0}^{N - 1} x_n log(\frac{x_n}{y_n}) - x_n + y_n

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    return torch.distributions.Poisson(rate=input).log_prob(target).sum()


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

    def __init__(self, var_names_g: Sequence[str], k: int, algorithm: Literal["mairal"] = "mairal") -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        g = len(self.var_names_g)
        self.n_vars = g
        self.algorithm = algorithm

        self.A_kk: torch.Tensor
        self.B_kg: torch.Tensor
        self.D_kg: torch.Tensor
        self.register_buffer("A_kk", torch.empty(k, k))
        self.register_buffer("B_kg", torch.empty(k, g))
        self.register_buffer("D_kg", torch.empty(k, g))
        self.register_buffer("ori_D", torch.empty(k, g))
        self._dummy_param = torch.nn.Parameter(torch.empty(()))

        self._D_tol = 0.005  # 0.05 #
        self._alpha_tol = 0.005  # 0.05 #

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.A_kk.zero_()
        self.B_kg.zero_()
        self.D_kg.uniform_(0.0, 2.0)  # TODO: figure out best initialization
        self._dummy_param.data.zero_()

        self.ori_D = self.D_kg.clone()

    def online_dictionary_learning(self, x_ng: torch.Tensor, factors_kg: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        """

        n = x_ng.shape[0]  # division by n is shown in Mairal section 3.4.3
        k = factors_kg.shape[0]

        # def loss(a_nk: torch.Tensor) -> torch.Tensor:
        #     return torch.linalg.matrix_norm(x_ng - torch.matmul(a_nk, factors_kg), ord='fro')

        # fit = torch.linalg.lstsq(factors_kg.T, x_ng.T)
        # alpha_nk = fit.solution.T

        # an added non-negativity constraint, quite possibly wrong
        # alpha_nk = torch.clamp(alpha_nk, min=0.0)

        # updata alpha
        # DDT_kk = torch.matmul(factors_kg, factors_kg.T)
        # DXT_kn = torch.matmul(factors_kg, x_ng.T)
        # alpha_nk = torch.zeros((n, k)).to(factors_kg.device)
        # alpha_nk = self.solve_alpha(alpha_nk, DDT_kk, DXT_kn, 200)

        alpha_nk = torch.zeros((n, k), requires_grad=True, device=factors_kg.device)
        alpha_nk = self.solve_alpha_wKL(alpha_nk, x_ng, 200)

        # alpha_nk = self.solve_alpha_wNN(x_ng, 200)

        # residual_loss = fit.residuals.mean()

        # update D
        self.A_kk = self.A_kk + torch.matmul(alpha_nk.T, alpha_nk) / n
        # TODO: see if this is faster than the matmul above: torch.einsum("ik,jk->ij", t, t)
        self.B_kg = self.B_kg + torch.matmul(alpha_nk.T, x_ng) / n
        updated_factors_kg = self.dictionary_update(factors_kg, 200)

        # residual_loss = torch.norm(torch.matmul(alpha_nk, self.D_kg) - x_ng, p=2) ** 2

        return updated_factors_kg  # , residual_loss

    def solve_alpha(
        self, alpha_nk: torch.Tensor, DDT_kk: torch.Tensor, DXT_kn: torch.Tensor, n_iterations: int
    ) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """

        alpha_buffer = alpha_nk.clone()

        k_dimension = alpha_nk.shape[1]

        for _ in range(n_iterations):
            for k in range(k_dimension):
                scalar = DDT_kk[k, k]
                a_1k = DDT_kk[k, :]
                b_1g = DXT_kn[k, :]

                u_1g = torch.clamp(
                    alpha_nk[:, k] + (b_1g - torch.matmul(a_1k, alpha_nk.T)) / scalar,
                    min=0.0,
                )

                alpha_nk[:, k] = u_1g

            alpha_diff = torch.linalg.norm(alpha_nk - alpha_buffer) / torch.linalg.norm(alpha_nk)
            if alpha_diff <= self._alpha_tol:
                break
            alpha_buffer = alpha_nk.clone()

        return alpha_nk

    def solve_alpha_wKL(self, alpha_nk: torch.Tensor, x_ng: torch.Tensor, n_iterations: int) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """

        # alpha_buffer = F.softplus(alpha_nk).clone()
        alpha_buffer = alpha_nk.exp().clone()

        optimizer = torch.optim.AdamW([alpha_nk], lr=0.2)
        # kl_loss = torch.nn.KLDivLoss(reduction="mean")
        # mse_loss = torch.nn.MSELoss(reduction="mean")

        for _ in range(n_iterations):
            optimizer.zero_grad()

            # alpha_nk_exp = F.softplus(alpha_nk) #.exp()
            alpha_nk_exp = alpha_nk.exp()
            # loss = kl_loss(torch.matmul(alpha_nk_exp, self.D_kg), x_ng)
            # loss = mse_loss(torch.matmul(alpha_nk_exp, self.D_kg), x_ng)
            # loss = torch.norm(torch.matmul(alpha_nk_exp, self.D_kg) - x_ng, p=2) ** 2
            # loss = torch.mean(loss)

            loss = euclidean(torch.matmul(alpha_nk_exp, self.D_kg), x_ng)
            loss = loss.mul(2).sqrt()

            # loss = kl_div(torch.matmul(alpha_nk_exp, self.D_kg), x_ng)
            # loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     alpha_nk.clamp_(min=0)

            alpha_diff = torch.linalg.norm(alpha_nk.exp() - alpha_buffer) / torch.linalg.norm(alpha_nk.exp())
            if alpha_diff <= self._alpha_tol:
                break
            # alpha_buffer = F.softplus(alpha_nk).clone()
            alpha_buffer = alpha_nk.exp().clone()

        # return F.softplus(alpha_nk).detach()
        return alpha_nk.exp().detach()

    def dictionary_update(self, factors_kg: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """
        D_buffer = factors_kg.clone()
        k_dimension = factors_kg.shape[0]
        updated_factors_kg = factors_kg.clone()

        for _ in range(n_iterations):
            for k in range(k_dimension):
                scalar = self.A_kk[k, k]
                a_1k = self.A_kk[k, :]
                b_1g = self.B_kg[k, :]

                # Algorithm 2 line 3 with added non-negativity constraint, also possibly wrong
                u_1g = torch.clamp(
                    updated_factors_kg[k, :] + (b_1g - torch.matmul(a_1k, updated_factors_kg)) / scalar,
                    min=0.0,
                )

                updated_factors_1g = u_1g / torch.clamp(torch.linalg.norm(u_1g), min=1.0)
                updated_factors_kg[k, :] = updated_factors_1g

            D_diff = torch.linalg.norm(updated_factors_kg - D_buffer) / torch.linalg.norm(updated_factors_kg)
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

        x_ = x_ng
        x_ = torch.log1p(x_)

        # x_ = x_.repeat(self.n_nmf, 1, 1)

        if self.algorithm == "mairal":
            self.D_kg = self.online_dictionary_learning(x_ng=x_, factors_kg=self.D_kg)
            # print('loss', loss.mean().cpu().numpy())
            # assert (self.D_kg < 0.0).sum() == 0, 'there are negative elements in the dictionary matrix'

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

    @property
    def factors_kg(self) -> torch.Tensor:
        """
        Inferred gene expression programs (i.e. "factors").
        """
        return self.D_kg

    @property
    def init_factors_kg(self) -> torch.Tensor:
        """
        Inferred gene expression programs (i.e. "factors").
        """
        return self.ori_D

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

        x_ = torch.log1p(x_ng)

        n = x_ng.shape[0]  # division by n is shown in Mairal section 3.4.3
        k = self.D_kg.shape[0]

        alpha_nk = torch.zeros((n, k), requires_grad=True, device=self.D_kg.device)
        alpha_nk = self.solve_alpha_wKL(alpha_nk, x_, 200)

        return {"alpha_nk": alpha_nk}
