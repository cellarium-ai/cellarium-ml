# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.data import get_rank_and_num_replicas
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

def anls_solve(loss: callable[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    The ANLS solver for the dictionary update step in the online NMF algorithm.

    Args:
        loss: The loss function to minimize.
    """
    # to be implemented
    raise NotImplementedError


class NonNegativeMatrixFactorization(CellariumModel):
    """
    Use the online NMF algorithm of Mairal et al. [1] to factorize the count matrix 
    into a dictionary of gene expression programs and a matrix of cell program loadings.

    **References:**

    1. `Online learning for matrix factorization and sparse coding. Mairal, Bach, Ponce, Sapiro. JMLR 2009.

    Args:
        var_names_g: The variable names schema for the input data validation.
    """

    def __init__(self, var_names_g: Sequence[str], algorithm: Literal["mairal"] = "mairal") -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        n_vars = len(self.var_names_g)
        self.n_vars = n_vars
        self.algorithm = algorithm

        self.A_kk: torch.Tensor
        self.B_kg: torch.Tensor
        self.D_kg: torch.Tensor
        self.register_buffer("A_kk", torch.empty(n_vars))
        self.register_buffer("B_kg", torch.empty(n_vars))
        self.register_buffer("D_kg", torch.empty(n_vars))
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.A_kk.zero_()
        self.B_kg.zero_()
        self.D_kg.uniform_(0.0, 2.0)  # TODO: figure out best initialization
        self._dummy_param.data.zero_()

    def online_dictionary_learning(self, x_ng: torch.Tensor, factors_kg: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
        """
        def loss(a_nk: torch.Tensor) -> torch.Tensor:
            return torch.linalg.matrix_norm(x_ng - torch.matmul(a_nk, factors_kg), ord='fro')
        
        alpha_nk = anls_solve(loss)  # to be implemented
        n = x_ng.shape[0]  # division by n is shown in Mairal section 3.4.3
        self.A_kk = self.A_kk + torch.matmul(alpha_nk.T, alpha_nk) / n
        # TODO: see if this is faster than the matmul above: torch.einsum("ik,jk->ij", t, t)
        self.B_kg = self.B_kg + torch.matmul(alpha_nk.T, x_ng) / n
        updated_factors_kg = self.dictionary_update(factors_kg)
        return updated_factors_kg
    
    def dictionary_update(self, factors_kg: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
        """
        Algorithm 2 from Mairal et al. [1] for computing the dictionary update.

        Args:
            factors_kg: The matrix of gene expression programs (Mairal's dictionary D).
            n_iterations: The number of iterations to perform.
        """
        updated_factors_kg = factors_kg.copy()

        for _ in range(n_iterations):
            for k in range(factors_kg.shape[0]):

                scalar = self.A_kk[k, k]
                a_1k = self.A_kk[k, :]
                b_1g = self.B_kg[k, :]
                b_1g = torch.clamp(b_1g, min=0.0)  # the non-negativity constraint

                # Algorithm 2 line 3
                u_1g = updated_factors_kg[k, :] + (b_1g - torch.matmul(a_1k, updated_factors_kg)) / scalar
                updated_factors_kg[k, :] = u_1g / torch.clamp(torch.linalg.norm(u_1g), min=1.0)

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

        if self.algorithm == "mairal":
            self.D_kg = self.online_dictionary_learning(x_ng=x_ng, factors_kg=self.D_kg)
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

    @property
    def factors_kg(self) -> torch.Tensor:
        """
        Inferred gene expression programs (i.e. "factors").
        """
        return self.D_kg
    
    def predict(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Predict the gene expression programs for the given gene counts matrix.
        """
        # something using the ANLS solver ...
        raise NotImplementedError
