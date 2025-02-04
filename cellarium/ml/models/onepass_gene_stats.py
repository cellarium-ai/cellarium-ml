# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.distributed import get_rank_and_num_replicas
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class StableOnlineGeneStats:
    """
    Compute gene-wise means, covariance, and correlation in a stable way using Welford's online algorithm [1].
    Computes correlation using both raw and rank-based statistics.

    ** References: **

    [1] `Welford's online algorithm for calculating variance
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_.
    """
    def __init__(self, num_genes):
        self.num_genes = num_genes

        # Raw statistics
        self.mean_raw_g = torch.zeros(num_genes)  # Running mean for raw values
        self.covariance_raw_gg = torch.zeros((num_genes, num_genes))  # Covariance accumulator for raw values
        self.n = 0  # Number of samples seen

        # Rank statistics
        self.mean_rank_g = torch.zeros(num_genes)  # Running mean for ranks
        self.covariance_rank_gg = torch.zeros((num_genes, num_genes))  # Covariance accumulator for ranks

    def update(self, x_ng: torch.Tensor) -> None:
        """
        Updates sufficient statistics for both raw and rank-based (Spearman) correlations.

        Args:
            batch (torch.Tensor): (num_cells, num_genes) matrix of raw values
        """
        batch_size = x_ng.shape[0]
        new_N = self.n + batch_size

        # Raw statistics update (Welford's algorithm)
        delta1_raw_ng = x_ng - self.mean_raw_g
        updated_mean_raw_g = self.mean_raw_g + delta1_raw_ng.sum(dim=0) / new_N
        delta2_raw_ng = x_ng - updated_mean_raw_g
        self.covariance_raw_gg += delta1_raw_ng.T @ delta2_raw_ng

        # Compute ranks within batch
        ranks_g = torch.argsort(torch.argsort(x_ng, dim=0), dim=0).float() + 1

        # Rank statistics update (similar to Welford's)
        delta1_rank_g = ranks_g - self.mean_rank_g
        updated_mean_rank_g = self.mean_rank_g + delta1_rank_g.sum(dim=0) / new_N
        delta2_rank = ranks_g - updated_mean_rank_g
        self.covariance_rank_gg += delta1_rank_g.T @ delta2_rank

        # Finalize updates
        self.mean_raw_g = updated_mean_raw_g
        self.mean_rank_g = updated_mean_rank_g
        self.n = new_N

    def covariance(self, use_rank=False) -> torch.Tensor:
        """
        Compute the covariance matrix using either raw or rank statistics.

        Args:
            use_rank (bool): Whether to use rank-based statistics
        """
        if self.n <= 1:
            return torch.zeros((self.num_genes, self.num_genes))
        
        cov_matrix = self.covariance_rank_gg if use_rank else self.covariance_raw_gg
        return cov_matrix / (self.n - 1)

    def correlation(self, use_rank=False) -> torch.Tensor:
        """
        Compute the correlation matrix using either raw or rank statistics.

        Args:
            use_rank (bool): Whether to use rank-based statistics
        """
        if self.n <= 1:
            return torch.zeros((self.num_genes, self.num_genes))
        
        cov_matrix = self.covariance(use_rank=use_rank)
        # mean_vector = self.mean_rank_g if use_rank else self.mean_raw_g

        # Compute variances and standard deviations
        variances = torch.diag(cov_matrix).clamp(min=1e-12)
        std_devs = torch.sqrt(variances)
        corr_matrix = cov_matrix / torch.outer(std_devs, std_devs)
        corr_matrix[torch.isnan(corr_matrix)] = 0
        return corr_matrix

    @staticmethod
    def combine_stats_on_gpus(stats_device_1, stats_device_2, n1, n2):
        # Compute combined means (running average)
        combined_mean = (n1 * stats_device_1.mean + n2 * stats_device_2.mean) / (n1 + n2)

        # Compute combined covariance (using the formula mentioned earlier)
        combined_covariance = ((n1 - 1) * stats_device_1.cov + (n2 - 1) * stats_device_2.cov + 
                            (n1 * n2 / (n1 + n2)) * (stats_device_1.mean - stats_device_2.mean).outer())
        combined_covariance /= (n1 + n2 - 1)
        
        return combined_mean, combined_covariance



class OnePassGeneStats(CellariumModel):
    """
    Calculate the per-gene mean, variance, standard deviation, and the 
    gene-gene covariance and correlation (Pearson and Spearman) in one pass
    (epoch) using an online approach which computes sufficient statistics.

    **References:**

    1. `Algorithms for calculating variance
       <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

    Args:
        var_names_g: The variable names schema for the input data validation.
    """

    def __init__(self, var_names_g: np.ndarray, algorithm: Literal["naive", "shifted_data"] = "naive") -> None:
        super().__init__()
        self.var_names_g = var_names_g
        n_vars = len(self.var_names_g)
        self.n_vars = n_vars
        self.algorithm = algorithm

        self.x_sums: torch.Tensor
        self.x_squared_sums: torch.Tensor
        self.x_size: torch.Tensor
        self.x_shift: torch.Tensor | None
        self.register_buffer("x_sums", torch.empty(n_vars))
        self.register_buffer("x_squared_sums", torch.empty(n_vars))
        self.register_buffer("x_size", torch.empty(()))
        if self.algorithm == "shifted_data":
            self.register_buffer("x_shift", torch.empty(n_vars))
        else:
            self.register_buffer("x_shift", None)
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.x_sums.zero_()
        self.x_squared_sums.zero_()
        self.x_size.zero_()
        if self.x_shift is not None:
            self.x_shift.zero_()
        self._dummy_param.data.zero_()

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

        if self.algorithm == "naive":
            self.x_sums = self.x_sums + x_ng.sum(dim=0)
            self.x_squared_sums = self.x_squared_sums + (x_ng**2).sum(dim=0)
            self.x_size = self.x_size + x_ng.shape[0]
        elif self.algorithm == "shifted_data":
            assert self.x_shift is not None
            if (self.x_shift == 0).all():
                _, world_size = get_rank_and_num_replicas()
                if world_size > 1:
                    gathered_x_ng = torch.zeros(
                        world_size * x_ng.shape[0], x_ng.shape[1], dtype=x_ng.dtype, device=x_ng.device
                    )
                    dist.all_gather_into_tensor(gathered_x_ng, x_ng)
                    x_shift = gathered_x_ng.mean(dim=0)
                else:
                    x_shift = x_ng.mean(dim=0)
                self.x_shift = x_shift
            self.x_sums = self.x_sums + (x_ng - self.x_shift).sum(dim=0)
            self.x_squared_sums = self.x_squared_sums + ((x_ng - self.x_shift) ** 2).sum(dim=0)
            self.x_size = self.x_size + x_ng.shape[0]
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return {}

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "OnePassMeanVarStd requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is False, (
                "OnePassMeanVarStd requires that broadcast_buffers is set to False."
            )

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        # no need to merge if only one process
        if trainer.world_size == 1:
            return

        # merge the running sums
        dist.reduce(self.x_sums, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_squared_sums, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_size, dst=0, op=dist.ReduceOp.SUM)

    @property
    def mean_g(self) -> torch.Tensor:
        """
        Mean of the data.
        """
        mean_g = self.x_sums / self.x_size
        if self.algorithm == "shifted_data":
            mean_g = mean_g + self.x_shift
        return mean_g

    @property
    def var_g(self) -> torch.Tensor:
        """
        Variance of the data.
        """
        return self.x_squared_sums / self.x_size - (self.x_sums / self.x_size) ** 2

    @property
    def std_g(self) -> torch.Tensor:
        """
        Standard deviation of the data.
        """
        return torch.sqrt(self.var_g)
