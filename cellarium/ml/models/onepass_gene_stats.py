# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.distributed import get_rank_and_num_replicas


class OnePassCellariumModel(CellariumModel, metaclass=ABCMeta):
    """
    Base class for models which take one pass (epoch) through the data to compute statistics.
    These models use registered buffers to store sufficient statistics, and are compatible with
    distributed training, as they must implement a method to reduce their buffers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        _, world_size = get_rank_and_num_replicas()
        self.world_size = world_size

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        self._dummy_param.data.zero_()

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "OnePassCellariumModel and derived classes require that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is False, (
                "OnePassCellariumModel and derived classes require that broadcast_buffers is set to False."
            )

    def _gather_tensor_list(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        Gather a tensor from all devices to a list on device rank 0.

        Args:
            tensor: The tensor to gather.

        Returns:
            The gathered list of tensors.
        """
        if self.world_size == 1:
            return [tensor]

        gathered_tensor_list = [torch.zeros_like(tensor)] * self.world_size
        dist.gather(tensor, gathered_tensor_list, dst=0)
        return gathered_tensor_list
        
    def _gather_batched_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather a tensor from all devices to a batched tensor on all devices.

        Args:
            tensor: The tensor to gather.

        Returns:
            The gathered list of tensors.
        """
        if self.world_size == 1:
            return tensor.unsqueeze(0)

        gathered_tensor = torch.zeros(self.world_size, *tensor.shape, dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(gathered_tensor, tensor)
        return gathered_tensor


class OnlineGeneStats(metaclass=ABCMeta):

    def __init__(self, n_vars):
        self.n_vars = n_vars

    @abstractmethod
    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        pass

    @abstractmethod
    def forward(self, x_ng: torch.Tensor) -> None:
        """
        Updates sufficient statistics for both raw and rank-based (Spearman) correlations.

        Args:
            x_ng: (num_cells, num_genes) matrix of values
        """
        pass

    @abstractmethod
    def _get_mean_g(self) -> torch.Tensor:
        """
        Compute the mean of the data using sufficient statistics.
        """
        pass

    @abstractmethod
    def _get_var_g(self) -> torch.Tensor:
        """
        Compute the variance of the data using sufficient statistics.
        """
        pass

    def _get_covariance(self) -> torch.Tensor | None:
        """
        Compute the covariance matrix using sufficient statistics.
        """
        return None
    
    def _get_correlation(self, use_rank: bool = False) -> torch.Tensor | None:
        """
        Compute the correlation matrix using sufficient statistics.

        Args:
            use_rank: Whether to use rank-based statistics (Spearman correlation)
        """
        return None

    @property
    def mean_g(self) -> torch.Tensor:
        """
        Mean of the data.
        """
        return self._get_mean_g()

    @property
    def var_g(self) -> torch.Tensor:
        """
        Variance of the data.
        """
        return self._get_var_g()

    @property
    def std_g(self) -> torch.Tensor:
        """
        Standard deviation of the data.
        """
        return torch.sqrt(self.var_g)
    
    @property
    def covariance_gg(self) -> torch.Tensor | None:
        """
        Covariance matrix of the data.
        """
        return self._get_covariance()
    
    @property
    def correlation_pearson_gg(self) -> torch.Tensor | None:
        """
        Pearson correlation matrix of the data.
        """
        return self._get_correlation(use_rank=False)
    
    @property
    def correlation_spearman_gg(self) -> torch.Tensor | None:
        """
        Spearman correlation matrix of the data.
        """
        return self._get_correlation(use_rank=True)


class NaiveOnlineGeneStats(OnePassCellariumModel, OnlineGeneStats):
    """
    Compute gene-wise means, variances, and standard deviations using a naive algorithm [1],
    with the option to use mean-shifted data.

    **References:**
    
    [1] `Algorithms for calculating variance
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.
    """

    def __init__(self, n_vars: int, shifted: bool):
        super().__init__(n_vars=n_vars)
        self.shifted = shifted
        self.x_sums_g: torch.Tensor
        self.x_squared_sums_g: torch.Tensor
        self.x_shift_g: torch.Tensor
        self.n: int

        self.register_buffer("x_sums_g", torch.empty(n_vars))
        self.register_buffer("x_squared_sums_g", torch.empty(n_vars))
        self.register_buffer("n", torch.empty(()))
        self.register_buffer("x_shift_g", torch.empty(n_vars))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.x_sums_g.zero_()
        self.x_squared_sums_g.zero_()
        self.x_shift_g.zero_()
        self.n.zero_()
        super().reset_parameters()

    @torch.no_grad()
    def forward(self, x_ng: torch.Tensor) -> None:
        """
        Updates sufficient statistics for both raw and rank-based (Spearman) correlations.

        Args:
            x_ng: (num_cells, num_genes) matrix of raw values
            shifted: Whether to use shifted data
        """
        if not self.shifted:
            self.x_sums_g += x_ng.sum(dim=0)
            self.x_squared_sums_g += (x_ng ** 2).sum(dim=0)
            self.n += x_ng.shape[0]
        else:
            assert self.x_shift_g is not None
            if (self.x_shift_g == 0).all():
                _, world_size = get_rank_and_num_replicas()
                if world_size > 1:
                    gathered_x_ng = torch.zeros(
                        world_size * x_ng.shape[0], x_ng.shape[1], dtype=x_ng.dtype, device=x_ng.device
                    )
                    dist.all_gather_into_tensor(gathered_x_ng, x_ng)
                    x_shift = gathered_x_ng.mean(dim=0)
                else:
                    x_shift = x_ng.mean(dim=0)
                self.x_shift_g = x_shift
            self.x_sums_g = self.x_sums_g + (x_ng - self.x_shift_g).sum(dim=0)
            self.x_squared_sums_g = self.x_squared_sums_g + ((x_ng - self.x_shift_g) ** 2).sum(dim=0)
            self.n = self.n + x_ng.shape[0]

        return {}

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        if trainer.world_size == 1:
            return
        
        # merge the running sums
        dist.reduce(self.x_sums_g, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.x_squared_sums_g, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(self.n, dst=0, op=dist.ReduceOp.SUM)

    def _get_mean_g(self) -> torch.Tensor:
        """
        Compute the mean of the data using sufficient statistics.
        """
        mean_g = self.x_sums_g / self.n
        if self.shifted:
            mean_g = mean_g + self.x_shift_g
        return mean_g
    
    def _get_var_g(self) -> torch.Tensor:
        """
        Compute the variance of the data using sufficient statistics.
        """
        return self.x_squared_sums_g / self.n - (self.x_sums_g / self.n) ** 2


class WelfordOnlineGeneStats(OnePassCellariumModel, OnlineGeneStats):
    """
    Compute gene-wise means, variances, and standard deviations
    using Welford's online algorithm [1].

    ** References: **

    [1] `Welford's online algorithm for calculating variance
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_.
    """
    def __init__(self, n_vars: int, use_rank: bool = False):
        super().__init__(n_vars=n_vars)
        self.use_rank = use_rank
        self.mean_g: torch.Tensor
        self.m2_g: torch.Tensor
        self.n: torch.Tensor

        self.register_buffer("mean_raw_g", torch.empty(n_vars))
        self.register_buffer("m2_raw_g", torch.empty(n_vars))
        self.register_buffer("n", torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mean_g.zero_()
        self.m2_g.zero_()
        self.n.zero_()
        super().reset_parameters()

    def update(self, x_ng: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates sufficient statistics for both raw and rank-based (Spearman) correlations.

        Args:
            x_ng: (num_cells, num_genes) matrix of raw values

        Returns:
            delta1_ng: Difference between the raw values and the mean
            delta2_ng: Difference between the raw values and the updated mean
            updated_mean_g: Updated mean
        """
        if self.use_rank:
            x_ng = torch.argsort(torch.argsort(x_ng, dim=0), dim=0).float() + 1

        batch_size = x_ng.shape[0]
        gathered_batch_size_list = self._gather_tensor_list(batch_size)
        self.n = self.n + sum(gathered_batch_size_list)

        # statistics update (Welford's algorithm)
        delta1_ng = x_ng - self.mean_g
        summed_delta1_g = delta1_ng.sum(dim=0)

        gathered_summed_delta1_wg = self._gather_batched_tensor(summed_delta1_g)
        summed_delta1_g = gathered_summed_delta1_wg.sum(dim=0)

        updated_mean_g = self.mean_g + summed_delta1_g / self.n
        self.mean_g = updated_mean_g

        delta2_ng = x_ng - updated_mean_g
        m2_update_g = (delta1_ng * delta2_ng).sum(dim=0)

        gathered_m2_g_update_wg = self._gather_batched_tensor(m2_update_g)
        m2_update_g = gathered_m2_g_update_wg.sum(dim=0)

        self.m2_g = self.m2_g + m2_update_g

        return delta1_ng, delta2_ng, updated_mean_g

    @torch.no_grad()
    def forward(self, x_ng: torch.Tensor) -> None:
        self.update(x_ng)
        return {}

    def _get_mean_g(self):
        return self.mean_g
    
    def _get_var_g(self):
        return self.m2_g / self.n


@torch.no_grad()
def combine_means_across_devices(
    mean_g_list: list[torch.Tensor],
    n_list: list[torch.Tensor],
) -> torch.Tensor:
    """
    Reduce per-gene means computed on multiple devices.
    """
    assert len(mean_g_list) == len(n_list)

    if len(mean_g_list) == 1:
        return mean_g_list[0]
    
    if len(mean_g_list) > 2:
        # use recursion to combine means pairwise
        return combine_means_across_devices(
            mean_g_list=[combine_means_across_devices(mean_g_list[:2], n_list[:2])] + mean_g_list[2:], 
            n_list=[sum(n_list[:2])] + n_list[2:],
        )

    # compute combined means
    combined_mean_g = (n_list[0] * mean_g_list[0] + n_list[1] * mean_g_list[1]) / (n_list[0] + n_list[1])
    return combined_mean_g


@torch.no_grad()
def combine_covariances_across_devices(
    cov_gg_list: list[torch.Tensor],
    mean_g_list: list[torch.Tensor],
    n_list: list[torch.Tensor],
) -> torch.Tensor:
    """
    Reduce gene-gene covariances computed on multiple devices [1].

    ** References: **

    [1] `Algorithms for calculating variance
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance>`_.
        See the "Online" subsection.
    """
    assert len(cov_gg_list) == len(mean_g_list) == len(n_list)

    if len(cov_gg_list) == 1:
        return cov_gg_list[0]
    
    if len(cov_gg_list) > 2:
        # use recursion to combine covariances pairwise
        return combine_covariances_across_devices(
            cov_gg_list=[
                combine_covariances_across_devices(
                    cov_gg_list=cov_gg_list[:2], 
                    mean_g_list=mean_g_list[:2], 
                    n_list=n_list[:2]
                )
            ] + cov_gg_list[2:],
            mean_g_list=[
                combine_means_across_devices(
                    mean_g_list=mean_g_list[:2], 
                    n_list=n_list[:2]
                )
            ] + mean_g_list[2:], 
            n_list=[sum(n_list[:2])] + n_list[2:],
        )

    # compute combined covariances
    combined_covariance_gg = (
        (n_list[0] - 1) * cov_gg_list[0] 
        + (n_list[1] - 1) * cov_gg_list[1] 
        + (n_list[0] * n_list[1] / (n_list[0] + n_list[1])) * (mean_g_list[0] - mean_g_list[1]).outer()
    ) / (sum(n_list) - 1)
    return combined_covariance_gg


class WelfordOnlineGeneGeneStats(WelfordOnlineGeneStats):
    """
    Compute gene-wise means, covariance, and correlation in a stable way using Welford's online algorithm [1].
    Computes correlation using both raw and rank-based statistics.

    ** References: **

    [1] `Welford's online algorithm for calculating variance
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_.
    """
    def __init__(self, n_vars, use_rank: bool = False):
        super().__init__(n_vars=n_vars, use_rank=use_rank)
        self.c_gg: torch.Tensor
        self.register_buffer("c_gg", torch.empty((n_vars, n_vars)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.c_gg.zero_()

    @torch.no_grad()
    def forward(self, x_ng: torch.Tensor) -> None:
        """
        Updates sufficient statistics for both raw and rank-based (Spearman) correlations.

        Args:
            x_ng: (num_cells, num_genes) matrix of raw values
        """
        n = x_ng.shape[0]
        delta1_ng, delta2_ng, updated_mean_g = super().update(x_ng)
        c_update_gg = delta1_ng.T @ delta2_ng

        gathered_c_update_gg_list = self._gather_tensor_list(c_update_gg)
        # gathered_mean_g_list = self._gather_tensor_list(updated_mean_g)
        gathered_n_list = self._gather_tensor_list(n)
        # c_update_gg = combine_covariances_across_devices(
        #     cov_gg_list=gathered_c_update_gg_list, 
        #     mean_g_list=gathered_mean_g_list, 
        #     n_list=gathered_n_list,
        # )
        combined_update_c_gg = sum(
            [(num / sum(gathered_n_list)) * update_gg 
             for num, update_gg in zip(gathered_n_list, gathered_c_update_gg_list)]
        )

        self.c_gg = self.c_gg + combined_update_c_gg

        return {}

    def _get_covariance(self) -> torch.Tensor:
        """
        Compute the covariance matrix using either raw or rank statistics.

        Args:
            use_rank (bool): Whether to use rank-based statistics
        """
        if self.n <= 1:
            return torch.zeros((self.n_vars, self.n_vars))
        
        return self.c_gg / self.n

    def _get_correlation(self) -> torch.Tensor:
        """
        Compute the correlation matrix using either raw or rank statistics.

        Args:
            use_rank (bool): Whether to use rank-based statistics
        """
        if self.n <= 1:
            return torch.zeros((self.n_vars, self.n_vars))
        
        cov_gg = self.covariance_gg
        var_g = torch.diag(cov_gg).clamp(min=1e-12)
        std_g = torch.sqrt(var_g)
        corr_gg = cov_gg / torch.outer(std_g, std_g)
        corr_gg[torch.isnan(corr_gg)] = 0
        return corr_gg
