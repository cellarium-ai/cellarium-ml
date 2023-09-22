# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn


class NormalizeTotal(nn.Module):
    """
    Normalize total gene counts per cell to target count.

    Args:
        target_count:
            Target gene epxression count.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        target_count: int = 10_000,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.target_count = target_count
        self.eps = eps

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        return self.target_count * x_ng / (x_ng.sum(dim=-1, keepdim=True) + self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_count={self.target_count}, eps={self.eps})"


class DivideByScale(nn.Module):
    """
    Divide gene counts by a scale.

    Args:
        scale_g:
            A scale for each gene.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(self, scale_g: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.scale_g: torch.Tensor
        self.register_buffer("scale_g", scale_g)
        self.eps = eps

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        return x_ng / (self.scale_g + self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_g={self.scale_g}, eps={self.eps})"


class ZScoreLog1pNormalize(nn.Module):
    """
    Log1pNormalize gene counts with target count and then ZScore with  mean and standard deviation.

    Args:
        mean_g:
            Means for each gene.
        std_g:
            Standard deviations for each gene.
        perform_scaling:
            A boolean value that when set to ``True``, scaling by ``std_g`` is applied.
        target_count:
            Target gene epxression count.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        mean_g: torch.Tensor | float,
        std_g: torch.Tensor | float | None,
        perform_scaling: bool,
        target_count: int = 10_000,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mean_g = mean_g
        self.std_g = std_g
        self.perform_scaling = perform_scaling
        self.target_count = target_count
        self.eps = eps

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        # Log1pNormalize
        l_n1 = x_ng.sum(dim=-1, keepdim=True)
        y_ng = torch.log1p(self.target_count * x_ng / (l_n1 + self.eps))

        # ZScore
        z_ng = y_ng - self.mean_g
        if self.perform_scaling:
            assert self.std_g is not None, "Must provide standard deviation `std_g`"
            z_ng = z_ng / (self.std_g + self.eps)

        return z_ng

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean_g={self.mean_g}, std_g={self.std_g}, "
            f"perform_scaling={self.perform_scaling}, target_count={self.target_count}, eps={self.eps})"
        )
