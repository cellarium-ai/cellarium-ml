# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn

from cellarium.ml.utilities.testing import (
    assert_nonnegative,
    assert_positive,
)
from cellarium.ml.utilities.types import BatchDict


class NormalizeTotal(nn.Module):
    """
    Normalize total gene counts per cell to target count.

    .. math::

        \\mathrm{total\\_mrna\\_umis}_n = \\sum_{g=1}^G x_{ng}

        y_{ng} = \\frac{\\mathrm{target\\_count} \\times x_{ng}}{\\mathrm{total\\_mrna\\_umis}_n + \\mathrm{eps}}

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
    ) -> None:
        super().__init__()
        assert_positive("target_count", target_count)
        self.target_count = target_count
        assert_nonnegative("eps", eps)
        self.eps = eps

    def forward(
        self,
        x_ng: torch.Tensor,
        total_mrna_umis_n: torch.Tensor | None = None,
    ) -> BatchDict:
        """
        Args:
            x_ng:
                Gene counts.
            total_mrna_umis_n:
                Total mRNA UMI counts per cell. If ``None``, it is computed from ``x_ng``.

        Returns:
            Gene counts normalized to target count.
        """
        if total_mrna_umis_n is None:
            total_mrna_umis_n = x_ng.sum(dim=-1)
        x_ng = self.target_count * x_ng / (total_mrna_umis_n[:, None] + self.eps)
        return BatchDict(x_ng=x_ng)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_count={self.target_count}, eps={self.eps})"
