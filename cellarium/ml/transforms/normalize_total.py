# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Callable

import numpy as np
import scipy.sparse
import torch
from torch import nn

from cellarium.ml.utilities.data import sparse_tensor_to_scipy_csr, to_torch_sparse_coo, to_torch_sparse_csr
from cellarium.ml.utilities.testing import (
    assert_nonnegative,
    assert_positive,
)


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
        x_ng: torch.Tensor | scipy.sparse.spmatrix,
        total_mrna_umis_n: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        .. note::

            When used with :class:`~cellarium.ml.core.CellariumModule` or :class:`~cellarium.ml.core.CellariumPipeline`,
            ``x_ng`` key in the input dictionary will be overwritten with the normalized values.

            When ``x_ng`` is a :class:`scipy.sparse.spmatrix`, a CPU :class:`torch.sparse_csr_tensor` or a CPU
            :class:`torch.sparse_coo_tensor` (the layouts a dataloader produces, see
            :func:`~cellarium.ml.utilities.data.to_torch_sparse_csr`), so that this transform can be the first
            ``cpu_transform`` (before a :class:`~cellarium.ml.transforms.Filter`), the normalization is done with
            scipy (torch cannot reduce or broadcast over a sparse CSR tensor) and the result is returned in the
            same torch sparse layout it arrived in (CSR stays CSR, COO stays COO; scipy input becomes CSR). The
            input is not modified.

        Args:
            x_ng:
                Gene counts. A dense :class:`torch.Tensor`, a scipy sparse matrix, or a CPU sparse CSR or COO
                :class:`torch.Tensor`.
            total_mrna_umis_n:
                Total mRNA UMI counts per cell. If ``None``, it is computed from ``x_ng``.

        Returns:
            A dictionary with the following keys:

            - ``x_ng``: The gene counts normalized to target count.
        """
        if scipy.sparse.issparse(x_ng):
            return self._forward_sparse(
                scipy.sparse.csr_matrix(x_ng), total_mrna_umis_n, to_torch_sparse_fn=to_torch_sparse_csr
            )

        if isinstance(x_ng, torch.Tensor) and x_ng.layout in (torch.sparse_csr, torch.sparse_coo):
            to_torch_sparse_fn = to_torch_sparse_csr if x_ng.layout == torch.sparse_csr else to_torch_sparse_coo
            return self._forward_sparse(
                sparse_tensor_to_scipy_csr(x_ng), total_mrna_umis_n, to_torch_sparse_fn=to_torch_sparse_fn
            )

        if total_mrna_umis_n is None:
            total_mrna_umis_n = x_ng.sum(dim=-1)
        x_ng = self.target_count * x_ng / (total_mrna_umis_n[:, None].float() + self.eps)
        return {"x_ng": x_ng}

    def _forward_sparse(
        self,
        x_ng: scipy.sparse.csr_matrix,
        total_mrna_umis_n: torch.Tensor | None,
        to_torch_sparse_fn: Callable[[scipy.sparse.spmatrix], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sparse-input path for :meth:`forward`, with the same arithmetic as the dense path (in float32)."""
        data = x_ng.data.astype(np.float32, copy=False)
        if total_mrna_umis_n is None:
            total_n = np.asarray(x_ng.sum(axis=1), dtype=np.float32).ravel()
        else:
            total_n = total_mrna_umis_n.detach().cpu().numpy().astype(np.float32, copy=False)
        denominator_n = total_n + np.float32(self.eps)
        normalized_data = self.target_count * data / np.repeat(denominator_n, np.diff(x_ng.indptr))
        normalized = scipy.sparse.csr_matrix((normalized_data, x_ng.indices, x_ng.indptr), shape=x_ng.shape)
        return {"x_ng": to_torch_sparse_fn(normalized)}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_count={self.target_count}, eps={self.eps})"
