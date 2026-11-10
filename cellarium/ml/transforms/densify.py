# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import scipy.sparse
import torch
from torch import nn


class Densify(nn.Module):
    """
    Convert a sparse ``x_ng`` to a dense tensor on the current device.

    Use this as the first entry in ``transforms`` (GPU transforms) when ``x_ng`` arrives as a
    :class:`torch.sparse_csr_tensor` (or, on the ``mps`` accelerator, a :class:`torch.sparse_coo_tensor`
    \u2014 see :func:`~cellarium.ml.utilities.data.to_torch_sparse_coo`) \u2014 for example when no
    :class:`~cellarium.ml.transforms.Filter` cpu_transform is in the pipeline and the sparse-transfer
    strategy is still desired (e.g. for statistics models that operate on all genes).

    If ``x_ng`` is already dense this transform is a no-op.
    """

    def forward(self, x_ng: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.  May be a sparse (CSR or COO) :class:`torch.Tensor` or a dense tensor.

        Returns:
            A dictionary with the key ``x_ng`` containing a dense :class:`torch.Tensor`.

        Raises:
            TypeError: If ``x_ng`` is a scipy sparse matrix (which should have been converted earlier).
        """
        # check for scipy sparse matrices that should have been converted earlier
        if scipy.sparse.issparse(x_ng):
            raise TypeError(
                "Densify received a scipy sparse matrix, but sparse inputs should already "
                "have been converted to a torch sparse tensor earlier in the pipeline.\n\n"
                f"Received type: {type(x_ng).__name__}\n\n"
                "For sparse x_ng with Densify, use one of the following convert_fn options:\n"
                "  - keep_sparse: use when Filter is in model.cpu_transforms, so Filter converts "
                "scipy sparse input to torch.sparse_csr_tensor before GPU transfer.\n"
                "  - to_torch_sparse_csr: use when there is no Filter, since the output is already "
                "a torch sparse CSR tensor.\n"
                "  - to_torch_sparse_coo: same as to_torch_sparse_csr, but for the mps accelerator, "
                "which has no sparse CSR support."
            )

        if x_ng.layout != torch.strided:
            x_ng = x_ng.to_dense()
        return {"x_ng": x_ng}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
