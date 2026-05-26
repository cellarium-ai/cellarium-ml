# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import scipy.sparse
import torch
from torch import nn


class Densify(nn.Module):
    """
    Convert a sparse ``x_ng`` to a dense tensor on the current device.

    Use this as the first entry in ``transforms`` (GPU transforms) when ``x_ng`` arrives as a
    :class:`torch.sparse_csr_tensor` \u2014 for example when no :class:`~cellarium.ml.transforms.Filter`
    cpu_transform is in the pipeline and the sparse-transfer strategy is still desired (e.g. for
    statistics models that operate on all genes).

    If ``x_ng`` is already dense this transform is a no-op.
    """

    def forward(self, x_ng: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.  May be a :class:`torch.sparse_csr_tensor` or a dense tensor.

        Returns:
            A dictionary with the key ``x_ng`` containing a dense :class:`torch.Tensor`.

        Raises:
            TypeError: If ``x_ng`` is a scipy sparse matrix (which should have been converted earlier).
        """
        # check for scipy sparse matrices that should have been converted earlier
        try:
            if scipy.sparse.issparse(x_ng):
                raise TypeError(
                    "Densify received a scipy sparse matrix, which should have been converted to " \
                    "torch.sparse_csr_tensor earlier in the pipeline. \n\n"
                    "The convert_fn used for x_ng (sparse mode, where Densify is used) must be either:\n"
                    "    - keep_sparse: ensure Filter is in model.cpu_transforms to convert scipy sparse to "
                    "torch.sparse_csr_tensor before GPU transfer.\n"
                    "    - to_torch_sparse_csr: can be used with no Filter, since CSR is torch.\n"
                )
        except TypeError:
            raise

        if x_ng.is_sparse_csr:
            x_ng = x_ng.to_dense()
        return {"x_ng": x_ng}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
