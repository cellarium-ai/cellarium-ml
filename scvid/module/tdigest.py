# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import crick
import torch

from .base_module import BaseModule


class TDigest(BaseModule):
    """
    Compute an approximate non-zero histogram of the distribution of each gene in a batch of
    cells using t-digests.

    **Reference**:

    1. Dunning, Ted, and Otmar Ertl. "Computing Extremely Accurate
       Quantiles Using T-Digests." https://github.com/tdunning/t-digest/blob/
       master/docs/t-digest-paper/histo.pdf

    Args:
        g_genes: Number of genes.
        transform: If not ``None`` is used to transform the input data.
    """

    def __init__(self, g_genes: int, transform: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.g_genes = g_genes
        self.transform = transform
        self.tdigests = [crick.tdigest.TDigest() for _ in range(g_genes)]
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, torch.Tensor]
    ) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng: torch.Tensor) -> None:
        if self.transform is not None:
            x_ng = self.transform(x_ng)

        for i, tdigest in enumerate(self.tdigests):
            x_n = x_ng[:, i]
            nonzero_mask = torch.nonzero(x_n)
            if len(nonzero_mask) > 0:
                tdigest.update(x_n[nonzero_mask])

    @property
    def median_g(self) -> torch.Tensor:
        return torch.as_tensor([tdigest.quantile(0.5) for tdigest in self.tdigests])
