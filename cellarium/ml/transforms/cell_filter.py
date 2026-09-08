# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn


class CellFilter(nn.Module):
    """
    Filter cells from the batch by a minimum quality threshold.

    Exactly one of ``min_count_per_cell`` or ``min_nonzero_genes_per_cell`` may be
    set to a positive value; setting both is an error.

    Any :class:`torch.Tensor` value in the batch whose first dimension matches the
    number of input cells (``x_ng.shape[0]``) is filtered consistently.  Non-tensor
    values and tensors whose first dimension differs from ``n`` (e.g. gene-indexed
    arrays) are left unchanged.

    When both thresholds are ``0`` (default) the transform is a no-op: an empty dict
    is returned and the pipeline merge leaves the batch unmodified.  The transform is
    also a no-op during predict mode (``_predict_mode=True`` in the batch dict, set
    automatically by :meth:`~cellarium.ml.core.CellariumPipeline.predict`).

    .. warning::

        ``CellFilter`` is **incompatible with** :class:`~cellarium.ml.models.ContrastiveMLP`.
        The NT-Xent loss asserts a fixed, world-size-divisible batch size and computes
        positive-pair offsets from it; a variable per-batch cell count breaks both the
        assertion and the pair alignment, even on a single device.

    Args:
        min_count_per_cell:
            Minimum total count (``x_ng`` row sum) required for a cell to be retained.
            ``0`` disables this filter.
        min_nonzero_genes_per_cell:
            Minimum number of genes with nonzero expression required for a cell to be
            retained, computed as ``(x_ng > 0).sum(dim=-1)``.  ``0`` disables this filter.
    """

    def __init__(self, min_count_per_cell: int = 0, min_nonzero_genes_per_cell: int = 0) -> None:
        super().__init__()
        if min_count_per_cell < 0:
            raise ValueError(f"min_count_per_cell must be >= 0, got {min_count_per_cell}")
        if min_nonzero_genes_per_cell < 0:
            raise ValueError(f"min_nonzero_genes_per_cell must be >= 0, got {min_nonzero_genes_per_cell}")
        if min_count_per_cell > 0 and min_nonzero_genes_per_cell > 0:
            raise ValueError(
                "min_count_per_cell and min_nonzero_genes_per_cell are mutually exclusive; "
                "set at most one to a positive value."
            )
        self.min_count_per_cell = min_count_per_cell
        self.min_nonzero_genes_per_cell = min_nonzero_genes_per_cell

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            kwargs:
                Full batch dictionary forwarded by :class:`~cellarium.ml.core.CellariumPipeline`
                via :func:`~cellarium.ml.utilities.core.call_func_with_batch`.  Must contain
                ``x_ng``.

        Returns:
            A dictionary of filtered tensors to merge back into the batch.  Only tensors
            whose first dimension equals ``n`` are included; all others are omitted so the
            pipeline merge leaves them unchanged.  Returns ``{}`` when both thresholds are
            ``0`` or when ``_predict_mode`` is set.
        """
        if kwargs.get("_predict_mode", False):
            return {}

        x_ng: torch.Tensor = kwargs["x_ng"]
        n = x_ng.shape[0]

        if self.min_count_per_cell > 0:
            mask = x_ng.sum(dim=-1) >= self.min_count_per_cell
        elif self.min_nonzero_genes_per_cell > 0:
            mask = (x_ng > 0).sum(dim=-1) >= self.min_nonzero_genes_per_cell
        else:
            return {}

        return {
            key: value[mask]
            for key, value in kwargs.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == n
        }

    def __repr__(self) -> str:
        if self.min_count_per_cell > 0:
            return f"{self.__class__.__name__}(min_count_per_cell={self.min_count_per_cell})"
        if self.min_nonzero_genes_per_cell > 0:
            return f"{self.__class__.__name__}(min_nonzero_genes_per_cell={self.min_nonzero_genes_per_cell})"
        return f"{self.__class__.__name__}()"
