# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from cellarium.ml.transforms import CenterPerCell, Log1p, NormalizeTotal


class PFlogPF(torch.nn.Module):
    """
    PFlog1pPF / shifted-CLR-style normalization [1] is:
        NormalizeTotal + Log1p + CenterPerCell

    This is a convenience wrapper for that sequence of transforms, but it provides
    no additional functionality.

    References:
    [1] Booeshaghi, Hallgrimsdottir, Galvez-Merchan, Pachter. Depth normalization for single-cell genomics count data.
        bioRxiv (2026). https://www.biorxiv.org/content/10.1101/2022.05.06.490859v3
    """

    def __init__(self, target_count: int = 10_000):
        super().__init__()
        self.normalize_total = NormalizeTotal(target_count=target_count)
        self.log1p = Log1p()
        self.center_per_cell = CenterPerCell()

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        x_ng = self.normalize_total(x_ng)["x_ng"]
        x_ng = self.log1p(x_ng)["x_ng"]
        x_ng = self.center_per_cell(x_ng)["x_ng"]
        return {"x_ng": x_ng}
