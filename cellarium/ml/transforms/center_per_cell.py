# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch


class CenterPerCell(torch.nn.Module):
    """
    Center each cell by subtracting its mean across genes.

    NOTE: PFlog1pPF / shifted-CLR-style normalization [1] is:
        NormalizeTotal + Log1p + CenterPerCell

    References:
    [1] Booeshaghi, Hallgrimsdottir, Galvez-Merchan, Pachter. Depth normalization for single-cell genomics count data.
        bioRxiv (2026). https://www.biorxiv.org/content/10.1101/2022.05.06.490859v3
    """

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        cell_mean_n1 = x_ng.mean(dim=-1, keepdim=True)
        x_ng = x_ng - cell_mean_n1
        return {"x_ng": x_ng}
