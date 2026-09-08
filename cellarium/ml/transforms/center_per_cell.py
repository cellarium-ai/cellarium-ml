# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch


class CenterPerCell(torch.nn.Module):
    """
    Center each cell by subtracting its mean across genes.
    """

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        cell_mean_n1 = x_ng.mean(dim=-1, keepdim=True)
        x_ng = x_ng - cell_mean_n1
        return {"x_ng": x_ng}
