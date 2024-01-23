# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import nn


class Log1p(nn.Module):
    """
    Log1p transform gene counts.

    .. math::

        y_{ng} = \\log(1 + x_{ng})
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        .. note::

            When used with :class:`~cellarium.ml.core.CellariumModule` or :class:`~cellarium.ml.core.CellariumPipeline`,
            ``x_ng`` key in the input dictionary will be overwritten with the log1p transformed values.

        Args:
            x_ng: Gene counts.

        Returns:
            A dictionary with the following keys:

            - ``x_ng``: The log1p transformed gene counts.
        """
        x_ng = torch.log1p(x_ng)
        return {"x_ng": x_ng}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
