# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class NormAdd(nn.Module):
    """
    Pre-norm layer where the layer normalization is applied before the sublayer.

    Args:
        norm_shape:
            The shape of the layer normalization.
        dropout_p:
            Dropout probability.
        use_bias:
            Whether to use bias in the layer normalization.
    """

    def __init__(self, norm_shape: int, dropout_p: float, use_bias: bool) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(norm_shape, bias=use_bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.ln.reset_parameters()

    def forward(
        self,
        hidden_state_ncd: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_state_ncd:
                Hidden state tensor of shape ``(n, c, d)``.
            sublayer:
                Sublayer function.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        return hidden_state_ncd + self.dropout(sublayer(self.ln(hidden_state_ncd)))  # _ncd
