# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cellarium.ml.utilities.layers import create_initializer


class PositionWiseFFN(nn.Module):
    """
    The positionwise feed-forward network.

    Args:
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        d_model:
            Dimensionality of the embeddings and hidden states.
        use_bias:
            Whether to use bias in the linear transformations.
        dense1_initializer:
            Initializer for the first dense layer.
        dense2_initializer:
            Initializer for the second dense layer.
    """

    def __init__(
        self,
        d_ffn: int,
        d_model: int,
        use_bias: bool,
        dense1_initializer: dict[str, Any],
        dense2_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_ffn, bias=use_bias)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(d_ffn, d_model, bias=use_bias)
        self.dense1_initializer = dense1_initializer
        self.dense2_initializer = dense2_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        create_initializer(self.dense1_initializer)(self.dense1.weight)
        if self.dense1.bias is not None:
            nn.init.zeros_(self.dense1.bias)

        create_initializer(self.dense2_initializer)(self.dense2.weight)
        if self.dense2.bias is not None:
            nn.init.zeros_(self.dense2.bias)

    def forward(self, hidden_state_ncd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state_ncd: Hidden state tensor of shape ``(n, c, d)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        return self.dense2(self.activation(self.dense1(hidden_state_ncd)))  # _ncd
