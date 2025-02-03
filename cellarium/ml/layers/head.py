# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import torch
from torch import nn

from cellarium.ml.utilities.layers import create_initializer


class MultiHeadReadout(nn.Module):
    """
    Multi-head readout.

    Args:
        categorical_vocab_sizes:
            Categorical token vocabulary sizes.
        d_model:
            Dimensionality of the embeddings and hidden states.
        use_bias:
            Whether to use bias in the linear transformations.
        output_logits_scale:
            Multiplier for the output logits.
        heads_initializer:
            Initializer for the output linear transformations.
    """

    def __init__(
        self,
        categorical_vocab_sizes: dict[str, int],
        d_model: int,
        use_bias: bool,
        output_logits_scale: float,
        heads_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.W = nn.ModuleDict(
            {key: nn.Linear(d_model, vocab_size, use_bias) for key, vocab_size in categorical_vocab_sizes.items()}
        )
        self.output_logits_scale = output_logits_scale
        self.heads_initializer = heads_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.W.children():
            assert isinstance(module.weight, torch.Tensor)
            create_initializer(self.heads_initializer)(module.weight)

            if module.bias is not None:
                assert isinstance(module.bias, torch.Tensor)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_state_ncd: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_state_ncd:
                Hidden state tensor of shape ``(n, c, d)``.

        Returns:
            Dictionary of output logits tensors of shape ``(n, c, vocab_size)``.
        """
        return {key: self.output_logits_scale * self.W[key](hidden_state_ncd) for key in self.W}
