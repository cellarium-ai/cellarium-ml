# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import torch
from torch import nn

from cellarium.ml.utilities.layers import create_initializer


class TokenEmbedding(nn.Module):
    """
    Gene and metadata tokens embedding.

    Args:
        categorical_vocab_sizes:
            Categorical token vocabulary sizes.
        continuous_tokens:
            Continuous tokens.
        d_model:
            Dimensionality of the embeddings and hidden states.
        embeddings_initializer:
            Initializer for the embeddings.
    """

    def __init__(
        self,
        categorical_vocab_sizes: dict[str, int],
        continuous_tokens: list[str],
        d_model: int,
        embeddings_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.embedding_dict = nn.ModuleDict()
        self.embedding_dict.update(
            {key: nn.Embedding(vocab_size, d_model) for key, vocab_size in categorical_vocab_sizes.items()}
        )
        self.embedding_dict.update({key: nn.Linear(1, d_model, bias=False) for key in continuous_tokens})
        self.categorical_vocab_sizes = categorical_vocab_sizes
        self.continuous_tokens = continuous_tokens
        self.embeddings_initializer = embeddings_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.embedding_dict.children():
            create_initializer(self.embeddings_initializer)(module.weight)

    def forward(self, token_nc_dict: dict[str, torch.Tensor], token_type_nc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_nc_dict:
                Dictionary of token tensors of shape ``(n, c)``.
            token_type_nc:
                Token type tensor of shape ``(n, c)``. Token type uses the bit representation to indicate which
                tokens are present in the input tensor element-wise. Digit positions (from the right) in the binary
                value correspond to the token index in the input tensor. A value of 1 indicates that the token is
                included, while a 0 means that the token is not included.

        Returns:
            Embedding tensor of shape ``(n, c, d)``.
        """
        return sum(
            self.embedding_dict[key](token_nc.unsqueeze(-1) if key in self.continuous_tokens else token_nc)
            * (token_type_nc >> i & 1).unsqueeze(-1)
            for i, (key, token_nc) in enumerate(token_nc_dict.items())
        )
