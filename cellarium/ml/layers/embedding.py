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

    def forward(
        self,
        token_value_nc_dict: dict[str, torch.Tensor],
        token_mask_nc_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            token_value_nc_dict:
                Dictionary of token value tensors of shape ``(n, c)``.
            token_mask_nc_dict:
                Dictionary of token mask tensors of shape ``(n, c)``.

        Returns:
            Embedding tensor of shape ``(n, c, d)``.
        """
        return sum(
            self.embedding_dict[key](token_value_nc.unsqueeze(-1) if key in self.continuous_tokens else token_value_nc)
            * token_mask_nc_dict[key].unsqueeze(-1)
            for i, (key, token_value_nc) in enumerate(token_value_nc_dict.items())
        )
