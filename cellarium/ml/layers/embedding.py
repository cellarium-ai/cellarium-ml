# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import torch
from torch import nn

from cellarium.ml.utilities.layers import create_initializer


class GeneExpressionEmbedding(nn.Module):
    """
    Gene embedding.

    Args:
        categorical_vocab_sizes:
            Categorical gene token vocabulary sizes.
        continuous_tokens:
            Continuous gene tokens.
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

    def forward(self, gene_tokens_nc_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gene_tokens_nc_dict:
                Dictionary of gene token tensors of shape ``(n, c)``.

        Returns:
            The gene embedding tensor of shape ``(n, c, d)``.
        """
        return sum(
            self.embedding_dict[key](gene_token_nc.unsqueeze(-1) if key in self.continuous_tokens else gene_token_nc)
            for key, gene_token_nc in gene_tokens_nc_dict.items()
        )


class MetadataEmbedding(nn.Module):
    """
    Metadata embedding.

    Args:
        categorical_vocab_sizes:
            Categorical metadata token vocabulary sizes.
        d_model:
            Dimensionality of the embeddings and hidden states.
        initializer:
            Initializer for the embeddings.
    """

    def __init__(
        self,
        categorical_vocab_sizes: dict[str, int],
        d_model: int,
        embeddings_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.embedding_dict = nn.ModuleDict(
            {key: nn.Embedding(vocab_size, d_model) for key, vocab_size in categorical_vocab_sizes.items()}
        )
        self.embeddings_initializer = embeddings_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.embedding_dict.children():
            create_initializer(self.embeddings_initializer)(module.weight)

    def forward(self, metadata_tokens_nc_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            metadata_token_nc_dict:
                Dictionary of metadata token tensors of shape ``(n, c)``.

        Returns:
            Dictionary of metadata embedding tensors of shape ``(n, c, d)``.
        """
        return {
            key: self.embedding_dict[key](metadata_token_nc)
            for key, metadata_token_nc in metadata_tokens_nc_dict.items()
        }
