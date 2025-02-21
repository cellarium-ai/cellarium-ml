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
        continuous_vocab_sizes:
            Continuous gene token vocabulary sizes.
        d_model:
            Dimensionality of the embeddings and hidden states.
        embeddings_initializer:
            Initializer for the embeddings.
    """

    def __init__(
        self,
        categorical_vocab_sizes: dict[str, int],
        continuous_vocab_sizes: dict[str, int],
        d_model: int,
        embeddings_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.E = nn.ModuleDict()
        self.E.update({key: nn.Embedding(vocab_size, d_model) for key, vocab_size in categorical_vocab_sizes.items()})
        self.E.update(
            {key: nn.Linear(vocab_size, d_model, bias=False) for key, vocab_size in continuous_vocab_sizes.items()}
        )
        self.embeddings_initializer = embeddings_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.E.children():
            create_initializer(self.embeddings_initializer)(module.weight)

    def forward(self, gene_tokens_nc: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gene_tokens_nc:
                Dictionary of gene token tensors of shape ``(n, c)``.

        Returns:
            The gene embedding tensor of shape ``(n, c, d)``.
        """
        return sum(self.E[key](gene_token_nc) for key, gene_token_nc in gene_tokens_nc.items())


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
        self.E = nn.ModuleDict(
            {key: nn.Embedding(vocab_size, d_model) for key, vocab_size in categorical_vocab_sizes.items()}
        )
        self.embeddings_initializer = embeddings_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.E.children():
            create_initializer(self.embeddings_initializer)(module.weight)

    def forward(self, metadata_tokens_n: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            metadata_token_n:
                Dictionary of metadata token tensors of shape ``(n,)``.

        Returns:
            The metadata embedding tensor of shape ``(n, m, d)``.
        """
        return torch.stack(
            [self.E[key](metadata_token_n) for key, metadata_token_n in metadata_tokens_n.items()],
            dim=1,
        )
