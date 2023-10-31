# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import BertConfig, BertForMaskedLM

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)
from cellarium.ml.utilities.types import BatchDict


class Geneformer(CellariumModel, PredictMixin):
    """
    Geneformer model.

    **References:**

    1. `Transfer learning enables predictions in network biology (Theodoris et al.)
       <https://www.nature.com/articles/s41586-023-06139-9>`_.

    Args:
        feature_schema:
            The variable names schema for the input data validation.
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act:
            The non-linear activation function (function or string) in the encoder and pooler. If string, ``"gelu"``,
            ``"relu"``, ``"silu"`` and ``"gelu_new"`` are supported.
        hidden_dropout_prob:
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob:
            The dropout ratio for the attention probabilities.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size:
            The vocabulary size of the ``token_type_ids`` passed when calling :class:`transformers.BertModel`.
        initializer_range:
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        position_embedding_type:
            Type of position embedding. Choose one of ``"absolute"``, ``"relative_key"``, ``"relative_key_query"``. For
            positional embeddings use ``"absolute"``. For more information on ``"relative_key"``, please refer to
            `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`_.
            For more information on ``"relative_key_query"``, please refer to *Method 4* in `Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`_.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        mlm_probability:
            Ratio of tokens to mask for masked language modeling loss.
    """

    def __init__(
        self,
        feature_schema: Sequence[str],
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 4,
        intermediate_size: int = 512,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.02,
        attention_probs_dropout_prob: float = 0.02,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        position_embedding_type: str = "absolute",
        layer_norm_eps: float = 1e-12,
        mlm_probability: float = 0.15,
    ) -> None:
        super().__init__()
        self.feature_schema = np.array(feature_schema)
        # model configuration
        config = {
            "vocab_size": len(self.feature_schema) + 2,  # number of genes + 2 for <mask> and <pad> tokens
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "hidden_act": hidden_act,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "hidden_dropout_prob": hidden_dropout_prob,
            "max_position_embeddings": max_position_embeddings,
            "type_vocab_size": type_vocab_size,
            "initializer_range": initializer_range,
            "position_embedding_type": position_embedding_type,
            "layer_norm_eps": layer_norm_eps,
            "pad_token_id": 0,
        }
        config = BertConfig(**config)
        self.bert = BertForMaskedLM(config)
        self.mlm_probability = mlm_probability
        self.feature_ids: torch.Tensor
        # ids for the features, 0 is for padding, 1 is for mask
        self.register_buffer("feature_ids", torch.arange(2, len(self.feature_schema) + 2))

    def tokenize(self, x_ng: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_ids.expand(x_ng.shape)
        # sort by median-scaled gene values
        sorted_indices = torch.argsort(x_ng, dim=1, descending=True)
        sorted_indices = sorted_indices[:, : self.bert.config.max_position_embeddings]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
        input_ids = tokens[ndx[:, None], sorted_indices]
        # mask out genes with zero expression
        sorted_x_ng = x_ng[ndx[:, None], sorted_indices]
        attention_mask = sorted_x_ng != 0
        # pad genes with zero expression
        input_ids[~attention_mask] = 0
        return input_ids, attention_mask

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray) -> BatchDict:
        """
        Args:
            x_ng:
                Gene counts matrix.
            feature_g:
                The list of the variable names in the input data.

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        input_ids, attention_mask = self.tokenize(x_ng)

        labels = input_ids.clone()
        labels_probs = torch.full(labels.shape, self.mlm_probability, device=x_ng.device)
        labels_probs[~attention_mask] = 0
        masked_indices = torch.bernoulli(labels_probs).bool()
        labels[~masked_indices] = -100  # we only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=x_ng.device)).bool() & masked_indices
        input_ids[indices_replaced] = 1  # tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=x_ng.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(x_ng.shape[1], labels.shape, dtype=torch.long, device=x_ng.device)
        input_ids[indices_random] = random_words[indices_random]

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": output.loss}

    def predict(
        self,
        x_ng: torch.Tensor,
        feature_g: np.ndarray,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
    ) -> BatchDict:
        """
        Args:
            x_ng:
                Gene counts matrix.
            feature_g:
                The list of the variable names in the input data.

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        input_ids, attention_mask = self.tokenize(x_ng)

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return output
