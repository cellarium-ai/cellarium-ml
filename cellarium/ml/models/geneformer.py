# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
import torch
from transformers import BertConfig, BertForMaskedLM

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class Geneformer(CellariumModel, PredictMixin):
    """
    Geneformer model.

    **References:**

    1. `Transfer learning enables predictions in network biology (Theodoris et al.)
       <https://www.nature.com/articles/s41586-023-06139-9>`_.

    Args:
        var_names_g:
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
        var_names_g: Sequence[str],
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
        self.var_names_g = np.array(var_names_g)
        # model configuration
        config = {
            "vocab_size": len(self.var_names_g) + 2,  # number of genes + 2 for <mask> and <pad> tokens
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
        self.config = BertConfig(**config)
        self.bert = BertForMaskedLM(self.config)
        self.mlm_probability = mlm_probability
        self.feature_ids: torch.Tensor
        # ids for the features, 0 is for padding, 1 is for mask
        self.register_buffer("feature_ids", torch.arange(2, len(self.var_names_g) + 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.feature_ids = torch.arange(2, len(self.var_names_g) + 2)
        self.bert.bert.embeddings.position_ids = torch.arange(self.config.max_position_embeddings).expand((1, -1))
        self.bert.bert.embeddings.token_type_ids = torch.zeros(
            self.bert.bert.embeddings.position_ids.size(), dtype=torch.long
        )
        self.bert.apply(lambda module: setattr(module, "_is_hf_initialized", False))
        self.bert.init_weights()

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

    def tokenize_with_perturbations(
        self,
        x_ng: torch.Tensor,
        feature_activation: list[str] | None = None,
        feature_deletion: list[str] | None = None,
        feature_map: dict[str, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # activation and deletion happen before sorting
        if feature_deletion:
            if not all([g in self.var_names_g for g in feature_deletion]):
                raise ValueError("Some feature_deletion elements are not in self.var_names_g")
            deletion_logic_g = np.isin(self.var_names_g, feature_deletion)
            x_ng[:, deletion_logic_g] = 0
        if feature_activation:
            max_val = x_ng.max()
            for i, g in enumerate(feature_activation[::-1]):
                feature_logic_g = self.var_names_g == g
                if feature_logic_g.sum() != 1:
                    raise ValueError(f"feature_activation element {g} is not in self.var_names_g")
                top_rank_value = max_val + i + 1
                x_ng[:, feature_logic_g] = top_rank_value

        # tokenize and sort to give rank-ordered inputs
        input_ids, attention_mask = self.tokenize(x_ng)

        # feature map is applied after tokenization and sorting
        if feature_map:
            for g, target_token in feature_map.items():
                feature_logic_g = self.var_names_g == g
                if feature_logic_g.sum() != 1:
                    raise ValueError(f"feature_map key {g} not in self.var_names_g")
                initial_token = self.feature_ids[feature_logic_g]
                input_ids[input_ids == initial_token] = target_token

        return input_ids, attention_mask

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
        Returns:
            A dictionary with the loss value.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

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
        var_names_g: np.ndarray,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        output_input_ids: bool = True,
        output_attention_mask: bool = True,
        feature_activation: list[str] | None = None,
        feature_deletion: list[str] | None = None,
        feature_map: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Send (transformed) data through the model and return outputs.
        Optionally perform in silico perturbations and masking.

        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
            output_hidden_states:
                Whether to return all hidden-states.
            output_attentions:
                Whether to return all attentions.
            output_input_ids:
                Whether to return input ids.
            output_attention_mask:
                Whether to return attention mask.
            feature_activation:
                Specify features whose expression should be set to > max(x_ng) before tokenization (top rank).
            feature_deletion:
                Specify features whose expression should be set to zero before tokenization (remove from inputs).
            feature_map:
                Specify a mapping for input tokens, to be applied before model.

        Returns:
            A dictionary with the inference results.

        .. note::
            In silico perturbations can be achieved in one of three ways:

            1. Use ``feature_map`` to replace a feature token with ``MASK`` (1) or ``PAD`` (0)
               (e.g. ``feature_map={"ENSG0001": 1}`` will replace ``var_names_g`` feature
               ``ENSG0001`` with a ``MASK`` token).
            2. Use ``feature_deletion`` to remove a feature from the cell's inputs, which instead of adding a
               ``PAD`` or ``MASK`` token, will allow another feature to take its place
               (e.g. ``feature_deletion=["ENSG0001"]`` will remove ``var_names_g`` feature ``ENSG0001`` from the input,
               and allow a new feature token to take its place).
            3. Use ``feature_activation`` to move a feature all the way to the top rank position in the input
               (e.g. ``feature_activation=["ENSG0001"]`` will make ``var_names_g`` feature ``ENSG0001`` the first in
               rank order. Multiple input features will be ranked according to their order in the input list).

            Number (2) and (3) are described in the Geneformer paper under "In silico perturbation" in the
            Methods section.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        input_ids, attention_mask = self.tokenize_with_perturbations(
            x_ng, feature_activation=feature_activation, feature_deletion=feature_deletion, feature_map=feature_map
        )

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if output_input_ids:
            output["input_ids"] = input_ids
        if output_attention_mask:
            output["attention_mask"] = attention_mask
        return output
