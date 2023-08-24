# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
import torch
from transformers import BertForMaskedLM

from scvid.module import BaseModule


class Geneformer(BaseModule):
    """
    Geneformer model.

    **Reference:**

    1. *Transfer learning enables predictions in network biology*,
       C. V. Theodoris, L. Xiao, A. Chopra, M. D. Chaffin, Z. R. Al Sayed,
       M. C. Hill, H. Mantineo, E. Brydon, Z. Zeng, X. S. Liu & P. T. Ellinor
       (https://www.nature.com/articles/s41586-023-06139-9)

    Args:
        var_names_schema:
            The list of the variable names in the input data.
        transform:
            The transform to apply to the input data.
        mlm_probability:
            Ratio of tokens to mask for masked language modeling loss.
        model:
            The bert model.
    """

    def __init__(
        self, var_names_schema: Sequence, transform: torch.nn.Module, mlm_probability: float, model: BertForMaskedLM
    ):
        super().__init__()
        self.var_names_schema = var_names_schema
        self.transform = transform
        self.mlm_probability = mlm_probability
        self.model = model
        self.register_buffer("var_ids", torch.arange(2, len(var_names_schema) + 2))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        var_names = tensor_dict["var_names"]
        return (x, var_names), {}

    def tokenize(self, x_ng: torch.Tensor, var_names: Sequence) -> None:
        tokens = self.var_ids.expand(x_ng.shape)
        # sort by median-scaled gene values
        position_ids = torch.argsort(x_ng, dim=1, descending=True)
        position_ids = position_ids[:, : self.model.config.max_position_embeddings]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
        input_ids = tokens[ndx[:, None], position_ids]
        # pad genes with zero expression
        sorted_x_ng = x_ng[ndx, position_ids]
        attention_mask = sorted_x_ng != 0
        input_ids.masked_fill_(~attention_mask, 0)
        return input_ids, attention_mask

    def forward(self, x_ng: torch.Tensor, var_names: Sequence) -> torch.Tensor:
        # validate
        assert x_ng.shape[1] == len(var_names), "x_ng must have the same number of columns as var_names."
        assert np.array_equal(var_names, self.var_names_schema), "var_names must match the schema."

        x_ng = self.transform(x_ng)

        input_ids, attention_mask = self.tokenize(x_ng, var_names)

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

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output.loss

    def predict(self, x_ng: torch.Tensor, var_names: Sequence):
        # validate
        assert x_ng.shape[1] == len(var_names), "x_ng must have the same number of columns as var_names."
        assert np.array_equal(var_names, self.var_names_schema), "var_names must match the schema."

        x_ng = self.transform(x_ng)

        input_ids, attention_mask = self.tokenize(x_ng, var_names)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        return output
