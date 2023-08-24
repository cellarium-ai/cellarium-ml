# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
import torch
from transformers import BertForMaskedLM

from scvid.module import BaseModule


class Geneformer(BaseModule):
    def __init__(
        self,
        var_names_schema: Sequence,
        model: BertForMaskedLM,
        transform: torch.nn.Module,
        mlm_probability: float = 0.15,
    ):
        super().__init__()

        self.model = model
        self.transform = transform

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        var_names = tensor_dict["var_names"]
        return (x, var_names), {}

    def tokenize(self, x_ng: torch.Tensor, var_names: np.ndarray) -> None:
        # tokenize
        gene_tokens = (torch.arange(x_ng.shape[1], device=x_ng.device) + 2).expand(x_ng.shape)
        # sort by median-scaled gene values
        position_ids = torch.argsort(x_ng, dim=1, descending=True)
        position_ids = position_ids[:, : self.model.config.max_position_embeddings]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device).unsqueeze(-1)
        input_ids = gene_tokens[ndx, position_ids]
        # pad genes with zero expression
        sorted_x_ng = x_ng[ndx, position_ids]
        attention_mask = sorted_x_ng != 0
        input_ids.masked_fill_(~attention_mask, 0)
        return input_ids, attention_mask

    def forward(self, x_ng: torch.Tensor, var_names: np.ndarray) -> None:
        x_ng = self.transform(x_ng)

        # tokenize
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
        # normalize
        x_ng = self.transform(x_ng)

        # tokenize
        input_ids, attention_mask = self.tokenize(x_ng, var_names)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        return output, attention_mask
