# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import BertForMaskedLM

from .base_module import BaseModule, PredictMixin


class Geneformer(BaseModule, PredictMixin):
    """
    Geneformer model.

    **References:**

    1. `Transfer learning enables predictions in network biology (Theodoris et al.)
       <https://www.nature.com/articles/s41586-023-06139-9>`_.

    Args:
        feature_schema:
            The list of the variable names in the input data.
        model:
            The bert model.
        mlm_probability:
            Ratio of tokens to mask for masked language modeling loss.
        transform:
            If not ``None`` is used to transform the input data.
        validate_input:
            If ``True`` the input data is validated.
    """

    def __init__(
        self,
        feature_schema: Sequence,
        model: BertForMaskedLM,
        mlm_probability: float,
        transform: torch.nn.Module | None = None,
        validate_input: bool = True,
    ):
        super().__init__()
        self.feature_schema = feature_schema
        self.model = model
        self.mlm_probability = mlm_probability
        self.transform = transform
        self.validate_input = validate_input
        self.feature_ids: torch.Tensor
        # ids for the features, 0 is for padding, 1 is for mask
        self.register_buffer("feature_ids", torch.arange(2, len(feature_schema) + 2))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        feature_list = tensor_dict["var_names"]
        return (x,), {"feature_list": feature_list}

    def tokenize(self, x_ng: torch.Tensor, feature_list: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_ids.expand(x_ng.shape)
        # sort by median-scaled gene values
        sorted_indices = torch.argsort(x_ng, dim=1, descending=True)
        sorted_indices = sorted_indices[:, : self.model.config.max_position_embeddings]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
        input_ids = tokens[ndx[:, None], sorted_indices]
        # mask out genes with zero expression
        sorted_x_ng = x_ng[ndx[:, None], sorted_indices]
        attention_mask = sorted_x_ng != 0
        # pad genes with zero expression
        input_ids[~attention_mask] = 0
        return input_ids, attention_mask

    def forward(self, x_ng: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert "feature_list" in kwargs, "feature_list must be provided."
        feature_list: Sequence = kwargs.pop("feature_list")

        if self.validate_input:
            assert x_ng.shape[1] == len(feature_list), "The number of x_ng columns must match the feature_list length."
            assert np.array_equal(feature_list, self.feature_schema), "feature_list must match the feature_schema."

        if self.transform is not None:
            x_ng = self.transform(x_ng)

        input_ids, attention_mask = self.tokenize(x_ng, feature_list)

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

    def predict(self, x_ng: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor | None]:
        """
        Send raw count data through the transformer and return outputs.
        Optionally perform in silico perturbations.

        NOTE:
            In silico perturbations can be achieved in one of three ways:
            1. use feature_map to replace a feature token with MASK (1) or PAD (0)
                e.g. feature_map={"ENSG0001": 0} will remove feature_schema feature
                ENSG0001 from the cell's inputs, replacing it with a PAD token
            2. use feature_deletion to remove a feature from the cell's
                inputs, which instead of adding a PAD or MASK token, will allow
                another feature to take its place.
                e.g. feature_zero_expression=["ENSG0001"] will remove feature_schema
                feature ENSG0001 from the cell's inputs, and allow a new feature
                token to take its place
            3.
            Number (2) and (3) are described in the Geneformer paper under
            "In silico perturbation" in the Methods section.

        Args:
            x_ng: cell by feature tensor to use for prediction
            **kwargs:
                feature_list: feature ids that specify order in x_ng
                feature_map: (optional) specify a mapping for in silico perturbations
                feature_deletion: (optional) specify features whose expression
                    should be set to zero before tokenization (remove from inputs)
                feature_activation: (optional) specify features whose expression
                    should be set to 1 + max(x_ng) before tokenization (front of inputs)
                output_hidden_states: (optional) True (default) to output the hidden states
                output_attentions: (optional) True (default) to output the attentions
        """

        assert "feature_list" in kwargs, "feature_list must be provided."
        feature_list: Sequence = kwargs.pop("feature_list")
        feature_map: dict[int, int] = kwargs.pop("feature_map", None)
        feature_deletion: Sequence = kwargs.pop("feature_deletion", None)
        feature_activation: Sequence = kwargs.pop("feature_activation", None)
        output_hidden_states: bool = kwargs.pop("output_hidden_states", True)
        output_attentions: bool = kwargs.pop("output_attentions", True)

        if self.validate_input:
            assert x_ng.shape[1] == len(feature_list), "The number of x_ng columns must match the feature_list length."
            assert np.array_equal(feature_list, self.feature_schema), "feature_list must match the feature_schema."
            if feature_map:
                for key in feature_map.keys():
                    assert key in self.feature_schema, f"feature_map key {key} is not in the feature_schema."
            if feature_deletion:
                assert all([f in self.feature_schema for f in feature_deletion]), \
                    "Elements of feature_deletion are not all in feature_schema"
            if feature_activation:
                assert all([f in self.feature_schema for f in feature_activation]), \
                    "Elements of feature_activation are not all in feature_schema"

        if feature_deletion:
            x_ng[:, np.array([f in feature_deletion for f in self.feature_schema])] = 0

        if feature_activation:
            large_value = x_ng.max() + 1
            x_ng[:, np.array([f in feature_activation for f in self.feature_schema])] = large_value

        if self.transform is not None:
            x_ng = self.transform(x_ng)

        if feature_map:
            # elementwise, apply map if entry is a key, otherwise do nothing
            x_ng.apply_(lambda x: feature_map.get(x, x))

        input_ids, attention_mask = self.tokenize(x_ng, feature_list)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return output
