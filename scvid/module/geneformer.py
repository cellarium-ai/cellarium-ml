# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from transformers import BertConfig, BertForMaskedLM

from scvid.module import BaseModule
from scvid.transforms.transforms import NonZeroMedianNormalize


class Geneformer(BaseModule):
    def __init__(self, g_genes: int, tdigest_path: str | None = None, median_path: str | None = None, model=None):
        super().__init__()

        if model is None:
            # set model parameters
            # model type
            model_type = "bert"
            # max input size
            max_input_size = 2**11  # 2048
            # number of layers
            num_layers = 6
            # number of attention heads
            num_attn_heads = 4
            # number of embedding dimensions
            num_embed_dim = 256
            # intermediate size
            intermed_size = num_embed_dim * 2
            # activation function
            activ_fn = "relu"
            # initializer range, layer norm, dropout
            initializer_range = 0.02
            layer_norm_eps = 1e-12
            attention_probs_dropout_prob = 0.02
            hidden_dropout_prob = 0.02

            # model configuration
            config = {
                "hidden_size": num_embed_dim,
                "num_hidden_layers": num_layers,
                "initializer_range": initializer_range,
                "layer_norm_eps": layer_norm_eps,
                "attention_probs_dropout_prob": attention_probs_dropout_prob,
                "hidden_dropout_prob": hidden_dropout_prob,
                "intermediate_size": intermed_size,
                "hidden_act": activ_fn,
                "max_position_embeddings": max_input_size,
                "model_type": model_type,
                "num_attention_heads": num_attn_heads,
                "pad_token_id": 0,
                "vocab_size": g_genes + 2,  # genes+2 for <mask> and <pad> tokens
            }

            config = BertConfig(**config)
            self.model = BertForMaskedLM(config)
        else:
            self.model = model

        # benchmark_v1 non-zero median
        if tdigest_path is not None:
            tdigest = torch.load(
                # "runs/benchmark_v1/tdigest/lightning_logs/version_2/checkpoints/module_checkpoint.pt"
                tdigest_path
            )
            self.transform = NonZeroMedianNormalize(
                tdigest.median_g,
                target_count=tdigest.transform.target_count,
                eps=tdigest.transform.eps,
            )
        elif median_path is not None:
            median = torch.load(median_path)
            self.transform = NonZeroMedianNormalize(
                median,
                target_count=10_000,
                eps=0,
            )

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, torch.Tensor]
    ) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng):
        # normalize
        x_ng = self.transform(x_ng)

        # tokenize
        gene_tokens = (torch.arange(x_ng.shape[1], device=x_ng.device) + 2).expand(
            x_ng.shape
        )
        # sort by median-scaled gene values
        position_ids = torch.argsort(x_ng, dim=1, descending=True)
        position_ids = position_ids[:, : self.model.config.max_position_embeddings]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device).unsqueeze(-1)
        input_ids = gene_tokens[ndx, position_ids]
        # pad genes with zero expression
        sorted_x_ng = x_ng[ndx, position_ids]
        attention_mask = sorted_x_ng != 0
        input_ids.masked_fill_(~attention_mask, 0)

        labels = input_ids.clone()
        mlm_probability = 0.15
        labels_probs = torch.full(labels.shape, mlm_probability, device=x_ng.device)
        labels_probs[~attention_mask] = 0
        masked_indices = torch.bernoulli(labels_probs).bool()
        labels[~masked_indices] = -100  # we only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=x_ng.device)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = 1  # tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=x_ng.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            x_ng.shape[1], labels.shape, dtype=torch.long, device=x_ng.device
        )
        input_ids[indices_random] = random_words[indices_random]

        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return x.loss

    def predict(self, x_ng):
        # normalize
        x_ng = self.transform(x_ng)

        # tokenize
        gene_tokens = (torch.arange(x_ng.shape[1], device=x_ng.device) + 2).expand(
            x_ng.shape
        )
        # sort by median-scaled gene values
        position_ids = torch.argsort(x_ng, dim=1, descending=True)
        position_ids = position_ids[:, : self.model.config.max_position_embeddings]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device).unsqueeze(-1)
        input_ids = gene_tokens[ndx, position_ids]
        # pad genes with zero expression
        sorted_x_ng = x_ng[ndx, position_ids]
        attention_mask = sorted_x_ng != 0
        input_ids.masked_fill_(~attention_mask, 0)


        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        return output, attention_mask
