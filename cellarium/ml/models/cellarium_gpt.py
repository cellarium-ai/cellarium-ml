# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin


def prefix_diagonal_mask(context_len: int, prefix_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate a prefix diagonal mask for self-attention.

    Args:
        context_len:
            The length of the context.
        prefix_len:
            The length of the prefix.
        device:
            The device to create the mask on.

    Returns:
        torch.Tensor: The prefix diagonal mask.

    Example:
        For context_len = 5, and prefix_len = 2, the mask is:
        [[1, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 0, 1, 0],
         [1, 1, 0, 0, 1]]
    """
    mask_cc = torch.eye(context_len, dtype=torch.bool, device=device)
    mask_cc[:, :prefix_len] = True
    return mask_cc


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        use_bias:
            Whether to use bias in the linear transformations.
        n_heads:
            Number of attention heads.
        dropout_p:
            Dropout probability.
        attn_mult:
            Multiplier for the attention scores.
        attn_backend:
            Backend for the attention computation.
    """

    backend_map = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    }

    def __init__(
        self,
        d_model: int,
        use_bias: bool,
        n_heads: int,
        dropout_p: float,
        attn_mult: float,
        attn_backend: Literal["math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wk = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wv = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wo = nn.Linear(d_model, d_model, bias=use_bias)
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.attn_mult = attn_mult
        self.attn_backend = attn_backend

    @staticmethod
    def split_heads(X_nqd: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads."""
        X_nqhk = X_nqd.reshape(X_nqd.shape[0], X_nqd.shape[1], n_heads, -1)
        X_nhqk = X_nqhk.permute(0, 2, 1, 3)
        return X_nhqk

    @staticmethod
    def merge_heads(X_nhqk: torch.Tensor) -> torch.Tensor:
        """Reverse of split_heads."""
        X_nqhk = X_nhqk.permute(0, 2, 1, 3)
        X_nqd = X_nqhk.reshape(X_nqhk.shape[0], X_nqhk.shape[1], -1)
        return X_nqd

    def forward(
        self,
        query_ncd: torch.Tensor,
        key_ncd: torch.Tensor,
        value_ncd: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        device = query_ncd.device
        c = query_ncd.shape[1]

        n_heads = self.n_heads
        query_ncd = self.Wq(query_ncd)
        key_ncd = self.Wk(key_ncd)
        value_ncd = self.Wv(value_ncd)
        # d = k * h
        query_nhck = self.split_heads(query_ncd, n_heads)
        key_nhck = self.split_heads(key_ncd, n_heads)
        value_nhck = self.split_heads(value_ncd, n_heads)

        attn_mask_cc = prefix_diagonal_mask(c, prefix_len, device)
        k = query_nhck.shape[3]
        scale_factor = self.attn_mult / k
        with sdpa_kernel(self.backend_map[self.attn_backend]):
            output_nhck = nn.functional.scaled_dot_product_attention(
                query_nhck,
                key_nhck,
                value_nhck,
                attn_mask=attn_mask_cc,
                dropout_p=self.dropout_p,
                scale=scale_factor,
            )

        output_ncd = self.merge_heads(output_nhck)
        return self.Wo(output_ncd)  # _ncd


class PositionWiseFFN(nn.Module):
    """
    The positionwise feed-forward network.

    Args:
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        d_model:
            Dimensionality of the embeddings and hidden states.
        use_bias:
            Whether to use bias in the linear transformations.
    """

    def __init__(self, d_ffn: int, d_model: int, use_bias: bool) -> None:
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_ffn, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(d_ffn, d_model, bias=use_bias)

    def forward(self, hidden_state_ncd: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(hidden_state_ncd)))  # _ncd


class ValueEmbedding(nn.Module):
    """
    Continuous value embedding.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        use_bias:
            Whether to use bias in the linear transformations.
    """

    def __init__(self, d_model: int, use_bias: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, d_model, bias=use_bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model, d_model, bias=use_bias)
        self.Em = nn.Embedding(1, d_model)

    def forward(self, value_nc: torch.Tensor) -> torch.Tensor:
        device = value_nc.device
        # value_nc[value_nc == float("-inf")] = -1
        # return self.fc2(self.relu(self.fc1(value_nc.unsqueeze(-1))))  # _ncd
        value_embedding_ncd = self.fc2(self.relu(self.fc1(value_nc.unsqueeze(-1))))  # _ncd
        mask_embedding_d = self.Em(torch.tensor(0, device=device))
        value_embedding_ncd[value_nc == float("-inf")] = mask_embedding_d
        return value_embedding_ncd


class GeneEmbedding(nn.Module):
    def __init__(self, n_genes: int, d_model: int, use_bias: bool) -> None:
        super().__init__()
        # gene ID embedding
        self.Ei = nn.Embedding(n_genes, d_model)
        # gene value embedding
        self.Ev = ValueEmbedding(d_model, use_bias=use_bias)
        # linear transformation for gene embeddings
        self.Wg = nn.Linear(3 * d_model, d_model, bias=use_bias)

    def forward(
        self, gene_id_nc: torch.Tensor, gene_value_nc: torch.Tensor, total_mrna_umis_n: torch.Tensor
    ) -> torch.Tensor:
        gene_id_embedding_ncd = self.Ei(gene_id_nc)
        gene_value_embedding_ncd = self.Ev(torch.log1p(gene_value_nc))
        total_mrna_umis_embedding_ncd = self.Ev(torch.log1p(total_mrna_umis_n[:, None])).expand(
            gene_id_embedding_ncd.shape
        )

        gene_embedding_ncd = self.Wg(
            torch.cat([gene_id_embedding_ncd, gene_value_embedding_ncd, total_mrna_umis_embedding_ncd], dim=-1)
        )

        return gene_embedding_ncd


class NormAdd(nn.Module):
    """
    Pre-norm layer where the layer normalization is applied before the sublayer.

    Args:
        norm_shape:
            The shape of the layer normalization.
        dropout_p:
            Dropout probability.
    """

    def __init__(self, norm_shape: int, dropout_p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, hidden_state_ncd: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return hidden_state_ncd + self.dropout(sublayer(self.ln(hidden_state_ncd)))  # _ncd


class TransformerBlock(nn.Module):
    """
    Transformer block.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        n_heads:
            Number of attention heads.
        dropout_p:
            Dropout probability.
        attn_mult:
            Multiplier for the attention scores.
        use_bias:
            Whether to use bias in the linear transformations.
        attn_backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        dropout_p: float,
        attn_mult: float,
        use_bias: bool,
        attn_backend: Literal["math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, use_bias, n_heads, dropout_p, attn_mult, attn_backend)
        self.normadd1 = NormAdd(d_model, dropout_p)
        self.ffn = PositionWiseFFN(d_ffn, d_model, use_bias)
        self.normadd2 = NormAdd(d_model, dropout_p)

    def forward(self, hidden_state_ncd: torch.Tensor, prefix_len: int) -> torch.Tensor:
        hidden_state_ncd = self.normadd1(hidden_state_ncd, lambda X: self.attention(X, X, X, prefix_len))
        return self.normadd2(hidden_state_ncd, lambda Y: self.ffn(Y))  # _ncd


class Transformer(nn.Module):
    """
    Transformer model.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        n_heads:
            Number of attention heads.
        n_blocks:
            Number of transformer blocks.
        dropout_p:
            Dropout probability.
        use_bias:
            Whether to use bias in the linear transformations.
        attn_mult:
            Multiplier for the attention scores.
        attn_backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        use_bias: bool,
        attn_mult: float,
        attn_backend: Literal["math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_ffn, n_heads, dropout_p, attn_mult, use_bias, attn_backend)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, hidden_state_ncd: torch.Tensor, prefix_len: int) -> torch.Tensor:
        for block in self.blocks:
            hidden_state_ncd = block(hidden_state_ncd, prefix_len)

        return hidden_state_ncd


class CellariumGPT(CellariumModel, ValidateMixin, PredictMixin):
    """
    Cellarium GPT model.

    Args:
        gene_categories:
            Gene ID categories.
        d_model:
            Dimensionality of the embeddings and hidden states.
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        n_heads:
            Number of attention heads.
        n_blocks:
            Number of transformer blocks.
        dropout_p:
            Dropout probability.
        use_bias:
            Whether to use bias in the linear transformations.
        max_prefix_len:
            Maximum length of the prefix. Must be greater than or equal to 1.
        min_suffix_len:
            Minimum length of the suffix. Must be greater than or equal to 1.
        max_value:
            Maximum count value (inclusive).
        attn_mult:
            Multiplier for the attention scores.
        input_mult:
            Multiplier for the input embeddings.
        output_mult:
            Multiplier for the output embeddings.
        initializer_range:
            The standard deviation of the truncated normal initializer.
        backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        gene_categories: pd.Index,
        d_model: int = 256,
        d_ffn: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        dropout_p: float = 0.0,
        use_bias: bool = False,
        max_prefix_len: int = 1000,
        suffix_len: int | None = None,
        context_len: int | None = None,
        max_value: int = 1000,
        attn_mult: float = 6.0,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        initializer_range: float = 0.02,
        attn_backend: Literal["math", "flash", "mem_efficient"] = "mem_efficient",
    ) -> None:
        super().__init__()
        self.gene_categories = gene_categories
        self.n_genes = len(gene_categories)
        self.max_value = max_value
        if (suffix_len is None) == (context_len is None):
            raise ValueError("Either `suffix_len` or `context_len` must be specified, but not both.")
        if context_len is not None:
            if context_len > self.n_genes:
                raise ValueError(
                    "`context_len` must be less than or equal to the number of genes. "
                    f"Got {context_len} > {self.n_genes}."
                )
            if max_prefix_len >= context_len:
                raise ValueError(
                    "`max_prefix_len` must be less than `context_len`. Got {max_prefix_len} >= {context_len}."
                )
        if suffix_len is not None:
            if max_prefix_len + suffix_len > self.n_genes:
                raise ValueError(
                    "`max_prefix_len + suffix_len` must be less than or equal to the number of genes. "
                    f"Got {max_prefix_len + suffix_len} > {self.n_genes}."
                )
        self.max_prefix_len = max_prefix_len
        self.suffix_len = suffix_len
        self.context_len = context_len
        self.input_mult = input_mult
        self.output_mult = output_mult

        self.gene_embedding = GeneEmbedding(self.n_genes, d_model, use_bias)
        self.transformer = Transformer(
            d_model,
            d_ffn,
            n_heads,
            n_blocks,
            dropout_p,
            use_bias,
            attn_mult,
            attn_backend,
        )
        self.head = nn.Linear(d_model, self.max_value + 1, use_bias)
        self.initializer_range = initializer_range
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # assert self.optimizer == "adam", "Only Adam(W) optimizer is supported for now."
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "dense2.weight" or name == "Wo.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.initializer_range / math.sqrt(2 * self.transformer.n_blocks)))

    @cached_property
    def token_to_id(self) -> dict[str, int]:
        return {var_name: i for i, var_name in enumerate(self.gene_categories)}

    @cached_property
    def vectorized_token_to_id(self):
        return np.vectorize(lambda x: self.token_to_id[x])

    def forward(
        self,
        gene_value_ng: torch.Tensor,
        gene_id_g: torch.Tensor,
        gene_categories: pd.Index,
        total_mrna_umis_n: torch.Tensor,
        batch_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        assert gene_categories.equals(self.gene_categories)

        device = gene_id_g.device
        gene_id_ng = gene_id_g.expand(gene_value_ng.shape)

        # sample the prefix length
        if batch_idx is not None:
            rng = torch.Generator(device=device)
            rng.manual_seed(batch_idx)
            prefix_len = int(torch.randint(0, self.max_prefix_len + 1, (), generator=rng, device=device))
        else:
            prefix_len = int(torch.randint(0, self.max_prefix_len + 1, (), device=device))

        # compute the context length and suffix length
        if self.context_len is not None:
            # use the fixed context length
            context_len = self.context_len
            suffix_len = context_len - prefix_len
        elif self.suffix_len is not None:
            # compute the context length dynamically based on the prefix length
            suffix_len = self.suffix_len
            context_len = prefix_len + suffix_len

        # shuffle the genes
        shuffle_idx_ng = torch.argsort(torch.rand_like(gene_value_ng), dim=-1)
        shuffle_idx_nc = shuffle_idx_ng[:, :context_len]
        gene_id_nc = torch.gather(gene_id_ng, dim=-1, index=shuffle_idx_nc)
        gene_value_nc = torch.gather(gene_value_ng, dim=-1, index=shuffle_idx_nc)

        # compute the target and mask target values
        label_ns = gene_value_nc[:, -suffix_len:].long()
        gene_value_nc[:, -suffix_len:] = -1

        # embed the gene IDs, values, and total mRNA UMIs
        gene_embedding_ncd = self.gene_embedding(gene_id_nc, gene_value_nc, total_mrna_umis_n)

        # compute logits
        hidden_state_ncd = gene_embedding_ncd * self.input_mult
        hidden_state_ncd = self.transformer(hidden_state_ncd, prefix_len)
        logits_nsm = self.head(hidden_state_ncd[:, -suffix_len:]) * self.output_mult

        # compute the loss
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        label_mask_ns = label_ns < self.max_value + 1
        label_weight_ns = 1 / label_mask_ns.sum(dim=-1, keepdim=True).expand(-1, suffix_len)
        label_weights = label_weight_ns[label_mask_ns]
        logits = logits_nsm[label_mask_ns]
        labels = label_ns[label_mask_ns]
        loss = (loss_fn(logits, labels) * label_weights).sum() / label_weights.sum()

        return {"loss": loss}

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        gene_id_g: torch.Tensor,
        gene_categories: pd.Series,
        gene_value_ng: torch.Tensor,
        total_mrna_umis_n: torch.Tensor,
        obs_names_n: np.ndarray,
        batch_idx: int,
    ) -> None:
        device = gene_id_g.device
        gene_id_ng = gene_id_g.expand(gene_value_ng.shape)
        n, g = gene_id_ng.shape

        loss = self.forward(gene_value_ng, gene_id_g, gene_categories, total_mrna_umis_n, batch_idx=batch_idx)["loss"]
        pl_module.log("val_loss", loss, sync_dist=True, on_epoch=True)

        losses: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
        suffix_len = 100
        n_seeds = 3
        prefix_lens = [0, 49, 499, 999, 1999, 3999, 7999]
        for i in range(n_seeds):
            rng_n = [torch.Generator(device=device) for _ in range(n)]
            [rng.manual_seed(int(obs_name) + i) for rng, obs_name in zip(rng_n, obs_names_n)]
            shuffle_idx_ng = torch.stack([torch.randperm(g, generator=rng, device=device) for rng in rng_n])

            for prefix_len in prefix_lens:
                shuffle_idx_nc = torch.cat(
                    [
                        shuffle_idx_ng[:, :prefix_len],
                        shuffle_idx_ng[:, -suffix_len:],
                    ],
                    dim=1,
                )
                gene_id_nc = torch.gather(gene_id_ng, dim=-1, index=shuffle_idx_nc)
                gene_value_nc = torch.gather(gene_value_ng, dim=-1, index=shuffle_idx_nc)
                label_ns = gene_value_nc[:, -suffix_len:].long()
                gene_value_nc[:, -suffix_len:] = -1

                logits = []
                for gene_id, gene_value, total_mrna_umis in zip(
                    torch.split(gene_id_nc, 25), torch.split(gene_value_nc, 25), torch.split(total_mrna_umis_n, 25)
                ):
                    gene_embedding = self.gene_embedding(gene_id, gene_value, total_mrna_umis)
                    hidden_state = gene_embedding * self.input_mult
                    hidden_state = self.transformer(hidden_state, prefix_len)
                    logits.append(self.head(hidden_state[:, -suffix_len:]) * self.output_mult)
                logits_nsm = torch.cat(logits, dim=0)

                label_mask_ns = label_ns < self.max_value + 1
                label_weight_ns = 1 / label_mask_ns.sum(dim=1, keepdim=True).expand(-1, suffix_len)
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                label_weights = label_weight_ns[label_mask_ns]
                loss = (
                    loss_fn(logits_nsm[label_mask_ns], label_ns[label_mask_ns]) * label_weights
                ).sum() / label_weights.sum()
                losses[prefix_len][i] = loss

                if trainer.global_rank == batch_idx == i == 0:
                    self._log_plots(
                        trainer,
                        label_ns,
                        logits_nsm,
                        total_mrna_umis_n,
                        prefix_len,
                    )

        loss_dict = {}
        for prefix_len in prefix_lens:
            loss = torch.mean(torch.stack([losses[prefix_len][i] for i in range(n_seeds)], dim=0))
            loss_dict[f"val_loss_prefix_{prefix_len + 1}"] = loss
        pl_module.log_dict(loss_dict, sync_dist=True, on_epoch=True)

    @torch.inference_mode()
    def predict(
        self,
        prompt_name_ns: np.ndarray | None,
        prompt_value_ns: torch.Tensor | None,
        query_name_nq: np.ndarray,
        total_mrna_umis_n: torch.Tensor,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        device = total_mrna_umis_n.device
        n = len(total_mrna_umis_n)

        values_nc = total_mrna_umis_n[:, None]
        ids_nc = torch.zeros((n, 1), dtype=torch.long, device=device)

        if prompt_name_ns is not None and prompt_value_ns is not None:
            prompt_id_ns = torch.tensor(self.vectorized_token_to_id(prompt_name_ns), dtype=torch.long, device=device)
            values_nc = torch.cat([values_nc, prompt_value_ns], dim=1)
            ids_nc = torch.cat([ids_nc, prompt_id_ns], dim=1)

        prefix_len = values_nc.shape[1]

        query_id_nq = torch.tensor(self.vectorized_token_to_id(query_name_nq), dtype=torch.long, device=device)
        values_nc = torch.cat([values_nc, torch.zeros_like(query_id_nq)], dim=1)
        ids_nc = torch.cat([ids_nc, query_id_nq], dim=1)

        hidden_state_ncd = self.transformer(ids_nc, values_nc, prefix_len)
        logits_ncp = self.head(hidden_state_ncd) * self.output_mult

        return {
            "logits_nqp": logits_ncp[:, prefix_len:],
            "query_name_nq": query_name_nq,
        }

    # def log_prob(self, var_names_ng: np.ndarray, x_ng: torch.Tensor, total_mrna_umis_n: torch.Tensor) -> torch.Tensor:
    #     prompt_value_ns = None
    #     log_probs = []
    #     ndx = torch.arange(x_ng.shape[0], device=x_ng.device)

    #     for i in tqdm.tqdm(range(var_names_ng.shape[1])):
    #         if prompt_value_ns is None:
    #             prompt_name_ns = None
    #         else:
    #             prompt_name_ns = var_names_ng[:, :i]

    #         logits_n1p: torch.Tensor = self.predict(
    #             prompt_name_ns, prompt_value_ns, var_names_ng[:, i : i + 1], total_mrna_umis_n
    #         )["logits_nqp"]
    #         logits_n1p = logits_n1p - logits_n1p.logsumexp(dim=-1, keepdim=True)
    #         value_n1 = x_ng[:, i : i + 1]
    #         log_prob_n1 = logits_n1p[ndx[:, None], 0, value_n1.long()]
    #         log_probs.append(log_prob_n1)
    #         if prompt_value_ns is None:
    #             prompt_value_ns = value_n1
    #         else:
    #             prompt_value_ns = torch.cat([prompt_value_ns, value_n1], dim=1)

    #     return torch.cat(log_probs, dim=1)

    # @torch.inference_mode()
    # def sample(
    #     self, var_names_ng: np.ndarray, total_mrna_umis_n: torch.Tensor, prompt_value_ns: torch.Tensor | None
    # ) -> dict[str, torch.Tensor]:
    #     if prompt_value_ns is None:
    #         prompt_len = 0
    #     else:
    #         prompt_len = prompt_value_ns.shape[1]
    #     total_len = var_names_ng.shape[1]

    #     for i in tqdm.tqdm(range(prompt_len, total_len)):
    #         if prompt_value_ns is None:
    #             prompt_name_ns = None
    #         else:
    #             prompt_name_ns = var_names_ng[:, :i]

    #         logits_n1p = self.predict(prompt_name_ns, prompt_value_ns, var_names_ng[:, i : i + 1], total_mrna_umis_n)[
    #             "logits_nqp"
    #         ]
    #         probs_np = logits_n1p[:, 0].softmax(dim=-1)
    #         value_n1 = torch.multinomial(probs_np, 1)
    #         if prompt_value_ns is None:
    #             prompt_value_ns = value_n1
    #         else:
    #             prompt_value_ns = torch.cat([prompt_value_ns, value_n1], dim=1)
    #     return prompt_value_ns

    def _log_plots(
        self,
        trainer: pl.Trainer,
        value_nq: torch.Tensor,
        logits_nqg: torch.Tensor,
        total_mrna_umis_n: torch.Tensor,
        prefix_len: int,
    ) -> None:
        import matplotlib.pyplot as plt

        nonzero_mask_nq = (value_nq > 0) & (value_nq < self.max_value + 1)
        zero_mask_nq = value_nq == 0
        q = value_nq.shape[1]

        total_mrna_umis_nq = total_mrna_umis_n[:, None].expand(-1, q)

        fig = plt.figure(figsize=(12, 8))

        for i in range(12):
            plt.subplot(3, 4, i + 1)
            if i < 8:
                label = value_nq[nonzero_mask_nq][i].item()
                logits = logits_nqg[nonzero_mask_nq][i].cpu()
                total_mrna_umis = total_mrna_umis_nq[nonzero_mask_nq][i].item()
            else:
                label = value_nq[zero_mask_nq][i].item()
                logits = logits_nqg[zero_mask_nq][i].cpu()
                total_mrna_umis = total_mrna_umis_nq[zero_mask_nq][i].item()

            x = torch.arange(0, min(label * 3 + 5, self.max_value + 1))
            y = logits.softmax(dim=-1)[: len(x)]
            plt.plot(x, y, "o")
            assert isinstance(label, int)
            plt.vlines(label, 0, y[label], color="r")
            plt.title(f"total UMIs={total_mrna_umis}, x={label}")

        plt.tight_layout()

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_figure(
                    f"val_pred_prefix_{prefix_len + 1}",
                    fig,
                    global_step=trainer.global_step,
                )
