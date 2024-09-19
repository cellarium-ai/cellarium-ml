# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import operator
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property, reduce
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cellarium.ml.models.model import CellariumModel, PredictMixin

try:
    from cerebras.pytorch.backend import use_cs
except ImportError:

    def use_cs() -> bool:
        return False


def prompt_diagonal_mask(prompt_mask_nc: torch.Tensor) -> torch.Tensor:
    """
    Generate a prompt diagonal mask for self-attention.

    Args:
        prompt_mask_nc:
            The prompt mask.

    Returns:
        torch.Tensor: The prompt diagonal mask.

    Example:
        For prompt_mask = [True, False, True, False, False], the attention mask is:
        [[True, False, True, False, False],
         [True, True,  True, False, False],
         [True, False, True, False, False],
         [True, False, True, True,  False],
         [True, False, True, False, True]]
    """
    device = prompt_mask_nc.device
    n, c = prompt_mask_nc.shape
    if use_cs():
        c_range = torch.arange(c, device=device, dtype=torch.float32)
        diag_mask_ncc = (c_range[:, None].expand(n, -1, 1) - c_range.expand(n, 1, -1)).abs()
        prompt_mask_n1c = 1 - prompt_mask_nc[:, None, :].float()
        attn_mask_ncc = diag_mask_ncc * prompt_mask_n1c
        return attn_mask_ncc == 0
    else:
        diag_mask_cc = torch.eye(c, dtype=torch.bool, device=device)
        attn_mask_ncc = prompt_mask_nc[:, None, :] | diag_mask_cc
        return attn_mask_ncc


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
        attn_backend: Literal["math", "flash", "mem_efficient", "torch"],
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
        attn_mask_ncc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_ncd:
                Query tensor of shape ``(n, c, d)``.
            key_ncd:
                Key tensor of shape ``(n, c, d)``.
            value_ncd:
                Value tensor of shape ``(n, c, d)``.
            attn_mask_ncc:
                Attention mask tensor of shape ``(n, c, c)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        n_heads = self.n_heads
        query_ncd = self.Wq(query_ncd)
        key_ncd = self.Wk(key_ncd)
        value_ncd = self.Wv(value_ncd)
        # d = k * h
        query_nhck = self.split_heads(query_ncd, n_heads)
        key_nhck = self.split_heads(key_ncd, n_heads)
        value_nhck = self.split_heads(value_ncd, n_heads)

        scale_factor = self.attn_mult / query_nhck.shape[-1]

        if self.attn_backend == "torch":
            key_nhck = key_nhck * torch.tensor(scale_factor, dtype=key_nhck.dtype)
            attn_logits_nhcc = torch.matmul(query_nhck, key_nhck.transpose(-1, -2))
            neg_inf = torch.tensor(float("-inf"), dtype=torch.float32)
            attn_bias_ncc = torch.where(attn_mask_ncc, 0, neg_inf).to(attn_logits_nhcc.dtype)
            attn_logits_nhcc += attn_bias_ncc.unsqueeze(1).expand(attn_logits_nhcc.shape)
            attn_weights_nhcc = torch.softmax(attn_logits_nhcc.float(), dim=-1).to(attn_logits_nhcc.dtype)
            attn_weights_nhcc = nn.functional.dropout(attn_weights_nhcc, self.dropout_p, training=self.training)
            output_nhck = torch.matmul(attn_weights_nhcc, value_nhck)
        else:
            with sdpa_kernel(self.backend_map[self.attn_backend]):
                output_nhck = nn.functional.scaled_dot_product_attention(
                    query_nhck,
                    key_nhck,
                    value_nhck,
                    attn_mask=attn_mask_ncc.unsqueeze(1),
                    dropout_p=self.dropout_p if self.training else 0.0,
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
        """
        Args:
            hidden_state_ncd: Hidden state tensor of shape ``(n, c, d)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        return self.dense2(self.relu(self.dense1(hidden_state_ncd)))  # _ncd


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

    def forward(
        self,
        hidden_state_ncd: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_state_ncd:
                Hidden state tensor of shape ``(n, c, d)``.
            sublayer:
                Sublayer function.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
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
        attn_backend: Literal["math", "flash", "mem_efficient", "torch"],
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, use_bias, n_heads, dropout_p, attn_mult, attn_backend)
        self.normadd1 = NormAdd(d_model, dropout_p)
        self.ffn = PositionWiseFFN(d_ffn, d_model, use_bias)
        self.normadd2 = NormAdd(d_model, dropout_p)

    def forward(
        self,
        hidden_state_ncd: torch.Tensor,
        attn_mask_ncc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state_ncd:
                Hidden state tensor of shape ``(n, c, d)``.
            attn_mask_ncc:
                Attention mask tensor of shape ``(n, c, c)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        hidden_state_ncd = self.normadd1(hidden_state_ncd, lambda X: self.attention(X, X, X, attn_mask_ncc))
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

    def forward(
        self,
        hidden_state_ncd: torch.Tensor,
        attn_mask_ncc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state_ncd:
                Hidden state tensor of shape ``(n, c, d)``.
            attn_mask_ncc:
                Attention mask tensor of shape ``(n, c, c)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        for block in self.blocks:
            hidden_state_ncd = block(hidden_state_ncd, attn_mask_ncc)

        return hidden_state_ncd


class GeneEmbedding(nn.Module):
    """
    Gene embedding.

    Args:
        categorical_vocab_sizes:
            Categorical gene token vocabulary sizes.
        continuous_vocab_sizes:
            Continuous gene token vocabulary sizes.
        d_model:
            Dimensionality of the embeddings and hidden states.
    """

    def __init__(
        self,
        categorical_vocab_sizes: dict[str, int],
        continuous_vocab_sizes: dict[str, int],
        d_model: int,
    ) -> None:
        super().__init__()
        self.E = nn.ModuleDict()
        self.E.update({key: nn.Embedding(vocab_size, d_model) for key, vocab_size in categorical_vocab_sizes.items()})
        self.E.update(
            {key: nn.Linear(vocab_size, d_model, bias=False) for key, vocab_size in continuous_vocab_sizes.items()}
        )

    def forward(self, gene_tokens_nc: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gene_tokens_nc:
                Dictionary of gene token tensors of shape ``(n, c)``.

        Returns:
            The gene embedding tensor of shape ``(n, c, d)``.
        """
        return reduce(
            operator.add,
            [self.E[key](gene_token_nc) for key, gene_token_nc in gene_tokens_nc.items()],
        )


class MetaDataEmbedding(nn.Module):
    """
    Metadata embedding.

    Args:
        categorical_vocab_sizes:
            Categorical metadata token vocabulary sizes.
        d_model:
            Dimensionality of the embeddings and hidden states.
    """

    def __init__(
        self,
        categorical_vocab_sizes: dict[str, int],
        d_model: int,
    ) -> None:
        super().__init__()
        self.E = nn.ModuleDict(
            {key: nn.Embedding(vocab_size, d_model) for key, vocab_size in categorical_vocab_sizes.items()}
        )

    def forward(self, metadata_tokens_n: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            metadata_token_n:
                Dictionary of metadata token tensors of shape ``(n,)``.

        Returns:
            The metadata embedding tensor of shape ``(n, d)``.
        """
        return torch.stack(
            [self.E[key](metadata_token_n) for key, metadata_token_n in metadata_tokens_n.items()],
            dim=1,
        )


class DummyTokenizer(torch.nn.Module):
    def forward(self, **kwargs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return kwargs


class Tokenizer(torch.nn.Module):
    def __init__(
        self,
        max_prefix_len: int,
        context_len: int,
        downsample_fraction: float,
        min_total_mrna_umis: int,
        max_total_mrna_umis: int,
        gene_vocab_sizes: dict[str, int],
        metadata_vocab_sizes: dict[str, int],
    ) -> None:
        super().__init__()
        self.max_prefix_len = max_prefix_len
        self.context_len = context_len
        self.downsample_fraction = downsample_fraction
        self.min_total_mrna_umis = min_total_mrna_umis
        self.max_total_mrna_umis = max_total_mrna_umis
        self.gene_vocab_sizes = gene_vocab_sizes
        self.metadata_vocab_sizes = metadata_vocab_sizes

    def forward(
        self,
        metadata_tokens_n: dict[str, torch.Tensor],
        gene_tokens_n: dict[str, torch.Tensor],
        gene_value_ng: torch.Tensor,
        total_mrna_umis_n: torch.Tensor,
        gene_id_g: torch.Tensor | None = None,
        measured_genes_mask_ng: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        n, g = gene_value_ng.shape
        device = gene_value_ng.device

        # gene tokens
        if gene_id_g is None:
            gene_id_g = torch.arange(g, device=device)
        gene_id_ng = gene_id_g.expand(gene_value_ng.shape)

        # shuffle and select context
        shuffle_idx_ng = torch.argsort(torch.rand_like(gene_value_ng), dim=-1)
        shuffle_idx_nc = shuffle_idx_ng[:, : self.context_len]
        gene_id_nc = torch.gather(gene_id_ng, dim=-1, index=shuffle_idx_nc)
        gene_value_nc = torch.gather(gene_value_ng, dim=-1, index=shuffle_idx_nc)
        total_mrna_umis_nc = total_mrna_umis_n[:, None].expand(-1, self.context_len)
        gene_tokens_nc = {key: gene_tokens_n[key][:, None].expand(-1, self.context_len).int() for key in gene_tokens_n}
        if measured_genes_mask_ng is not None:
            measured_genes_mask_nc = torch.gather(measured_genes_mask_ng, dim=-1, index=shuffle_idx_nc)
        else:
            measured_genes_mask_nc = None

        # downsample gene values
        max_total_mrna_umis = torch.tensor(self.max_total_mrna_umis, device=device)
        downsampled_total_mrna_umis_nc = torch.minimum(total_mrna_umis_nc, max_total_mrna_umis).float()
        if self.downsample_fraction > 0:
            downsample_p_nc = torch.minimum(
                torch.rand((n, self.context_len), device=device) / self.downsample_fraction,
                torch.tensor(1.0, device=device),
            )
            downsampled_total_mrna_umis_nc = torch.lerp(
                torch.full_like(downsample_p_nc, self.min_total_mrna_umis),
                downsampled_total_mrna_umis_nc,
                downsample_p_nc,
            )
        downsample_p_nc = downsampled_total_mrna_umis_nc / total_mrna_umis_nc
        gene_value_nc = torch.binomial(gene_value_nc, downsample_p_nc)
        total_mrna_umis_nc = torch.round(downsampled_total_mrna_umis_nc)

        prefix_weights = self.max_prefix_len / torch.arange(self.max_prefix_len, dtype=torch.float32)
        prefix_weights[0] = 1
        prefix_len_n = torch.multinomial(prefix_weights, n, replacement=True)
        # prefix_len_n = torch.randint(0, self.max_prefix_len, (num_prefixes,), device=device)

        suffix_mask_nc = torch.arange(self.context_len, device=device) >= prefix_len_n[:, None].expand(n, -1)
        gene_query_mask_nc = suffix_mask_nc
        gene_prompt_mask_nc = ~suffix_mask_nc
        if measured_genes_mask_nc is not None:
            gene_query_mask_nc = gene_query_mask_nc & measured_genes_mask_nc
            gene_prompt_mask_nc = gene_prompt_mask_nc & measured_genes_mask_nc

        gene_value_vocab_size = self.gene_vocab_sizes["gene_value"]
        gene_label_nc = gene_value_nc.clamp(0, gene_value_vocab_size - 1).int()
        gene_value_nc3 = torch.stack(
            [
                torch.log1p(gene_value_nc) * suffix_mask_nc.logical_not().float(),
                suffix_mask_nc.float(),
                torch.log1p(total_mrna_umis_nc),
            ],
            dim=2,
        )
        gene_tokens_nc["gene_id"] = gene_id_nc
        gene_tokens_nc["gene_value"] = gene_value_nc3

        # metadata tokens
        metadata_weights_n = prefix_len_n / (self.max_prefix_len + 1)
        m = len(self.metadata_vocab_sizes)
        metadata_query_mask_nm = torch.bernoulli(metadata_weights_n[:, None].expand(-1, m)).bool()
        metadata_prompt_mask_nm = ~metadata_query_mask_nm
        metadata_measured_mask_nm = torch.stack(
            [metadata_token_n < 0 for metadata_token_n in metadata_tokens_n.values()], dim=1
        ).bool()
        metadata_prompt_mask_nm = metadata_prompt_mask_nm & metadata_measured_mask_nm
        metadata_query_mask_nm = metadata_query_mask_nm & metadata_measured_mask_nm

        for key, metadata_token_n in metadata_tokens_n.items():
            metadata_tokens_n[key] = metadata_token_n.clamp(0).int()
        block_label_nc = torch.block_diag(
            gene_label_nc,
            *[metadata_token_n.unsqueeze(-1) for metadata_token_n in metadata_tokens_n.values()],
        )
        labels_nc = {
            key: block_label_nc[n * i : n * (i + 1)] for i, key in enumerate(["gene"] + list(metadata_tokens_n))
        }
        # downsample metadata based on ontology
        for key, metadata_token_n in metadata_tokens_n.items():
            vocab_size = self.metadata_vocab_sizes[key]
            ancestors_matrix = torch.triu(torch.ones(vocab_size, vocab_size, device=device))
            metadata_tokens_n[key] = (
                torch.multinomial(ancestors_matrix[metadata_token_n], num_samples=1).squeeze(-1).int()
            )

        for i, (key, metadata_token_n) in enumerate(metadata_tokens_n.items()):
            metadata_tokens_n[key] = torch.where(
                metadata_query_mask_nm[:, i], self.metadata_vocab_sizes[key], metadata_token_n
            ).int()

        # combine
        prompt_mask_nc = torch.cat([gene_prompt_mask_nc, metadata_prompt_mask_nm], dim=1)

        block_label_weight_nc = torch.block_diag(
            gene_query_mask_nc / gene_query_mask_nc.sum(dim=-1, keepdim=True),
            *[metadata_query_mask_nm[:, i].unsqueeze(-1).float() for i in range(m)],
        )
        label_weights_nc = {
            key: block_label_weight_nc[n * i : n * (i + 1)] for i, key in enumerate(["gene"] + list(metadata_tokens_n))
        }

        return {
            "gene_tokens_nc": gene_tokens_nc,
            "metadata_tokens_n": metadata_tokens_n,
            "prompt_mask_nc": prompt_mask_nc,
            "labels_nc": labels_nc,
            "label_weights_nc": label_weights_nc,
        }


class CellariumGPT(CellariumModel, PredictMixin):
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
        attn_backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        gene_vocab_sizes: dict[str, int],
        metadata_vocab_sizes: dict[str, int],
        d_model: int = 256,
        d_ffn: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        dropout_p: float = 0.0,
        use_bias: bool = False,
        attn_mult: float = 6.0,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        initializer_range: float = 0.02,
        attn_backend: Literal["math", "flash", "mem_efficient", "torch"] = "mem_efficient",
        gene_categories: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if gene_categories is not None:
            assert len(gene_categories) == gene_vocab_sizes["gene_id"]
        self.gene_categories = gene_categories
        self.input_mult = input_mult
        self.output_mult = output_mult
        gene_value_vocab_size = gene_vocab_sizes.pop("gene_value")

        self.gene_embedding = GeneEmbedding(
            categorical_vocab_sizes=gene_vocab_sizes,
            continuous_vocab_sizes={"gene_value": 3},
            d_model=d_model,
        )
        self.metadata_embedding = MetaDataEmbedding(
            categorical_vocab_sizes={key: vocab_size + 1 for key, vocab_size in metadata_vocab_sizes.items()},
            d_model=d_model,
        )
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
        self.head = nn.ModuleDict(
            {
                "gene": nn.Linear(d_model, gene_value_vocab_size, use_bias),
                **{key: nn.Linear(d_model, vocab_size, use_bias) for key, vocab_size in metadata_vocab_sizes.items()},
            }
        )
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
        gene_tokens_nc: dict[str, torch.Tensor],
        metadata_tokens_n: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
        labels_nc: dict[str, torch.Tensor],
        label_weights_nc: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # embed the gene IDs, values, and total mRNA UMIs
        embedding_ncd = torch.cat(
            [
                self.gene_embedding(gene_tokens_nc),
                self.metadata_embedding(metadata_tokens_n),
            ],
            dim=1,
        )

        # create attention mask
        attn_mask_ncc = prompt_diagonal_mask(prompt_mask_nc)

        # transformer blocks
        hidden_state_ncd = embedding_ncd * self.input_mult
        hidden_state_ncd = self.transformer(hidden_state_ncd, attn_mask_ncc)

        # compute loss
        loss = 0.0
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        for key, label_nc in labels_nc.items():
            logits_ncr = self.head[key](hidden_state_ncd) * self.output_mult
            loss += torch.sum(
                loss_fn(logits_ncr.view(label_nc.numel(), -1), label_nc.view(-1).long())
                * label_weights_nc[key].view(-1)
            )

        # loss /= (
        #     gene_label_weight_nc.sum()
        #     + cell_type_label_weight_nc.sum()
        #     + development_stage_label_weight_nc.sum()
        #     + sex_label_weight_nc.sum()
        # )

        return {"loss": loss}

    # @torch.inference_mode()
    def test(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        gene_id_g: torch.Tensor,
        # gene_categories: pd.Series,
        gene_value_ng: torch.Tensor,
        total_mrna_umis_n: torch.Tensor,
        obs_names_n: np.ndarray,
        batch_idx: int,
        measured_genes_mask_ng: torch.Tensor | None = None,
    ) -> None:
        device = gene_id_g.device
        gene_id_ng = gene_id_g.expand(gene_value_ng.shape)
        n, g = gene_id_ng.shape

        # loss = self(gene_id_g, gene_value_ng, total_mrna_umis_n, measured_genes_mask_ng)["loss"]
        # pl_module.log("test_loss", loss, sync_dist=True, on_epoch=True)

        losses: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
        suffix_len = 500
        n_seeds = 3
        prefix_lens = [0, 1, 10, 50, 500, 1000, 3000]
        for i in range(n_seeds):
            rng_n = [torch.Generator(device=device) for _ in range(n)]
            [rng.manual_seed(abs(hash(obs_name)) + i) for rng, obs_name in zip(rng_n, obs_names_n)]
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
                total_mrna_umis_nc = total_mrna_umis_n[:, None].expand(gene_id_nc.shape)
                if measured_genes_mask_ng is not None:
                    measured_genes_mask_nc = torch.gather(measured_genes_mask_ng, dim=-1, index=shuffle_idx_nc)
                else:
                    measured_genes_mask_nc = None
                label_ns = gene_value_nc[:, -suffix_len:].long()
                gene_value_nc[:, -suffix_len:] = -1

                context_len = prefix_len + suffix_len
                prefix_len_n = torch.tensor([prefix_len], device=device)
                attn_mask_ncc = prefix_diagonal_mask(context_len, prefix_len_n, measured_genes_mask_nc)

                # gene type embedding
                gene_type_nc = torch.zeros((n, context_len), dtype=torch.long, device=device)
                suffix_mask_nc = torch.arange(context_len, device=device) >= prefix_len_n[:, None].expand(n, -1)
                gene_type_nc.masked_fill_(suffix_mask_nc, 1)

                gene_embedding = self.gene_embedding(gene_type_nc, gene_id_nc, gene_value_nc, total_mrna_umis_nc)
                hidden_state = gene_embedding * self.input_mult
                hidden_state = self.transformer(hidden_state, attn_mask_ncc)
                logits_nsm = self.head(hidden_state[:, -suffix_len:]) * self.output_mult

                label_mask_ns = label_ns < self.max_value + 1
                label_weight_ns = 1 / label_mask_ns.sum(dim=1, keepdim=True).expand(-1, suffix_len)
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                label_weights = label_weight_ns[label_mask_ns]
                loss = (
                    loss_fn(logits_nsm[label_mask_ns], label_ns[label_mask_ns]) * label_weights
                ).sum() / label_weights.sum()
                losses[prefix_len][i] = loss

                # if trainer.global_rank == batch_idx == i == 0:
                #     self._log_plots(
                #         trainer,
                #         label_ns,
                #         logits_nsm,
                #         total_mrna_umis_n,
                #         prefix_len,
                #     )

        loss_dict = {}
        for prefix_len in prefix_lens:
            loss = torch.mean(torch.stack([losses[prefix_len][i] for i in range(n_seeds)], dim=0))
            loss_dict[f"test_loss_prefix_{prefix_len}"] = loss
        pl_module.log_dict(loss_dict, sync_dist=True, on_epoch=True)

    def predict(
        self,
        prompt_name_ns: np.ndarray | None,
        prompt_value_ns: torch.Tensor | None,
        prompt_total_mrna_umis_n: torch.Tensor | None,
        prompt_measured_genes_mask_ns: torch.Tensor | None,
        query_name_nq: np.ndarray | None,
        query_total_mrna_umis_n: torch.Tensor | None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        if prompt_name_ns is not None and query_name_nq is not None:
            n, q = query_name_nq.shape
            device = query_total_mrna_umis_n.device
            gene_name_nc = np.concatenate([prompt_name_ns, query_name_nq], axis=1)
            gene_value_nc = torch.cat([prompt_value_ns, -torch.ones((n, q), device=device)], dim=1)
            total_mrna_umis_nc = torch.cat(
                [
                    prompt_total_mrna_umis_n[:, None].expand(prompt_name_ns.shape),
                    query_total_mrna_umis_n[:, None].expand(query_name_nq.shape),
                ],
                dim=-1,
            )
            measured_genes_mask_nc = torch.cat(
                [
                    prompt_measured_genes_mask_ns,
                    torch.ones((n, q), dtype=torch.bool, device=device),
                ],
                dim=1,
            )
        elif prompt_name_ns is not None:
            gene_name_nc = prompt_name_ns
            gene_value_nc = prompt_value_ns
            total_mrna_umis_nc = prompt_total_mrna_umis_n[:, None].expand(gene_name_nc.shape)
            measured_genes_mask_nc = prompt_measured_genes_mask_ns
        else:
            n, q = query_name_nq.shape
            device = query_total_mrna_umis_n.device
            gene_name_nc = query_name_nq
            gene_value_nc = -torch.ones((n, q), device=device)
            total_mrna_umis_nc = query_total_mrna_umis_n[:, None].expand(gene_name_nc.shape)
            measured_genes_mask_nc = torch.ones((n, q), dtype=torch.bool, device=device)

        device = gene_value_nc.device
        gene_id_nc = torch.tensor(self.vectorized_token_to_id(gene_name_nc), dtype=torch.long, device=device)

        if prompt_name_ns is not None and prompt_value_ns is not None:
            prefix_len = prompt_value_ns.shape[1]
        else:
            prefix_len = 0

        attn_mask_cc = prefix_diagonal_mask(
            gene_id_nc.shape[1], torch.tensor([prefix_len], device=device), measured_genes_mask_nc
        )
        gene_embedding_ncd = self.gene_embedding(gene_id_nc, gene_value_nc, total_mrna_umis_nc)
        hidden_state_ncd = gene_embedding_ncd * self.input_mult
        hidden_state_ncd = self.transformer(hidden_state_ncd, attn_mask_cc)
        logits_nqm = self.head(hidden_state_ncd[:, prefix_len:]) * self.output_mult

        return {
            "logits_nqm": logits_nqm,
            # "query_name_nq": query_name_nq,
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
