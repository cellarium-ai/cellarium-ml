# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property, reduce
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cellarium.ml.models.model import CellariumModel, PredictMixin

try:
    from cerebras.modelzoo.common.utils.model.mup_utils import LRAdjustmentGroup
    from cerebras.pytorch.backend import use_cs
except ImportError:
    from cellarium.ml.utilities.mup import LRAdjustmentGroup

    def use_cs() -> bool:
        return False


def create_initializer(initializer: dict[str, Any]) -> Callable[[torch.Tensor], None]:
    initializer_fn = getattr(nn.init, initializer["name"])
    initializer_kwargs = initializer.copy()
    del initializer_kwargs["name"]
    return lambda x: initializer_fn(x, **initializer_kwargs)


def scale_initializers_by_dimension(
    initializers: dict[str, Any] | list[dict[str, Any]],
    width_scale: float | None = None,
    depth_scale: float | None = None,
) -> None:
    """
    Scales the std of an initializer or list of initializers by the specified
    width and depth scalars. Unsupported initializers are ignored and a warning
    is printed to the user.
    """
    if not width_scale:
        width_scale = 1.0
    if not depth_scale:
        depth_scale = 1.0
    mup_scalar = width_scale * depth_scale

    if not isinstance(initializers, list):
        initializers = [initializers]

    for initializer in initializers:
        if "name" not in initializer:
            raise ValueError("Initializer name must be provided")
        initializer_name = initializer["name"].lower()

        if initializer_name == "normal_":
            initializer["std"] = initializer.get("std", 1.0) * mup_scalar
        elif initializer_name == "trunc_normal_":
            std = initializer.get("std", 1.0)
            initializer["std"] = std * mup_scalar
            initializer["a"] = initializer.get("a", -2 * std) * mup_scalar
            initializer["b"] = initializer.get("b", 2 * std) * mup_scalar
            std = None
        else:
            raise ValueError(f"Initializer {initializer_name} is not supported for muP")


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
        attention_mask_ncc = diag_mask_ncc * prompt_mask_n1c
        return attention_mask_ncc == 0
    else:
        diag_mask_cc = torch.eye(c, dtype=torch.bool, device=device)
        attention_mask_ncc = prompt_mask_nc[:, None, :] | diag_mask_cc
        return attention_mask_ncc


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
        attention_logits_scale:
            Multiplier for the attention scores.
        attention_backend:
            Backend for the attention computation.
        Wqkv_initializer:
            Initializer for the query, key, and value linear transformations.
        Wo_initializer:
            Initializer for the output linear transformation.
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
        attention_logits_scale: float,
        attention_backend: Literal["math", "flash", "mem_efficient", "torch"],
        attention_softmax_fp32: bool,
        Wqkv_initializer: dict[str, Any],
        Wo_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wk = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wv = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wo = nn.Linear(d_model, d_model, bias=use_bias)
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.attention_logits_scale = attention_logits_scale
        self.attention_backend = attention_backend
        self.attention_softmax_fp32 = attention_softmax_fp32
        self.Wqkv_initializer = Wqkv_initializer
        self.Wo_initializer = Wo_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in [self.Wq, self.Wk, self.Wv]:
            create_initializer(self.Wqkv_initializer)(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        create_initializer(self.Wo_initializer)(self.Wo.weight)
        if self.Wo.bias is not None:
            nn.init.zeros_(self.Wo.bias)

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
        attention_mask_ncc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_ncd:
                Query tensor of shape ``(n, c, d)``.
            key_ncd:
                Key tensor of shape ``(n, c, d)``.
            value_ncd:
                Value tensor of shape ``(n, c, d)``.
            attention_mask_ncc:
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

        scale_factor = self.attention_logits_scale / query_nhck.shape[-1]

        if self.attention_backend == "torch":
            key_nhck = key_nhck * torch.tensor(scale_factor, dtype=key_nhck.dtype)
            attention_logits_nhcc = torch.matmul(query_nhck, key_nhck.transpose(-1, -2))
            neg_inf = torch.tensor(float("-inf"), dtype=torch.float32)
            attention_bias_ncc = torch.where(attention_mask_ncc, 0, neg_inf).to(attention_logits_nhcc.dtype)
            attention_logits_nhcc += attention_bias_ncc.unsqueeze(1).expand(attention_logits_nhcc.shape)
            if self.attention_softmax_fp32 and attention_logits_nhcc.dtype != torch.float32:
                attention_weights_nhcc = nn.functional.softmax(attention_logits_nhcc.float(), dim=-1).to(
                    attention_logits_nhcc.dtype
                )
            else:
                attention_weights_nhcc = nn.functional.softmax(attention_logits_nhcc, dim=-1)
            attention_weights_nhcc = nn.functional.dropout(
                attention_weights_nhcc, self.dropout_p, training=self.training
            )
            output_nhck = torch.matmul(attention_weights_nhcc, value_nhck)
        else:
            with sdpa_kernel(self.backend_map[self.attention_backend]):
                output_nhck = nn.functional.scaled_dot_product_attention(
                    query_nhck,
                    key_nhck,
                    value_nhck,
                    attention_mask=attention_mask_ncc.unsqueeze(1),
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
        dense1_initializer:
            Initializer for the first dense layer.
        dense2_initializer:
            Initializer for the second dense layer.
    """

    def __init__(
        self,
        d_ffn: int,
        d_model: int,
        use_bias: bool,
        dense1_initializer: dict[str, Any],
        dense2_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_ffn, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(d_ffn, d_model, bias=use_bias)
        self.dense1_initializer = dense1_initializer
        self.dense2_initializer = dense2_initializer

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        create_initializer(self.dense1_initializer)(self.dense1.weight)
        if self.dense1.bias is not None:
            nn.init.zeros_(self.dense1.bias)

        create_initializer(self.dense2_initializer)(self.dense2.weight)
        if self.dense2.bias is not None:
            nn.init.zeros_(self.dense2.bias)

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

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.ln.reset_parameters()

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
        attention_logits_scale:
            Multiplier for the attention scores.
        use_bias:
            Whether to use bias in the linear transformations.
        attention_backend:
            Backend for the attention computation.
        Wqkv_initializer:
            Initializer for the query, key, and value linear transformations.
        Wo_initializer:
            Initializer for the output linear transformation.
        dense1_initializer:
            Initializer for the first dense layer.
        dense2_initializer:
            Initializer for the second dense layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        dropout_p: float,
        attention_logits_scale: float,
        use_bias: bool,
        attention_backend: Literal["math", "flash", "mem_efficient", "torch"],
        attention_softmax_fp32: bool,
        Wqkv_initializer: dict[str, Any],
        Wo_initializer: dict[str, Any],
        dense1_initializer: dict[str, Any],
        dense2_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model,
            use_bias,
            n_heads,
            dropout_p,
            attention_logits_scale,
            attention_backend,
            attention_softmax_fp32,
            Wqkv_initializer,
            Wo_initializer,
        )
        self.normadd1 = NormAdd(d_model, dropout_p)
        self.ffn = PositionWiseFFN(
            d_ffn,
            d_model,
            use_bias,
            dense1_initializer,
            dense2_initializer,
        )
        self.normadd2 = NormAdd(d_model, dropout_p)

    def reset_parameters(self) -> None:
        self.attention._reset_parameters()
        self.normadd1._reset_parameters()
        self.ffn._reset_parameters()
        self.normadd2._reset_parameters()

    def forward(
        self,
        hidden_state_ncd: torch.Tensor,
        attention_mask_ncc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state_ncd:
                Hidden state tensor of shape ``(n, c, d)``.
            attention_mask_ncc:
                Attention mask tensor of shape ``(n, c, c)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        hidden_state_ncd = self.normadd1(hidden_state_ncd, lambda X: self.attention(X, X, X, attention_mask_ncc))
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
        attention_logits_scale:
            Multiplier for the attention scores.
        attention_backend:
            Backend for the attention computation.
        Wqkv_initializer:
            Initializer for the query, key, and value linear transformations.
        Wo_initializer:
            Initializer for the output linear transformation.
        dense1_initializer:
            Initializer for the first dense layer.
        dense2_initializer:
            Initializer for the second dense layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        use_bias: bool,
        attention_logits_scale: float,
        attention_backend: Literal["math", "flash", "mem_efficient"],
        attention_softmax_fp32: bool,
        Wqkv_initializer: dict[str, Any],
        Wo_initializer: dict[str, Any],
        dense1_initializer: dict[str, Any],
        dense2_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    d_ffn,
                    n_heads,
                    dropout_p,
                    attention_logits_scale,
                    use_bias,
                    attention_backend,
                    attention_softmax_fp32,
                    Wqkv_initializer,
                    Wo_initializer,
                    dense1_initializer,
                    dense2_initializer,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        hidden_state_ncd: torch.Tensor,
        attention_mask_ncc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state_ncd:
                Hidden state tensor of shape ``(n, c, d)``.
            attention_mask_ncc:
                Attention mask tensor of shape ``(n, c, c)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        for block in self.blocks:
            hidden_state_ncd = block(hidden_state_ncd, attention_mask_ncc)

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
        result = reduce(
            operator.add,
            [self.E[key](gene_token_nc) for key, gene_token_nc in gene_tokens_nc.items()],
        )
        return result


class MetaDataEmbedding(nn.Module):
    """
    Metadata embedding.

    Args:
        categorical_vocab_sizes:
            Categorical metadata token vocabulary sizes.
        d_model:
            Dimensionality of the embeddings and hidden states.
        embeddings_initializer:
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
            The metadata embedding tensor of shape ``(n, d)``.
        """
        return torch.stack(
            [self.E[key](metadata_token_n) for key, metadata_token_n in metadata_tokens_n.items()],
            dim=1,
        )


class DummyTokenizer(torch.nn.Module):
    def forward(self, **kwargs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return kwargs


class PredictTokenizer(torch.nn.Module):
    def __init__(
        self,
        max_total_mrna_umis: int,
        gene_vocab_sizes: dict[str, int],
        metadata_vocab_sizes: dict[str, int],
        ontology_infos: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()
        self.max_total_mrna_umis = max_total_mrna_umis
        self.gene_vocab_sizes = gene_vocab_sizes
        self.metadata_vocab_sizes = metadata_vocab_sizes
        self.ontology_infos = ontology_infos

    def forward(
        self,
        metadata_tokens_n: dict[str, torch.Tensor],
        metadata_prompt_masks_n: dict[str, torch.Tensor],
        gene_tokens_nc: dict[str, torch.Tensor],
        gene_prompt_mask_nc: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        ### GENE TOKENS ###

        ## gene value ##
        gene_value_nc = gene_tokens_nc.pop("gene_value")
        total_mrna_umis_nc = gene_tokens_nc.pop("total_mrna_umis")
        device = gene_value_nc.device
        # downsample gene values
        max_total_mrna_umis = torch.tensor(self.max_total_mrna_umis, device=device)
        downsampled_total_mrna_umis_nc = torch.minimum(total_mrna_umis_nc, max_total_mrna_umis).float()
        gene_downsample_p_nc = downsampled_total_mrna_umis_nc / total_mrna_umis_nc
        gene_value_nc = torch.binomial(gene_value_nc, gene_downsample_p_nc)
        total_mrna_umis_nc = torch.round(downsampled_total_mrna_umis_nc)

        gene_query_mask_nc = ~gene_prompt_mask_nc
        gene_value_nc3 = torch.stack(
            [
                torch.log1p(gene_value_nc) * gene_prompt_mask_nc.float(),
                gene_query_mask_nc.float(),
                torch.log1p(total_mrna_umis_nc),
            ],
            dim=2,
        )
        gene_tokens_nc["gene_value"] = gene_value_nc3

        ### METADATA TOKENS ###

        ## metadata tokens ##
        # assign token codes based on the ontology info
        # token values not in the ontology are treated as unmeasured and assigned a code value of -1
        for key, ontology_info in self.ontology_infos.items():
            assert self.metadata_vocab_sizes[key] == len(ontology_info["labels"])
            metadata_tokens_n[key] = torch.tensor(
                pd.Categorical(metadata_tokens_n[key], categories=ontology_info["labels"]).codes,
                dtype=torch.int,
            )
        # create metadata query and prompt masks
        metadata_prompt_mask_nm = torch.stack([metadata_prompt_masks_n[key] for key in metadata_tokens_n], dim=1)
        metadata_query_mask_nm = ~metadata_prompt_mask_nm

        # clamp unmeasured tokens to 0
        # for key, metadata_token_n in metadata_tokens_n.items():
        #     metadata_tokens_n[key] = metadata_token_n.clamp(0).int()

        # impute mask token for unmeasured metadata
        # mask token is the last token in the vocabulary
        for i, (key, metadata_token_n) in enumerate(metadata_tokens_n.items()):
            metadata_tokens_n[key] = torch.where(
                metadata_query_mask_nm[:, i], self.metadata_vocab_sizes[key], metadata_token_n
            ).int()

        ### PROMPT MASK ###
        prompt_mask_nc = torch.cat([gene_prompt_mask_nc, metadata_prompt_mask_nm], dim=1)

        return {
            "gene_tokens_nc": gene_tokens_nc,
            "metadata_tokens_n": metadata_tokens_n,
            "prompt_mask_nc": prompt_mask_nc,
        }


class Tokenizer(torch.nn.Module):
    """
    Tokenizer for the Cellarium GPT model.

    Args:
        max_prefix_len:
            Maximum prefix length.
        context_len:
            Context length.
        downsample_fraction:
            Downsample fraction.
        min_total_mrna_umis:
            Minimum total mRNA UMIs.
        max_total_mrna_umis:
            Maximum total mRNA UMIs.
        gene_vocab_sizes:
            Gene token vocabulary sizes.
        metadata_vocab_sizes:
            Metadata token vocabulary sizes.
        ontology_infos:
            Ontology information.
    """

    def __init__(
        self,
        max_prefix_len: int,
        context_len: int,
        downsample_fraction: float,
        min_total_mrna_umis: int,
        max_total_mrna_umis: int,
        gene_vocab_sizes: dict[str, int],
        metadata_vocab_sizes: dict[str, int],
        ontology_infos: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()
        self.max_prefix_len = max_prefix_len
        self.context_len = context_len
        self.downsample_fraction = downsample_fraction
        self.min_total_mrna_umis = min_total_mrna_umis
        self.max_total_mrna_umis = max_total_mrna_umis
        self.gene_vocab_sizes = gene_vocab_sizes
        self.metadata_vocab_sizes = metadata_vocab_sizes
        self.ontology_infos = ontology_infos

    def forward(
        self,
        metadata_tokens_n: dict[str, torch.Tensor],
        gene_tokens_n: dict[str, torch.Tensor],
        gene_tokens_ng: dict[str, torch.Tensor],
        gene_id_g: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        ### GENE TOKENS ###
        n, g = gene_tokens_ng["gene_value"].shape
        device = gene_tokens_ng["gene_value"].device

        ## gene measurement tokens (assay, suspension type, etc.) ##
        gene_tokens_nc = {key: gene_tokens_n[key][:, None].expand(-1, self.context_len).int() for key in gene_tokens_n}

        ## gene id ##
        if gene_id_g is None:
            gene_id_g = torch.arange(g, device=device)
        gene_tokens_ng["gene_id"] = gene_id_g.expand(n, g)

        shuffle_idx_ng = torch.argsort(torch.rand((n, g), dtype=torch.float32, device=device), dim=-1)
        shuffle_idx_nc = shuffle_idx_ng[:, : self.context_len]

        for key, gene_token_ng in gene_tokens_ng.items():
            gene_tokens_nc[key] = torch.gather(gene_token_ng, dim=-1, index=shuffle_idx_nc)

        ## gene value ##
        gene_value_nc = gene_tokens_nc.pop("gene_value")
        total_mrna_umis_nc = gene_tokens_nc.pop("total_mrna_umis")
        # downsample gene values
        max_total_mrna_umis = torch.tensor(self.max_total_mrna_umis, device=device)
        downsampled_total_mrna_umis_nc = torch.minimum(total_mrna_umis_nc, max_total_mrna_umis).float()
        if self.downsample_fraction > 0:
            gene_downsample_p_nc = torch.minimum(
                torch.rand((n, self.context_len), device=device) / self.downsample_fraction,
                torch.tensor(1.0, device=device),
            )
            downsampled_total_mrna_umis_nc = torch.lerp(
                torch.full_like(gene_downsample_p_nc, self.min_total_mrna_umis),
                downsampled_total_mrna_umis_nc,
                gene_downsample_p_nc,
            )
        gene_downsample_p_nc = downsampled_total_mrna_umis_nc / total_mrna_umis_nc
        gene_value_nc = torch.binomial(gene_value_nc, gene_downsample_p_nc)
        total_mrna_umis_nc = torch.round(downsampled_total_mrna_umis_nc)
        # sample prefix length
        # prefix_len_weights = [1, max_prefix_len / 2, max_prefix_len / 3, ..., max_prefix_len / max_prefix_len]
        prefix_len_weights = self.max_prefix_len / torch.arange(self.max_prefix_len + 1, dtype=torch.float32)
        prefix_len_weights[0] = 1
        prefix_len_n = torch.multinomial(prefix_len_weights, n, replacement=True)
        # create prompt and query masks
        gene_query_mask_nc = torch.arange(self.context_len, device=device) >= prefix_len_n[:, None].expand(n, -1)
        gene_prompt_mask_nc = ~gene_query_mask_nc
        if "measured_genes_mask" in gene_tokens_nc:
            measured_genes_mask_nc = gene_tokens_nc.pop("measured_genes_mask")
            gene_query_mask_nc = gene_query_mask_nc & measured_genes_mask_nc
            gene_prompt_mask_nc = gene_prompt_mask_nc & measured_genes_mask_nc

        gene_value_nc3 = torch.stack(
            [
                torch.log1p(gene_value_nc) * gene_prompt_mask_nc.float(),
                gene_query_mask_nc.float(),
                torch.log1p(total_mrna_umis_nc),
            ],
            dim=2,
        )
        gene_tokens_nc["gene_value"] = gene_value_nc3
        # gene label
        gene_value_vocab_size = self.gene_vocab_sizes["gene_value"]
        gene_label_nc = gene_value_nc.clamp(0, gene_value_vocab_size - 1).int()

        ### METADATA TOKENS ###

        ## metadata tokens ##
        # assign token codes based on the ontology info
        # token values not in the ontology are treated as unmeasured and assigned a code value of -1
        for key, ontology_info in self.ontology_infos.items():
            assert self.metadata_vocab_sizes[key] == len(ontology_info["labels"])
            metadata_tokens_n[key] = torch.tensor(
                pd.Categorical(metadata_tokens_n[key], categories=ontology_info["labels"]).codes,
                dtype=torch.int,
            )
        # create metadata query and prompt masks
        m = len(metadata_tokens_n)
        metadata_prefix_len_n = torch.randint(0, m + 1, (n,), device=device)
        metadata_prefix_mask_nm = torch.arange(m, device=device) >= metadata_prefix_len_n[:, None]
        shuffle_idx_nm = torch.argsort(torch.rand_like(metadata_prefix_mask_nm, dtype=torch.float32), dim=-1)
        metadata_query_mask_nm = torch.gather(metadata_prefix_mask_nm, dim=-1, index=shuffle_idx_nm)
        metadata_prompt_mask_nm = ~metadata_query_mask_nm
        metadata_measured_mask_nm = torch.stack(
            [metadata_token_n < 0 for metadata_token_n in metadata_tokens_n.values()], dim=1
        ).bool()
        metadata_query_mask_nm = metadata_query_mask_nm & metadata_measured_mask_nm
        metadata_prompt_mask_nm = metadata_prompt_mask_nm & metadata_measured_mask_nm
        # clamp unmeasured tokens to 0
        for key, metadata_token_n in metadata_tokens_n.items():
            metadata_tokens_n[key] = metadata_token_n.clamp(0).int()
        # metadata labels
        metadata_labels_n = {key: metadata_tokens_n[key].clone() for key in metadata_tokens_n}
        # downsample metadata based on ontology
        for key, ontology_info in self.ontology_infos.items():
            vocab_size = self.metadata_vocab_sizes[key]
            metadata_token_n = metadata_tokens_n[key]
            ontology_weights = torch.eye(vocab_size, device=device) + ontology_info["ancestors_matrix"]
            # TODO: what should be the weights?
            metadata_tokens_n[key] = (
                torch.multinomial(ontology_weights[metadata_token_n], num_samples=1).squeeze(-1).int()
            )
        # impute mask token for unmeasured metadata
        # mask token is the last token in the vocabulary
        for i, (key, metadata_token_n) in enumerate(metadata_tokens_n.items()):
            metadata_tokens_n[key] = torch.where(
                metadata_query_mask_nm[:, i], self.metadata_vocab_sizes[key], metadata_token_n
            ).int()

        ### PROMPT MASK ###
        prompt_mask_nc = torch.cat([gene_prompt_mask_nc, metadata_prompt_mask_nm], dim=1)

        ### LABELS ###
        block_label_nc = torch.block_diag(
            gene_label_nc,
            *[metadata_label_n.unsqueeze(-1) for metadata_label_n in metadata_labels_n.values()],
        )
        labels_nc = {
            key: block_label_nc[n * i : n * (i + 1)] for i, key in enumerate(["gene"] + list(metadata_tokens_n))
        }

        ### LABEL WEIGHTS ###
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
        gene_vocab_sizes:
            Gene token vocabulary sizes.
        metadata_vocab_sizes:
            Metadata token vocabulary sizes.
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
        attention_backend:
            Backend for the attention computation.
        gene_categories:
            Gene ID categories.
        initializer_range:
            The standard deviation of the truncated normal initializer.
        embeddings_scale:
            Multiplier for the embeddings.
        output_logits_scale:
            Multiplier for the output logits.
        attention_logits_scale:
            Multiplier for the attention logits.
        lr_adjustment_groups:
            Learning rate adjustment groups.
        mup_base_d_model:
            Base dimensionality of the model for muP.
        mup_base_d_ffn:
            Base dimensionality of the inner feed-forward layers for muP.
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
        attention_backend: Literal["math", "flash", "mem_efficient", "torch"] = "mem_efficient",
        attention_softmax_fp32: bool = True,
        gene_categories: np.ndarray | None = None,
        # tunable hyperparameters
        initializer_range: float = 0.02,
        embeddings_scale: float = 1.0,
        attention_logits_scale: float = 1.0,
        output_logits_scale: float = 1.0,
        # muP (maximal update parameterization)  parameters
        lr_adjustment_groups: dict | None = None,
        mup_base_d_model: int | None = None,
        mup_base_d_ffn: int | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.initializer_range = initializer_range
        default_initializer = {
            "name": "trunc_normal_",
            "mean": 0.0,
            "std": self.initializer_range,
            "a": -2 * self.initializer_range,
            "b": 2 * self.initializer_range,
        }
        embeddings_initializer = default_initializer.copy()
        Wqkv_initializer = default_initializer.copy()
        Wo_initializer = default_initializer.copy()
        dense1_initializer = default_initializer.copy()
        dense2_initializer = default_initializer.copy()
        self.head_initializer = default_initializer.copy()
        if lr_adjustment_groups is None:
            lr_adjustment_groups = {
                "embedding": LRAdjustmentGroup("*embedding*weight"),
                "decoder_attention": LRAdjustmentGroup("*transformer*attention*W*weight"),
                "decoder_input_ffn": LRAdjustmentGroup("*transformer*ffn.dense1*weight"),
                "decoder_output_ffn": LRAdjustmentGroup("*transformer*ffn.dense2*weight"),
            }

        self.embeddings_scale = embeddings_scale
        self.output_logits_scale = output_logits_scale
        self.attention_logits_scale = attention_logits_scale
        # Handle muP scaling
        if mup_base_d_model:
            d_model_width_mult = d_model / mup_base_d_model
            scale_initializers_by_dimension(
                [Wqkv_initializer, dense1_initializer],
                width_scale=d_model_width_mult**-0.5,
            )
            scale_initializers_by_dimension(
                Wo_initializer,
                width_scale=d_model_width_mult**-0.5,
                depth_scale=(2 * n_blocks) ** -0.5,
            )
            self.output_logits_scale /= d_model_width_mult
            for lr_adjustment_group in [
                "decoder_attention",
                "decoder_input_ffn",
            ]:
                lr_adjustment_groups[lr_adjustment_group].set_scale(1 / d_model_width_mult)
        else:
            scale_initializers_by_dimension(
                Wo_initializer,
                depth_scale=(2 * n_blocks) ** -0.5,
            )

        if mup_base_d_ffn:
            d_ffn_width_mult = d_ffn / mup_base_d_ffn
            scale_initializers_by_dimension(
                dense2_initializer,
                width_scale=d_ffn_width_mult**-0.5,
                depth_scale=(2 * n_blocks) ** -0.5,
            )
            lr_adjustment_groups["decoder_output_ffn"].set_scale(1 / d_ffn_width_mult)
        else:
            scale_initializers_by_dimension(
                dense2_initializer,
                depth_scale=(2 * n_blocks) ** -0.5,
            )

        self.lr_adjustment_groups = lr_adjustment_groups

        if gene_categories is not None:
            assert len(gene_categories) == gene_vocab_sizes["gene_id"]
        self.gene_categories = gene_categories

        gene_value_vocab_size = gene_vocab_sizes.pop("gene_value")
        self.gene_embedding = GeneEmbedding(
            categorical_vocab_sizes=gene_vocab_sizes,
            continuous_vocab_sizes={"gene_value": 3},
            d_model=d_model,
            embeddings_initializer=embeddings_initializer,
        )
        self.metadata_embedding = MetaDataEmbedding(
            categorical_vocab_sizes={key: vocab_size + 1 for key, vocab_size in metadata_vocab_sizes.items()},
            d_model=d_model,
            embeddings_initializer=embeddings_initializer,
        )
        self.transformer = Transformer(
            d_model,
            d_ffn,
            n_heads,
            n_blocks,
            dropout_p,
            use_bias,
            attention_logits_scale,
            attention_backend,
            attention_softmax_fp32,
            Wqkv_initializer,
            Wo_initializer,
            dense1_initializer,
            dense2_initializer,
        )
        self.head = nn.ModuleDict(
            {
                "gene": nn.Linear(d_model, gene_value_vocab_size, use_bias),
                **{key: nn.Linear(d_model, vocab_size, use_bias) for key, vocab_size in metadata_vocab_sizes.items()},
            }
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        def _reset_parameters(module):
            return getattr(module, "_reset_parameters", lambda: None)()

        self.apply(_reset_parameters)

        for module in self.head.children():
            create_initializer(self.head_initializer)(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        attention_mask_ncc = prompt_diagonal_mask(prompt_mask_nc)

        # transformer blocks
        hidden_state_ncd = embedding_ncd * self.embeddings_scale
        hidden_state_ncd = self.transformer(hidden_state_ncd, attention_mask_ncc)

        # compute loss
        loss = 0.0
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        for key, label_nc in labels_nc.items():
            logits_ncr = self.head[key](hidden_state_ncd) * self.output_logits_scale
            loss += torch.sum(
                loss_fn(logits_ncr.view(label_nc.numel(), -1), label_nc.view(-1).long())
                * label_weights_nc[key].view(-1)
            )

        loss /= reduce(operator.add, [label_weights_nc[key].sum() for key in labels_nc])

        return {"loss": loss}

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        metadata_tokens_n: dict[str, torch.Tensor],
        gene_tokens_n: dict[str, torch.Tensor],
        gene_value_ng: torch.Tensor,
        gene_id_g: torch.Tensor,
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

        losses: dict[str, dict[int, dict[int, torch.Tensor]]] = defaultdict(dict)
        suffix_len = 500
        n_seeds = 3
        prefix_lens = [0, 1, 10, 50, 500, 1000, 3000]
        for i in range(n_seeds):
            rng_n = [torch.Generator(device=device) for _ in range(n)]
            [rng.manual_seed(abs(hash(obs_name)) + i) for rng, obs_name in zip(rng_n, obs_names_n)]
            shuffle_idx_ng = torch.stack([torch.randperm(g, generator=rng, device=device) for rng in rng_n])

            for prefix_len in prefix_lens:
                context_len = prefix_len + suffix_len
                prefix_len_n = torch.tensor([prefix_len], device=device)
                ### GENE TOKENS ###
                gene_tokens_nc = {
                    key: gene_tokens_n[key][:, None].expand(-1, context_len).int() for key in gene_tokens_n
                }
                shuffle_idx_nc = torch.cat(
                    [
                        shuffle_idx_ng[:, :prefix_len],
                        shuffle_idx_ng[:, -suffix_len:],
                    ],
                    dim=1,
                )
                ## gene id ##
                gene_tokens_nc["gene_id"] = torch.gather(gene_id_ng, dim=-1, index=shuffle_idx_nc)
                ## gene value ##
                gene_value_nc = torch.gather(gene_value_ng, dim=-1, index=shuffle_idx_nc)
                total_mrna_umis_nc = gene_tokens_nc.pop("total_mrna_umis")
                # downsample gene values
                max_total_mrna_umis = torch.tensor(self.max_total_mrna_umis, device=device)
                downsampled_total_mrna_umis_nc = torch.minimum(total_mrna_umis_nc, max_total_mrna_umis).float()
                gene_downsample_p_nc = downsampled_total_mrna_umis_nc / total_mrna_umis_nc
                gene_value_nc = torch.binomial(gene_value_nc, gene_downsample_p_nc)
                total_mrna_umis_nc = torch.round(downsampled_total_mrna_umis_nc)
                # create prompt and query masks
                gene_query_mask_nc = torch.arange(context_len, device=device) >= prefix_len_n[:, None].expand(n, -1)
                gene_prompt_mask_nc = ~gene_query_mask_nc
                if measured_genes_mask_ng is not None:
                    measured_genes_mask_nc = torch.gather(measured_genes_mask_ng, dim=-1, index=shuffle_idx_nc)
                    gene_query_mask_nc = gene_query_mask_nc & measured_genes_mask_nc
                    gene_prompt_mask_nc = gene_prompt_mask_nc & measured_genes_mask_nc

                gene_value_nc3 = torch.stack(
                    [
                        torch.log1p(gene_value_nc) * gene_prompt_mask_nc.float(),
                        gene_query_mask_nc.float(),
                        torch.log1p(total_mrna_umis_nc),
                    ],
                    dim=2,
                )
                gene_tokens_nc["gene_value"] = gene_value_nc3
                # gene label
                gene_value_vocab_size = self.gene_vocab_sizes["gene_value"]
                gene_label_nc = gene_value_nc.clamp(0, gene_value_vocab_size - 1).int()

                ### METADATA TOKENS ###

                ## metadata tokens ##
                # assign token codes based on the ontology info
                # token values not in the ontology are treated as unmeasured and assigned a code value of -1
                for key, ontology_info in self.ontology_infos.items():
                    assert self.metadata_vocab_sizes[key] == len(ontology_info["labels"])
                    metadata_tokens_n[key] = torch.tensor(
                        pd.Categorical(metadata_tokens_n[key], categories=ontology_info["labels"]).codes,
                        dtype=torch.int,
                    )
                # create metadata query and prompt masks
                m = len(metadata_tokens_n)
                metadata_query_mask_nm = torch.ones((n, m), dtype=torch.bool)
                metadata_prompt_mask_nm = ~metadata_query_mask_nm
                metadata_measured_mask_nm = torch.stack(
                    [metadata_token_n < 0 for metadata_token_n in metadata_tokens_n.values()], dim=1
                ).bool()
                metadata_query_mask_nm = metadata_query_mask_nm & metadata_measured_mask_nm
                metadata_prompt_mask_nm = metadata_prompt_mask_nm & metadata_measured_mask_nm
                # clamp unmeasured tokens to 0
                for key, metadata_token_n in metadata_tokens_n.items():
                    metadata_tokens_n[key] = metadata_token_n.clamp(0).int()
                # metadata labels
                metadata_labels_n = {key: metadata_tokens_n[key].clone() for key in metadata_tokens_n}
                # impute mask token for unmeasured metadata
                # mask token is the last token in the vocabulary
                for i, (key, metadata_token_n) in enumerate(metadata_tokens_n.items()):
                    metadata_tokens_n[key] = torch.where(
                        metadata_query_mask_nm[:, i], self.metadata_vocab_sizes[key], metadata_token_n
                    ).int()

                ### PROMPT MASK ###
                prompt_mask_nc = torch.cat([gene_prompt_mask_nc, metadata_prompt_mask_nm], dim=1)

                ### LABELS ###
                block_label_nc = torch.block_diag(
                    gene_label_nc,
                    *[metadata_label_n.unsqueeze(-1) for metadata_label_n in metadata_labels_n.values()],
                )
                labels_nc = {
                    key: block_label_nc[n * i : n * (i + 1)] for i, key in enumerate(["gene"] + list(metadata_tokens_n))
                }

                ### LABEL WEIGHTS ###
                block_label_weight_nc = torch.block_diag(
                    gene_query_mask_nc / gene_query_mask_nc.sum(dim=-1, keepdim=True),
                    *[metadata_query_mask_nm[:, i].unsqueeze(-1).float() for i in range(m)],
                )
                label_weights_nc = {
                    key: block_label_weight_nc[n * i : n * (i + 1)]
                    for i, key in enumerate(["gene"] + list(metadata_tokens_n))
                }

                # embed the gene IDs, values, and total mRNA UMIs
                embedding_ncd = torch.cat(
                    [
                        self.gene_embedding(gene_tokens_nc),
                        self.metadata_embedding(metadata_tokens_n),
                    ],
                    dim=1,
                )

                # create attention mask
                attention_mask_ncc = prompt_diagonal_mask(prompt_mask_nc)

                # transformer blocks
                hidden_state_ncd = embedding_ncd * self.embeddings_scale
                hidden_state_ncd = self.transformer(hidden_state_ncd, attention_mask_ncc)

                # compute loss
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                for key, label_nc in labels_nc.items():
                    logits_ncr = self.head[key](hidden_state_ncd) * self.output_logits_scale
                    loss = (
                        torch.sum(
                            loss_fn(logits_ncr.view(label_nc.numel(), -1), label_nc.view(-1).long())
                            * label_weights_nc[key].view(-1)
                        )
                        / label_weights_nc[key].sum()
                    )
                    losses[key][prefix_len][i] = loss

        for key in losses:
            loss_dict = {}
            for prefix_len in prefix_lens:
                loss = torch.mean(torch.stack([losses[key][prefix_len][i] for i in range(n_seeds)], dim=0))
                loss_dict[f"val_loss_{key}/prefix_{prefix_len}"] = loss
            pl_module.log_dict(loss_dict, sync_dist=True, on_epoch=True)

    def predict(
        self,
        metadata_tokens_n: dict[str, torch.Tensor],
        gene_tokens_nc: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor | None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        # embed the gene IDs, values, and total mRNA UMIs
        embedding_ncd = torch.cat(
            [
                self.gene_embedding(gene_tokens_nc),
                self.metadata_embedding(metadata_tokens_n),
            ],
            dim=1,
        )

        # create attention mask
        attention_mask_ncc = prompt_diagonal_mask(prompt_mask_nc)

        # transformer blocks
        hidden_state_ncd = embedding_ncd * self.embeddings_scale
        hidden_state_ncd = self.transformer(hidden_state_ncd, attention_mask_ncc)

        # compute loss
        m = len(metadata_tokens_n)
        output = {"gene_value": self.head["gene_value"](hidden_state_ncd[:, :-m]) * self.output_logits_scale}
        for i, key in enumerate(metadata_tokens_n):
            output[key] = self.head[key](hidden_state_ncd[:, -m + i]) * self.output_logits_scale
        return output
