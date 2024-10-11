# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cellarium.ml.utilities.layers import create_initializer

try:
    # use_cs returns True if the active device is a CSX device.
    from cerebras.pytorch.backend import use_cs
except ImportError:

    def use_cs() -> bool:
        return False


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
        attention_softmax_fp32:
            Whether to use float32 for softmax computation.
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
        x_query_ncd: torch.Tensor,
        x_key_ncd: torch.Tensor,
        x_value_ncd: torch.Tensor,
        attention_mask_ncc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_query_ncd:
                Input query tensor of shape ``(n, c, d)``.
            x_key_ncd:
                Input key tensor of shape ``(n, c, d)``.
            x_value_ncd:
                Input value tensor of shape ``(n, c, d)``.
            attention_mask_ncc:
                Attention mask tensor of shape ``(n, c, c)``.

        Returns:
            The output hidden state tensor of shape ``(n, c, d)``.
        """
        n_heads = self.n_heads
        query_ncd = self.Wq(x_query_ncd)
        key_ncd = self.Wk(x_key_ncd)
        value_ncd = self.Wv(x_value_ncd)
        # d = k * h
        query_nhck = self.split_heads(query_ncd, n_heads)
        key_nhck = self.split_heads(key_ncd, n_heads)
        value_nhck = self.split_heads(value_ncd, n_heads)

        # scale_factor is computed according to the muP paper
        scale_factor = self.attention_logits_scale / query_nhck.shape[-1]

        if (self.attention_backend == "torch") or use_cs():
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
                    attn_mask=attention_mask_ncc.unsqueeze(1),
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=scale_factor,
                )

        output_ncd = self.merge_heads(output_nhck)
        return self.Wo(output_ncd)  # _ncd
