# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal

import torch
from torch import nn

from cellarium.ml.layers.attention import MultiHeadAttention
from cellarium.ml.layers.ffn import PositionWiseFFN
from cellarium.ml.layers.normadd import NormAdd


class TransformerBlock(nn.Module):
    """
    Transformer block.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        use_bias:
            Whether to use bias in the linear transformations and norm-add layers.
        n_heads:
            Number of attention heads.
        dropout_p:
            Dropout probability.
        attention_logits_scale:
            Multiplier for the attention scores.
        attention_backend:
            Backend for the attention computation.
        attention_softmax_fp32:
            Whether to use FP32 for the softmax computation.
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
        use_bias: bool,
        n_heads: int,
        dropout_p: float,
        attention_logits_scale: float,
        attention_backend: Literal["math", "flash", "mem_efficient", "torch"],
        attention_softmax_fp32: bool,
        Wqkv_initializer: dict[str, Any],
        Wo_initializer: dict[str, Any],
        dense1_initializer: dict[str, Any],
        dense2_initializer: dict[str, Any],
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"`d_model` ({d_model}) must be divisible by `n_heads` ({n_heads})")
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
        self.normadd1 = NormAdd(d_model, dropout_p, use_bias)
        self.ffn = PositionWiseFFN(
            d_ffn,
            d_model,
            use_bias,
            dense1_initializer,
            dense2_initializer,
        )
        self.normadd2 = NormAdd(d_model, dropout_p, use_bias)

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
        use_bias:
            Whether to use bias in the linear transformations and norm-add layers.
        n_heads:
            Number of attention heads.
        n_blocks:
            Number of transformer blocks.
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
        dense1_initializer:
            Initializer for the first dense layer.
        dense2_initializer:
            Initializer for the second dense layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        use_bias: bool,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        attention_logits_scale: float,
        attention_backend: Literal["math", "flash", "mem_efficient", "torch"],
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
                    use_bias,
                    n_heads,
                    dropout_p,
                    attention_logits_scale,
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
        self.ln = nn.LayerNorm(d_model, use_bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.ln.reset_parameters()

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

        return self.ln(hidden_state_ncd)
