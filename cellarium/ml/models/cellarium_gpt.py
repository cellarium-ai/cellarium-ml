# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.transforms import DivideByScale, NormalizeTotal, Filter


class DotProductAttention(nn.Module):  # @save
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries_nqd, keys_nsd, values_nsv, attention_mask_nqs):
        d = queries_nqd.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores_nqs = queries_nqd @ keys_nsd.transpose(1, 2) / math.sqrt(d)
        scores_nqs[~attention_mask_nqs] = -1e9
        # scores_nqs = scores_nqs.masked_fill(~attention_mask, -1e9)
        self.attention_probs_nqs = scores_nqs.softmax(dim=-1)
        return self.dropout(self.attention_probs_nqs) @ values_nsv  # _nqv


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, d_hiddens, h_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.h_heads = h_heads
        self.attention = DotProductAttention(dropout)
        self.Wq = nn.Linear(d_hiddens, d_hiddens, bias=bias)
        self.Wk = nn.Linear(d_hiddens, d_hiddens, bias=bias)
        self.Wv = nn.Linear(d_hiddens, d_hiddens, bias=bias)
        self.Wo = nn.Linear(d_hiddens, d_hiddens, bias=bias)

    @staticmethod
    def split_heads(X, h_heads):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], h_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    @staticmethod
    def split_attention_mask(attention_mask, h_heads):
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.repeat(1, h_heads, 1, 1)
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[2], attention_mask.shape[3])
        return attention_mask

    @staticmethod
    def merge_heads(X, h_heads):
        # reverse of split_heads
        X = X.reshape(-1, h_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(
        self,
        queries_nqd,
        keys_nsd,
        values_nsd,
        attention_mask_nqs,
    ):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        h_heads = self.h_heads
        queries_nqd = self.Wq(queries_nqd)
        keys_nsd = self.Wk(keys_nsd)
        values_nsd = self.Wv(values_nsd)
        # m = n * h
        # k = d / h
        queries_mqk = self.split_heads(queries_nqd, h_heads)
        keys_msk = self.split_heads(keys_nsd, h_heads)
        values_msk = self.split_heads(values_nsd, h_heads)
        attention_mask_mqs = self.split_attention_mask(attention_mask_nqs, h_heads)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output_mqk = self.attention(queries_mqk, keys_msk, values_msk, attention_mask_mqs)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_nqd = self.merge_heads(output_mqk, h_heads)
        return self.Wo(output_nqd)  # _nqd


class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""

    def __init__(self, mlp_hiddens, d_hiddens):
        super().__init__()
        self.dense1 = nn.Linear(d_hiddens, mlp_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(mlp_hiddens, d_hiddens)

    def forward(self, X_nd):
        return self.dense2(self.relu(self.dense1(X_nd)))


class NormAdd(nn.Module):
    """The residual connection followed by layer normalization."""

    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return X + self.ln(self.dropout(Y))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_hiddens: int,
        f_hiddens: int,
        h_heads: int,
        dropout: float,
        use_bias: bool = False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_hiddens, h_heads, dropout, use_bias)
        self.normadd1 = NormAdd(d_hiddens, dropout)
        self.ffn = PositionWiseFFN(f_hiddens, d_hiddens)
        self.normadd2 = NormAdd(d_hiddens, dropout)

    def forward(self, X, attention_mask):
        Y = self.normadd1(X, self.attention(X, X, X, attention_mask))
        return self.normadd2(Y, self.ffn(Y))


class CellariumGPT(CellariumModel):
    """
    Cellarium GPT model.
    Args:
        feature_schema:
            The list of the variable names in the input data.
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            Has to be one of the following options:
                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].
            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
    """

    def __init__(
        self,
        feature_schema: Sequence[str],
        b_bins: int = 5,
        d_hiddens: int = 32,
        f_hiddens: int = 64,
        h_heads: int = 4,
        num_blocks: int = 6,
        dropout: float = 0.1,
        use_bias: bool = False,
        max_len: int = 32,
        tdigest_path: str | None = None,
    ):
        super().__init__()
        self.feature_schema = np.array(feature_schema)

        self.feature_ids: torch.Tensor
        # ids for the features, 0 is for padding, 1 is for mask
        self.register_buffer("feature_ids", torch.arange(2, len(feature_schema) + 2))
        self.b_bins = b_bins
        self.g_genes = len(feature_schema)

        self.id_embedding = nn.Embedding(len(feature_schema) + 2, d_hiddens)
        self.value_embedding = nn.Embedding(b_bins + 1, d_hiddens)
        self.max_len = max_len

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_hiddens,
                    f_hiddens,
                    h_heads,
                    dropout,
                    use_bias,
                )
                for i in range(num_blocks)
            ]
        )
        self.dense = nn.Linear(d_hiddens, b_bins + 1, bias=False)

        assert tdigest_path is not None
        from cellarium.ml import CellariumModule
        from crick import TDigest

        tdigest = CellariumModule.load_from_checkpoint(tdigest_path).model
        median_g = tdigest.median_g
        self.normalize = NormalizeTotal(
            target_count=tdigest.transform.target_count,
            eps=tdigest.transform.eps,
        )
        self.divide = DivideByScale(
            scale_g=median_g,
            feature_schema=feature_schema,
            eps=tdigest.transform.eps,
        )
        self.filter = Filter(self.feature_schema[median_g.isfinite()])
        normalized_tdigest = TDigest()
        for median, tdigest in zip(median_g, tdigest.tdigests):
            state = tdigest.__getstate__()
            state[0]["mean"] = state[0]["mean"] / median
            tdigest.__setstate__(state)
            normalized_tdigest.merge(tdigest)
        probs = np.linspace(0, 1, b_bins)[1:-1]
        quantiles = normalized_tdigest.quantile(probs)
        quantiles = np.concatenate([[-1, 0], quantiles, [float("inf")]])
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x_ng = tensor_dict["X"]
        feature_g = tensor_dict["var_names"]
        total_mrna_umis_n = tensor_dict["total_mrna_umis_n"]
        return (x_ng, feature_g, total_mrna_umis_n), {}

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray, total_mrna_umis_n: torch.Tensor) -> torch.Tensor:
        x_ng = self.normalize(x_ng, total_mrna_umis_n)
        x_ng = self.divide(x_ng, feature_g)
        x_ng, feature_g = self.filter(x_ng, feature_g)
        #  randomize(x_ng)
        #  clip(x_ng)
        random_indices = torch.argsort(torch.rand_like(x_ng))[:, : self.max_len]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
        x_ng = x_ng[ndx[:, None], random_indices]
        gene_ids = self.feature_ids[random_indices]
        # right=False: boundaries[i-1] < input[m][n]...[l][x] <= boundaries[i]
        value_ids = torch.bucketize(x_ng, self.quantiles, right=False)
        labels = value_ids.clone()
        atten_len = torch.randint(1, self.max_len, (x_ng.shape[0],), device=x_ng.device)
        masked_indices = (
            torch.arange((self.max_len), dtype=torch.float32, device=x_ng.device)[None, :] >= atten_len[:, None]
        )
        labels[~masked_indices] = -100
        value_ids[masked_indices] = 0

        gene_gd = self.id_embedding(gene_ids)
        value_ngd = self.value_embedding(value_ids)
        hidden_ngd = gene_gd + value_ngd

        # per cell
        # 1 1 1 0 0
        # 1 1 1 0 0
        # 1 1 1 0 0
        # 1 1 1 1 0
        # 1 1 1 0 1
        block_mask = (
            torch.arange((self.max_len), dtype=torch.short, device=x_ng.device).expand([self.max_len, self.max_len])
            < atten_len[:, None, None]
        )
        diag_mask = (
            torch.diag(torch.ones((self.max_len,), dtype=torch.bool, device=x_ng.device))
            .bool()
            .expand([x_ng.shape[0], self.max_len, self.max_len])
        )
        attention_mask = block_mask | diag_mask

        self.attention_weights = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            hidden_ngd = block(
                hidden_ngd,
                attention_mask=attention_mask,
            )
            self.attention_weights[i] = block.attention.attention.attention_probs_nqs

        logits = self.dense(hidden_ngd)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.b_bins + 1), labels.view(-1))
        nonzero_mask = labels > 1
        zero_mask = labels == 1
        # nonzero_loss = loss_fn(logits[mask], labels[mask])
        # zero_loss = loss_fn(logits[~mask], labels[~mask])
        # loss = nn.cross_entropy(logits, feature_g)
        return {
            "loss": loss,
            "zero_logits": logits[zero_mask],
            "zero_labels": labels[zero_mask],
            "nonzero_logits": logits[nonzero_mask],
            "nonzero_labels": labels[nonzero_mask],
        }
