# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence

import numpy as np
import torch
from scvi.distributions import NegativeBinomial
from torch import nn

from cellarium.ml.models.model import CellariumModel

# from cellarium.ml.models.mu_linear import MuLinear
# from cellarium.ml.transforms import DivideByScale, Filter, NormalizeTotal


class DotProductAttention(nn.Module):  # @save
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries_nqd, keys_nsd, values_nsv, prefix_len_n, use_keops=False):
        n, q, d = queries_nqd.shape
        s = keys_nsd.shape[1]
        if use_keops:
            from pykeops.torch import LazyTensor, Vi, Vj

            prefix_len_n = LazyTensor(prefix_len_n[:, None, None, None].expand([n, q, 1, 1]).float())
            q_range = LazyTensor(torch.arange(q, device=keys_nsd.device)[:, None].float(), axis=0)
            s_range = LazyTensor(torch.arange(s, device=keys_nsd.device)[:, None].float(), axis=1)
            block_mask = 1 - (prefix_len_n - s_range - 1).step()
            diag_mask = 1 - (-(q_range - s_range).abs()).step()
            attention_mask_nqs = block_mask * diag_mask
            scores_nqs = (
                LazyTensor(queries_nqd[:, :, None, :].contiguous()) * LazyTensor(keys_nsd[:, None, :, :].contiguous())
            ).sum(-1) / math.sqrt(d)
            scores_nqs = scores_nqs - 1e9 * attention_mask_nqs
            return scores_nqs.sumsoftmaxweight(LazyTensor(values_nsv.unsqueeze(1).contiguous()), dim=2)
        else:
            block_mask = (
                torch.arange((s), dtype=torch.short, device=queries_nqd.device).expand([s, s])
                < prefix_len_n[:, None, None]
            )
            diag_mask = (
                torch.diag(torch.ones((s,), dtype=torch.bool, device=queries_nqd.device)).bool().expand([n, s, s])
            )
            attention_mask_nqs = block_mask | diag_mask
            # Swap the last two dimensions of keys with keys.transpose(1, 2)
            scores_nqs = queries_nqd @ keys_nsd.transpose(1, 2) / math.sqrt(d)
            scores_nqs[~attention_mask_nqs] = -1e9
            self.attention_probs_nqs = scores_nqs.softmax(dim=-1)
            return self.dropout(self.attention_probs_nqs) @ values_nsv  # _nqv


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, d_hiddens, h_heads, dropout, bias=True, **kwargs):
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
        prefix_len_n,
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
        # attention_mask_mqs = self.split_attention_mask(attention_mask_nqs, h_heads)
        prefix_len_m = prefix_len_n.repeat(1, h_heads).reshape(-1)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output_mqk = self.attention(queries_mqk, keys_msk, values_msk, prefix_len_m)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_nqd = self.merge_heads(output_mqk, h_heads)
        return self.Wo(output_nqd)  # _nqd


class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""

    def __init__(self, mlp_hiddens, d_hiddens):
        super().__init__()
        self.dense1 = nn.Linear(d_hiddens, mlp_hiddens, bias=True)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(mlp_hiddens, d_hiddens, bias=True)

    def forward(self, X_nd):
        return self.dense2(self.relu(self.dense1(X_nd)))


class ValueEmbedding(nn.Module):
    """The positionwise feed-forward network."""

    def __init__(self, d_hiddens):
        super().__init__()
        self.dense1 = nn.Linear(1, d_hiddens, bias=True)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(d_hiddens, d_hiddens, bias=True)

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

    def forward(self, X, prefix_len_n):
        Y = self.normadd1(X, self.attention(X, X, X, prefix_len_n))
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
        # b_bins: int = 20,  # including 0
        d_hiddens: int = 128,
        f_hiddens: int = 512,
        h_heads: int = 4,
        num_blocks: int = 6,
        dropout: float = 0.02,
        use_bias: bool = True,
        c_context: int | None = None,
        tdigest_path: str | None = None,
        initializer_range: float = 0.02,
        optimizer: str = "adam",
    ):
        super().__init__()
        self.feature_schema = np.array(feature_schema)

        self.feature_ids: torch.Tensor
        # ids for the features, 0 is for cls
        self.register_buffer("feature_ids", torch.arange(1, len(feature_schema) + 1))
        # self.b_bins = b_bins
        self.g_genes = len(feature_schema)

        # +1 for masking
        self.id_embedding = nn.Embedding(len(feature_schema) + 1, d_hiddens)
        self.mask_embedding = nn.Embedding(1, d_hiddens)
        self.value_embedding = ValueEmbedding(d_hiddens)

        self.num_blocks = num_blocks
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
        # no mask token in predictions
        self.dense = nn.Linear(d_hiddens, 2)

        # assert tdigest_path is not None
        # from crick import TDigest

        # from cellarium.ml import CellariumModule

        # tdigest = CellariumModule.load_from_checkpoint(tdigest_path).model
        # median_g = tdigest.median_g
        # self.normalize = NormalizeTotal(
        #     target_count=tdigest.transform.target_count,
        #     eps=tdigest.transform.eps,
        # )
        # self.divide = DivideByScale(
        #     scale_g=median_g,
        #     feature_schema=feature_schema,
        #     eps=tdigest.transform.eps,
        # )
        # self.filter = Filter(self.feature_schema[median_g.isfinite()])
        self.c_context = c_context  # or len(self.filter.filter_list)
        # normalized_tdigest = TDigest()
        # for median, tdigest in zip(median_g, tdigest.tdigests):
        #     state = tdigest.__getstate__()
        #     state[0]["mean"] = state[0]["mean"] / median
        #     tdigest.__setstate__(state)
        #     normalized_tdigest.merge(tdigest)
        # probs = np.linspace(0, 1, b_bins)[1:-1]
        # quantiles = normalized_tdigest.quantile(probs)
        # quantiles = np.concatenate([[-1, 0], quantiles, [float("inf")]])
        # self.quantiles: torch.Tensor
        # self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.optimizer = optimizer
        self.initializer_range = initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            assert self.optimizer == "adam", "Only Adam(W) optimizer is supported for now."
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
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
                p.data.normal_(mean=0.0, std=(self.initializer_range / math.sqrt(2 * self.num_blocks)))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x_ng = tensor_dict["X"]
        feature_g = tensor_dict["var_names"]
        total_mrna_umis_n = tensor_dict["total_mrna_umis_n"]
        return (x_ng, feature_g, total_mrna_umis_n), {}

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray, total_mrna_umis_n: torch.Tensor) -> torch.Tensor:
        # random_indices = torch.argsort(torch.rand_like(x_ng))[:, : self.c_context - 1]
        # ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
        # x_nc = x_ng[ndx[:, None], random_indices]
        # x_nc = torch.cat([total_mrna_umis_n[:, None], x_nc], dim=1)
        # gene_ids = self.feature_ids[random_indices]
        # gene_ids = torch.cat([torch.zeros((x_ng.shape[0], 1), dtype=torch.long, device=x_ng.device), gene_ids], dim=1)
        random_indices_ng = torch.argsort(torch.rand_like(x_ng))
        prefix_len_n = torch.randint(1, self.c_context, (x_ng.shape[0],), device=x_ng.device)
        masked_indices_ng = (
            torch.arange((self.g_genes), dtype=torch.float32, device=x_ng.device)[None, :] >= prefix_len_n[:, None]
        )
        masked_indices_nc = masked_indices_ng[:, : self.c_context]
        random_indices_nc = random_indices_ng[:, : self.c_context]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
        randomized_x_ng = x_ng[ndx[:, None], random_indices_ng]
        randomized_x_nc = randomized_x_ng[:, : self.c_context]

        zero_indices_ng = (randomized_x_ng == 0) & masked_indices_ng
        nonzero_indices_ng = (randomized_x_ng > 0) & masked_indices_ng
        zero_counts_n1 = zero_indices_ng.sum(dim=1, keepdim=True)
        nonzero_counts_n1 = nonzero_indices_ng.sum(dim=1, keepdim=True)
        total_counts_n1 = masked_indices_ng.sum(dim=1, keepdim=True)
        zero_weights_ng = (0.5 / zero_counts_n1).expand(-1, self.g_genes)
        nonzero_weights_ng = (0.5 / nonzero_counts_n1).expand(-1, self.g_genes)
        # zero_weights_ng = (1 / total_counts_n1).expand(-1, self.g_genes)
        # nonzero_weights_ng = (1 / total_counts_n1).expand(-1, self.g_genes)
        weights_ng = torch.zeros_like(x_ng, dtype=torch.float32)
        weights_ng[zero_indices_ng] = zero_weights_ng[zero_indices_ng]
        weights_ng[nonzero_indices_ng] = nonzero_weights_ng[nonzero_indices_ng]
        label_indices_nc = torch.multinomial(weights_ng, num_samples=self.c_context, replacement=True)
        # label_indices_nc = random_indices_nc.clone()
        importance_weights_ng = 1 / (weights_ng * total_counts_n1)
        label_weights_nc = importance_weights_ng[ndx[:, None], label_indices_nc]
        sample_weights_nc = 1 / masked_indices_nc.sum(dim=1, keepdim=True).expand(-1, self.c_context)
        # 1
        labels_nc = x_ng[ndx[:, None], label_indices_nc]
        labels_nc[~masked_indices_nc] = -100
        randomized_x_nc[masked_indices_nc] = labels_nc[masked_indices_nc]
        random_indices_nc[masked_indices_nc] = label_indices_nc[masked_indices_nc]
        label_weights = label_weights_nc[masked_indices_nc]
        # label_weights = 1
        # 2
        # import pdb

        # pdb.set_trace()
        # labels_nc = randomized_x_ng[:, : self.c_context].clone()
        # labels_nc[~masked_indices_nc] = -100
        # label_weights = 1

        gene_ids = self.feature_ids[random_indices_nc]
        # value_ids[masked_indices] = self.b_bins
        value_ids = torch.log1p(randomized_x_nc)

        gene_ncd = self.id_embedding(gene_ids)
        value_ncd = self.value_embedding(value_ids.unsqueeze(-1))
        mask_embedding = self.mask_embedding(torch.zeros(1, device=x_ng.device).long())
        value_ncd[masked_indices_nc] = mask_embedding.squeeze()
        hidden_ncd = gene_ncd + value_ncd

        # per cell
        # 1 1 1 0 0
        # 1 1 1 0 0
        # 1 1 1 0 0
        # 1 1 1 1 0
        # 1 1 1 0 1

        self.attention_weights = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            hidden_ncd = block(
                hidden_ncd,
                prefix_len_n=prefix_len_n,
            )
            # self.attention_weights[i] = block.attention.attention.attention_probs_nqs

        logits = self.dense(hidden_ncd)
        mu_nc = logits[:, :, 0].exp()
        theta_nc = logits[:, :, 1].exp()
        mu = mu_nc[masked_indices_nc]
        theta = theta_nc[masked_indices_nc]
        dist = NegativeBinomial(mu=mu, theta=theta)
        log_prob = dist.log_prob(labels_nc[masked_indices_nc])
        sample_weights = sample_weights_nc[masked_indices_nc]
        loss = -(log_prob * label_weights * sample_weights).sum() / sample_weights.sum()

        # loss_logits = logits[masked_indices]
        # loss_logits = loss_logits - loss_logits.logsumexp(dim=-1, keepdim=True)
        # loss_labels = labels[masked_indices].unsqueeze(-1)
        # log_prob = loss_logits.gather(-1, loss_labels).squeeze(-1)
        #  loss_fn = nn.CrossEntropyLoss(weight=weights.view(-1), ignore_index=-100, reduction="mean")
        #  loss = loss_fn(logits.view(-1, self.b_bins), labels.view(-1))
        with torch.no_grad():
            nonzero_mask = labels_nc > 0
            nonzero_log_prob = NegativeBinomial(mu=mu_nc[nonzero_mask], theta=theta_nc[nonzero_mask]).log_prob(
                labels_nc[nonzero_mask]
            )
            nonzero_sample_weights = sample_weights_nc[nonzero_mask]
            nonzero_loss = -(nonzero_log_prob * nonzero_sample_weights).sum() / nonzero_sample_weights.sum()

            zero_mask = labels_nc == 0
            zero_log_prob = NegativeBinomial(mu=mu_nc[zero_mask], theta=theta_nc[zero_mask]).log_prob(
                labels_nc[zero_mask]
            )
            zero_sample_weights = sample_weights_nc[zero_mask]
            zero_loss = -(zero_log_prob * zero_sample_weights).sum() / zero_sample_weights.sum()

            prefix_len_nc = prefix_len_n[:, None].expand(-1, self.c_context)
            # nonzero_prefix_len = prefix_len_nc[nonzero_mask]
            # # split context size of 2048 into 8 parts
            # nonzero_prefix = torch.bucketize(
            #     nonzero_prefix_len, torch.arange(0, self.c_context, self.c_context // 8, device=x_nc.device), right=True
            # )
        return {
            "loss": loss,
            "nonzero_loss": nonzero_loss,
            "zero_loss": zero_loss,
            "logits": logits,
            "nonzero_mask": nonzero_mask,
            "zero_mask": zero_mask,
            "labels": labels_nc,
            "prefix_len_nc": prefix_len_nc,
            # "zero_preds": torch.argmax(logits[zero_mask], dim=-1),
            # "zero_labels": labels[zero_mask],
            # "nonzero_preds": torch.argmax(logits[nonzero_mask], dim=-1),
            # "nonzero_labels": labels[nonzero_mask],
            # "nonzero_prefix": nonzero_prefix,
            "mu_nc": mu_nc,
            "theta_nc": theta_nc,
            "nonzero_log_prob": nonzero_log_prob,
            "zero_log_prob": zero_log_prob,
        }
