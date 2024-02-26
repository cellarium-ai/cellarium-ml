# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.distributions as dist
from lightning.pytorch.core.mixins import HyperparametersMixin
from pykeops.torch import LazyTensor
from torch import nn
from torch.backends.cuda import sdp_kernel
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

backend_map = {
    "math": {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    "flash": {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    "mem_efficient": {"enable_math": False, "enable_flash": False, "enable_mem_efficient": True},
}


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


class NegativeBinomial(dist.Distribution):
    r"""Negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        mu: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8

        mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        self.scale = scale
        super().__init__(validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.mean + (self.mean**2) / self.theta

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: torch.Size | tuple | None = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = dist.Poisson(l_train).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)

    def _gamma(self) -> dist.Gamma:
        pass
        # return _gamma(self.theta, self.mu)

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: {self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
                if self.__dict__[p] is not None
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(
        self,
        dropout: float,
        attn_mult: float,
        backend: Literal["pykeops", "pytorch", "math", "flash", "mem_efficient"],
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn_mult = attn_mult
        self.backend = backend

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(
        self,
        queries_nqd: torch.Tensor,
        keys_nsd: torch.Tensor,
        values_nsv: torch.Tensor,
        prefix_len_n=None,
        use_softmax: bool = True,
    ):
        n, q, d = queries_nqd.shape
        s = keys_nsd.shape[1]
        if self.backend == "pykeops":

            prefix_len_n = LazyTensor(prefix_len_n[:, None, None, None].expand([n, q, 1, 1]).float())
            q_range = LazyTensor(torch.arange(q, device=keys_nsd.device)[:, None].float(), axis=0)
            s_range = LazyTensor(torch.arange(s, device=keys_nsd.device)[:, None].float(), axis=1)
            block_mask = 1 - (prefix_len_n - s_range - 1).step()
            diag_mask = 1 - (-(q_range - s_range).abs()).step()
            attention_mask_nqs = block_mask * diag_mask
            scores_nqs = (
                (
                    LazyTensor(queries_nqd[:, :, None, :].contiguous())
                    * LazyTensor(keys_nsd[:, None, :, :].contiguous())
                ).sum(-1)
                * self.attn_mult
                / d
            )
            scores_nqs = scores_nqs - 1e9 * attention_mask_nqs
            return scores_nqs.sumsoftmaxweight(LazyTensor(values_nsv.unsqueeze(1).contiguous()), dim=2)
        elif self.backend in ["math", "flash", "mem_efficient"]:
            block_mask = (
                torch.arange((s), dtype=torch.short, device=queries_nqd.device).expand([s, s])
                < prefix_len_n[:, None, None]
            )
            diag_mask = (
                torch.diag(torch.ones((s,), dtype=torch.bool, device=queries_nqd.device)).bool().expand([n, s, s])
            )
            attention_mask_nqs = block_mask | diag_mask
            with sdp_kernel(**backend_map[self.backend]):
                return nn.functional.scaled_dot_product_attention(
                    queries_nqd.unsqueeze(0),
                    keys_nsd.unsqueeze(0),
                    values_nsv.unsqueeze(0),
                    attention_mask_nqs,
                    scale=self.attn_mult / d,
                ).squeeze(0)
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
            scores_nqs = queries_nqd @ keys_nsd.transpose(1, 2) * self.attn_mult / d
            scores_nqs[~attention_mask_nqs] = -1e9
            self.attention_probs_nqs = scores_nqs.softmax(dim=-1)
            self.scores_nqs = scores_nqs
            self.attention_mask_nqs = attention_mask_nqs
            return self.dropout(self.attention_probs_nqs) @ values_nsv  # _nqv


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, d_hiddens, h_heads, dropout, attn_mult, bias=True):
        super().__init__()
        self.h_heads = h_heads
        self.attention = DotProductAttention(dropout, attn_mult)
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
        output_mqk = self.attention(queries_mqk, keys_msk, values_msk, prefix_len_n=prefix_len_m)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_nqd = self.merge_heads(output_mqk, h_heads)
        return self.Wo(output_nqd)  # _nqd


class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""

    def __init__(self, mlp_hiddens, d_hiddens, use_bias):
        super().__init__()
        self.dense1 = nn.Linear(d_hiddens, mlp_hiddens, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(mlp_hiddens, d_hiddens, bias=use_bias)

    def forward(self, X_nd):
        return self.dense2(self.relu(self.dense1(X_nd)))


class ValueEmbedding(nn.Module):
    """The positionwise feed-forward network."""

    def __init__(self, d_hiddens, use_bias):
        super().__init__()
        self.dense1 = nn.Linear(1, d_hiddens, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(d_hiddens, d_hiddens, bias=use_bias)

    def forward(self, X_nd):
        return self.dense2(self.relu(self.dense1(X_nd)))


class NormAdd(nn.Module):
    """Pre-norm layer where the layer normalization is applied before the sublayer."""

    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    # def forward(self, X, sublayer):
    #     return X + self.dropout(sublayer(self.ln(X)))
    def forward(self, X, Y):
        return X + self.ln(self.dropout(Y))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_hiddens: int,
        f_hiddens: int,
        h_heads: int,
        dropout: float,
        attn_mult: float,
        use_bias: bool = False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_hiddens, h_heads, dropout, attn_mult, use_bias)
        self.normadd1 = NormAdd(d_hiddens, dropout)
        self.ffn = PositionWiseFFN(f_hiddens, d_hiddens, use_bias)
        self.normadd2 = NormAdd(d_hiddens, dropout)

    # def forward(self, X, prefix_len_n):
    #     Y = self.normadd1(X, lambda X: self.attention(X, X, X, prefix_len_n))
    #     return self.normadd2(Y, lambda Y: self.ffn(Y))
    def forward(self, X, prefix_len_n, attention_type="block"):
        Y = self.normadd1(X, self.attention(X, X, X, prefix_len_n))
        return self.normadd2(Y, self.ffn(Y))


class CellariumGPT(CellariumModel, HyperparametersMixin):
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
        n_vars: int,
        n_hiddens: int = 128,
        n_intermediates: int = 256,
        n_heads: int = 4,
        n_blocks: int = 6,
        dropout: float = 0.02,
        use_bias: bool = False,
        n_context: int | None = None,
        attn_mult: float = 1.0,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        initializer_range: float = 0.02,
        importance_sampling: bool = False,
        skip: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        # self.var_names_g = np.array(var_names_g)
        # self.n_vars = len(var_names_g)
        self.n_vars = n_vars
        self.attn_mult = attn_mult
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.importance_sampling = importance_sampling
        self.skip = skip
        # skip
        if self.skip:
            self.query_embedding = nn.Embedding(self.n_vars + 1, n_hiddens)
            self.skip_attention = MultiHeadAttention(n_hiddens, n_heads, dropout, attn_mult, use_bias)
            self.skip_normadd1 = NormAdd(n_hiddens, dropout)
            self.skip_ffn = PositionWiseFFN(n_intermediates, n_hiddens, use_bias)
            self.skip_normadd2 = NormAdd(n_hiddens, dropout)

        # self.var_ids_g: torch.Tensor
        # ids for the features, 0 is for cls
        # self.register_buffer("var_ids_g", torch.arange(self.n_vars + 1))

        # +1 for cls
        self.id_embedding = nn.Embedding(self.n_vars + 1, n_hiddens)
        # +1 for masking
        self.mask_embedding = nn.Embedding(1, n_hiddens)
        self.value_embedding = ValueEmbedding(n_hiddens, use_bias=use_bias)

        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_hiddens,
                    n_intermediates,
                    n_heads,
                    dropout,
                    attn_mult,
                    use_bias,
                )
                for i in range(n_blocks)
            ]
        )
        # no mask token in predictions
        self.dense = nn.Linear(n_hiddens, 2, bias=use_bias)

        self.n_context = n_context  # or len(self.filter.filter_list)
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
                p.data.normal_(mean=0.0, std=(self.initializer_range / math.sqrt(2 * self.n_blocks)))

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray, total_mrna_umis_n: torch.Tensor) -> torch.Tensor:
        # assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        # assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        ### masking
        # n_context = 5
        # choices = [1, 2, 3, 4]
        # prefix = 3
        # indices = [0, 1, 2, 3, 4]
        # labels mask = [False, False, False, True, True]
        prefix_len_n = torch.randint(1, self.n_context, (x_ng.shape[0],), device=x_ng.device)
        # prefix_len = torch.randint(1, self.n_context, (1,), device=x_ng.device)

        # importance sampling
        if self.importance_sampling:
            labels_mask_ng = (
                torch.arange((self.n_vars + 1), dtype=torch.float32, device=x_ng.device)[None, :]
                >= prefix_len_n[:, None]
            )
            labels_mask_nc = labels_mask_ng[:, : self.n_context]
            ### prefix
            random_indices_ng = torch.argsort(torch.rand_like(x_ng))
            random_indices_ng = torch.cat(
                [torch.zeros((x_ng.shape[0], 1), dtype=torch.long, device=x_ng.device), random_indices_ng + 1], dim=1
            )
            x_ng = torch.cat([total_mrna_umis_n[:, None], x_ng], dim=1)
            random_indices_nc = random_indices_ng[:, : self.n_context]
            ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
            random_x_ng = x_ng[ndx[:, None], random_indices_ng]
            random_x_nc = random_x_ng[:, : self.n_context]

            zero_mask_ng = (random_x_ng == 0) & labels_mask_ng
            nonzero_mask_ng = (random_x_ng > 0) & labels_mask_ng
            zero_counts_n1 = zero_mask_ng.sum(dim=1, keepdim=True)
            nonzero_counts_n1 = nonzero_mask_ng.sum(dim=1, keepdim=True)
            total_counts_n1 = labels_mask_ng.sum(dim=1, keepdim=True)
            zero_weights_ng = (0.5 / zero_counts_n1).expand(-1, self.n_vars + 1)
            nonzero_weights_ng = (0.5 / nonzero_counts_n1).expand(-1, self.n_vars + 1)
            weights_ng = torch.zeros_like(x_ng, dtype=torch.float32)
            weights_ng[zero_mask_ng] = zero_weights_ng[zero_mask_ng]
            weights_ng[nonzero_mask_ng] = nonzero_weights_ng[nonzero_mask_ng]
            label_indices_nc = torch.multinomial(weights_ng, num_samples=self.n_context, replacement=True)
            importance_weights_ng = 1 / (weights_ng * total_counts_n1)
            label_weights_nc = importance_weights_ng[ndx[:, None], label_indices_nc]
            # 1
            labels_nc = x_ng[ndx[:, None], label_indices_nc]
            labels_nc[~labels_mask_nc] = -100
            random_x_nc[labels_mask_nc] = labels_nc[labels_mask_nc]
            random_indices_nc[labels_mask_nc] = label_indices_nc[labels_mask_nc]
            label_weights = label_weights_nc[labels_mask_nc]
        else:
            labels_mask_nc = (
                torch.arange((self.n_context), dtype=torch.float32, device=x_ng.device)[None, :]
                >= prefix_len_n[:, None]
            )
            ### prefix
            ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
            random_indices_ng = torch.argsort(torch.rand_like(x_ng))
            random_indices_nc = random_indices_ng[:, : self.n_context - 1]
            random_x_nc = x_ng[ndx[:, None], random_indices_nc]
            random_x_nc = torch.cat([total_mrna_umis_n[:, None], random_x_nc], dim=1)
            random_indices_nc = torch.cat(
                [torch.zeros((x_ng.shape[0], 1), dtype=torch.long, device=x_ng.device), random_indices_nc + 1], dim=1
            )
            labels_nc = random_x_nc.clone()
            labels_nc[~labels_mask_nc] = -100
            label_weights = 1

        sample_weights_nc = 1 / labels_mask_nc.sum(dim=1, keepdim=True).expand(-1, self.n_context)
        gene_ids = random_indices_nc
        value_ids = torch.log1p(random_x_nc)

        gene_ncd = self.id_embedding(gene_ids)
        value_ncd = self.value_embedding(value_ids.unsqueeze(-1))
        mask_embedding = self.mask_embedding(torch.zeros(1, device=x_ng.device).long())
        value_ncd[labels_mask_nc] = mask_embedding.squeeze()
        hidden_ncd = gene_ncd + value_ncd

        # attention_weights = [None] * len(self.blocks)

        hidden_ncd = hidden_ncd * self.input_mult
        for i, block in enumerate(self.blocks):
            hidden_ncd = block(
                hidden_ncd,
                prefix_len_n=prefix_len_n,
            )
            # attention_weights[i] = block.attention.attention.attention_probs_nqs

        attention_weights = None
        if self.skip:
            queries = self.query_embedding(gene_ids)
            hidden_ncd[labels_mask_nc] = queries[labels_mask_nc]
            Y = self.skip_normadd1(0, self.skip_attention(queries, hidden_ncd, hidden_ncd, prefix_len_n))
            hidden_ncd = self.skip_normadd2(0, self.skip_ffn(Y))
            attention_weights = self.skip_attention.attention.attention_probs_nqs

        logits = self.dense(hidden_ncd) * self.output_mult
        mu_nc = logits[:, :, 0].sigmoid()
        theta_nc = logits[:, :, 1].exp()
        mu = mu_nc[labels_mask_nc]
        theta = theta_nc[labels_mask_nc]
        total_mrna_umis_nc = total_mrna_umis_n.unsqueeze(1).expand(-1, self.n_context)
        total_mrna_umis = total_mrna_umis_nc[labels_mask_nc]
        model_dist = NegativeBinomial(mu=mu * total_mrna_umis, theta=theta)
        log_prob = model_dist.log_prob(labels_nc[labels_mask_nc])
        sample_weights = sample_weights_nc[labels_mask_nc]
        loss = -(log_prob * label_weights * sample_weights).sum() / sample_weights.sum()

        with torch.no_grad():
            nonzero_mask = labels_nc > 0
            nonzero_log_prob = NegativeBinomial(
                mu=mu_nc[nonzero_mask] * total_mrna_umis_nc[nonzero_mask], theta=theta_nc[nonzero_mask]
            ).log_prob(labels_nc[nonzero_mask])
            nonzero_sample_weights = sample_weights_nc[nonzero_mask]
            nonzero_loss = -(nonzero_log_prob * nonzero_sample_weights).sum() / nonzero_sample_weights.sum()

            zero_mask = labels_nc == 0
            zero_log_prob = NegativeBinomial(
                mu=mu_nc[zero_mask] * total_mrna_umis_nc[zero_mask], theta=theta_nc[zero_mask]
            ).log_prob(labels_nc[zero_mask])
            zero_sample_weights = sample_weights_nc[zero_mask]
            zero_loss = -(zero_log_prob * zero_sample_weights).sum() / zero_sample_weights.sum()

            prefix_len_nc = prefix_len_n[:, None].expand(-1, self.n_context)
        return {
            "loss": loss,
            "nonzero_loss": nonzero_loss,
            "zero_loss": zero_loss,
            "logits": logits,
            "nonzero_mask": nonzero_mask,
            "zero_mask": zero_mask,
            "labels": labels_nc,
            "prefix_len_nc": prefix_len_nc,
            "mu_nc": mu_nc,
            "theta_nc": theta_nc,
            "nonzero_log_prob": nonzero_log_prob,
            "zero_log_prob": zero_log_prob,
            "attention_weights": attention_weights,
        }
