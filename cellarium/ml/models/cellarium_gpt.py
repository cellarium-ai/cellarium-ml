# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
from collections.abc import Callable, Sequence
from numbers import Number
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributions as dist
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


class NegativeBinomial(dist.Distribution):
    """Negative binomial distribution.

    Args:
        mu:
            Mean of the distribution.
        theta:
            Inverse dispersion.
    """

    arg_constraints = {"mu": constraints.greater_than_eq(0), "theta": constraints.greater_than_eq(0)}
    support = constraints.nonnegative_integer

    def __init__(self, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8, validate_args: bool = False) -> None:
        self.mu, self.theta = broadcast_all(mu, theta)
        if isinstance(mu, Number) and isinstance(theta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        self.eps = eps
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.mu + (self.mu**2) / self.theta

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        # log_theta_mu_eps = torch.log(self.theta + self.mu + self.eps)
        # return (
        #     self.theta * (torch.log(self.theta + self.eps) - log_theta_mu_eps)
        #     + value * (torch.log(self.mu + self.eps) - log_theta_mu_eps)
        #     + torch.lgamma(value + self.theta)
        #     - torch.lgamma(self.theta)
        #     - torch.lgamma(value + 1)
        # )
        delta = torch.where(
            (value / self.theta < 1e-2) & (self.theta > 1e2),
            (value + self.theta - 0.5) * torch.log1p(value / self.theta) - value,
            (value + self.theta).lgamma() - self.theta.lgamma() - torch.xlogy(value, self.theta),
        )
        return (
            delta
            - (value + self.theta) * torch.log1p(self.mu / self.theta)
            - (value + 1).lgamma()
            + torch.xlogy(value, self.mu)
        )


backend_map = {
    "math": {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    "flash": {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    "mem_efficient": {"enable_math": False, "enable_flash": False, "enable_mem_efficient": True},
}


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot product attention.

    Args:
        dropout:
            Dropout probability.
        attn_mult:
            Multiplier for the attention scores.
        backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        dropout: float,
        attn_mult: float,
        backend: Literal["keops", "torch", "math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn_mult = attn_mult
        self.backend = backend

    def forward(
        self,
        queries_nqd: torch.Tensor,
        keys_nsd: torch.Tensor,
        values_nsv: torch.Tensor,
        prefix_len_n: torch.Tensor | None,
        attention_type: Literal["block", "block_diagonal", "full"],
    ) -> torch.Tensor:
        n, q, d = queries_nqd.shape
        s = keys_nsd.shape[1]
        scale_factor = self.attn_mult / d
        device = queries_nqd.device
        if self.backend == "keops":
            if self.dropout.p > 0:
                raise NotImplementedError("Dropout is not supported with PyKeOps.")

            if attention_type == "block":
                assert prefix_len_n is not None
                prefix_len_n = LazyTensor(prefix_len_n[:, None, None, None].expand([n, q, 1, 1]).float())
                s_range = LazyTensor(torch.arange(s, device=device)[:, None].float(), axis=1)
                block_mask_nqs = 1 - (s_range < prefix_len_n)
                attention_mask_nqs = block_mask_nqs
            elif attention_type == "block_diagonal":
                assert prefix_len_n is not None
                prefix_len_n = LazyTensor(prefix_len_n[:, None, None, None].expand([n, q, 1, 1]).float())
                q_range = LazyTensor(torch.arange(q, device=device)[:, None].float(), axis=0)
                s_range = LazyTensor(torch.arange(s, device=device)[:, None].float(), axis=1)
                block_mask_nqs = 1 - (s_range < prefix_len_n)
                diag_mask_qs = 1 - (q_range == s_range)
                attention_mask_nqs = block_mask_nqs * diag_mask_qs
            elif attention_type == "full":
                attention_mask_nqs = 0
            else:
                raise ValueError(f"Unsupported attention type: {attention_type}")

            scores_nqs = (
                LazyTensor(queries_nqd[:, :, None, :].contiguous()) * LazyTensor(keys_nsd[:, None, :, :].contiguous())
            ).sum(-1) * scale_factor
            scores_nqs = scores_nqs - 1e9 * attention_mask_nqs
            return scores_nqs.sumsoftmaxweight(LazyTensor(values_nsv.unsqueeze(1).contiguous()), dim=2)
        else:
            if attention_type == "block":
                assert prefix_len_n is not None
                block_mask_nqs = torch.arange(s, device=device).expand([q, s]) < prefix_len_n[:, None, None]
                attention_mask_nqs = block_mask_nqs
            elif attention_type == "block_diagonal":
                assert prefix_len_n is not None
                block_mask_nqs = torch.arange(s, device=device).expand([q, s]) < prefix_len_n[:, None, None]
                diag_mask_qs = torch.arange(q, device=device)[:, None] == torch.arange(s, device=device)
                attention_mask_nqs = block_mask_nqs | diag_mask_qs
            elif attention_type == "full":
                attention_mask_nqs = None
            else:
                raise ValueError(f"Unsupported attention type: {attention_type}")

            if self.backend in ["math", "flash", "mem_efficient"]:
                with sdp_kernel(**backend_map[self.backend]):
                    return nn.functional.scaled_dot_product_attention(
                        queries_nqd.unsqueeze(0),
                        keys_nsd.unsqueeze(0),
                        values_nsv.unsqueeze(0),
                        attention_mask_nqs,
                        dropout_p=self.dropout.p,
                        scale=scale_factor,
                    ).squeeze(0)
            elif self.backend == "torch":
                scores_nqs = queries_nqd @ keys_nsd.transpose(1, 2) * scale_factor
                if attention_mask_nqs is not None:
                    scores_nqs[~attention_mask_nqs] = float("-inf")
                self.attention_probs_nqs = scores_nqs.softmax(dim=-1)
                return self.dropout(self.attention_probs_nqs) @ values_nsv  # _nqv
            else:
                raise ValueError(f"Unknown backend: {self.backend}")


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        n_heads:
            Number of attention heads.
        dropout:
            Dropout probability.
        attn_mult:
            Multiplier for the attention scores.
        use_bias:
            Whether to use bias in the linear transformations.
        backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        attn_mult: float,
        use_bias: bool,
        backend: Literal["keops", "torch", "math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention(dropout, attn_mult, backend=backend)
        self.Wq = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wk = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wv = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wo = nn.Linear(d_model, d_model, bias=use_bias)

    @staticmethod
    def split_heads(X_nqd: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads."""
        # m = n * h
        # k = d / h
        X_nqhk = X_nqd.reshape(X_nqd.shape[0], X_nqd.shape[1], n_heads, -1)
        X_nhqk = X_nqhk.permute(0, 2, 1, 3)
        return X_nhqk.reshape(-1, X_nhqk.shape[2], X_nhqk.shape[3])  # _mqk

    @staticmethod
    def merge_heads(X_mqk: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Reverse of split_heads."""
        X_nhqk = X_mqk.reshape(-1, n_heads, X_mqk.shape[1], X_mqk.shape[2])
        X_nqhk = X_nhqk.permute(0, 2, 1, 3)
        return X_nqhk.reshape(X_nqhk.shape[0], X_nqhk.shape[1], -1)  # _nqd

    def forward(
        self,
        queries_nqd: torch.Tensor,
        keys_nsd: torch.Tensor,
        values_nsd: torch.Tensor,
        prefix_len_n: torch.Tensor | None,
        attention_type: Literal["block", "block_diag", "full"],
    ) -> torch.Tensor:
        n_heads = self.n_heads
        queries_nqd = self.Wq(queries_nqd)
        keys_nsd = self.Wk(keys_nsd)
        values_nsd = self.Wv(values_nsd)
        # m = n * h
        # k = d / h
        queries_mqk = self.split_heads(queries_nqd, n_heads)
        keys_msk = self.split_heads(keys_nsd, n_heads)
        values_msk = self.split_heads(values_nsd, n_heads)
        if prefix_len_n is not None:
            prefix_len_m = prefix_len_n.repeat_interleave(n_heads)
        else:
            prefix_len_m = None

        output_mqk = self.attention(queries_mqk, keys_msk, values_msk, prefix_len_m, attention_type)
        output_nqd = self.merge_heads(output_mqk, n_heads)
        return self.Wo(output_nqd)  # _nqd


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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(X)))


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
        self.dense1 = nn.Linear(1, d_model, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(X)))


class NormAdd(nn.Module):
    """
    Pre-norm layer where the layer normalization is applied before the sublayer.

    Args:
        norm_shape:
            The shape of the layer normalization.
        dropout:
            Dropout probability.
    """

    def __init__(self, norm_shape: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return X + self.dropout(sublayer(self.ln(X)))


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
        dropout:
            Dropout probability.
        attn_mult:
            Multiplier for the attention scores.
        use_bias:
            Whether to use bias in the linear transformations.
        backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        dropout: float,
        attn_mult: float,
        use_bias: bool,
        backend: Literal["keops", "torch", "math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, attn_mult, use_bias, backend)
        self.normadd1 = NormAdd(d_model, dropout)
        self.ffn = PositionWiseFFN(d_ffn, d_model, use_bias)
        self.normadd2 = NormAdd(d_model, dropout)

    def forward(self, X: torch.Tensor, prefix_len_n: torch.Tensor | None, attention_type) -> torch.Tensor:
        Y = self.normadd1(X, lambda X: self.attention(X, X, X, prefix_len_n, attention_type))
        return self.normadd2(Y, lambda Y: self.ffn(Y))


class GPTModel(nn.Module):
    """
    GPT model.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        n_heads:
            Number of attention heads.
        n_blocks:
            Number of transformer blocks.
        dropout:
            Dropout probability.
        use_bias:
            Whether to use bias in the linear transformations.
        attn_mult:
            Multiplier for the attention scores.
        input_mult:
            Multiplier for the input embeddings.
        backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int = 128,
        d_ffn: int = 256,
        n_heads: int = 4,
        n_blocks: int = 6,
        dropout: float = 0.02,
        use_bias: bool = False,
        attn_mult: float = 1.0,
        input_mult: float = 1.0,
        backend: Literal["keops", "torch", "math", "flash", "mem_efficient"] = "keops",
    ) -> None:
        super().__init__()
        # +1 token for total_mrna_umis
        self.id_embedding = nn.Embedding(n_vars + 1, d_model)
        # continuous value embedding
        self.value_embedding = ValueEmbedding(d_model, use_bias=use_bias)
        # mask token for target values
        self.mask_embedding = nn.Embedding(1, d_model)

        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, d_ffn, n_heads, dropout, attn_mult, use_bias, backend) for _ in range(n_blocks)]
        )

        self.input_mult = input_mult

    def forward(
        self,
        ids_nc: torch.Tensor,
        values_nc: torch.Tensor,
        labels_mask_nc: torch.Tensor,
        prefix_len_n: torch.Tensor,
        attention_type: Literal["block", "block_diagonal", "full"],
    ) -> torch.Tensor:
        device = ids_nc.device

        id_tokens_ncd = self.id_embedding(ids_nc)
        value_tokens_ncd = self.value_embedding(torch.log1p(values_nc.float()).unsqueeze(-1))
        mask_token = self.mask_embedding(torch.tensor(0, device=device))
        value_tokens_ncd[labels_mask_nc] = mask_token

        hidden_state_ncd = (id_tokens_ncd + value_tokens_ncd) * self.input_mult
        for block in self.blocks:
            hidden_state_ncd = block(
                hidden_state_ncd,
                prefix_len_n=prefix_len_n,
                attention_type=attention_type,
            )

        return hidden_state_ncd


class NegativeBinomialHead(nn.Module):
    """
    Negative binomial head.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        use_bias:
            Whether to use bias in the linear transformations.
        output_mult:
            Multiplier for the output embeddings.
    """

    def __init__(self, d_model: int, use_bias: bool, output_mult: float) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, 2, bias=use_bias)
        self.output_mult = output_mult

    def forward(self, hidden_state_ncd) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.dense(hidden_state_ncd) * self.output_mult
        mu_nc = output[..., 0].exp()
        theta_nc = output[..., 1].exp()
        return mu_nc, theta_nc


class CellariumGPT(CellariumModel):
    """
    Cellarium GPT model.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        d_model:
            Dimensionality of the embeddings and hidden states.
        d_ffn:
            Dimensionality of the inner feed-forward layers.
        n_heads:
            Number of attention heads.
        n_blocks:
            Number of transformer blocks.
        dropout:
            Dropout probability.
        use_bias:
            Whether to use bias in the linear transformations.
        n_context:
            Number of context variables.
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
        log_metrics:
            Whether to log the zero and nonzero losses and predictions.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        d_model: int = 128,
        d_ffn: int = 256,
        n_heads: int = 4,
        n_blocks: int = 6,
        dropout: float = 0.0,
        use_bias: bool = False,
        n_context: int = 2048,
        attn_mult: float = 1.0,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        initializer_range: float = 0.02,
        backend: Literal["keops", "torch", "math", "flash", "mem_efficient"] = "keops",
        log_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.n_vars = len(var_names_g)
        self.gpt_model = GPTModel(
            self.n_vars,
            d_model,
            d_ffn,
            n_heads,
            n_blocks,
            dropout,
            use_bias,
            attn_mult,
            input_mult,
            backend,
        )
        self.nb_head = NegativeBinomialHead(d_model, use_bias, output_mult)
        assert n_context <= self.n_vars + 1, "n_context must be less than or equal to the number of genes + 1"
        self.n_context = n_context
        self.initializer_range = initializer_range
        self.reset_parameters()

        self.log_metrics = log_metrics

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
                p.data.normal_(mean=0.0, std=(self.initializer_range / math.sqrt(2 * self.gpt_model.n_blocks)))

    def tokenize(
        self, x_ng: torch.Tensor, total_mrna_umis_n: torch.Tensor | None, context_size: int, shuffle: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x_ng.shape[0]
        device = x_ng.device
        ndx = torch.arange(n, device=device)

        genes_context_size = context_size if total_mrna_umis_n is None else context_size - 1
        if shuffle:
            indices_ng = torch.argsort(torch.rand_like(x_ng))
            indices_nc = indices_ng[:, :genes_context_size]
        else:
            indices_nc = torch.arange(genes_context_size, device=device).expand(n, -1)
        values_nc = x_ng[ndx[:, None], indices_nc]
        if total_mrna_umis_n is not None:
            # concatenate total_mrna_umis to the random subset of genes
            values_nc = torch.cat([total_mrna_umis_n[:, None], values_nc], dim=1)
            ids_nc = torch.cat([torch.zeros((n, 1), dtype=torch.long, device=device), indices_nc + 1], dim=1)
        else:
            ids_nc = indices_nc + 1
        return ids_nc, values_nc

    def forward(
        self, x_ng: torch.Tensor, var_names_g: np.ndarray, total_mrna_umis_n: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        n = x_ng.shape[0]
        device = x_ng.device

        # prefix includes the total_mrna_umis and a random subset of genes
        # the length of the prefix is sampled uniformly from 1 to n_context - 1 (inclusive)
        prefix_len_n = torch.randint(1, self.n_context, (n,), device=device)
        labels_mask_nc = torch.arange(self.n_context, device=device)[None, :] >= prefix_len_n[:, None]

        ids_nc, values_nc = self.tokenize(x_ng, total_mrna_umis_n, self.n_context, shuffle=True)
        hidden_state_ncd = self.gpt_model(ids_nc, values_nc, labels_mask_nc, prefix_len_n, "block_diagonal")
        mu_nc, theta_nc = self.nb_head(hidden_state_ncd)

        sample_weights_nc = 1 / labels_mask_nc.sum(dim=1, keepdim=True).expand(-1, self.n_context)
        nb_dist = NegativeBinomial(mu=mu_nc[labels_mask_nc], theta=theta_nc[labels_mask_nc])
        log_prob = nb_dist.log_prob(values_nc[labels_mask_nc])
        sample_weights = sample_weights_nc[labels_mask_nc]
        loss = -(log_prob * sample_weights).sum() / sample_weights.sum()

        return {
            "loss": loss,
            "values_nc": values_nc,
            "total_mrna_umis_n": total_mrna_umis_n,
            "prefix_len_n": prefix_len_n,
            "labels_mask_nc": labels_mask_nc,
            "sample_weights_nc": sample_weights_nc,
            "mu_nc": mu_nc,
            "theta_nc": theta_nc,
        }

    @torch.no_grad()
    def on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
    ) -> None:

        if not self.log_metrics:
            return

        if (trainer.global_step + 1) % trainer.log_every_n_steps != 0:  # type: ignore[attr-defined]
            return

        values_nc = outputs["values_nc"]
        total_mrna_umis_n = outputs["total_mrna_umis_n"]
        prefix_len_n = outputs["prefix_len_n"]
        labels_mask_nc = outputs["labels_mask_nc"]
        sample_weights_nc = outputs["sample_weights_nc"]
        mu_nc = outputs["mu_nc"]
        theta_nc = outputs["theta_nc"]

        nonzero_mask = (values_nc > 0) & labels_mask_nc
        nonzero_log_prob = NegativeBinomial(mu=mu_nc[nonzero_mask], theta=theta_nc[nonzero_mask]).log_prob(
            values_nc[nonzero_mask]
        )
        nonzero_sample_weights = sample_weights_nc[nonzero_mask]
        nonzero_loss = -(nonzero_log_prob * nonzero_sample_weights).sum() / nonzero_sample_weights.sum()

        zero_mask = (values_nc == 0) & labels_mask_nc
        zero_log_prob = NegativeBinomial(mu=mu_nc[zero_mask], theta=theta_nc[zero_mask]).log_prob(values_nc[zero_mask])
        zero_sample_weights = sample_weights_nc[zero_mask]
        zero_loss = -(zero_log_prob * zero_sample_weights).sum() / zero_sample_weights.sum()

        pl_module.log("zero_loss", zero_loss, sync_dist=True)
        pl_module.log("nonzero_loss", nonzero_loss, sync_dist=True)

        if trainer.global_rank != 0:
            return

        if (trainer.global_step + 1) % (trainer.log_every_n_steps * 10) != 0:  # type: ignore[attr-defined]
            return

        import matplotlib.pyplot as plt

        prefix_len_nc = prefix_len_n[:, None].expand(-1, self.n_context)
        total_mrna_umis_nc = total_mrna_umis_n[:, None].expand(-1, self.n_context)

        fig = plt.figure(figsize=(12, 15))
        nonzero_indices = random.sample(range(int(nonzero_mask.sum().item())), 24)

        for idx, i in enumerate(nonzero_indices):
            plt.subplot(8, 4, idx + 1)
            prefix = prefix_len_nc[nonzero_mask][i].item()
            label = values_nc[nonzero_mask][i].item()
            mu = mu_nc[nonzero_mask][i].cpu()
            theta = theta_nc[nonzero_mask][i].cpu()
            n_umis = total_mrna_umis_nc[nonzero_mask][i].item()

            nb_dist = NegativeBinomial(mu=mu, theta=theta)
            x = torch.arange(0, label * 3 + 5)
            y = nb_dist.log_prob(x).exp()
            plt.plot(x, y, "o")
            plt.vlines(label, 0, nb_dist.log_prob(torch.tensor(label)).exp(), color="r")
            plt.title(f"umis={n_umis}, prfx={prefix}, lbl={label}")

        plt.tight_layout()

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_figure(
                    "nonzero predictions",
                    fig,
                    global_step=trainer.global_step,
                )

        zero_fig = plt.figure(figsize=(12, 15))
        zero_indices = random.sample(range(int(zero_mask.sum().item())), 24)

        for idx, i in enumerate(zero_indices):
            plt.subplot(8, 4, idx + 1)
            prefix = prefix_len_nc[zero_mask][i].item()
            label = values_nc[zero_mask][i].item()
            mu = mu_nc[zero_mask][i].cpu()
            theta = theta_nc[zero_mask][i].cpu()
            n_umis = total_mrna_umis_nc[zero_mask][i].item()

            nb_dist = NegativeBinomial(mu=mu, theta=theta)
            x = torch.arange(0, label * 3 + 5)
            y = nb_dist.log_prob(x).exp()
            plt.plot(x, y, "o")
            plt.vlines(label, 0, nb_dist.log_prob(torch.tensor(label)).exp(), color="r")
            plt.title(f"umis={n_umis}, prfx={prefix}, lbl={label}")

        plt.tight_layout()

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_figure(
                    "zero predictions",
                    zero_fig,
                    global_step=trainer.global_step,
                )
