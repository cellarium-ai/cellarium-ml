# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


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
        prefix_len: int | None,
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
        self.dense1 = nn.Linear(1, d_model, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, value_nc: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(value_nc.unsqueeze(-1))))  # _ncd


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
        input_mult:
            Multiplier for the input embeddings.
        attn_backend:
            Backend for the attention computation.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        use_bias: bool,
        attn_mult: float,
        input_mult: float,
        attn_backend: Literal["math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        # +1 token for total_mrna_umis
        self.Et = nn.Embedding(n_vars + 1, d_model)
        # continuous value embedding
        self.Ev = ValueEmbedding(d_model, use_bias=use_bias)
        # mask embedding for target values
        self.Em = nn.Embedding(1, d_model)

        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_ffn, n_heads, dropout_p, attn_mult, use_bias, attn_backend)
                for _ in range(n_blocks)
            ]
        )
        self.input_mult = input_mult

    def forward(
        self,
        token_id_nc: torch.Tensor,
        value_nc: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        device = token_id_nc.device

        token_embedding_ncd = self.Et(token_id_nc)
        value_embedding_ncd = self.Ev(torch.log1p(value_nc))
        mask_embedding_d = self.Em(torch.tensor(0, device=device))
        value_embedding_ncd[:, prefix_len:] = mask_embedding_d

        hidden_state_ncd = (token_embedding_ncd + value_embedding_ncd) * self.input_mult
        for block in self.blocks:
            hidden_state_ncd = block(hidden_state_ncd, prefix_len)

        return hidden_state_ncd


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
        dropout_p:
            Dropout probability.
        use_bias:
            Whether to use bias in the linear transformations.
        context_len:
            Length of the context.
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
        log_metrics:
            Whether to log the zero and nonzero losses and predictions.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        d_model: int = 256,
        d_ffn: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        dropout_p: float = 0.0,
        use_bias: bool = False,
        context_len: int = 2048,
        max_value: int = 1000,
        attn_mult: float = 6.0,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        initializer_range: float = 0.02,
        attn_backend: Literal["math", "flash", "mem_efficient"] = "mem_efficient",
        log_metrics: bool = True,
        # log_plots: bool = True,
        # log_plots_every_n_steps_multiplier: int = 10,
    ) -> None:
        super().__init__()
        self.var_names_g = np.array(var_names_g)
        self.n_vars = len(var_names_g)
        self.max_value = max_value
        if context_len > self.n_vars + 1:
            raise ValueError(
                "`context_len` must be less than or equal to the number of genes + 1. "
                f"Got {context_len} > {self.n_vars + 1}."
            )
        self.context_len = context_len
        self.output_mult = output_mult

        self.transformer = Transformer(
            self.n_vars,
            d_model,
            d_ffn,
            n_heads,
            n_blocks,
            dropout_p,
            use_bias,
            attn_mult,
            input_mult,
            attn_backend,
        )
        self.head = nn.Linear(d_model, self.max_value + 1, use_bias)
        self.initializer_range = initializer_range
        self.reset_parameters()

        self.log_metrics = log_metrics
        # self.log_plots = log_plots
        # self.log_plots_every_n_steps_multiplier = log_plots_every_n_steps_multiplier

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

    def tokenize(
        self, x_ng: torch.Tensor, total_mrna_umis_n: torch.Tensor, shuffle: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x_ng.shape[0]
        device = x_ng.device
        ndx = torch.arange(n, device=device)

        if shuffle:
            index_ng = torch.argsort(torch.rand_like(x_ng))
            index_nc = index_ng[:, : self.context_len - 1]
        else:
            index_nc = torch.arange(self.context_len, device=device).expand(n, -1)

        # add total_mrna_umis to the prefix
        token_id_nc = torch.cat([torch.zeros((n, 1), dtype=torch.long, device=device), index_nc + 1], dim=1)
        value_nc = torch.cat([total_mrna_umis_n[:, None], x_ng[ndx[:, None], index_nc]], dim=1)
        return token_id_nc, value_nc

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        total_mrna_umis_n: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        device = x_ng.device

        token_id_nc, value_nc = self.tokenize(x_ng, total_mrna_umis_n, shuffle=True)

        # prefix includes the total_mrna_umis and a random subset of genes
        # the length of the prefix is sampled uniformly from 1 to context_len - 1 (inclusive)
        prefix_len = random.randint(1, self.context_len - 1)
        hidden_state_ncd = self.transformer(token_id_nc, value_nc, prefix_len)
        logits_ncg = self.head(hidden_state_ncd) * self.output_mult

        label_mask_c = torch.arange(self.context_len, device=device) >= prefix_len
        label_mask_nc = label_mask_c & (value_nc < self.max_value + 1)
        sample_weight_nc = 1 / label_mask_nc.sum(dim=1, keepdim=True).expand(-1, self.context_len)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        sample_weights = sample_weight_nc[label_mask_nc]
        value_nc = value_nc.long()
        loss = (
            loss_fn(logits_ncg[label_mask_nc], value_nc[label_mask_nc]) * sample_weights
        ).sum() / sample_weights.sum()

        return {
            "loss": loss,
            "value_nc": value_nc,
            "total_mrna_umis_n": total_mrna_umis_n,
            "prefix_len": prefix_len,
            "sample_weight_nc": sample_weight_nc,
            "logits_ncg": logits_ncg,
            "label_mask_nc": label_mask_nc,
        }

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        x_ng: torch.Tensor,
        obs_names_n: np.ndarray,
        var_names_g: np.ndarray,
        total_mrna_umis_n: torch.Tensor,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        device = x_ng.device
        n = x_ng.shape[0]
        ndx = torch.arange(n, device=device)

        loss = self(x_ng, var_names_g, total_mrna_umis_n)["loss"]
        pl_module.log("val_loss", loss, sync_dist=True, on_epoch=True)

        losses = defaultdict(dict)
        suffix_len = 100
        n_seeds = 3
        prefix_lens = [1, 50, 500, 1000, 2000, 4000, 8000]
        for i in range(n_seeds):
            rng_n = [torch.Generator(device=device) for _ in range(x_ng.shape[0])]
            [rng.manual_seed(int(obs_name) + i) for rng, obs_name in zip(rng_n, obs_names_n)]
            index_ng = torch.stack([torch.randperm(x_ng.shape[1], generator=rng, device=device) for rng in rng_n])

            for prefix_len in prefix_lens:
                context_len = prefix_len + suffix_len
                token_id_nc = torch.cat(
                    [
                        torch.zeros((n, 1), dtype=torch.long, device=device),
                        index_ng[:, : prefix_len - 1] + 1,
                        index_ng[:, -suffix_len:] + 1,
                    ],
                    dim=1,
                )
                value_nc = torch.cat(
                    [
                        total_mrna_umis_n[:, None],
                        x_ng[ndx[:, None], index_ng[:, : prefix_len - 1]],
                        x_ng[ndx[:, None], index_ng[:, -suffix_len:]],
                    ],
                    dim=1,
                )

                logits = []
                for token_id, value in zip(torch.split(token_id_nc, 25), torch.split(value_nc, 25)):
                    hidden_state = self.transformer(token_id, value, prefix_len)
                    logits.append(self.head(hidden_state) * self.output_mult)
                logits_ncg = torch.cat(logits, dim=0)

                label_mask_c = torch.arange(context_len, device=device) >= prefix_len
                label_mask_nc = label_mask_c & (value_nc < self.max_value + 1)
                sample_weight_nc = 1 / label_mask_nc.sum(dim=1, keepdim=True).expand(-1, context_len)
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                sample_weights = sample_weight_nc[label_mask_nc]
                value_nc = value_nc.long()
                loss = (
                    loss_fn(logits_ncg[label_mask_nc], value_nc[label_mask_nc]) * sample_weights
                ).sum() / sample_weights.sum()
                losses[prefix_len][i] = loss

                if trainer.global_rank == batch_idx == i == 0:
                    self._log_plots(
                        trainer,
                        value_nc[:, prefix_len:],
                        logits_ncg[:, prefix_len:],
                        total_mrna_umis_n,
                        prefix_len,
                    )

        loss_dict = {}
        for prefix_len in prefix_lens:
            loss = torch.mean(torch.stack([losses[prefix_len][i] for i in range(n_seeds)], dim=0))
            loss_dict[f"val_loss_prefix_{prefix_len}"] = loss
        pl_module.log_dict(loss_dict, sync_dist=True, on_epoch=True)

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

        self._log_metrics(pl_module, outputs)

        # if not self.log_plots:
        #     return

        # if trainer.global_rank != 0:
        #     return

        # if (trainer.global_step + 1) % (trainer.log_every_n_steps * self.log_plots_every_n_steps_multiplier) != 0:  # type: ignore[attr-defined]
        #     return

        # self._log_plots(trainer, outputs)

    def _log_metrics(self, pl_module: pl.LightningModule, outputs: dict[str, torch.Tensor]) -> None:
        value_nc = outputs["value_nc"]
        sample_weight_nc = outputs["sample_weight_nc"]
        logits_ncg = outputs["logits_ncg"]
        label_mask_nc = outputs["label_mask_nc"]
        nonzero_mask = (value_nc > 0) & label_mask_nc
        zero_mask = (value_nc == 0) & label_mask_nc

        nonzero_log_prob = nn.functional.cross_entropy(
            logits_ncg[nonzero_mask], value_nc[nonzero_mask], reduction="none"
        )
        nonzero_sample_weights = sample_weight_nc[nonzero_mask]
        nonzero_loss = (nonzero_log_prob * nonzero_sample_weights).sum() / nonzero_sample_weights.sum()

        zero_log_prob = nn.functional.cross_entropy(logits_ncg[zero_mask], value_nc[zero_mask], reduction="none")
        zero_sample_weights = sample_weight_nc[zero_mask]
        zero_loss = (zero_log_prob * zero_sample_weights).sum() / zero_sample_weights.sum()

        pl_module.log("zero_loss", zero_loss, sync_dist=True)
        pl_module.log("nonzero_loss", nonzero_loss, sync_dist=True)

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
            plt.vlines(label, 0, y[label], color="r")
            plt.title(f"total UMIs={total_mrna_umis}, x={label}")

        plt.tight_layout()

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_figure(
                    f"val_pred_prefix_{prefix_len}",
                    fig,
                    global_step=trainer.global_step,
                )
