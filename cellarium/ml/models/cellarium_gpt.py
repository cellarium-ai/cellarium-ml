# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

backend_map = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
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
        dropout_p: float,
        attn_mult: float,
        backend: Literal["math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.attn_mult = attn_mult
        self.backend = backend

    def forward(
        self,
        queries_nhqk: torch.Tensor,
        keys_nhsk: torch.Tensor,
        values_nhsv: torch.Tensor,
        prefix_len: int | None,
        attention_type: Literal["block", "block_diagonal", "full"],
    ) -> torch.Tensor:
        q = queries_nhqk.shape[2]
        s = keys_nhsk.shape[2]
        k = keys_nhsk.shape[3]

        scale_factor = self.attn_mult / k
        device = queries_nhqk.device
        if attention_type == "block":
            assert prefix_len is not None
            range_s = torch.arange(s, device=device)
            attention_mask_qs = range_s.expand([q, s]) < prefix_len
        elif attention_type == "block_diagonal":
            assert prefix_len is not None
            range_s = torch.arange(s, device=device)
            range_q = torch.arange(q, device=device)
            block_mask_qs = range_s.expand([q, s]) < prefix_len
            diag_mask_qs = range_q[:, None] == range_s
            attention_mask_qs = block_mask_qs | diag_mask_qs
        elif attention_type == "full":
            attention_mask_qs = None
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        with sdpa_kernel(backend_map[self.backend]):
            return nn.functional.scaled_dot_product_attention(
                queries_nhqk,
                keys_nhsk,
                values_nhsv,
                attention_mask_qs,
                dropout_p=self.dropout_p,
                scale=scale_factor,
            )


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention.

    Args:
        d_model:
            Dimensionality of the embeddings and hidden states.
        n_heads:
            Number of attention heads.
        dropout_p:
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
        dropout_p: float,
        attn_mult: float,
        use_bias: bool,
        backend: Literal["math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention(dropout_p, attn_mult, backend=backend)
        self.Wq = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wk = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wv = nn.Linear(d_model, d_model, bias=use_bias)
        self.Wo = nn.Linear(d_model, d_model, bias=use_bias)

    @staticmethod
    def split_heads(X_nqd: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads."""
        # k = d / h
        X_nqhk = X_nqd.reshape(X_nqd.shape[0], X_nqd.shape[1], n_heads, -1)
        X_nhqk = X_nqhk.permute(0, 2, 1, 3)
        return X_nhqk

    @staticmethod
    def merge_heads(X_nhqk: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Reverse of split_heads."""
        X_nqhk = X_nhqk.permute(0, 2, 1, 3)
        return X_nqhk.reshape(X_nqhk.shape[0], X_nqhk.shape[1], -1)  # _nqd

    def forward(
        self,
        queries_nqd: torch.Tensor,
        keys_nsd: torch.Tensor,
        values_nsd: torch.Tensor,
        prefix_len: int | None,
        attention_type: Literal["block", "block_diag", "full"],
    ) -> torch.Tensor:
        n_heads = self.n_heads
        queries_nqd = self.Wq(queries_nqd)
        keys_nsd = self.Wk(keys_nsd)
        values_nsd = self.Wv(values_nsd)
        # m = n * h
        # k = d / h
        queries_nhqk = self.split_heads(queries_nqd, n_heads)
        keys_nhsk = self.split_heads(keys_nsd, n_heads)
        values_nhsk = self.split_heads(values_nsd, n_heads)

        output_nhqk = self.attention(queries_nhqk, keys_nhsk, values_nhsk, prefix_len, attention_type)
        output_nqd = self.merge_heads(output_nhqk, n_heads)
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
        dropout_p:
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
        dropout_p: float,
        attn_mult: float,
        use_bias: bool,
        backend: Literal["keops", "torch", "math", "flash", "mem_efficient"],
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout_p, attn_mult, use_bias, backend)
        self.normadd1 = NormAdd(d_model, dropout_p)
        self.ffn = PositionWiseFFN(d_ffn, d_model, use_bias)
        self.normadd2 = NormAdd(d_model, dropout_p)

    def forward(self, X: torch.Tensor, prefix_len: int | None, attention_type) -> torch.Tensor:
        Y = self.normadd1(X, lambda X: self.attention(X, X, X, prefix_len, attention_type))
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
        dropout_p:
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
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        use_bias: bool,
        attn_mult: float,
        input_mult: float,
        backend: Literal["math", "flash", "mem_efficient"],
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
            [
                TransformerBlock(d_model, d_ffn, n_heads, dropout_p, attn_mult, use_bias, backend)
                for _ in range(n_blocks)
            ]
        )

        self.input_mult = input_mult

    def forward(
        self,
        ids_nc: torch.Tensor,
        values_nc: torch.Tensor,
        prefix_len: int,
        attention_type: Literal["block", "block_diagonal", "full"],
    ) -> torch.Tensor:
        device = ids_nc.device
        n_context = ids_nc.shape[1]

        id_tokens_ncd = self.id_embedding(ids_nc)
        value_tokens_ncd = self.value_embedding(torch.log1p(values_nc.float()).unsqueeze(-1))
        mask_token = self.mask_embedding(torch.tensor(0, device=device))
        labels_mask_c = torch.arange(n_context, device=device) >= prefix_len
        value_tokens_ncd[:, labels_mask_c] = mask_token

        hidden_state_ncd = (id_tokens_ncd + value_tokens_ncd) * self.input_mult
        for block in self.blocks:
            hidden_state_ncd = block(
                hidden_state_ncd,
                prefix_len=prefix_len,
                attention_type=attention_type,
            )

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
        dropout_p: float = 0.0,
        use_bias: bool = False,
        n_context: int | None = 2048,
        attn_mult: float = 1.0,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        max_count: int = 10_000,
        initializer_range: float = 0.02,
        backend: Literal["math", "flash", "mem_efficient"] = "mem_efficient",
        log_metrics: bool = True,
        log_plots: bool = False,
        log_plots_every_n_steps_multiplier: int = 10,
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
            dropout_p,
            use_bias,
            attn_mult,
            input_mult,
            backend,
        )
        self.max_count = max_count
        self.head = nn.Linear(d_model, self.max_count + 1, use_bias)
        if n_context is None:
            n_context = self.n_vars + 1
        assert n_context <= self.n_vars + 1, "n_context must be less than or equal to the number of genes + 1"
        self.n_context = n_context
        self.initializer_range = initializer_range
        self.output_mult = output_mult
        self.reset_parameters()

        self.log_metrics = log_metrics
        self.log_plots = log_plots
        self.log_plots_every_n_steps_multiplier = log_plots_every_n_steps_multiplier

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

    @cached_property
    def token_to_id(self) -> dict[str, int]:
        return {var_name: i + 1 for i, var_name in enumerate(self.var_names_g)}

    @cached_property
    def vectorized_token_to_id(self):
        return np.vectorize(lambda x: self.token_to_id[x])

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

        device = x_ng.device

        # prefix includes the total_mrna_umis and a random subset of genes
        # the length of the prefix is sampled uniformly from 1 to n_context - 1 (inclusive)
        prefix_len = random.randint(1, self.n_context - 1)
        labels_mask_c = torch.arange(self.n_context, device=device) >= prefix_len

        ids_nc, values_nc = self.tokenize(x_ng, total_mrna_umis_n, self.n_context, shuffle=True)
        hidden_state_ncd = self.gpt_model(ids_nc, values_nc, prefix_len, "block_diagonal")
        logits_ncg = self.head(hidden_state_ncd) * self.output_mult

        labels_mask_nc = labels_mask_c & (values_nc < self.max_count + 1)
        sample_weights_nc = 1 / labels_mask_nc.sum(dim=1, keepdim=True).expand(-1, self.n_context)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        sample_weights = sample_weights_nc[labels_mask_nc]
        values_nc = values_nc.long()
        loss = (
            loss_fn(logits_ncg[labels_mask_nc], values_nc[labels_mask_nc]) * sample_weights
        ).sum() / sample_weights.sum()

        return {
            "loss": loss,
            "values_nc": values_nc,
            "total_mrna_umis_n": total_mrna_umis_n,
            "prefix_len": prefix_len,
            "labels_mask_nc": labels_mask_nc,
            "sample_weights_nc": sample_weights_nc,
            "logits_ncg": logits_ncg,
            "ids_nc": ids_nc,
        }

    @torch.inference_mode()
    def predict(
        self,
        prompt_name_ns: np.ndarray | None,
        prompt_value_ns: torch.Tensor | None,
        query_name_nq: np.ndarray,
        total_mrna_umis_n: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = total_mrna_umis_n.device
        n = len(total_mrna_umis_n)

        values_nc = total_mrna_umis_n[:, None]
        ids_nc = torch.zeros((n, 1), dtype=torch.long, device=device)

        if prompt_name_ns is not None:
            prompt_id_ns = torch.tensor(self.vectorized_token_to_id(prompt_name_ns), dtype=torch.long, device=device)
            values_nc = torch.cat([values_nc, prompt_value_ns], dim=1)
            ids_nc = torch.cat([ids_nc, prompt_id_ns], dim=1)

        prefix_len = values_nc.shape[1]

        query_id_nq = torch.tensor(self.vectorized_token_to_id(query_name_nq), dtype=torch.long, device=device)
        values_nc = torch.cat([values_nc, torch.zeros_like(query_id_nq)], dim=1)
        ids_nc = torch.cat([ids_nc, query_id_nq], dim=1)

        hidden_state_ncd = self.gpt_model(ids_nc, values_nc, prefix_len, "block_diagonal")
        logits_ncp = self.head(hidden_state_ncd) * self.output_mult

        return {
            "logits_nqp": logits_ncp[:, prefix_len:],
            "query_name_nq": query_name_nq,
        }

    @torch.inference_mode()
    def generate(
        self, var_names_ng: np.ndarray, total_mrna_umis_n: torch.Tensor, prompt_value_ns: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        if prompt_value_ns is None:
            prompt_len = 0
        else:
            prompt_len = prompt_value_ns.shape[1]
        total_len = var_names_ng.shape[1]

        for i in tqdm.tqdm(range(prompt_len, total_len)):
            if prompt_value_ns is None:
                prompt_name_ns = None
            else:
                prompt_name_ns = var_names_ng[:, :i]

            logits_n1p = self.predict(prompt_name_ns, prompt_value_ns, var_names_ng[:, i : i + 1], total_mrna_umis_n)[
                "logits_nqp"
            ]
            probs_np = logits_n1p[:, 0].softmax(dim=-1)
            value_n1 = torch.multinomial(probs_np, 1)
            if prompt_value_ns is None:
                prompt_value_ns = value_n1
            else:
                prompt_value_ns = torch.cat([prompt_value_ns, value_n1], dim=1)
        return prompt_value_ns

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

        if not self.log_plots:
            return

        if trainer.global_rank != 0:
            return

        if (trainer.global_step + 1) % (trainer.log_every_n_steps * self.log_plots_every_n_steps_multiplier) != 0:  # type: ignore[attr-defined]
            return

        self._log_plots(trainer, outputs)

    def _log_metrics(self, pl_module: pl.LightningModule, outputs: dict[str, torch.Tensor]) -> None:
        values_nc = outputs["values_nc"]
        labels_mask_nc = outputs["labels_mask_nc"]
        sample_weights_nc = outputs["sample_weights_nc"]
        logits_ncg = outputs["logits_ncg"]
        nonzero_mask = (values_nc > 0) & labels_mask_nc
        zero_mask = (values_nc == 0) & labels_mask_nc

        nonzero_log_prob = nn.functional.cross_entropy(
            logits_ncg[nonzero_mask], values_nc[nonzero_mask], reduction="none"
        )
        nonzero_sample_weights = sample_weights_nc[nonzero_mask]
        nonzero_loss = (nonzero_log_prob * nonzero_sample_weights).sum() / nonzero_sample_weights.sum()

        zero_log_prob = nn.functional.cross_entropy(logits_ncg[zero_mask], values_nc[zero_mask], reduction="none")
        zero_sample_weights = sample_weights_nc[zero_mask]
        zero_loss = (zero_log_prob * zero_sample_weights).sum() / zero_sample_weights.sum()

        pl_module.log("zero_loss", zero_loss, sync_dist=True)
        pl_module.log("nonzero_loss", nonzero_loss, sync_dist=True)

    def _log_plots(self, trainer: pl.Trainer, outputs: dict[str, torch.Tensor]):
        import matplotlib.pyplot as plt

        values_nc = outputs["values_nc"]
        total_mrna_umis_n = outputs["total_mrna_umis_n"]
        prefix_len = outputs["prefix_len"]
        labels_mask_nc = outputs["labels_mask_nc"]
        logits_ncg = outputs["logits_ncg"]
        nonzero_mask = (values_nc > 0) & labels_mask_nc
        zero_mask = (values_nc == 0) & labels_mask_nc

        prefix_len_nc = torch.full(values_nc.shape, prefix_len, device=values_nc.device)
        total_mrna_umis_nc = total_mrna_umis_n[:, None].expand(-1, self.n_context)

        fig = plt.figure(figsize=(12, 15))
        nonzero_indices = random.sample(range(int(nonzero_mask.sum().item())), 24)

        for idx, i in enumerate(nonzero_indices):
            plt.subplot(8, 4, idx + 1)
            prefix = prefix_len_nc[nonzero_mask][i].item()
            label = values_nc[nonzero_mask][i].item()
            logits = logits_ncg[nonzero_mask][i].cpu()
            n_umis = total_mrna_umis_nc[nonzero_mask][i].item()

            x = torch.arange(0, min(label * 3 + 5, self.max_count + 1))
            y = logits.softmax(dim=-1)[: len(x)]
            plt.plot(x, y, "o")
            plt.vlines(label, 0, y[label], color="r")
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
            logits = logits_ncg[zero_mask][i].cpu()
            n_umis = total_mrna_umis_nc[zero_mask][i].item()

            x = torch.arange(0, min(label * 3 + 5, self.max_count + 1))
            y = logits.softmax(dim=-1)[: len(x)]
            plt.plot(x, y, "o")
            plt.vlines(label, 0, y[label], color="r")
            plt.title(f"umis={n_umis}, prfx={prefix}, lbl={label}")

        plt.tight_layout()

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_figure(
                    "zero predictions",
                    zero_fig,
                    global_step=trainer.global_step,
                )
