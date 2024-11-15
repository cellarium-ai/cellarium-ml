# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from functools import cached_property
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from cellarium.ml.layers import GeneExpressionEmbedding, MetadataEmbedding, Transformer
from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.layers import create_initializer
from cellarium.ml.utilities.mup import scale_initializers_by_dimension

try:
    from cerebras.modelzoo.common.utils.model.mup_utils import LRAdjustmentGroup
    from cerebras.pytorch.backend import use_cs
except ImportError:
    from cellarium.ml.utilities.mup import LRAdjustmentGroup

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
        attention_mask_ncc = diag_mask_ncc * prompt_mask_n1c
        return attention_mask_ncc == 0
    else:
        diag_mask_cc = torch.eye(c, dtype=torch.bool, device=device)
        attention_mask_ncc = prompt_mask_nc[:, None, :] | diag_mask_cc
        return attention_mask_ncc


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


class TrainTokenizer(torch.nn.Module):
    """
    Tokenizer for the Cellarium GPT model.

    Args:
        context_len:
            Context length.
        gene_downsample_fraction:
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
        context_len: int,
        gene_downsample_fraction: float,
        min_total_mrna_umis: int,
        max_total_mrna_umis: int,
        gene_vocab_sizes: dict[str, int],
        metadata_vocab_sizes: dict[str, int],
        ontology_downsample_p: float,
        ontology_infos_path: str,
        prefix_len: int | None = None,
        metadata_prompt_tokens: list[str] | None = None,
        obs_names_rng: bool = False,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.gene_downsample_fraction = gene_downsample_fraction
        self.min_total_mrna_umis = min_total_mrna_umis
        self.max_total_mrna_umis = max_total_mrna_umis
        self.gene_vocab_sizes = gene_vocab_sizes
        self.metadata_vocab_sizes = metadata_vocab_sizes
        ontology_infos = torch.load(ontology_infos_path)
        self.ontology_infos = ontology_infos
        self.ontology_downsample_p = ontology_downsample_p
        self.prefix_len = prefix_len
        self.metadata_prompt_tokens = metadata_prompt_tokens
        self.obs_names_rng = obs_names_rng

    def forward(
        self,
        metadata_tokens_n: dict[str, torch.Tensor],
        gene_tokens_n: dict[str, torch.Tensor],
        gene_tokens_ng: dict[str, torch.Tensor],
        gene_id_g: torch.Tensor | None = None,
        obs_names_n: np.ndarray | None = None,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        ### GENE TOKENS ###
        n, g = gene_tokens_ng["gene_value"].shape
        m = len(metadata_tokens_n)
        gene_context_len = self.context_len - m
        device = gene_tokens_ng["gene_value"].device

        ## gene measurement tokens (assay, suspension type, etc.) ##
        gene_tokens_nc = {key: gene_tokens_n[key][:, None].expand(-1, gene_context_len).int() for key in gene_tokens_n}

        ## gene id ##
        if gene_id_g is None:
            gene_id_g = torch.arange(g, device=device)
        gene_tokens_ng["gene_id"] = gene_id_g.expand(n, g)

        if self.obs_names_rng:
            rng_n = [torch.Generator(device=device) for _ in range(n)]
            [rng.manual_seed(int(obs_name)) for rng, obs_name in zip(rng_n, obs_names_n)]
            shuffle_idx_ng = torch.stack([torch.randperm(g, generator=rng, device=device) for rng in rng_n])
        else:
            shuffle_idx_ng = torch.argsort(torch.rand((n, g), dtype=torch.float32, device=device), dim=-1)
        shuffle_idx_nc = shuffle_idx_ng[:, :gene_context_len]

        for key, gene_token_ng in gene_tokens_ng.items():
            gene_tokens_nc[key] = torch.gather(gene_token_ng, dim=-1, index=shuffle_idx_nc)

        ## gene value ##
        gene_value_nc = gene_tokens_nc.pop("gene_value")
        total_mrna_umis_nc = gene_tokens_nc.pop("total_mrna_umis")
        # downsample gene values
        max_total_mrna_umis = torch.tensor(self.max_total_mrna_umis, device=device)
        downsampled_total_mrna_umis_nc = torch.minimum(total_mrna_umis_nc, max_total_mrna_umis).float()
        if self.gene_downsample_fraction > 0:
            gene_downsample_p_nc = torch.minimum(
                torch.rand((n, gene_context_len), device=device) / self.gene_downsample_fraction,
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
        if self.prefix_len is not None:
            prefix_len_n = torch.full((n,), self.prefix_len, dtype=torch.float32)
        else:
            # sample prefix length
            # prefix_len_weights = [1, max_prefix_len / 2, max_prefix_len / 3, ..., max_prefix_len / max_prefix_len]
            max_prefix_len = gene_context_len - 1
            prefix_len_weights = 1 / torch.arange(max_prefix_len + 1, dtype=torch.float32)
            prefix_len_weights[0] = 1 / 10
            prefix_len_n = torch.multinomial(prefix_len_weights, n, replacement=True)
        # create prompt and query masks
        gene_query_mask_nc = torch.arange(gene_context_len, device=device) >= prefix_len_n[:, None].expand(n, -1)
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
            assert self.metadata_vocab_sizes[key] == len(ontology_info["names"])
            metadata_tokens_n[key] = torch.tensor(
                pd.Categorical(metadata_tokens_n[key], categories=ontology_info["names"]).codes,
                dtype=torch.int,
            )
        # create metadata query and prompt masks
        if self.metadata_prompt_tokens is not None:
            metadata_prompt_mask_nm = torch.zeros((n, m), dtype=torch.bool, device=device)
            for metadata_token_idx, metadata_token in enumerate(metadata_tokens_n):
                if metadata_token in self.metadata_prompt_tokens:
                    metadata_prompt_mask_nm[:, metadata_token_idx] = True
        else:
            metadata_prefix_len_n = torch.randint(0, m + 1, (n,), device=device)
            metadata_prefix_mask_nm = torch.arange(m, device=device) < metadata_prefix_len_n[:, None]
            shuffle_idx_nm = torch.argsort(torch.rand_like(metadata_prefix_mask_nm, dtype=torch.float32), dim=-1)
            metadata_prompt_mask_nm = torch.gather(metadata_prefix_mask_nm, dim=-1, index=shuffle_idx_nm)
        metadata_query_mask_nm = ~metadata_prompt_mask_nm
        metadata_measured_mask_nm = torch.stack(
            [metadata_token_n >= 0 for metadata_token_n in metadata_tokens_n.values()], dim=1
        ).bool()
        metadata_query_mask_nm = metadata_query_mask_nm & metadata_measured_mask_nm
        metadata_prompt_mask_nm = metadata_prompt_mask_nm & metadata_measured_mask_nm
        # clamp unmeasured tokens to 0
        for key, metadata_token_n in metadata_tokens_n.items():
            metadata_tokens_n[key] = metadata_token_n.clamp(0).int()
        # metadata labels
        metadata_labels_n = {key: metadata_tokens_n[key].clone() for key in metadata_tokens_n}
        if self.ontology_downsample_p != 0:
            # downsample metadata based on ontology
            for key, ontology_info in self.ontology_infos.items():
                if "shortest_distances_matrix" not in ontology_info:
                    continue
                metadata_token_n = metadata_tokens_n[key]
                shortest_distances_matrix = ontology_info["shortest_distances_matrix"]
                ontology_weights = (
                    self.ontology_downsample_p * (1 - self.ontology_downsample_p) ** shortest_distances_matrix
                )
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
            key: block_label_nc[n * i : n * (i + 1)] for i, key in enumerate(["gene_value"] + list(metadata_tokens_n))
        }

        ### LABEL WEIGHTS ###
        block_label_weight_nc = (
            torch.block_diag(
                gene_query_mask_nc / torch.maximum(gene_query_mask_nc.sum(dim=-1, keepdim=True), torch.tensor(1.0)),
                *[metadata_query_mask_nm[:, i].unsqueeze(-1).float() for i in range(m)],
            )
            / n
        )
        label_weights_nc = {
            key: block_label_weight_nc[n * i : n * (i + 1)]
            for i, key in enumerate(["gene_value"] + list(metadata_tokens_n))
        }

        return {
            "gene_tokens_nc": gene_tokens_nc,
            "metadata_tokens_n": metadata_tokens_n,
            "prompt_mask_nc": prompt_mask_nc,
            "labels_nc": labels_nc,
            "label_weights_nc": label_weights_nc,
        }


class CellariumGPT(CellariumModel, PredictMixin, ValidateMixin):
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
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        use_bias: bool,
        attention_backend: Literal["flash", "flex", "math", "mem_efficient", "torch"],
        attention_softmax_fp32: bool,
        loss_scales: dict[str, float],
        # tunable hyperparameters
        initializer_range: float = 0.02,
        embeddings_scale: float = 1.0,
        attention_logits_scale: float = 1.0,
        output_logits_scale: float = 1.0,
        # muP (maximal update parameterization)  parameters
        lr_adjustment_groups: dict | None = None,
        mup_base_d_model: int | None = None,
        mup_base_d_ffn: int | None = None,
        gene_categories: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.gene_vocab_sizes = gene_vocab_sizes.copy()
        self.metadata_vocab_sizes = metadata_vocab_sizes.copy()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.attention_backend = attention_backend
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
        self.gene_embedding = GeneExpressionEmbedding(
            categorical_vocab_sizes=gene_vocab_sizes,
            continuous_vocab_sizes={"gene_value": 3},
            d_model=d_model,
            embeddings_initializer=embeddings_initializer,
        )
        self.metadata_embedding = MetadataEmbedding(
            categorical_vocab_sizes={key: vocab_size + 1 for key, vocab_size in metadata_vocab_sizes.items()},
            d_model=d_model,
            embeddings_initializer=embeddings_initializer,
        )
        self.transformer = Transformer(
            d_model,
            d_ffn,
            use_bias,
            n_heads,
            n_blocks,
            dropout_p,
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
                "gene_value": nn.Linear(d_model, gene_value_vocab_size, use_bias),
                **{key: nn.Linear(d_model, vocab_size, use_bias) for key, vocab_size in metadata_vocab_sizes.items()},
            }
        )
        self.loss_scales = loss_scales

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

    def predict(
        self,
        gene_tokens_nc: dict[str, torch.Tensor],
        metadata_tokens_n: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor | None,
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
        if self.attention_backend == "flex":

            def prompt_diagonal_mask_mod(b, h, q_idx, kv_idx):
                return prompt_mask_nc[b, kv_idx] | (q_idx == kv_idx)

            n, c = prompt_mask_nc.shape
            attention_mask_ncc = create_block_mask(prompt_diagonal_mask_mod, B=n, H=None, Q_LEN=c, KV_LEN=c)
        else:
            attention_mask_ncc = prompt_diagonal_mask(prompt_mask_nc)

        # transformer blocks
        hidden_state_ncd = embedding_ncd * self.embeddings_scale
        hidden_state_ncd = self.transformer(hidden_state_ncd, attention_mask_ncc)

        # compute logits
        logits_nck = {}
        for key in ["gene_value"] + list(metadata_tokens_n):
            logits_nck[key] = self.head[key](hidden_state_ncd) * self.output_logits_scale

        return logits_nck

    def forward(
        self,
        gene_tokens_nc: dict[str, torch.Tensor],
        metadata_tokens_n: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
        labels_nc: dict[str, torch.Tensor],
        label_weights_nc: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits_nck = self.predict(gene_tokens_nc, metadata_tokens_n, prompt_mask_nc)

        # compute loss
        losses = {}
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        for key, label_nc in labels_nc.items():
            losses[key] = torch.sum(
                loss_fn(logits_nck[key].view(label_nc.numel(), -1), label_nc.view(-1).long())
                * label_weights_nc[key].view(-1)
            )

        losses["loss"] = sum(losses[key] * self.loss_scales[key] for key in losses)

        return losses

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        metadata_tokens_n: dict[str, torch.Tensor],
        gene_tokens_nc: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
        labels_nc: dict[str, torch.Tensor],
        label_weights_nc: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        n = gene_tokens_nc["gene_value"].shape[0]
        loss_dict = self(gene_tokens_nc, metadata_tokens_n, prompt_mask_nc, labels_nc, label_weights_nc)

        pl_module.log_dict(loss_dict, sync_dist=True, on_epoch=True, batch_size=n)
