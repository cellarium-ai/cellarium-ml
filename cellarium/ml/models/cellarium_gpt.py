# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from cellarium.ml.layers import GeneExpressionEmbedding, MetadataEmbedding, MultiHeadReadout, Transformer
from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.layers import scale_initializers_by_dimension
from cellarium.ml.utilities.mup import LRAdjustmentGroup

try:
    from cerebras.pytorch.backend import use_cs
except ImportError:

    def use_cs() -> bool:
        return False


torch.set_float32_matmul_precision("high")


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
        attention_softmax_fp32:
            Whether to use float32 for softmax computation when ``torch`` backend is used.
        loss_scales:
            A dictionary of loss scales for each label type.
        initializer_range:
            The standard deviation of the truncated normal initializer.
        embeddings_scale:
            Multiplier for the embeddings.
        attention_logits_scale:
            Multiplier for the attention logits.
        output_logits_scale:
            Multiplier for the output logits.
        mup_base_d_model:
            Base dimensionality of the model for muP.
        mup_base_d_ffn:
            Base dimensionality of the inner feed-forward layers for muP.
    """

    def __init__(
        self,
        # Vocab sizes
        gene_vocab_sizes: dict[str, int],
        metadata_vocab_sizes: dict[str, int],
        # Model parameters
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        use_bias: bool,
        attention_backend: Literal["flex", "math", "mem_efficient", "torch"],
        attention_softmax_fp32: bool,
        loss_scales: dict[str, float],
        # Tunable parameters
        initializer_range: float = 0.02,
        embeddings_scale: float = 1.0,
        attention_logits_scale: float = 1.0,
        output_logits_scale: float = 1.0,
        # muP (maximal update parameterization) parameters
        mup_base_d_model: int | None = None,
        mup_base_d_ffn: int | None = None,
    ) -> None:
        super().__init__()

        # Vocab sizes
        self.gene_vocab_sizes = gene_vocab_sizes
        self.metadata_vocab_sizes = metadata_vocab_sizes

        # Initializers
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
        heads_initializer = default_initializer.copy()
        self.lr_adjustment_groups = {
            "embedding": LRAdjustmentGroup("*embedding*weight"),
            "decoder_attention": LRAdjustmentGroup("*transformer*attention*W*weight"),
            "decoder_input_ffn": LRAdjustmentGroup("*transformer*ffn.dense1*weight"),
            "decoder_output_ffn": LRAdjustmentGroup("*transformer*ffn.dense2*weight"),
        }

        # Multipliers
        self.embeddings_scale = embeddings_scale
        self.attention_logits_scale = attention_logits_scale
        self.output_logits_scale = output_logits_scale

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
                self.lr_adjustment_groups[lr_adjustment_group].set_scale(1 / d_model_width_mult)
            self.width_mult = d_model_width_mult
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
            self.lr_adjustment_groups["decoder_output_ffn"].set_scale(1 / d_ffn_width_mult)
            assert self.width_mult == d_ffn_width_mult
        else:
            scale_initializers_by_dimension(
                dense2_initializer,
                depth_scale=(2 * n_blocks) ** -0.5,
            )

        gene_categorical_vocab_sizes = gene_vocab_sizes.copy()
        gene_value_vocab_size = gene_categorical_vocab_sizes.pop("gene_value")
        self.gene_embedding = GeneExpressionEmbedding(
            categorical_vocab_sizes=gene_categorical_vocab_sizes,
            continuous_tokens=["gene_value", "gene_query_mask", "total_mrna_umis"],
            d_model=d_model,
            embeddings_initializer=embeddings_initializer,
        )
        # Add 1 to the vocab size for the metadata embeddings to account for the mask token
        self.metadata_embedding = MetadataEmbedding(
            categorical_vocab_sizes={key: vocab_size + 1 for key, vocab_size in metadata_vocab_sizes.items()},
            d_model=d_model,
            embeddings_initializer=embeddings_initializer,
        )
        self.transformer = Transformer(
            d_model=d_model,
            d_ffn=d_ffn,
            use_bias=use_bias,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout_p=dropout_p,
            attention_logits_scale=attention_logits_scale,
            attention_backend=attention_backend,
            attention_softmax_fp32=attention_softmax_fp32,
            Wqkv_initializer=Wqkv_initializer,
            Wo_initializer=Wo_initializer,
            dense1_initializer=dense1_initializer,
            dense2_initializer=dense2_initializer,
        )
        self.head = MultiHeadReadout(
            categorical_vocab_sizes={"gene_value": gene_value_vocab_size, **metadata_vocab_sizes},
            d_model=d_model,
            use_bias=use_bias,
            output_logits_scale=output_logits_scale,
            heads_initializer=heads_initializer,
        )
        self.loss_scales = loss_scales

        self.reset_parameters()

    def reset_parameters(self) -> None:
        def _reset_parameters(module):
            return getattr(module, "_reset_parameters", lambda: None)()

        self.apply(_reset_parameters)

    @property
    def d_model(self) -> int:
        return self.transformer.blocks[0].d_model

    @property
    def d_ffn(self) -> int:
        return self.transformer.blocks[0].d_ffn

    @property
    def n_heads(self) -> int:
        return self.transformer.blocks[0].n_heads

    @property
    def n_blocks(self) -> int:
        return len(self.transformer.blocks)

    @property
    def attention_backend(self) -> Literal["flex", "math", "mem_efficient", "torch"]:
        return self.transformer.blocks[0].attention.attention_backend

    @attention_backend.setter
    def attention_backend(self, value: Literal["flex", "math", "mem_efficient", "torch"]) -> None:
        for block in self.transformer.blocks:
            block.attention.attention_backend = value

    def predict(
        self,
        gene_token_nc_dict: dict[str, torch.Tensor],
        gene_token_mask_nc: torch.Tensor,
        metadata_token_nc_dict: dict[str, torch.Tensor],
        metadata_token_mask_nc_dict: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        # Create embeddings
        gene_embedding_ncd = self.gene_embedding(gene_token_nc_dict) * gene_token_mask_nc.unsqueeze(-1)
        metadata_embedding_ncd_dict = self.metadata_embedding(metadata_token_nc_dict)
        metadata_embedding_ncd = sum(
            metadata_embedding_ncd_dict[key] * metadata_token_mask_nc_dict[key].unsqueeze(-1)
            for key in metadata_token_nc_dict
        )
        embedding_ncd = gene_embedding_ncd + metadata_embedding_ncd

        # Create attention mask
        attention_mask_ncc: torch.Tensor | BlockMask
        if self.attention_backend == "flex":

            def prompt_diagonal_mask_mod(b, h, q_idx, kv_idx):
                return prompt_mask_nc[b, kv_idx] | (q_idx == kv_idx)

            n, c = prompt_mask_nc.shape
            attention_mask_ncc = create_block_mask(prompt_diagonal_mask_mod, B=n, H=None, Q_LEN=c, KV_LEN=c)
        else:
            attention_mask_ncc = prompt_diagonal_mask(prompt_mask_nc)

        # Transformer blocks
        hidden_state_ncd = embedding_ncd * self.embeddings_scale
        hidden_state_ncd = self.transformer(hidden_state_ncd, attention_mask_ncc)

        # Compute logits
        logits_nck_dict = self.head(hidden_state_ncd)

        return logits_nck_dict

    def forward(
        self,
        gene_token_nc_dict: dict[str, torch.Tensor],
        gene_token_mask_nc: torch.Tensor,
        metadata_token_nc_dict: dict[str, torch.Tensor],
        metadata_token_mask_nc_dict: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
        label_nc_dict: dict[str, torch.Tensor],
        label_weight_nc_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits_nck_dict = self.predict(
            gene_token_nc_dict=gene_token_nc_dict,
            gene_token_mask_nc=gene_token_mask_nc,
            metadata_token_nc_dict=metadata_token_nc_dict,
            metadata_token_mask_nc_dict=metadata_token_mask_nc_dict,
            prompt_mask_nc=prompt_mask_nc,
        )

        # Compute loss
        losses = {}
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        # Make sure that label_nc_dict is created by concatenating the gene_value and metadata labels
        # in the same order as the embeddings.
        for key, label_nc in label_nc_dict.items():
            logits_nck = logits_nck_dict[key]
            assert isinstance(logits_nck, torch.Tensor)
            label_weight_nc = label_weight_nc_dict[key]
            assert isinstance(label_weight_nc, torch.Tensor)
            losses[key] = torch.sum(
                loss_fn(logits_nck.view(label_nc.numel(), -1), label_nc.view(-1).long()) * label_weight_nc.view(-1)
            )

        loss = sum(losses[key] * self.loss_scales[key] for key in losses)
        assert isinstance(loss, torch.Tensor)
        losses["loss"] = loss

        return losses

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        gene_token_ns_dict: dict[str, torch.Tensor],
        gene_prompt_mask_ns: torch.Tensor,
        metadata_token_n_dict: dict[str, torch.Tensor],
        metadata_prompt_mask_n_dict: dict[str, torch.Tensor],
        labels_nc: dict[str, torch.Tensor],
        label_weights_nc: dict[str, torch.Tensor],
    ) -> None:
        n = gene_prompt_mask_ns.shape[0]
        loss_dict = self.forward(
            gene_token_ns_dict=gene_token_ns_dict,
            gene_prompt_mask_ns=gene_prompt_mask_ns,
            metadata_token_n_dict=metadata_token_n_dict,
            metadata_prompt_mask_n_dict=metadata_prompt_mask_n_dict,
            label_nc_dict=labels_nc,
            label_weight_nc_dict=label_weights_nc,
        )

        pl_module.log_dict(loss_dict, sync_dist=True, on_epoch=True, batch_size=n)
