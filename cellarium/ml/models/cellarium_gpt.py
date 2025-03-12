# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from cellarium.ml.layers import MultiHeadReadout, TokenEmbedding, Transformer
from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.layers import scale_initializers_by_dimension
from cellarium.ml.utilities.mup import LRAdjustmentGroup

try:
    from cerebras.pytorch.backend import use_cs
except ImportError:

    def use_cs() -> bool:
        return False


torch.set_float32_matmul_precision("highest")


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
    CellariumGPT model.

    Args:
        categorical_token_size_dict:
            Categorical token vocabulary sizes. Must include "gene_value" and "gene_id". Additionally, it can include
            experimental conditions, such as "assay" and "suspension_type", and metadata tokens such as "cell_type",
            "tissue", "sex", "development_stage", and "disease".
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
        loss_scale_dict:
            A dictionary of loss scales for each label type. These are the query tokens that are used
            to compute the loss.
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
        categorical_token_size_dict: dict[str, int],
        # Model parameters
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        use_bias: bool,
        attention_backend: Literal["flex", "math", "mem_efficient", "torch"],
        attention_softmax_fp32: bool,
        loss_scale_dict: dict[str, float],
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
        self.categorical_token_size_dict = categorical_token_size_dict

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

        # Handle muP scaling for Adam and AdamW optimizers
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

        embedding_token_size_dict = {}
        for key, vocab_size in categorical_token_size_dict.items():
            if key in loss_scale_dict:
                # Add 1 to the vocab size for the query tokens to account for the mask token
                embedding_token_size_dict[key] = vocab_size + 1
            elif key != "gene_value":
                embedding_token_size_dict[key] = vocab_size
        self.token_embedding = TokenEmbedding(
            categorical_token_size_dict=embedding_token_size_dict,
            continuous_token_list=["gene_value", "gene_query_mask", "total_mrna_umis"],
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
            categorical_token_size_dict={key: categorical_token_size_dict[key] for key in loss_scale_dict},
            d_model=d_model,
            use_bias=use_bias,
            output_logits_scale=output_logits_scale,
            heads_initializer=heads_initializer,
        )
        self.loss_scale_dict = loss_scale_dict

        self.reset_parameters()

    def reset_parameters(self) -> None:
        def _reset_parameters(module):
            return getattr(module, "_reset_parameters", lambda: None)()

        self.apply(_reset_parameters)

    @property
    def d_model(self) -> int:
        block = self.transformer.blocks[0]
        # assert isinstance(block, TransformerBlock)
        return block.d_model

    @property
    def d_ffn(self) -> int:
        block = self.transformer.blocks[0]
        # assert isinstance(block, TransformerBlock)
        return block.d_ffn

    @property
    def n_heads(self) -> int:
        block = self.transformer.blocks[0]
        # assert isinstance(block, TransformerBlock)
        return block.attention.n_heads

    @property
    def n_blocks(self) -> int:
        return len(self.transformer.blocks)

    @property
    def attention_backend(self) -> Literal["flex", "math", "mem_efficient", "torch"]:
        block = self.transformer.blocks[0]
        # assert isinstance(block, TransformerBlock)
        return block.attention.attention_backend

    @attention_backend.setter
    def attention_backend(self, value: Literal["flex", "math", "mem_efficient", "torch"]) -> None:
        for block in self.transformer.blocks:
            # assert isinstance(block, TransformerBlock)
            block.attention.attention_backend = value

    def get_embeddings(
        self,
        token_value_nc_dict: dict[str, torch.Tensor],
        token_mask_nc_dict: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
        to_cpu = True
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Args:
            token_value_nc_dict:
                Dictionary of token value tensors of shape ``(n, c)``.
            token_mask_nc_dict:
                Dictionary of token mask tensors of shape ``(n, c)``.

        Returns:
            Dictionary of logits tensors of shape ``(n, c, k)``.
        """
        # Create embeddings
        embedding_ncd = self.token_embedding(token_value_nc_dict, token_mask_nc_dict)

        # Create attention mask
        attention_mask_ncc: torch.Tensor | BlockMask
        if self.attention_backend == "flex":

            def prompt_diagonal_mask_mod(b, h, q_idx, kv_idx):
                return prompt_mask_nc[b, kv_idx] | (q_idx == kv_idx)

            n, c = prompt_mask_nc.shape
            attention_mask_ncc = create_block_mask(prompt_diagonal_mask_mod, B=n, H=None, Q_LEN=c, KV_LEN=c)
        else:
            attention_mask_ncc = prompt_diagonal_mask(prompt_mask_nc)

        # transformer blocks
        hidden_state_ncd = embedding_ncd * self.embeddings_scale

        hidden_states = self.transformer.forward_all_hidden_states(hidden_state_ncd, attention_mask_ncc, 
                                                                   to_cpu=to_cpu)

        return hidden_states

    def predict(
        self,
        token_value_nc_dict: dict[str, torch.Tensor],
        token_mask_nc_dict: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Args:
            token_value_nc_dict:
                Dictionary of token value tensors of shape ``(n, c)``.
            token_mask_nc_dict:
                Dictionary of token mask tensors of shape ``(n, c)``.

        Returns:
            Dictionary of logits tensors of shape ``(n, c, k)``.
        """
        # Create embeddings
        embedding_ncd = self.token_embedding(token_value_nc_dict, token_mask_nc_dict)

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
        token_value_nc_dict: dict[str, torch.Tensor],
        token_mask_nc_dict: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
        label_nc_dict: dict[str, torch.Tensor],
        label_weight_nc_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits_nck_dict = self.predict(
            token_value_nc_dict=token_value_nc_dict,
            token_mask_nc_dict=token_mask_nc_dict,
            prompt_mask_nc=prompt_mask_nc,
        )

        # Compute loss
        if not (set(self.loss_scale_dict) == set(label_nc_dict) == set(label_weight_nc_dict)):
            raise ValueError("The keys of loss_scale_dict, label_nc_dict, and label_weight_nc_dict must be the same.")
        loss_dict = {}
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        # Make sure that label_nc_dict is created by concatenating the gene_value and metadata labels
        # in the same order as the embeddings.
        for key, label_nc in label_nc_dict.items():
            logits_nck = logits_nck_dict[key]
            assert isinstance(logits_nck, torch.Tensor)
            label_weight_nc = label_weight_nc_dict[key]
            assert isinstance(label_weight_nc, torch.Tensor)
            loss_dict[key] = torch.sum(
                loss_fn(logits_nck.view(label_nc.numel(), -1), label_nc.view(-1).long()) * label_weight_nc.view(-1)
            )

        loss = sum(loss_dict[key] * self.loss_scale_dict[key] for key in loss_dict)
        assert isinstance(loss, torch.Tensor)
        loss_dict["loss"] = loss

        return loss_dict

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        token_value_nc_dict: dict[str, torch.Tensor],
        token_mask_nc_dict: dict[str, torch.Tensor],
        prompt_mask_nc: torch.Tensor,
        label_nc_dict: dict[str, torch.Tensor],
        label_weight_nc_dict: dict[str, torch.Tensor],
    ) -> None:
        n = prompt_mask_nc.shape[0]
        loss_dict = self.forward(
            token_value_nc_dict=token_value_nc_dict,
            token_mask_nc_dict=token_mask_nc_dict,
            prompt_mask_nc=prompt_mask_nc,
            label_nc_dict=label_nc_dict,
            label_weight_nc_dict=label_weight_nc_dict,
        )

        pl_module.log_dict(loss_dict, sync_dist=True, on_epoch=True, batch_size=n)
