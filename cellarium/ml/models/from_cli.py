# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence

import torch
from transformers import BertConfig, BertForMaskedLM

from cellarium.ml.models.geneformer import Geneformer
from cellarium.ml.models.incremental_pca import IncrementalPCA
from cellarium.ml.models.onepass_mean_var_std import OnePassMeanVarStd
from cellarium.ml.models.probabilistic_pca import ProbabilisticPCA
from cellarium.ml.models.tdigest import TDigest
from cellarium.ml.transforms import DivideByScale, Log1p, NormalizeTotal


class OnePassMeanVarStdFromCLI(OnePassMeanVarStd):
    """
    Preset default values for the LightningCLI.

    Args:
        g_genes:
            Number of genes.
        target_count:
            Target gene expression count.
    """

    def __init__(self, g_genes: int, target_count: int = 10_000) -> None:
        transform = torch.nn.Sequential(NormalizeTotal(target_count), Log1p())
        super().__init__(g_genes, transform=transform)


class ProbabilisticPCAFromCLI(ProbabilisticPCA):
    """
    Preset default values for the LightningCLI.

    Args:
        n_cells:
            Number of cells.
        g_genes:
            Number of genes.
        k_components:
            Number of principcal components.
        ppca_flavor:
            Type of the PPCA model. Has to be one of ``marginalized`` or ``linear_vae``.
        W_init_variance_ratio:
            Ratio of variance of W_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``W_init_scale`` is set to
            ``W_init_variance_ratio``.
        sigma_init_variance_ratio:
            Ratio of variance of sigma_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``sigma_init_scale`` is set to
            ``sigma_init_variance_ratio``.
        seed:
            Random seed used to initialize parameters.
        target_count:
            Target gene expression count.
        mean_var_std_ckpt_path:
            Path to checkpoint containing OnePassMeanVarStd.
    """

    def __init__(
        self,
        n_cells: int,
        g_genes: int,
        k_components: int = 256,
        ppca_flavor: str = "marginalized",
        W_init_variance_ratio: float = 0.5,
        sigma_init_variance_ratio: float = 0.5,
        seed: int = 0,
        target_count: int = 10_000,
        mean_var_std_ckpt_path: str | None = None,
    ) -> None:
        if mean_var_std_ckpt_path is None:
            # compute W_init_scale and sigma_init_scale
            W_init_scale = W_init_variance_ratio
            sigma_init_scale = sigma_init_variance_ratio
            mean_g = None
        else:
            # load OnePassMeanVarStdFromCLI from checkpoint
            onepass: OnePassMeanVarStdFromCLI = torch.load(mean_var_std_ckpt_path)
            assert isinstance(onepass.transform, torch.nn.Sequential)
            assert isinstance(onepass.transform[0], NormalizeTotal)
            assert target_count == onepass.transform.target_count
            assert g_genes == onepass.g_genes
            # compute W_init_scale and sigma_init_scale
            total_variance = onepass.var_g.sum().item()
            W_init_scale = math.sqrt(W_init_variance_ratio * total_variance / (g_genes * k_components))
            sigma_init_scale = math.sqrt(sigma_init_variance_ratio * total_variance / g_genes)
            mean_g = onepass.mean_g
        # create transform
        transform = torch.nn.Sequential(NormalizeTotal(target_count), Log1p())
        self.mean_var_std_ckpt_path = mean_var_std_ckpt_path
        super().__init__(
            n_cells=n_cells,
            g_genes=g_genes,
            k_components=k_components,
            ppca_flavor=ppca_flavor,
            mean_g=mean_g,
            W_init_scale=W_init_scale,
            sigma_init_scale=sigma_init_scale,
            seed=seed,
            transform=transform,
        )


class IncrementalPCAFromCLI(IncrementalPCA):
    """
    Preset default values for the LightningCLI.

    Args:
        k_components:
            Number of principal components.
        svd_lowrank_niter:
            Number of iterations for the low-rank SVD algorithm.
        perform_mean_correction:
            If ``True`` then the mean correction is applied to the update step.
            If ``False`` then the data is assumed to be centered and the mean correction
            is not applied to the update step.
        target_count:
            Target gene epxression count.
    """

    def __init__(
        self,
        g_genes: int,
        k_components: int,
        svd_lowrank_niter: int = 2,
        perform_mean_correction: bool = False,
        target_count: int = 10_000,
    ) -> None:
        transform = torch.nn.Sequential(NormalizeTotal(target_count), Log1p())
        super().__init__(
            g_genes=g_genes,
            k_components=k_components,
            svd_lowrank_niter=svd_lowrank_niter,
            perform_mean_correction=perform_mean_correction,
            transform=transform,
        )


class TDigestFromCLI(TDigest):
    """
    Preset default values for the LightningCLI.

    Args:
        g_genes:
            Number of genes.
        target_count:
            Target gene epxression count.
        eps:
            A value added to the denominator for numerical stability.
    """

    def __init__(self, g_genes: int, target_count: int = 10_000, eps: float = 1e-6) -> None:
        transform = NormalizeTotal(target_count=target_count, eps=eps)
        super().__init__(g_genes, transform=transform)


class GeneformerFromCLI(Geneformer):
    """
    Preset default values for the LightningCLI.

    Args:
        feature_schema:
            The list of the variable names in the input data.
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act:
            The non-linear activation function (function or string) in the encoder and pooler. If string, ``"gelu"``,
            ``"relu"``, ``"silu"`` and ``"gelu_new"`` are supported.
        hidden_dropout_prob:
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob:
            The dropout ratio for the attention probabilities.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size:
            The vocabulary size of the ``token_type_ids`` passed when calling :class:`transformers.BertModel`.
        initializer_range:
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        position_embedding_type:
            Type of position embedding. Choose one of ``"absolute"``, ``"relative_key"``, ``"relative_key_query"``. For
            positional embeddings use ``"absolute"``. For more information on ``"relative_key"``, please refer to
            `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`_.
            For more information on ``"relative_key_query"``, please refer to *Method 4* in `Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`_.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        mlm_probability:
            Ratio of tokens to mask for masked language modeling loss.
        tdigest_path:
            Path to the tdigest checkpoint. The tdigest checkpoint is used to normalize the input data by the non-zero
            median gene count values. If ``None`` then no normalization is applied.
        validate_input:
            If ``True`` the input data is validated.
    """

    def __init__(
        self,
        feature_schema: Sequence,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 4,
        intermediate_size: int = 512,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.02,
        attention_probs_dropout_prob: float = 0.02,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        position_embedding_type: str = "absolute",
        layer_norm_eps: float = 1e-12,
        mlm_probability: float = 0.15,
        tdigest_path: str | None = None,
        validate_input: bool = True,
    ):
        # model configuration
        config = {
            "vocab_size": len(feature_schema) + 2,  # number of genes + 2 for <mask> and <pad> tokens
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "hidden_act": hidden_act,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "hidden_dropout_prob": hidden_dropout_prob,
            "max_position_embeddings": max_position_embeddings,
            "type_vocab_size": type_vocab_size,
            "initializer_range": initializer_range,
            "position_embedding_type": position_embedding_type,
            "layer_norm_eps": layer_norm_eps,
            "pad_token_id": 0,
        }
        config = BertConfig(**config)
        model = BertForMaskedLM(config)

        if tdigest_path is not None:
            tdigest = torch.load(tdigest_path)
            transform = torch.nn.Sequential(
                NormalizeTotal(
                    target_count=tdigest.transform.target_count,
                    eps=tdigest.transform.eps,
                ),
                DivideByScale(
                    scale_g=tdigest.median_g,
                    feature_schema=feature_schema,
                    eps=tdigest.transform.eps,
                ),
            )
        else:
            transform = None
        super().__init__(
            feature_schema=feature_schema,
            model=model,
            mlm_probability=mlm_probability,
            transform=transform,
            validate_input=validate_input,
        )
