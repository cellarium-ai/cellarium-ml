# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
from transformers import BertConfig, BertForMaskedLM

from scvid.transforms import NormalizeTotal, ZScoreLog1pNormalize
from scvid.transforms.transforms import NonZeroMedianNormalize

from .geneformer import Geneformer
from .incremental_pca import IncrementalPCA
from .onepass_mean_var_std import OnePassMeanVarStd
from .probabilistic_pca import ProbabilisticPCA
from .tdigest import TDigest


class OnePassMeanVarStdFromCLI(OnePassMeanVarStd):
    """
    Preset default values for the LightningCLI.

    Args:
        g_genes: Number of genes.
        target_count: Target gene expression count. Default: ``10_000``.
    """

    def __init__(self, g_genes, target_count: int = 10_000) -> None:
        transform = ZScoreLog1pNormalize(mean_g=0, std_g=None, perform_scaling=False, target_count=target_count)
        super().__init__(g_genes, transform=transform)


class ProbabilisticPCAFromCLI(ProbabilisticPCA):
    """
    Preset default values for the LightningCLI.

    Args:
        n_cells: Number of cells.
        g_genes: Number of genes.
        k_components: Number of principcal components.
        ppca_flavor: Type of the PPCA model. Has to be one of ``marginalized`` or ``linear_vae``.
        W_init_variance_ratio: Ratio of variance of W_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``W_init_scale`` is set to
            ``W_init_variance_ratio``.
        sigma_init_variance_ratio: Ratio of variance of sigma_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``sigma_init_scale`` is set to
            ``sigma_init_variance_ratio``.
        seed: Random seed used to initialize parameters. Default: ``0``.
        target_count: Target gene expression count. Default: ``10_000``.
        mean_var_std_ckpt_path: Path to checkpoint containing OnePassMeanVarStd.
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
            assert isinstance(onepass.transform, ZScoreLog1pNormalize)
            assert target_count == onepass.transform.target_count
            assert g_genes == onepass.g_genes
            # compute W_init_scale and sigma_init_scale
            total_variance = onepass.var_g.sum().item()
            W_init_scale = math.sqrt(W_init_variance_ratio * total_variance / (g_genes * k_components))
            sigma_init_scale = math.sqrt(sigma_init_variance_ratio * total_variance / g_genes)
            mean_g = onepass.mean_g
        # create transform
        transform = ZScoreLog1pNormalize(mean_g=0, std_g=None, perform_scaling=False, target_count=target_count)
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
        k_components: Number of principal components.
        svd_lowrank_niter: Number of iterations for the low-rank SVD algorithm. Default: ``2``.
        perform_mean_correction: If ``True`` then the mean correction is applied to the update step.
            If ``False`` then the data is assumed to be centered and the mean correction
            is not applied to the update step.
        target_count: Target gene epxression count. Default: ``10_000``
    """

    def __init__(
        self,
        g_genes: int,
        k_components: int,
        svd_lowrank_niter: int = 2,
        perform_mean_correction: bool = False,
        target_count: int = 10_000,
    ) -> None:
        transform = ZScoreLog1pNormalize(mean_g=0, std_g=None, perform_scaling=False, target_count=target_count)
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
        g_genes: Number of genes.
        target_count: Target gene epxression count. Default: ``10_000``
        eps: A value added to the denominator for numerical stability. Default: ``1e-6``
    """

    def __init__(self, g_genes, target_count: int = 10_000, eps: float = 1e-6) -> None:
        transform = NormalizeTotal(target_count=target_count, eps=eps)
        super().__init__(g_genes, transform=transform)


class GeneformerFromCLI(Geneformer):
    def __init__(
        self,
        model_type="bert",
        max_input_size=2**11,  # 2048
        num_layers=6,
        num_attn_heads=4,
        num_embed_dim=256,
        intermed_size=None,
        activ_fn="relu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.02,
        hidden_dropout_prob=0.02,
        var_names_schema=None,
        tdigest_path=None,
        mlm_probability=0.15,
    ):
        intermed_size = intermed_size or num_embed_dim * 2
        # model configuration
        config = {
            "hidden_size": num_embed_dim,
            "num_hidden_layers": num_layers,
            "initializer_range": initializer_range,
            "layer_norm_eps": layer_norm_eps,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "hidden_dropout_prob": hidden_dropout_prob,
            "intermediate_size": intermed_size,
            "hidden_act": activ_fn,
            "max_position_embeddings": max_input_size,
            "model_type": model_type,
            "num_attention_heads": num_attn_heads,
            "pad_token_id": 0,
            "vocab_size": len(var_names_schema) + 2,  # genes+2 for <mask> and <pad> tokens
        }
        config = BertConfig(**config)
        model = BertForMaskedLM(config)

        if tdigest_path is not None:
            tdigest = torch.load(tdigest_path)
            transform = NonZeroMedianNormalize(
                tdigest.median_g,
                target_count=tdigest.transform.target_count,
                eps=tdigest.transform.eps,
            )
        super().__init__(
            var_names_schema=var_names_schema, model=model, transform=transform, mlm_probability=mlm_probability
        )
