# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import torch


class CellariumGPTTrainTokenizer(torch.nn.Module):
    """
    Tokenizer for the Cellarium GPT model.

    Args:
        context_len:
            Context length.
        gene_downsample_fraction:
            Fraction of genes to downsample.
        min_total_mrna_umis:
            Minimum total mRNA UMIs.
        max_total_mrna_umis:
            Maximum total mRNA UMIs.
        gene_vocab_sizes:
            Gene token vocabulary sizes.
        metadata_vocab_sizes:
            Metadata token vocabulary sizes.
        ontology_infos_path:
            Path to ontology information.
        prefix_len:
            Prefix length. If ``None``, the prefix length is sampled.
        metadata_prompt_token_list:
            List of metadata tokens to prompt. If ``None``, the metadata prompt tokens are sampled.
        obs_names_rng:
            Cell IDs are used as random seeds for shuffling gene tokens.
            If ``None``, gene tokens are shuffled without a random seed.
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
        metadata_prompt_token_list: list[str] | None = None,
        obs_names_rng: bool = False,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.gene_downsample_fraction = gene_downsample_fraction
        self.min_total_mrna_umis = min_total_mrna_umis
        self.max_total_mrna_umis = max_total_mrna_umis
        self.gene_vocab_sizes = gene_vocab_sizes
        self.metadata_vocab_sizes = metadata_vocab_sizes
        self.ontology_infos = torch.load(ontology_infos_path, weights_only=True)
        self.ontology_downsample_p = ontology_downsample_p
        self.prefix_len = prefix_len
        self.metadata_prompt_token_list = metadata_prompt_token_list
        self.obs_names_rng = obs_names_rng

    def forward(
        self,
        metadata_token_n_dict: dict[str, torch.Tensor],
        gene_token_n_dict: dict[str, torch.Tensor],
        gene_token_ng_dict: dict[str, torch.Tensor],
        obs_names_n: np.ndarray | None = None,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        ### GENE TOKENS ###
        n, g = gene_token_ng_dict["gene_value"].shape
        m = len(metadata_token_n_dict)
        c = self.context_len
        # gene context length
        j = c - m
        device = gene_token_ng_dict["gene_value"].device

        ## gene measurement tokens (assay, suspension type, etc.) ##
        gene_token_nj_dict = {key: gene_token_n_dict[key][:, None].expand(-1, j).int() for key in gene_token_n_dict}

        ## gene id ##
        gene_token_ng_dict["gene_id"] = torch.arange(g, device=device).expand(n, g)

        if self.obs_names_rng:
            rng_n = [torch.Generator(device=device) for _ in range(n)]
            [rng.manual_seed(int(obs_name)) for rng, obs_name in zip(rng_n, obs_names_n)]
            shuffle_idx_ng = torch.stack([torch.randperm(g, generator=rng, device=device) for rng in rng_n])
        else:
            shuffle_idx_ng = torch.argsort(torch.rand((n, g), dtype=torch.float32, device=device), dim=-1)
        shuffle_idx_nj = shuffle_idx_ng[:, :j]

        for key, gene_token_ng in gene_token_ng_dict.items():
            gene_token_nj_dict[key] = torch.gather(gene_token_ng, dim=-1, index=shuffle_idx_nj)

        ## gene value ##
        gene_value_nj = gene_token_nj_dict.pop("gene_value")
        total_mrna_umis_nj = gene_token_nj_dict.pop("total_mrna_umis")
        # downsample gene values
        max_total_mrna_umis = torch.tensor(self.max_total_mrna_umis, device=device)
        downsampled_total_mrna_umis_nj = torch.minimum(total_mrna_umis_nj, max_total_mrna_umis).float()
        if self.gene_downsample_fraction > 0:
            gene_downsample_p_nj = torch.minimum(
                torch.rand((n, j), device=device) / self.gene_downsample_fraction,
                torch.tensor(1.0, device=device),
            )
            downsampled_total_mrna_umis_nj = torch.lerp(
                torch.full_like(gene_downsample_p_nj, self.min_total_mrna_umis),
                downsampled_total_mrna_umis_nj,
                gene_downsample_p_nj,
            )
        gene_downsample_p_nj = downsampled_total_mrna_umis_nj / total_mrna_umis_nj
        gene_value_nj = torch.binomial(gene_value_nj, gene_downsample_p_nj)
        total_mrna_umis_nj = torch.round(downsampled_total_mrna_umis_nj)
        if self.prefix_len is not None:
            prefix_len_n = torch.full((n,), self.prefix_len, dtype=torch.float32)
        else:
            # sample prefix length
            # prefix_len_weights = [1, max_prefix_len / 2, max_prefix_len / 3, ..., max_prefix_len / max_prefix_len]
            max_prefix_len = j - 1
            prefix_len_weights = 1 / torch.arange(max_prefix_len + 1, dtype=torch.float32)
            prefix_len_weights[0] = 1 / 10
            prefix_len_n = torch.multinomial(prefix_len_weights, n, replacement=True)
        # create prompt and query masks
        gene_query_mask_nj = torch.arange(j, device=device) >= prefix_len_n[:, None].expand(n, -1)
        gene_prompt_mask_nj = ~gene_query_mask_nj
        if "measured_genes_mask" in gene_token_nj_dict:
            measured_genes_mask_nj = gene_token_nj_dict.pop("measured_genes_mask")
            gene_query_mask_nj = gene_query_mask_nj & measured_genes_mask_nj
            gene_prompt_mask_nj = gene_prompt_mask_nj & measured_genes_mask_nj

        gene_token_nj_dict["gene_value"] = torch.log1p(gene_value_nj) * gene_prompt_mask_nj.float()
        gene_token_nj_dict["gene_query_mask"] = gene_query_mask_nj.float()
        gene_token_nj_dict["total_mrna_umis"] = torch.log1p(total_mrna_umis_nj)

        gene_token_value_nc_dict = {
            key: torch.cat([gene_token_nj, torch.zeros((n, m), device=device, dtype=gene_token_nj.dtype)], dim=1)
            for key, gene_token_nj in gene_token_nj_dict.items()
        }
        gene_token_mask_nc = torch.cat(
            [torch.ones((n, j), dtype=torch.bool, device=device), torch.zeros((n, m), dtype=torch.bool, device=device)],
            dim=1,
        )
        gene_token_mask_nc_dict = {key: gene_token_mask_nc for key in gene_token_nj_dict}

        # gene label
        gene_value_vocab_size = self.gene_vocab_sizes["gene_value"]
        gene_label_nj = gene_value_nj.clamp(0, gene_value_vocab_size - 1).int()

        ### METADATA TOKENS ###

        ## metadata tokens ##
        # assign token codes based on the ontology info
        # token values not in the ontology are treated as unmeasured and assigned a code value of -1
        for key, ontology_info in self.ontology_infos.items():
            assert self.metadata_vocab_sizes[key] == len(ontology_info["names"])
            metadata_token_n_dict[key] = torch.tensor(
                pd.Categorical(metadata_token_n_dict[key], categories=ontology_info["names"]).codes,
                dtype=torch.int,
            )
        # create metadata query and prompt masks
        if self.metadata_prompt_token_list is not None:
            metadata_prompt_mask_nm = torch.zeros((n, m), dtype=torch.bool, device=device)
            for metadata_token_idx, metadata_token in enumerate(metadata_token_n_dict):
                if metadata_token in self.metadata_prompt_token_list:
                    metadata_prompt_mask_nm[:, metadata_token_idx] = True
        else:
            metadata_prefix_len_n = torch.randint(0, m + 1, (n,), device=device)
            metadata_prefix_mask_nm = torch.arange(m, device=device) < metadata_prefix_len_n[:, None]
            shuffle_idx_nm = torch.argsort(torch.rand_like(metadata_prefix_mask_nm, dtype=torch.float32), dim=-1)
            metadata_prompt_mask_nm = torch.gather(metadata_prefix_mask_nm, dim=-1, index=shuffle_idx_nm)
        metadata_query_mask_nm = ~metadata_prompt_mask_nm
        metadata_measured_mask_nm = torch.stack(
            [metadata_token_n >= 0 for metadata_token_n in metadata_token_n_dict.values()], dim=1
        ).bool()
        metadata_query_mask_nm = metadata_query_mask_nm & metadata_measured_mask_nm
        metadata_prompt_mask_nm = metadata_prompt_mask_nm & metadata_measured_mask_nm
        # clamp unmeasured tokens to 0 in order to avoid error during embedding
        # the value of unmeasured tokens doesn't matter since they will be masked out by the attention mask
        for key, metadata_token_n in metadata_token_n_dict.items():
            metadata_token_n_dict[key] = metadata_token_n.clamp(0).int()
        # metadata labels
        metadata_label_n_dict = {key: metadata_token_n_dict[key].clone() for key in metadata_token_n_dict}
        if self.ontology_downsample_p != 0:
            # downsample metadata based on ontology
            for key, ontology_info in self.ontology_infos.items():
                if "shortest_distances_matrix" not in ontology_info:
                    continue
                metadata_token_n = metadata_token_n_dict[key]
                shortest_distances_matrix = ontology_info["shortest_distances_matrix"]
                ontology_weights = (
                    self.ontology_downsample_p * (1 - self.ontology_downsample_p) ** shortest_distances_matrix
                )
                metadata_token_n_dict[key] = (
                    torch.multinomial(ontology_weights[metadata_token_n], num_samples=1).squeeze(-1).int()
                )
        # impute mask token for unmeasured metadata
        # mask token is the last token in the vocabulary
        for i, (key, metadata_token_n) in enumerate(metadata_token_n_dict.items()):
            metadata_token_n_dict[key] = torch.where(
                metadata_query_mask_nm[:, i], self.metadata_vocab_sizes[key], metadata_token_n
            ).int()

        block_metadata_token_nm = torch.block_diag(
            *[metadata_token_n_dict[key].unsqueeze(-1) for key in metadata_token_n_dict],
        )
        metadata_token_value_nc_dict = {
            key: torch.cat(
                [torch.zeros((n, j), dtype=torch.int, device=device), block_metadata_token_nm[n * i : n * (i + 1)]],
                dim=1,
            )
            for i, key in enumerate(metadata_token_n_dict)
        }
        block_metadata_token_mask_nm = torch.block_diag(
            *[torch.ones((n, 1), dtype=torch.bool, device=device) for _ in metadata_token_n_dict],
        )
        metadata_token_mask_nc_dict = {
            key: torch.cat(
                [
                    torch.zeros((n, j), dtype=torch.bool, device=device),
                    block_metadata_token_mask_nm[n * i : n * (i + 1)],
                ],
                dim=1,
            )
            for i, key in enumerate(metadata_token_n_dict)
        }

        ### PROMPT MASK ###
        prompt_mask_nc = torch.cat([gene_prompt_mask_nj, metadata_prompt_mask_nm], dim=1)

        ### LABELS ###
        block_label_nc = torch.block_diag(
            gene_label_nj,
            *[metadata_label_n.unsqueeze(-1) for metadata_label_n in metadata_label_n_dict.values()],
        )
        label_nc_dict = {
            key: block_label_nc[n * i : n * (i + 1)]
            for i, key in enumerate(["gene_value"] + list(metadata_token_n_dict))
        }

        ### LABEL WEIGHTS ###
        block_label_weight_nc = (
            torch.block_diag(
                gene_query_mask_nj / torch.maximum(gene_query_mask_nj.sum(dim=-1, keepdim=True), torch.tensor(1.0)),
                *[metadata_query_mask_nm[:, i].unsqueeze(-1).float() for i in range(m)],
            )
            / n
        )
        label_weight_nc_dict = {
            key: block_label_weight_nc[n * i : n * (i + 1)]
            for i, key in enumerate(["gene_value"] + list(metadata_token_n_dict))
        }

        return {
            "token_value_nc_dict": gene_token_value_nc_dict | metadata_token_value_nc_dict,
            "token_mask_nc_dict": gene_token_mask_nc_dict | metadata_token_mask_nc_dict,
            "prompt_mask_nc": prompt_mask_nc,
            "label_nc_dict": label_nc_dict,
            "label_weight_nc_dict": label_weight_nc_dict,
        }


class CellariumGPTPredictTokenizer(torch.nn.Module):
    def __init__(
        self,
        max_total_mrna_umis: int,
        gene_vocab_sizes: dict[str, int],
        metadata_vocab_sizes: dict[str, int],
        ontology_infos_path: str,
    ) -> None:
        super().__init__()
        self.max_total_mrna_umis = max_total_mrna_umis
        self.gene_vocab_sizes = gene_vocab_sizes
        self.metadata_vocab_sizes = metadata_vocab_sizes
        self.ontology_infos = torch.load(ontology_infos_path, weights_only=True)

    def forward(
        self,
        gene_token_nj_dict: dict[str, torch.Tensor],  # set query genes to neg value?
        metadata_token_n_dict: dict[str, torch.Tensor],  # set query metadata to -1?
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        ### GENE TOKENS ###

        ## gene value ##
        gene_value_nj = gene_token_nj_dict.pop("gene_value")
        total_mrna_umis_nj = gene_token_nj_dict.pop("total_mrna_umis")
        device = gene_value_nj.device
        n, j = gene_value_nj.shape
        m = len(metadata_token_n_dict)
        # downsample library size to max_total_mrna_umis
        max_total_mrna_umis = torch.tensor(self.max_total_mrna_umis, device=device)
        downsampled_total_mrna_umis_nj = torch.minimum(total_mrna_umis_nj, max_total_mrna_umis).float()
        gene_downsample_p_nj = downsampled_total_mrna_umis_nj / total_mrna_umis_nj
        gene_value_nj = torch.binomial(gene_value_nj, gene_downsample_p_nj)

        gene_prompt_mask_nj = gene_value_nj >= 0
        gene_query_mask_nj = ~gene_prompt_mask_nj
        gene_token_nj_dict["gene_value"] = torch.log1p(gene_value_nj) * gene_prompt_mask_nj.float()
        gene_token_nj_dict["gene_query_mask"] = gene_query_mask_nj.float()
        gene_token_nj_dict["total_mrna_umis"] = torch.log1p(downsampled_total_mrna_umis_nj)

        gene_token_value_nc_dict = {
            key: torch.cat([gene_token_nj, torch.zeros((n, m), device=device, dtype=gene_token_nj.dtype)], dim=1)
            for key, gene_token_nj in gene_token_nj_dict.items()
        }
        gene_token_mask_nc = torch.cat(
            [torch.ones((n, j), dtype=torch.bool, device=device), torch.zeros((n, m), dtype=torch.bool, device=device)],
            dim=1,
        )
        gene_token_mask_nc_dict = {key: gene_token_mask_nc for key in gene_token_nj_dict}

        ### METADATA TOKENS ###

        ## metadata tokens ##
        # assign token codes based on the ontology info
        # token values not in the ontology are treated as unmeasured and assigned a code value of -1
        for key, ontology_info in self.ontology_infos.items():
            assert self.metadata_vocab_sizes[key] == len(ontology_info["names"])
            metadata_token_n_dict[key] = torch.tensor(
                pd.Categorical(metadata_token_n_dict[key], categories=ontology_info["names"]).codes,
                dtype=torch.int,
            )
        # create metadata query and prompt masks
        metadata_prompt_mask_nm = torch.stack(
            [metadata_token_n >= 0 for metadata_token_n in metadata_token_n_dict.values()], dim=1
        ).bool()
        metadata_query_mask_nm = ~metadata_prompt_mask_nm

        # impute mask token for unmeasured metadata
        # mask token is the last token in the vocabulary
        for i, (key, metadata_token_n) in enumerate(metadata_token_n_dict.items()):
            metadata_token_n_dict[key] = torch.where(
                metadata_query_mask_nm[:, i], self.metadata_vocab_sizes[key], metadata_token_n
            ).int()

        block_metadata_token_nm = torch.block_diag(
            *[metadata_token_n_dict[key].unsqueeze(-1) for key in metadata_token_n_dict],
        )
        metadata_token_value_nc_dict = {
            key: torch.cat(
                [torch.zeros((n, j), dtype=torch.int, device=device), block_metadata_token_nm[n * i : n * (i + 1)]],
                dim=1,
            )
            for i, key in enumerate(metadata_token_n_dict)
        }
        block_metadata_token_mask_nm = torch.block_diag(
            *[torch.ones((n, 1), dtype=torch.bool, device=device) for _ in metadata_token_n_dict],
        )
        metadata_token_mask_nc_dict = {
            key: torch.cat(
                [
                    torch.zeros((n, j), dtype=torch.bool, device=device),
                    block_metadata_token_mask_nm[n * i : n * (i + 1)],
                ],
                dim=1,
            )
            for i, key in enumerate(metadata_token_n_dict)
        }

        ### PROMPT MASK ###
        prompt_mask_nc = torch.cat([gene_prompt_mask_nj, metadata_prompt_mask_nm], dim=1)

        return {
            "token_value_nc_dict": gene_token_value_nc_dict | metadata_token_value_nc_dict,
            "token_mask_nc_dict": gene_token_mask_nc_dict | metadata_token_mask_nc_dict,
            "prompt_mask_nc": prompt_mask_nc,
        }
