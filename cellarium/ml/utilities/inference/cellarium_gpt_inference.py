# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Inference utilities for the CellariumGPT model."""

import logging
import typing as t
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scanpy import AnnData
from tqdm.auto import tqdm
from functools import cached_property
from more_itertools import chunked

from cellarium.ml import CellariumModule, CellariumPipeline
from cellarium.ml.models.cellarium_gpt import PredictTokenizer
from cellarium.ml.utilities.inference.gene_network_analysis import load_gene_info_table

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a handler
handler = logging.StreamHandler()

# Create and set a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# To suppress the stupid AnnData warning ...
warnings.filterwarnings("ignore", category=UserWarning, message="Transforming to str index.")


def load_gene_info_table(gene_info_tsv_path: str, included_gene_ids: list[str]) -> t.Tuple[pd.DataFrame, dict, dict]:
    gene_info_df = pd.read_csv(gene_info_tsv_path, sep="\t")

    gene_symbol_to_gene_id_map = dict()
    for gene_symbol, gene_id in zip(gene_info_df['Gene Symbol'], gene_info_df['ENSEMBL Gene ID']):
        if gene_symbol != float('nan'):
            gene_symbol_to_gene_id_map[gene_symbol] = gene_id

    gene_id_to_gene_symbol_map = {
        gene_id: gene_symbol for gene_symbol, gene_id in gene_symbol_to_gene_id_map.items()}
    for gene_id in included_gene_ids:
        if gene_id not in gene_id_to_gene_symbol_map:
            gene_id_to_gene_symbol_map[gene_id] = gene_id

    return gene_info_df, gene_symbol_to_gene_id_map, gene_id_to_gene_symbol_map


class CellariumGPTInferenceContext:
    MAX_TOTAL_MRNA_UMIS = 100_000
    ASSAY_VOCAB_SIZE = 19
    GENE_ID_VOCAB_SIZE = 36_601
    GENE_VALUE_VOCAB_SIZE = 2_001
    SUSPENSION_TYPE_VOCAB_SIZE = 2

    CELL_TYPE_VOCAB_SIZE = 890
    DEVELOPMENT_STAGE_VOCAB_SIZE = 191
    DISEASE_VOCAB_SIZE = 350
    SEX_VOCAB_SIZE = 2
    TISSUE_VOCAB_SIZE = 822

    DEFAULT_METADATA_PROMPT_MASKS_DICT = {
        "cell_type": False,
        "tissue": False,
        "development_stage": False,
        "disease": False,
        "sex": False,
    }

    def __init__(
            self,
            cellarium_gpt_ckpt_path: str,
            ref_adata_path: str,
            gene_info_tsv_path: str,
            device: torch.device,
            attention_backend: str = "mem_efficient",
            verbose: bool = True):
        
        # for logging
        self.verbose = verbose

        # load an anndata extract as reference
        self._adata = sc.read_h5ad(ref_adata_path)

        # get gene ontology infos from the reference anndata
        self.gene_ontology_infos = self._get_gene_ontology_infos(self._adata)

        # load the model
        self.device = device
        self.gpt_pipeline: CellariumModule = CellariumModule.load_from_checkpoint(
            cellarium_gpt_ckpt_path, map_location=device
        )

        # inject gene categories
        self.gpt_pipeline.model.gene_categories = torch.from_numpy(np.asarray(self._adata.var_names))

        # change attention backend to memory efficient
        self.gpt_pipeline.model.set_attention_backend(attention_backend)

        # gene info related
        self.model_var_names = np.asarray(self._adata.var_names)
        self.model_var_names_set = set(self.model_var_names)
        self.var_name_to_index_map = {var_name: i for i, var_name in enumerate(self.model_var_names)}
        self.gene_info_df, self.gene_symbol_to_gene_id_map, self.gene_id_to_gene_symbol_map = load_gene_info_table(
            gene_info_tsv_path, self.model_var_names
        )

        # get the metadata ontology infos from the TrainTokenizer, which is the first step of the pipeline
        assert isinstance(self.gpt_pipeline.pipeline, CellariumPipeline)
        self.metadata_ontology_infos: dict[str, dict[str, t.Any]] = self.gpt_pipeline.pipeline[0].ontology_infos  # type: ignore[assignment]

        # rewire the pipeline with a PredictTokenizer
        self.predict_tokenizer = self._instantiate_predict_tokenizer()

        self.gpt_pipeline.pipeline = CellariumPipeline([
            self.predict_tokenizer,
            self.gpt_pipeline.model,
        ])

    def log(self, message: str):
        if self.verbose:
            logger.info(message)

    def _instantiate_predict_tokenizer(self) -> PredictTokenizer:
        return PredictTokenizer(
            max_total_mrna_umis=self.MAX_TOTAL_MRNA_UMIS,
            gene_vocab_sizes={
                "assay": self.ASSAY_VOCAB_SIZE,
                "gene_id": self.GENE_ID_VOCAB_SIZE,
                "gene_value": self.GENE_VALUE_VOCAB_SIZE,
                "suspension_type": self.SUSPENSION_TYPE_VOCAB_SIZE,
            },
            metadata_vocab_sizes={
                "cell_type": self.CELL_TYPE_VOCAB_SIZE,
                "tissue": self.TISSUE_VOCAB_SIZE,
                "development_stage": self.DEVELOPMENT_STAGE_VOCAB_SIZE,
                "disease": self.DISEASE_VOCAB_SIZE,
                "sex": self.SEX_VOCAB_SIZE,
            },
            ontology_infos=self.metadata_ontology_infos,
        )

    def _get_gene_ontology_infos(self, adata: AnnData) -> dict:
        gene_ontology_infos: dict[str, dict[str, t.Any]] = dict()

        gene_ontology_infos["assay_ontology_term_id"] = dict()
        gene_ontology_infos["assay_ontology_term_id"]["names"] = list(
            adata.obs["assay_ontology_term_id"].cat.categories
        )
        gene_ontology_infos["assay_ontology_term_id"]["labels"] = list(adata.obs["assay"].cat.categories)

        gene_ontology_infos["suspension_type"] = dict()
        gene_ontology_infos["suspension_type"]["names"] = list(adata.obs["suspension_type"].cat.categories)
        gene_ontology_infos["suspension_type"]["labels"] = list(adata.obs["suspension_type"].cat.categories)

        return gene_ontology_infos

    def generate_tokens_from_adata(
        self,
        adata: sc.AnnData,
        obs_index: int | list[int] | None,
        query_var_names: list[str],
        query_total_mrna_umis: float | None = None,
        metadata_prompt_masks_dict: dict[str, bool] | None = None,
    ) -> tuple[dict, dict]:
        """

        .. note::
        All variables in the AnnData are treated as prompts.


        """

        # slice the anndata
        if isinstance(obs_index, int):
            obs_index = [obs_index]

        # save obs before slicing
        if obs_index is not None:
            adata = adata[obs_index]

        # generate gene ids and masks
        n_cells = len(adata)
        adata_var_names = adata.var_names
        assert all([var_name in self.var_name_to_index_map for var_name in adata_var_names])
        assert all([var_name in self.var_name_to_index_map for var_name in query_var_names])
        prompt_var_index = [self.var_name_to_index_map[var_name] for var_name in adata_var_names]
        query_var_index = [self.var_name_to_index_map[var_name] for var_name in query_var_names]
        n_prompt_vars = len(prompt_var_index)
        n_query_vars = len(query_var_index)
        n_total_vars = n_prompt_vars + n_query_vars

        # cpu device for intermediate
        cpu_device = torch.device("cpu")

        # gene id
        gene_ids_nc = torch.tensor(prompt_var_index + query_var_index, dtype=torch.int64, device=cpu_device)[
            None, :
        ].expand(n_cells, n_total_vars)

        # gene prompt mask
        gene_prompt_mask_nc = torch.tensor(
            [1] * n_prompt_vars + [0] * n_query_vars, dtype=torch.bool, device=cpu_device
        )[None, :].expand(n_cells, n_total_vars)

        # gene value
        try:
            prompt_X_ng = np.asarray(adata.X.todense())
        except AttributeError:
            prompt_X_ng = adata.X
        prompt_gene_value_nc = torch.tensor(prompt_X_ng, dtype=torch.float32, device=cpu_device)
        query_gene_value_nc = torch.zeros(n_cells, n_query_vars, dtype=torch.float32, device=cpu_device)
        gene_value_nc = torch.cat([prompt_gene_value_nc, query_gene_value_nc], dim=1)

        # total mrna umis
        prompt_total_mrna_umis_nc = torch.tensor(
            adata.obs["total_mrna_umis"].values, dtype=torch.float32, device=cpu_device
        )[:, None].expand(n_cells, n_prompt_vars)
        if query_total_mrna_umis is None:
            # the same as prompt
            query_total_mrna_umis_nc = torch.tensor(
                adata.obs["total_mrna_umis"].values, dtype=torch.float32, device=cpu_device
            )[:, None].expand(n_cells, n_query_vars)
        else:
            query_total_mrna_umis_nc = torch.tensor(
                [query_total_mrna_umis] * n_cells, dtype=torch.float32, device=cpu_device
            )[:, None].expand(n_cells, n_query_vars)
        total_mrna_umis_nc = torch.cat([prompt_total_mrna_umis_nc, query_total_mrna_umis_nc], dim=1)

        # convert assay and suspension_type to codes
        assay_nc = torch.tensor(
            pd.Categorical(
                adata.obs["assay_ontology_term_id"].values,
                categories=self.gene_ontology_infos["assay_ontology_term_id"]["names"],
            ).codes,
            dtype=torch.int64,
            device=cpu_device,
        )[:, None].expand(n_cells, n_total_vars)
        suspension_type_nc = torch.tensor(
            pd.Categorical(
                adata.obs["suspension_type"].values, categories=self.gene_ontology_infos["suspension_type"]["names"]
            ).codes,
            dtype=torch.int64,
            device=cpu_device,
        )[:, None].expand(n_cells, n_total_vars)

        gene_tokens_dict = {
            "assay": assay_nc,  # categorical
            "suspension_type": suspension_type_nc,  # categorical
            "gene_id": gene_ids_nc,  # categorical
            "gene_value": gene_value_nc,  # continuous
            "total_mrna_umis": total_mrna_umis_nc,  # continuous
        }

        # metadata prompt masks
        if metadata_prompt_masks_dict is None:
            metadata_prompt_masks_dict = self.DEFAULT_METADATA_PROMPT_MASKS_DICT
        expanded_metadata_prompt_masks_dict = dict()
        for key in self.metadata_ontology_infos.keys():  # note: key order is important ...
            expanded_metadata_prompt_masks_dict[key] = torch.tensor(
                [metadata_prompt_masks_dict[key]] * n_cells, dtype=torch.bool, device=cpu_device
            )

        # generate metadata tokens dicts; `PredictTokenizer` will convert these to codes
        metadata_tokens_dict = {
            "cell_type": adata.obs["cell_type_ontology_term_id"].values,  # categorical
            "tissue": adata.obs["tissue_ontology_term_id"].values,  # categorical
            "development_stage": adata.obs["development_stage_ontology_term_id"].values,  # categorical
            "disease": adata.obs["disease_ontology_term_id"].values,  # categorical
            "sex": adata.obs["sex_ontology_term_id"].values,  # categorical
        }

        # where to find each thing in the context?
        context_indices = dict()
        context_indices["prompt_genes"] = np.arange(0, n_prompt_vars).tolist()
        context_indices["query_genes"] = np.arange(n_prompt_vars, n_query_vars + n_prompt_vars).tolist()
        offset = 0
        for metadata_key in self.metadata_ontology_infos.keys():
            if metadata_prompt_masks_dict[metadata_key]:  # prompted
                prefix = "prompt"
            else:
                prefix = "query"
            context_indices[f'{prefix}_{metadata_key}'] = n_query_vars + n_prompt_vars + offset
            offset += 1

        # return gene_tokens_dict, metadata_tokens_dict
        tokens_dict = self.predict_tokenizer(
            metadata_tokens_n=metadata_tokens_dict,
            metadata_prompt_masks_n=expanded_metadata_prompt_masks_dict,
            gene_tokens_nc=gene_tokens_dict,
            gene_prompt_mask_nc=gene_prompt_mask_nc,
        )

        return tokens_dict, context_indices

    def get_gene_value_logits_by_metadata(
        self,
        assay: str,
        suspension_type: str,
        prompt_metadata_dict: dict,
        total_mrna_umis: int,
        query_gene_ids: list[str],
        max_counts: int | None = None
    ) -> t.Tuple[dict, dict]:
        
        metadata_prompt_masks_dict, metadata_dict = self.process_user_metadata(
            assay, suspension_type, prompt_metadata_dict, total_mrna_umis)

        obs_df = pd.DataFrame({key: [value] for key, value in metadata_dict.items()})
        obs_df.index = obs_df.index.astype(str)
        var_df = pd.DataFrame()
        adata = sc.AnnData(X=np.zeros((1, 0)), obs=obs_df, var=var_df)

        # Tokenize
        tokens_dict, context_indices = self.generate_tokens_from_adata(
            adata=adata,
            obs_index=None,
            query_var_names=query_gene_ids,
            metadata_prompt_masks_dict=metadata_prompt_masks_dict
        )
        
        gene_logits_nqk = self.get_gene_value_logits_from_tokens(tokens_dict, context_indices, max_counts)

        return gene_logits_nqk


    def generate_gene_tokens_by_metadata(
        self,
        assay: str,
        suspension_type: str,
        prompt_metadata_dict: dict,
        total_mrna_umis: int,
        query_gene_ids: list[str],
        perturb_gene_ids: list[str] | None,
        perturb_gene_values: np.ndarray | None = None,
    ) -> t.Tuple[dict, dict]:
        
        """

        .. note::
        If `perturb_gene_ids` is None, there will be only a single cell in the generated tokens.
        If `perturb_gene_ids` is not None, there first cell will be unperturbed, and the subsequent
            cells will have the genes in `perturb_gene_ids` set to 0, in order.

        .. note::
        The first token is reversed for perturbation. The first cell is always control unperturbed (which
        will be the only cell if `perturb_gene_ids` is None). In that cell, the first token is arbitrarily
        set to the gene corresponding to index 0 in the gene ID dictionary and is marked to be queried.
        Please ignore that. More generally, use `context_indices` to determine the indices of the actual
        query genes in the generated context.

        """

        metadata_prompt_masks_dict, metadata_dict = self.process_user_metadata(
            assay, suspension_type, prompt_metadata_dict, total_mrna_umis)

        if perturb_gene_ids is None:
            n_cells = 1
        else:
            n_cells = len(perturb_gene_ids) + 1

        # Generate a placeholder AnnData. There is always only gene at the prompt, which will
        # be replaced with the gene to be perturbed. If no perturbation is required, the gene
        # will be set to the first gene in the gene ID dictionary.
        obs_df = pd.DataFrame({key: [value] * n_cells for key, value in metadata_dict.items()})
        obs_df.index = obs_df.index.astype(str)
        var_df = pd.DataFrame(index=[self.model_var_names[0]])
        pert_adata = sc.AnnData(X=np.zeros((n_cells, 1)), obs=obs_df, var=var_df)

        # Tokenize
        tokens_dict, context_indices = self.generate_tokens_from_adata(
            adata=pert_adata,
            obs_index=None,
            query_var_names=query_gene_ids,
            metadata_prompt_masks_dict=metadata_prompt_masks_dict
        )

        # The first cell is control unperturbed, so we will ensure that the first gene is marked as queried  (not perturbed)
        PERTURB_GENE_CONTEXT_INDEX = 0
        tokens_dict['prompt_mask_nc'][0, 0] = False
        tokens_dict['gene_tokens_nc']['gene_value'][0, PERTURB_GENE_CONTEXT_INDEX, 1] = 1  # Mark as queried

        # In subsequent cells, prompt genes are set to 0 sequentially
        if perturb_gene_ids is not None:
            assert all(gene_id in self.var_name_to_index_map for gene_id in perturb_gene_ids)
            tokens_dict['gene_tokens_nc']['gene_id'] = tokens_dict['gene_tokens_nc']['gene_id'].clone()
            tokens_dict['gene_tokens_nc']['gene_id'][1:, 0] = torch.tensor([
                self.var_name_to_index_map[var_name] for var_name in perturb_gene_ids])
            if perturb_gene_values is None:
                self.log("Perturbed gene values are not provided, assuming in silico deletion (set to 0) ...")
                perturb_gene_values = np.zeros((len(perturb_gene_ids),))
            else:
                self.log("Perturbed gene values are provided, injecting the provided values into the prompt ...")
                assert len(perturb_gene_values) == len(perturb_gene_ids)
                tokens_dict['gene_tokens_nc']['gene_value'] = tokens_dict['gene_tokens_nc']['gene_value'].clone()
                tokens_dict['gene_tokens_nc']['gene_value'][1:, PERTURB_GENE_CONTEXT_INDEX, 0] = torch.tensor(
                    perturb_gene_values).log1p()

        return tokens_dict, context_indices, pert_adata, metadata_prompt_masks_dict


    def process_user_metadata(
            self,
            assay: str,
            suspension_type: str,
            prompt_metadata_dict: dict[str, str],
            total_mrna_umis: int | float) -> t.Tuple[dict[str, bool], dict[str, str]]:
        """ Given user provided metadata, generate a complete metadata dictionary with ontology term IDs
        where applicable.
        """
        
        METADATA_KEYS = [
            "cell_type",
            "tissue",
            "development_stage",
            "disease",
            "sex",
        ]

        # Use the first label as the default value (actual value does not matter)
        default_metadata_dict = {
            metadata_key: self.metadata_ontology_infos[metadata_key]["labels"][0] for metadata_key in METADATA_KEYS
        }

        # True if provided, False otherwise
        metadata_prompt_masks_dict = {
            metadata_key: metadata_key in prompt_metadata_dict for metadata_key in METADATA_KEYS
        }

        # Generate a complete metadata dictionary, including default values for missing keys
        metadata_dict = {
            metadata_key: prompt_metadata_dict.get(metadata_key, default_metadata_dict[metadata_key])
            for metadata_key in METADATA_KEYS
        }
        metadata_dict |= {"assay": assay, "suspension_type": suspension_type, "total_mrna_umis": total_mrna_umis}

        # Augment metadata dictionary with ontology term IDs
        metadata_dict |= {
            f"{metadata_key}_ontology_term_id": self.metadata_ontology_infos[metadata_key]["names"][
                self.metadata_ontology_infos[metadata_key]["labels"].index(metadata_dict[metadata_key])
            ]
            for metadata_key in METADATA_KEYS
        }
        metadata_dict |= {
            "assay_ontology_term_id": self.gene_ontology_infos["assay_ontology_term_id"]["names"][
                self.gene_ontology_infos["assay_ontology_term_id"]["labels"].index(assay)
            ]
        }
        
        return metadata_prompt_masks_dict, metadata_dict


    def get_marginal_mean_std(
            self,
            adata: sc.AnnData,
            query_var_names: list[str],
            query_total_mrna_umis: float | None = None,
            prompt_gene_values_g: torch.Tensor | None = None,
            metadata_prompt_masks_dict: dict[str, bool] | None = None
        ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        .. note: This is a legacy implemetation used for Jacobian calculation which assumes only a single cell.
          For most applications, please consider using the other methods provided in this class.
        """
    
        assert len(adata) == 1, "Only a single cell is allowed"

        if metadata_prompt_masks_dict is None:
            metadata_prompt_masks_dict = self.DEFAULT_METADATA_PROMPT_MASKS_DICT

        tokens_dict, context_indices = self.generate_tokens_from_adata(
            adata=adata,
            obs_index=None,
            query_var_names=query_var_names,
            query_total_mrna_umis=query_total_mrna_umis,
            metadata_prompt_masks_dict=metadata_prompt_masks_dict,
        )

        # convert to cuda
        tokens_dict = self.gpt_pipeline.transfer_batch_to_device(tokens_dict, self.device, 0)

        # get a reference to prompt gene values
        FIRST_CELL_DIM = 0
        GENE_VALUE_DIM = 0
        prompt_gene_log1p_values_g = tokens_dict["gene_tokens_nc"]["gene_value"][
            FIRST_CELL_DIM, context_indices["prompt_genes"], GENE_VALUE_DIM
        ]

        # this is the "source", in case it is provided, e.g., for Jacobian calculation
        if prompt_gene_values_g is None:
            prompt_gene_values_g = torch.expm1(prompt_gene_log1p_values_g).clone()

        # inject back to tokens_dict to re-establish the reference for Jacobian calculation
        tokens_dict["gene_tokens_nc"]["gene_value"][FIRST_CELL_DIM, context_indices["prompt_genes"], GENE_VALUE_DIM] = (
            torch.log1p(prompt_gene_values_g)
        )

        # get model predictions
        logits_dict = self.gpt_pipeline.model.predict(
            gene_tokens_nc=tokens_dict["gene_tokens_nc"],
            metadata_tokens_n=tokens_dict["metadata_tokens_n"],
            prompt_mask_nc=tokens_dict["prompt_mask_nc"],
        )

        # note: we use `q` to denote query genes
        gene_logits_qk = logits_dict["gene_value"][FIRST_CELL_DIM, context_indices["query_genes"], :]
        gene_logits_qk = gene_logits_qk - torch.logsumexp(gene_logits_qk, dim=-1, keepdim=True)
        MAX_COUNTS = gene_logits_qk.shape[-1]
        log_counts_1_k = torch.arange(0, MAX_COUNTS, device=gene_logits_qk.device).log()
        log_counts_2_k = torch.arange(0, MAX_COUNTS, device=gene_logits_qk.device).pow(2).log()
        gene_mom_1_q = torch.logsumexp(gene_logits_qk + log_counts_1_k[None, :], dim=-1).exp()
        gene_mom_2_q = torch.logsumexp(gene_logits_qk + log_counts_2_k[None, :], dim=-1).exp()
        gene_marginal_means_q = gene_mom_1_q
        gene_marginal_std_q = torch.clamp(gene_mom_2_q - gene_mom_1_q.pow(2), 0.0).sqrt()

        return gene_marginal_means_q, gene_marginal_std_q


    def get_gene_value_logits_from_tokens(
            self,
            tokens_dict: dict,
            context_indices: dict,
            max_counts: int | None = None
        ) -> torch.Tensor:

        # convert to cuda
        tokens_dict = self.gpt_pipeline.transfer_batch_to_device(tokens_dict, self.device, 0)
        
        # get model predictions
        logits_dict = self.gpt_pipeline.model.predict(
            gene_tokens_nc=tokens_dict["gene_tokens_nc"],
            metadata_tokens_n=tokens_dict["metadata_tokens_n"],
            prompt_mask_nc=tokens_dict["prompt_mask_nc"],
            predict_keys=["gene_value"],
        )

        # note: we use `q` to denote query genes
        query_gene_indices = torch.tensor(context_indices['query_genes'], device=self.device, dtype=torch.int64)
        gene_logits_nqk = logits_dict['gene_value'][:, query_gene_indices, :]
        
        if max_counts is None:
            max_counts = gene_logits_nqk.shape[-1]
        else:
            assert max_counts > 0
        gene_logits_nqk = gene_logits_nqk[:, :, :max_counts]  # truncate to max_counts
        gene_logits_nqk = gene_logits_nqk - torch.logsumexp(gene_logits_nqk, dim=-1, keepdim=True)  # renormalize

        return gene_logits_nqk
     

    def get_marginal_mean_std_from_tokens(
            self,
            tokens_dict: dict,
            context_indices: dict,
            use_logsumexp: bool = True,
            max_counts: int | None = None,
        ) -> t.Tuple[torch.Tensor, torch.Tensor]:
    
        gene_logits_nqk = self.get_gene_value_logits_from_tokens(tokens_dict, context_indices, max_counts)

        if max_counts is None:
            max_counts = gene_logits_nqk.shape[-1]
        
        return self.calculate_gene_mean_std_from_logits(gene_logits_nqk, max_counts, use_logsumexp)


    def calculate_gene_mean_std_from_logits(
            self,
            gene_logits_nqk: torch.Tensor,
            max_counts: int,
            use_logsumexp: bool = True) -> t.Tuple[torch.Tensor, torch.Tensor]:
    
        if use_logsumexp:
            log_counts_1_k = torch.arange(0, max_counts, device=gene_logits_nqk.device).log()
            log_counts_2_k = torch.arange(0, max_counts, device=gene_logits_nqk.device).pow(2).log()
            gene_mom_1_nq = torch.logsumexp(gene_logits_nqk + log_counts_1_k[None, None, :], dim=-1).exp()
            gene_mom_2_nq = torch.logsumexp(gene_logits_nqk + log_counts_2_k[None, None, :], dim=-1).exp()
            gene_marginal_means_nq = gene_mom_1_nq
            gene_marginal_std_nq = torch.clamp(gene_mom_2_nq - gene_mom_1_nq.pow(2), 0.0).sqrt()
        else:
            gene_probs_nqk = torch.exp(gene_logits_nqk)
            counts_1_k = torch.arange(0, max_counts, device=gene_logits_nqk.device)
            counts_2_k = torch.arange(0, max_counts, device=gene_logits_nqk.device).pow(2)
            gene_mom_1_nq = (gene_probs_nqk * counts_1_k[None, None, :]).sum(dim=-1)
            gene_mom_2_nq = (gene_probs_nqk * counts_2_k[None, None, :]).sum(dim=-1)
            gene_marginal_means_nq = gene_mom_1_nq
            gene_marginal_std_nq = torch.clamp(gene_mom_2_nq - gene_mom_1_nq.pow(2), 0.).sqrt()

        return gene_marginal_means_nq, gene_marginal_std_nq


    def get_marginal_mean_std_multi_cell(
        self,
        adata: sc.AnnData,
        query_var_names: list[str],
        query_total_mrna_umis: float | None = None,
        metadata_prompt_masks_dict: dict[str, bool] | None = None,
        use_logsumexp: bool = True,
        max_counts: int | None = None,
        verbose: bool = False,
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if metadata_prompt_masks_dict is None:
            metadata_prompt_masks_dict = self.DEFAULT_METADATA_PROMPT_MASKS_DICT

        if verbose:
            logger.info("Tokenizing the AnnData ...")

        tokens_dict, context_indices = self.generate_tokens_from_adata(
            adata=adata,
            obs_index=None,
            query_var_names=query_var_names,
            query_total_mrna_umis=query_total_mrna_umis,
            metadata_prompt_masks_dict=metadata_prompt_masks_dict,
        )

        if verbose:
            logger.info("Done.")

        return self.get_marginal_mean_std_from_tokens(
            tokens_dict=tokens_dict,
            context_indices=context_indices,
            use_logsumexp=use_logsumexp,
            max_counts=max_counts,
            verbose=verbose,
        )


    def predict_gene_expression_range_for_metadata(
        self,
        assay: str,
        suspension_type: str,
        prompt_metadata_dict: dict[str, str],
        total_mrna_umis: int | float,
        query_gene_ids: list[str],
        query_chunk_size: int | None = None,
        total_prob_mass: float = 0.9,
        symmetric_range_pad: int = 1,
        max_counts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the maximum counts indices based on cumulative gene probabilities.

        This function chunks the list of gene identifiers, retrieves gene logits
        for each chunk using `ctx.get_gene_value_logits_by_metadata`, concatenates
        the results, and then computes the cumulative sum of the exponentiated logits.
        For each query, it finds the first (smallest) index along the k dimension where
        the cumulative probability exceeds the provided upper_percentile threshold,
        and then adds an upper_pad value.

        Args:
            ctx: An object that provides:
                - `ctx.get_gene_value_logits_by_metadata(...)` to compute gene logits.
            query_gene_ids (list): List of gene identifiers.
            assay (str): Name of the assay (e.g., "10x 3' v3").
            suspension_type (str): Type of suspension (e.g., "cell").
            prompt_metadata_dict (dict): Metadata dictionary (e.g., cell type, tissue).
            total_mrna_umis (int): Total number of mRNA UMIs.
            query_chunk_size (int): Number of gene IDs to process in one chunk.
            total_prob_mass (float): The amount of probability mass to capture within the to be determined
              expression range (e.g., 0.999).
            symmetric_range_pad (int): Value to add symmetrically as padding to the found range.
            max_counts (int): Maximum number of counts to consider. If None, use the full range. Otherwise, truncate.

        Returns:
            torch.Tensor: A tensor containing, for each query, the smallest index along k
                        where the cumulative probability exceeds upper_percentile plus upper_pad.
        """

        assert 0.0 < total_prob_mass < 1.0, "The upper percentile must be between 0 and 1."

        if query_chunk_size is None:
            query_chunk_size = len(query_gene_ids)
        
        partial_gene_logits_list = []
        with torch.inference_mode():
            # Process gene IDs in chunks to limit memory usage
            for partial_query_gene_ids in tqdm(list(chunked(query_gene_ids, query_chunk_size)),
                                            desc="Processing gene chunks"):
                partial_gene_logits_nqk = self.get_gene_value_logits_by_metadata(
                    assay=assay,
                    suspension_type=suspension_type,
                    prompt_metadata_dict=prompt_metadata_dict,
                    total_mrna_umis=total_mrna_umis,
                    query_gene_ids=partial_query_gene_ids,
                    max_counts=max_counts,
                )
                partial_gene_logits_list.append(partial_gene_logits_nqk)

            gene_logits_qk = torch.cat(partial_gene_logits_list, dim=1)[0]

        # first, find the mode of the counts distribution for each gene
        gene_logits_mode_q = torch.argmax(gene_logits_qk, dim=1)

        # symmetric lower and upper counts about the mode for each gene
        x_lo_qm = torch.clamp(
            gene_logits_mode_q[:, None] - torch.arange(0, max_counts, device=self.device)[None, :], min=0)
        x_hi_qm = torch.clamp(
            gene_logits_mode_q[:, None] + torch.arange(0, max_counts, device=self.device)[None, :], max=max_counts - 1)

        # compute the CDF of counts for each gene
        pdf_qk = gene_logits_qk.exp()
        cdf_qk = pdf_qk.cumsum(dim=1)
        q_indices = torch.arange(cdf_qk.size(0), device=self.device)
        symm_prob_mass_qm = (
            cdf_qk[q_indices[:, None], x_hi_qm]  # add total prob mass at the right point (inclusive)
            - cdf_qk[q_indices[:, None], x_lo_qm]  # subtract total prob mass at the left point (inclusive)
            + pdf_qk[q_indices[:, None], x_lo_qm]  # add back the prob mass of the left point
        )
        mask_qm = symm_prob_mass_qm > total_prob_mass
        range_q = torch.clamp(mask_qm.float().argmax(dim=-1) + symmetric_range_pad, max=max_counts - 1)
        x_lo_q = x_lo_qm[q_indices, range_q]
        x_hi_q = x_hi_qm[q_indices, range_q]

        # calculate the mean and std of expression for bookkeeping
        gene_marginal_mean_nq, gene_marginal_std_nq = self.calculate_gene_mean_std_from_logits(
            gene_logits_nqk=gene_logits_qk[None, :, :],
            max_counts=max_counts,
            use_logsumexp=True)

        return {
            'x_lo_q': x_lo_q,
            'x_hi_q': x_hi_q,
            'range_q': range_q,
            'gene_logits_qk': gene_logits_qk,
            'gene_logits_mode_q': gene_logits_mode_q,
            'gene_marginal_mean_q': gene_marginal_mean_nq[0],
            'gene_marginal_std_q': gene_marginal_std_nq[0],
        }

    def generate_gene_dose_response_for_metadata(
        self,
        assay: str,
        suspension_type: str,
        prompt_metadata_dict: dict[str, str],
        total_mrna_umis: int | float,
        query_gene_ids: list[str],
        perturb_gene_ids: list[str],
        x_lo_p: np.ndarray,
        x_hi_p: np.ndarray,
        n_points: int,
        query_chunk_size: int,
        max_counts: int | None = None,
    ):
        
        assert n_points >= 2

        # initialize arrays to store results
        n_query_genes = len(query_gene_ids)
        n_perturb_genes = len(perturb_gene_ids)
        doses_pi = np.zeros((n_perturb_genes, n_points))
        responses_mean_pqi = np.zeros((n_perturb_genes, n_query_genes, n_points))
        responses_std_pqi = np.zeros((n_perturb_genes, n_query_genes, n_points))

        # outer loop (dose quantiles)
        for i_point in tqdm(range(n_points), desc="Processing dose quantiles"):

            # values of genes to perturb
            perturb_gene_values = x_lo_p +  (x_hi_p - x_lo_p) * i_point / (n_points - 1)
            doses_pi[:, i_point] = perturb_gene_values

            # inner loop (responses)
            gene_marginal_mean_nq_chunks = []
            gene_marginal_std_nq_chunks = []
            for query_gene_ids_chunk in tqdm(list(chunked(query_gene_ids, query_chunk_size)),
                                             desc="Processing query gene chunks"):
                with torch.inference_mode():
                    tokens_dict, context_indices, _, _ = self.generate_gene_tokens_by_metadata(
                        assay=assay,
                        suspension_type=suspension_type,
                        prompt_metadata_dict=prompt_metadata_dict,
                        total_mrna_umis=total_mrna_umis,
                        query_gene_ids=query_gene_ids_chunk,
                        perturb_gene_ids=perturb_gene_ids,
                        perturb_gene_values=perturb_gene_values
                    )
                    tokens_dict = self.gpt_pipeline.transfer_batch_to_device(tokens_dict, self.device, 0)
                    gene_marginal_mean_nq, gene_marginal_std_nq = self.get_marginal_mean_std_from_tokens(
                        tokens_dict=tokens_dict,
                        context_indices=context_indices,
                        max_counts=max_counts)
                gene_marginal_mean_nq_chunks.append(gene_marginal_mean_nq.cpu().numpy())
                gene_marginal_std_nq_chunks.append(gene_marginal_std_nq.cpu().numpy())
            gene_marginal_mean_nq = np.concatenate(gene_marginal_mean_nq_chunks, axis=1)
            gene_marginal_std_nq = np.concatenate(gene_marginal_std_nq_chunks, axis=1)

            control_mean_q = gene_marginal_mean_nq[0, :]
            control_std_q = gene_marginal_std_nq[0, :]
            responses_mean_pqi[:, :, i_point] = gene_marginal_mean_nq[1:, :]
            responses_std_pqi[:, :, i_point] = gene_marginal_std_nq[1:, :]

        return {
            'doses_pi': doses_pi,
            'responses_mean_pqi': responses_mean_pqi,
            'responses_std_pqi': responses_std_pqi,
            'control_mean_q': control_mean_q,
            'control_std_q': control_std_q,
        }
    

    def predict_metadata_chunked(self, adata: sc.AnnData, chunk_size: int = 128) -> dict[str, np.ndarray]:
        metadata_prediction_chunks_dict = []
        for i in tqdm(range(0, len(adata), chunk_size)):
            first = i
            last = min(len(adata), i + chunk_size)
            if last == first:
                continue
            metadata_prediction_dict = self.predict_metadata(adata[first:last])
            metadata_prediction_chunks_dict.append(metadata_prediction_dict)
        metadata_prediction_dict = dict()
        for key in self.metadata_ontology_infos.keys():
            metadata_prediction_dict[key] = np.concatenate(
                [chunk[key] for chunk in metadata_prediction_chunks_dict], axis=0
            )
        return metadata_prediction_dict

    def predict_metadata(self, adata: sc.AnnData) -> dict[str, np.ndarray]:
        # tokenize the given anndata: no query genes, just metadata (which will be included as query by default)
        tokens_dict, context_indices = self.generate_tokens_from_adata(
            adata=adata,
            obs_index=None,
            query_var_names=[],
        )

        # convert to cuda
        tokens_dict = self.gpt_pipeline.transfer_batch_to_device(tokens_dict, self.device, 0)

        # get model predictions
        metadata_prediction_dict = dict()
        with torch.inference_mode():
            logits_dict = self.gpt_pipeline.model.predict(
                gene_tokens_nc=tokens_dict["gene_tokens_nc"],
                metadata_tokens_n=tokens_dict["metadata_tokens_n"],
                prompt_mask_nc=tokens_dict["prompt_mask_nc"],
            )
            for metadata_key in self.metadata_ontology_infos.keys():
                metadata_logits_nk = logits_dict[metadata_key][:, context_indices[f"query_{metadata_key}"], :]
                metadata_logits_nk = metadata_logits_nk - torch.logsumexp(metadata_logits_nk, dim=1, keepdim=True)
                metadata_probs_nk = torch.exp(metadata_logits_nk)
                metadata_prediction_dict[metadata_key] = metadata_probs_nk.cpu().numpy()

        return metadata_prediction_dict

    def convert_adata_to_metacell(
        self,
        adata: sc.AnnData,
        target_total_mrna_umis: float | None = None,
    ) -> sc.AnnData:
        # make a metacell
        X_meta_g = np.asarray(adata.X.sum(0))

        # set total mrna umis to the mean of the dataset
        if target_total_mrna_umis is None:
            target_total_mrna_umis = adata.obs["total_mrna_umis"].mean()
        else:
            assert target_total_mrna_umis > 0
        X_meta_g = X_meta_g * target_total_mrna_umis / X_meta_g.sum()

        # make a metacell anndata
        adata_meta = adata[0, :].copy()
        adata_meta.X = X_meta_g
        adata_meta.obs["total_mrna_umis"] = [target_total_mrna_umis]

        return adata_meta

    def compute_jacobian(
        self,
        adata: sc.AnnData,
        prompt_gene_ids: list[str],
        query_gene_ids: list[str],
        jacobian_point: t.Literal["actual", "marginal_mean"],
        query_chunk_size: int = 500,
        convert_to_metacell: bool = True,
    ):
        if not convert_to_metacell:
            assert len(adata) == 1, (
                "The provided AnnData has more than one cell. Please set `convert_to_metacell` to True."
            )
            adata_meta = adata.copy()
            print("Total mRNA UMIs in the AnnData: ", adata_meta.obs["total_mrna_umis"].values[0])
        else:
            print(f"The provided AnnData has {len(adata)} cells and will be converted to a metacell ...")
            adata_meta = self.convert_adata_to_metacell(adata)
            print(
                "Total mRNA UMIs in the AnnData after metacell conversion: ",
                adata_meta.obs["total_mrna_umis"].values[0],
            )

        # subset to prompt gene ids
        adata_meta = adata_meta[:, prompt_gene_ids].copy()

        # query var names
        print(f"Number of prompt genes: {len(prompt_gene_ids)}")
        print(f"Number of query genes: {len(query_gene_ids)}")

        with torch.inference_mode():
            print("Calculating marginal mean and std ...")
            prompt_marginal_mean_p, prompt_marginal_std_p = self.get_marginal_mean_std(
                adata=adata_meta,
                query_var_names=prompt_gene_ids,
            )
            query_marginal_mean_q, query_marginal_std_q = self.get_marginal_mean_std(
                adata=adata_meta,
                query_var_names=query_gene_ids,
            )

        # jacobian point
        adata_meta.layers["original"] = adata_meta.X.copy()
        if jacobian_point == "actual":
            print("Calculating the Jacobian at the actual metacell point ...")
        elif jacobian_point == "marginal_mean":
            # inject into the adata
            print("Calculating the Jacobian at the marginal mean point ...")
            adata_meta.X = prompt_marginal_mean_p.cpu().numpy()[None, :]
        else:
            raise ValueError

        query_var_names_chunks = list(chunked(query_gene_ids, query_chunk_size))
        jacobian_chunks = []

        print("Calculating the Jacobian ...")

        prompt_gene_values_g = torch.tensor(adata_meta.X[0], device=self.gpt_pipeline.device, dtype=torch.float32)

        for query_var_names_chunk in tqdm(query_var_names_chunks):

            def _wrapped_snap_to_marginal_mean_manifold(prompt_gene_values_g: torch.Tensor) -> torch.Tensor:
                gene_marginal_mean_q, _ = self.get_marginal_mean_std(
                    adata=adata_meta, query_var_names=query_var_names_chunk, prompt_gene_values_g=prompt_gene_values_g
                )
                return gene_marginal_mean_q

            chunk_jacobian_qg = torch.autograd.functional.jacobian(
                func=_wrapped_snap_to_marginal_mean_manifold,
                inputs=prompt_gene_values_g,
                create_graph=False,
                vectorize=False,
            )

            jacobian_chunks.append(chunk_jacobian_qg)

        jacobian_qp = torch.cat(jacobian_chunks, dim=0)

        return {
            'adata_obs': adata_meta.obs,
            'jacobian_point': jacobian_point,
            'query_var_names': query_gene_ids,
            'prompt_var_names': prompt_gene_ids,
            'jacobian_qp': jacobian_qp,
            'adata_meta_gene_values_p': np.asarray(adata_meta.layers['original']).flatten(),
            'prompt_marginal_mean_p': prompt_marginal_mean_p,
            'prompt_marginal_std_p': prompt_marginal_std_p,
            'query_marginal_mean_q': query_marginal_mean_q,
            'query_marginal_std_q': query_marginal_std_q,
        }
