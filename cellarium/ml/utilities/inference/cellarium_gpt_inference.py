import os
import logging
import warnings
import torch
import pymde
import igraph as ig
import leidenalg
import numpy as np
import typing as t
import scanpy as sc
import pandas as pd
from scanpy import AnnData
from tqdm.notebook import tqdm
from functools import cached_property
from more_itertools import chunked

from cellarium.ml import CellariumModule, CellariumPipeline
from cellarium.ml.models.cellarium_gpt import PredictTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc

from scipy.stats import linregress
from scipy.linalg import eigh

import typing as t


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a handler
handler = logging.StreamHandler()

# Create and set a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        self.gpt_pipeline = CellariumModule.load_from_checkpoint(cellarium_gpt_ckpt_path, map_location=device)

        # inject gene categories
        self.gpt_pipeline.model.gene_categories = np.asarray(self._adata.var_names)
        
        # change attention backend to memory efficient
        self.gpt_pipeline.model.set_attention_backend(attention_backend)

        # gene info related
        self.model_var_names = np.asarray(self._adata.var_names)
        self.model_var_names_set = set(self.model_var_names)
        self.var_name_to_index_map = {var_name: i for i, var_name in enumerate(self.model_var_names)}
        self.gene_info_df, self.gene_symbol_to_gene_id_map, self.gene_id_to_gene_symbol_map = \
            load_gene_info_table(gene_info_tsv_path, self.model_var_names)
        
        # get the metadata ontology infos from the TrainTokenizer, which is the first step of the pipeline
        self.metadata_ontology_infos = self.gpt_pipeline.pipeline[0].ontology_infos

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
        gene_ontology_infos = dict()

        gene_ontology_infos["assay_ontology_term_id"] = dict()
        gene_ontology_infos["assay_ontology_term_id"]["names"] = list(adata.obs['assay_ontology_term_id'].cat.categories)
        gene_ontology_infos["assay_ontology_term_id"]["labels"] = list(adata.obs['assay'].cat.categories)

        gene_ontology_infos["suspension_type"] = dict()
        gene_ontology_infos["suspension_type"]["names"] = list(adata.obs['suspension_type'].cat.categories)
        gene_ontology_infos["suspension_type"]["labels"] = list(adata.obs['suspension_type'].cat.categories)

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
        gene_ids_nc = torch.tensor(
            prompt_var_index + query_var_index,
            dtype=torch.int64, device=cpu_device)[None, :].expand(n_cells, n_total_vars)
        
        # gene prompt mask
        gene_prompt_mask_nc = torch.tensor(
            [1] * n_prompt_vars + [0] * n_query_vars,
            dtype=torch.bool, device=cpu_device)[None, :].expand(n_cells, n_total_vars)
        
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
            adata.obs["total_mrna_umis"].values,
            dtype=torch.float32, device=cpu_device)[:, None].expand(n_cells, n_prompt_vars)
        if query_total_mrna_umis is None:
            # the same as prompt
            query_total_mrna_umis_nc = torch.tensor(
                adata.obs["total_mrna_umis"].values,
                dtype=torch.float32, device=cpu_device)[:, None].expand(n_cells, n_query_vars)
        else:
            query_total_mrna_umis_nc = torch.tensor(
                [query_total_mrna_umis] * n_cells,
                dtype=torch.float32, device=cpu_device)[:, None].expand(n_cells, n_query_vars)
        total_mrna_umis_nc = torch.cat([prompt_total_mrna_umis_nc, query_total_mrna_umis_nc], dim=1)

        # convert assay and suspension_type to codes
        assay_nc = torch.tensor(
            pd.Categorical(
                adata.obs["assay_ontology_term_id"].values,
                categories=self.gene_ontology_infos["assay_ontology_term_id"]["names"]).codes,
            dtype=torch.int64, device=cpu_device)[:, None].expand(n_cells, n_total_vars)
        suspension_type_nc = torch.tensor(
            pd.Categorical(
                adata.obs["suspension_type"].values,
                categories=self.gene_ontology_infos["suspension_type"]["names"]).codes,
            dtype=torch.int64, device=cpu_device)[:, None].expand(n_cells, n_total_vars)

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
                [metadata_prompt_masks_dict[key]] * n_cells, dtype=torch.bool, device=cpu_device)
        
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
        context_indices['prompt_genes'] = np.arange(0, n_prompt_vars).tolist()
        context_indices['query_genes'] = np.arange(n_prompt_vars, n_query_vars + n_prompt_vars).tolist()
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
        query_gene_ids: list[list],
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
        query_gene_ids: list[list],
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
            metadata_key: self.metadata_ontology_infos[metadata_key]["labels"][0]
            for metadata_key in METADATA_KEYS
        }

        # True if provided, False otherwise    
        metadata_prompt_masks_dict = {
            metadata_key: metadata_key in prompt_metadata_dict
            for metadata_key in METADATA_KEYS
        }

        # Generate a complete metadata dictionary, including default values for missing keys
        metadata_dict = {
            metadata_key: prompt_metadata_dict.get(metadata_key, default_metadata_dict[metadata_key])
            for metadata_key in METADATA_KEYS
        }
        metadata_dict |= {
            "assay": assay,
            "suspension_type": suspension_type,
            "total_mrna_umis": total_mrna_umis
        }

        # Augment metadata dictionary with ontology term IDs
        metadata_dict |= {
            f"{metadata_key}_ontology_term_id": self.metadata_ontology_infos[metadata_key]["names"][
                self.metadata_ontology_infos[metadata_key]["labels"].index(metadata_dict[metadata_key])]
            for metadata_key in METADATA_KEYS
        }
        metadata_dict |= {
            "assay_ontology_term_id": self.gene_ontology_infos["assay_ontology_term_id"]["names"][
                self.gene_ontology_infos["assay_ontology_term_id"]["labels"].index(assay)]
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
        prompt_gene_log1p_values_g = tokens_dict['gene_tokens_nc']['gene_value'][
            FIRST_CELL_DIM, context_indices['prompt_genes'], GENE_VALUE_DIM]
        
        # this is the "source", in case it is provided, e.g., for Jacobian calculation
        if prompt_gene_values_g is None:
            prompt_gene_values_g = torch.expm1(prompt_gene_log1p_values_g).clone()
        
        # inject back to tokens_dict to re-establish the reference for Jacobian calculation
        tokens_dict['gene_tokens_nc']['gene_value'][
            FIRST_CELL_DIM, context_indices['prompt_genes'], GENE_VALUE_DIM] = torch.log1p(prompt_gene_values_g)

        # get model predictions
        logits_dict = self.gpt_pipeline.model.predict(
            gene_tokens_nc=tokens_dict["gene_tokens_nc"],
            metadata_tokens_n=tokens_dict["metadata_tokens_n"],
            prompt_mask_nc=tokens_dict["prompt_mask_nc"],
        )

        # note: we use `q` to denote query genes
        gene_logits_qk = logits_dict['gene_value'][FIRST_CELL_DIM, context_indices['query_genes'], :]
        gene_logits_qk = gene_logits_qk - torch.logsumexp(gene_logits_qk, dim=-1, keepdim=True)
        MAX_COUNTS = gene_logits_qk.shape[-1]
        log_counts_1_k = torch.arange(0, MAX_COUNTS, device=gene_logits_qk.device).log()
        log_counts_2_k = torch.arange(0, MAX_COUNTS, device=gene_logits_qk.device).pow(2).log()
        gene_mom_1_q = torch.logsumexp(gene_logits_qk + log_counts_1_k[None, :], dim=-1).exp()
        gene_mom_2_q = torch.logsumexp(gene_logits_qk + log_counts_2_k[None, :], dim=-1).exp()
        gene_marginal_means_q = gene_mom_1_q
        gene_marginal_std_q = torch.clamp(gene_mom_2_q - gene_mom_1_q.pow(2), 0.).sqrt()

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
            predict_keys=['gene_value']
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
            gene_marginal_std_nq = torch.clamp(gene_mom_2_nq - gene_mom_1_nq.pow(2), 0.).sqrt()
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
            metadata_prediction_dict[key] = np.concatenate([
                chunk[key] for chunk in metadata_prediction_chunks_dict], axis=0)   
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
                prompt_mask_nc=tokens_dict["prompt_mask_nc"])
            for metadata_key in self.metadata_ontology_infos.keys():
                metadata_logits_nk = logits_dict[metadata_key][:, context_indices[f'query_{metadata_key}'], :]
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
            target_total_mrna_umis = adata.obs['total_mrna_umis'].mean()
        else:
            assert target_total_mrna_umis > 0
        X_meta_g = X_meta_g * target_total_mrna_umis / X_meta_g.sum()

        # make a metacell anndata
        adata_meta = adata[0, :].copy()
        adata_meta.X = X_meta_g
        adata_meta.obs['total_mrna_umis'] = [target_total_mrna_umis]

        return adata_meta


    def compute_jacobian(
            self,
            adata: sc.AnnData,
            prompt_gene_ids: list[str],
            query_gene_ids: list[str],
            jacobian_point: t.Literal["actual", "marginal_mean"],
            query_chunk_size: int = 500,
            convert_to_metacell: bool = True
        ):

        if not convert_to_metacell:
            assert len(adata) == 1, "The provided AnnData has more than one cell. Please set `convert_to_metacell` to True."
            adata_meta = adata.copy()
            print("Total mRNA UMIs in the AnnData: ", adata_meta.obs['total_mrna_umis'].values[0])
        else:
            print(f"The provided AnnData has {len(adata)} cells and will be converted to a metacell ...")
            adata_meta = self.convert_adata_to_metacell(adata)
            print("Total mRNA UMIs in the AnnData after metacell conversion: ", adata_meta.obs['total_mrna_umis'].values[0])


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
        adata_meta.layers['original'] = adata_meta.X.copy()
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

        prompt_gene_values_g = torch.tensor(
            adata_meta.X[0], device=self.gpt_pipeline.device, dtype=torch.float32)

        for query_var_names_chunk in tqdm(query_var_names_chunks):

            def _wrapped_snap_to_marginal_mean_manifold(prompt_gene_values_g: torch.Tensor) -> torch.Tensor:
                gene_marginal_mean_q, _ = self.get_marginal_mean_std(
                    adata=adata_meta,
                    query_var_names=query_var_names_chunk,
                    prompt_gene_values_g=prompt_gene_values_g)
                return gene_marginal_mean_q

            chunk_jacobian_qg = torch.autograd.functional.jacobian(
                func=_wrapped_snap_to_marginal_mean_manifold, 
                inputs=prompt_gene_values_g,
                create_graph=False,
                vectorize=False)

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


def quantile_normalize_select(
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int,
        top_k: int,
        min_x: int | None = None,
        max_x: int | None = None):
    """
    Filter, normalize, and select top_k elements.
    
    Parameters:
    -----------
    x : np.ndarray
        1D numpy array of covariate values.
    y : np.ndarray
        1D numpy array of response values.
    n_bins : int
        Number of bins to subdivide the range [min_x, max_x].
    top_k : int
        Number of top largest normalized y values to select.
    min_x : float
        Minimum x value (x must be > min_x).
    max_x : float
        Maximum x value (x must be < max_x).

    Returns:
    --------
    selected_indices : np.ndarray
        Indices in the original arrays corresponding to the top_k selected values.
    top_normalized_y : np.ndarray
        The normalized y values corresponding to these selected indices.
    """
    
    # Create bin edges (n_bins bins from min_x to max_x).
    bin_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)

    # Assign each x_valid to a bin.
    # np.digitize returns bin indices in 1..n_bins, so subtract 1 to have 0-indexed bins.
    bin_indices = np.digitize(x, bin_edges) - 1

    # Prepare an array for the normalized y values.
    y_normalized = np.empty_like(y, dtype=float)

    # Process each bin separately.
    for i in range(n_bins):
        # Find indices in x_valid that fall in bin i.
        in_bin = np.where(bin_indices == i)[0]
        if in_bin.size > 0:
            # Compute the mean of y values in this bin.
            bin_mean = np.mean(y[in_bin])
            # Avoid division by zero (if bin_mean happens to be zero, leave values unchanged).
            if bin_mean == 0:
                y_normalized[in_bin] = y[in_bin]
            else:
                y_normalized[in_bin] = y[in_bin] / bin_mean

    if min_x is None:
        min_x = np.min(x)
    if max_x is None:
        max_x = np.max(x)

    _y_normalized = y_normalized.copy()
    _y_normalized[x < min_x] = -np.inf
    _y_normalized[x > max_x] = -np.inf

    sorted_idx = np.argsort(_y_normalized)[::-1]
    top_k = min(top_k, np.sum(_y_normalized > -np.inf))
    top_idx = sorted_idx[:top_k]

    return top_idx, y_normalized


class GeneNetworkAnalysisBase:
    def __init__(
            self,
            adata_obs: pd.DataFrame,
            gene_info_tsv_path: str,
            query_var_names: list[str],
            prompt_var_names: list[str],
            response_qp: np.ndarray,
            prompt_marginal_mean_p: np.ndarray,
            prompt_marginal_std_p: np.ndarray,
            query_marginal_mean_q: np.ndarray,
            query_marginal_std_q: np.ndarray,
            verbose: bool = True):
        
        self.verbose = verbose

        n_query_vars = len(query_var_names)
        n_prompt_vars = len(prompt_var_names)

        assert response_qp.shape == (n_query_vars, n_prompt_vars)
        assert prompt_marginal_mean_p.shape == (n_prompt_vars,)
        assert prompt_marginal_std_p.shape == (n_prompt_vars,)
        assert query_marginal_mean_q.shape == (n_query_vars,)
        assert query_marginal_std_q.shape == (n_query_vars,)

        self.adata_obs = adata_obs
        self.query_var_names = query_var_names
        self.prompt_var_names = prompt_var_names
        self.response_qp = response_qp
        self.prompt_marginal_mean_p = prompt_marginal_mean_p
        self.prompt_marginal_std_p = prompt_marginal_std_p
        self.query_marginal_mean_q = query_marginal_mean_q
        self.query_marginal_std_q = query_marginal_std_q

        self.gene_info_df, self.gene_symbol_to_gene_id_map, self.gene_id_to_gene_symbol_map = \
            load_gene_info_table(gene_info_tsv_path, query_var_names + prompt_var_names)

        self.processed = False

    @property
    def cell_type(self) -> str:
        return self.adata_obs['cell_type'].values[0]
    
    @property
    def tissue(self) -> str:
        return self.adata_obs['tissue'].values[0]

    @property
    def disease(self) -> str:
        return self.adata_obs['disease'].values[0]

    @property
    def development_stage(self) -> str:
        return self.adata_obs['development_stage'].values[0]

    @property
    def sex(self) -> str:
        return self.adata_obs['sex'].values[0]

    @property
    def total_mrna_umis(self) -> float:
        return self.adata_obs['total_mrna_umis'].values[0]

    @property
    def query_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.query_var_names]
    
    @property
    def prompt_gene_symbols(self) -> list[str]:
        return [self.gene_id_to_gene_symbol_map[gene_id] for gene_id in self.prompt_var_names]
    
    @cached_property
    def query_gene_id_to_idx_map(self) -> dict[str, int]:
        assert self.processed, "Must process before accessing"
        return {gene_id: idx for idx, gene_id in enumerate(self.query_var_names)}
    
    @cached_property
    def prompt_gene_id_to_idx_map(self) -> dict[str, int]:
        assert self.processed, "Must process before accessing"
        return {gene_id: idx for idx, gene_id in enumerate(self.prompt_var_names)}

    def __str__(self) -> str:
        return (
            f"GeneNetworkAnalysisBase({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__

    def process(
            self,
            response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
            feature_normalization_strategy: t.Literal["l2", "z_score", "none"] = "z_score",
            min_prompt_gene_tpm: float = 0.,
            min_query_gene_tpm: float = 0.,
            query_response_amp_min_pct: float | None = None,
            feature_max_value: float | None = None,
            norm_pseudo_count: float = 1e-3,
            query_hv_top_k: int | None = None,
            query_hv_n_bins: int | None = 50,
            query_hv_min_x: float | None = 1e-2,
            query_hv_max_x: float | None = np.inf,
            eps: float = 1e-8,
            z_trans_func: t.Callable[[np.ndarray], np.ndarray] | None = None,
        ) -> None:

        assert not self.processed, "Already processed -- please create a new instance"
        if response_normalization_strategy == "mean":
            z_p = self.prompt_marginal_mean_p
            z_q = self.query_marginal_mean_q
        elif response_normalization_strategy == "std":
            z_p = self.prompt_marginal_std_p
            z_q = self.query_marginal_std_q
        elif response_normalization_strategy == "none":
            z_p = np.ones_like(self.prompt_marginal_mean_p)
            z_q = np.ones_like(self.query_marginal_mean_q)
        else:
            raise ValueError("Invalid Jacobian normalization strategy")

        # linear proportional activation
        self.z_qp = self.response_qp * (z_p[None, :] + norm_pseudo_count) / (z_q[:, None] + norm_pseudo_count)

        if self.verbose:
            logger.info(f"Maximum value of z_qp: {np.max(self.z_qp):.3f}")
            logger.info(f"Minimum value of z_qp: {np.min(self.z_qp):.3f}")

        self.mask_q = (1e6 * self.query_marginal_mean_q / self.total_mrna_umis) >= min_query_gene_tpm
        self.mask_p = (1e6 * self.prompt_marginal_mean_p / self.total_mrna_umis) >= min_prompt_gene_tpm

        logger.info(f"Number of query genes after TPM filtering: {np.sum(self.mask_q)} / {len(self.mask_q)}")
        logger.info(f"Number of prompt genes after TPM filtering: {np.sum(self.mask_p)} / {len(self.mask_p)}")
        
        if query_response_amp_min_pct is not None:
            z_norm_q = np.linalg.norm(self.z_qp, axis=-1)
            z_norm_thresh = np.percentile(z_norm_q, query_response_amp_min_pct)
            self.mask_q = self.mask_q & (z_norm_q >= z_norm_thresh)
            logger.info(f"Number of query genes after z-norm filtering: {np.sum(self.mask_q)} / {len(self.mask_q)}")
        
        if query_hv_top_k is not None:
            assert query_hv_n_bins is not None
            assert query_hv_min_x is not None
            assert query_hv_max_x is not None
            top_idx, _ = quantile_normalize_select(
                x=np.log1p(self.query_marginal_mean_q),
                y=np.std(self.z_qp, axis=1),
                n_bins=query_hv_n_bins,
                top_k=query_hv_top_k,
                min_x=query_hv_min_x,
                max_x=query_hv_max_x)
            hv_mask_q = np.zeros_like(self.mask_q, dtype=bool)
            hv_mask_q[top_idx] = True
            self.mask_q = self.mask_q & hv_mask_q
            logger.info(f"Number of query genes after highly-variable filtering: {np.sum(self.mask_q)} / {len(self.mask_q)}")

        # apply the mask to everything else
        self.prompt_var_names = [self.prompt_var_names[i] for i in range(len(self.prompt_var_names)) if self.mask_p[i]]
        self.prompt_marginal_mean_p = self.prompt_marginal_mean_p[self.mask_p]
        self.prompt_marginal_std_p = self.prompt_marginal_std_p[self.mask_p]

        self.query_var_names = [self.query_var_names[i] for i in range(len(self.query_var_names)) if self.mask_q[i]]
        self.query_marginal_mean_q = self.query_marginal_mean_q[self.mask_q]
        self.query_marginal_std_q = self.query_marginal_std_q[self.mask_q]

        # apply the mask to z_qp
        self.z_qp = self.z_qp[self.mask_q, :][:, self.mask_p]

        # clip and transform features
        self.z_qp[np.isnan(self.z_qp)] = 0.
        self.z_qp[np.isinf(self.z_qp)] = 0.

        if feature_max_value is not None:
            assert feature_max_value > 0
            self.z_qp = np.clip(self.z_qp, -feature_max_value, feature_max_value)

        if z_trans_func is not None:
            self.z_qp = z_trans_func(self.z_qp)

        if feature_normalization_strategy == "z_score":
            # z-score each query gene separately in response to prompt genes
            self.z_qp = (self.z_qp - np.mean(self.z_qp, axis=0, keepdims=True)) / (
                eps + np.std(self.z_qp, axis=0, keepdims=True))
        elif feature_normalization_strategy == "l2":
            # l2-normalize query genes separately for each prompt gene
            self.z_qp = self.z_qp / (eps + np.linalg.norm(self.z_qp, axis=0, keepdims=True))
        elif feature_normalization_strategy == "none":
            pass
        else:
            raise ValueError("Invalid feature normalization strategy")

        self.processed = True

        # adj
        self.a_pp = None
        
        # leiden
        self.leiden_membership = None
        
        # spectral analysis
        self.eigs = None
        self.spectral_dim = None


    def compute_adjacency_matrix(
            self,
            adjacency_strategy: str = t.Literal[
                "shifted_correlation", "unsigned_correlation", "positive_correlation", "binary"],
            n_neighbors: int | None = 50,
            self_loop: bool = False,
            **kwargs) -> None:

        n_query_genes = self.z_qp.shape[0]
        rho_pp = self.z_qp.T @ self.z_qp / n_query_genes

        if adjacency_strategy == "shifted_correlation":
            assert "beta" in kwargs, "Must provide beta for shifted correlation"
            beta = kwargs["beta"]
            a_pp = np.power(0.5 * (1 + rho_pp), beta)
        elif adjacency_strategy == "unsigned_correlation":
            assert "beta" in kwargs, "Must provide beta for unsigned correlation"
            beta = kwargs["beta"]
            a_pp = np.power(np.abs(rho_pp), beta)
        elif adjacency_strategy == "positive_correlation":
            assert "beta" in kwargs, "Must provide beta for positive correlation"
            beta = kwargs["beta"]
            a_pp = np.power(np.maximum(0, rho_pp), beta)
        elif adjacency_strategy == "positive_correlation_binary":
            assert n_neighbors is None, "n_neighbors must be None for binary adjacency"
            assert "tau" in kwargs, "Must provide correlation threshold for binary adjacency"
            tau = kwargs["tau"]
            a_pp = (np.maximum(0, rho_pp) > tau).astype(float)
        else:
            raise ValueError("Invalid adjacency strategy")

        assert np.isclose(a_pp, a_pp.T).all(), "Adjacency matrix must be symmetric -- something is wrong!"

        if n_neighbors is not None:
            assert n_neighbors > 0, "n_neighbors must be positive"
            t_pp = np.argsort(a_pp, axis=-1)[:, -n_neighbors:]  # take the top n_neighbors

            # make a mask for the top n_neighbors
            _a_pp = np.zeros_like(a_pp)
            for p in range(a_pp.shape[0]):
                _a_pp[p, t_pp[p]] = a_pp[p, t_pp[p]]
                _a_pp[t_pp[p], p] = a_pp[t_pp[p], p]
        else:
            _a_pp = a_pp
        a_pp = _a_pp
        
        if not self_loop:
            np.fill_diagonal(a_pp, 0)

        self.a_pp = a_pp

    def _compute_igraph_from_adjacency(self, directed: bool = False) -> ig.Graph:
        assert self.a_pp is not None, "Must compute adjacency matrix first"
        sources, targets = self.a_pp.nonzero()
        weights = self.a_pp[sources, targets]
        g = ig.Graph(directed=directed)
        g.add_vertices(self.a_pp.shape[0])  # this adds adjacency.shape[0] vertices
        g.add_edges(list(zip(sources, targets)))
        g.es["weight"] = weights
        return g

    def compute_leiden_communites(
            self,
            resolution: float = 3.0,
            min_community_size: int = 2,
        ):

        g = self._compute_igraph_from_adjacency()
        
        leiden_partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
        
        self.leiden_membership = np.array(leiden_partition.membership)
        
        # remove small communities
        n_leiden = len(np.unique(self.leiden_membership))
        sizes = np.array([np.sum(self.leiden_membership == i) for i in range(n_leiden)])
        for i_leiden in range(n_leiden):
            if sizes[i_leiden] < min_community_size:
                self.leiden_membership[self.leiden_membership == i_leiden] = -1
    
    def compute_spectral_dimension(
            self,
            offset: int = 2,
            n_lambda_for_estimation: int = 5) -> float:
        assert self.a_pp is not None, "Must compute adjacency matrix first"
        
        # calculate normalized laplacian and its eigenvalues
        norm_p = 1. / (1e-9 + np.sqrt(self.a_pp.sum(0)))
        lap_pp = np.eye(self.a_pp.shape[0]) - norm_p[:, None] * norm_p[None, :] * self.a_pp
        eigs = eigh(lap_pp.astype(np.float64), eigvals_only=True)
        eigs[0] = 0
        eigs = np.clip(eigs, 0, np.inf)  # roundoff error guardrail
        self.eigs = eigs

        n_lambda = np.cumsum(eigs)
        n_lambda = n_lambda / n_lambda[-1]
        first_nonzero = np.where(eigs > 0)[0][0] + offset
        xx = np.log(eigs[first_nonzero:first_nonzero + n_lambda_for_estimation])
        yy = np.log(n_lambda[first_nonzero:first_nonzero + n_lambda_for_estimation])

        lin = linregress(xx, yy)
        slope, intercept = lin.slope, lin.intercept
 
        # save a few thigs for later
        self.spectral_dim = 2 * linregress(xx, yy).slope
        self.eigs = eigs
        self.n_lambda = n_lambda
        self.log_eigs_asymptotic = xx
        self.log_n_lambda_asymptotic = yy
        self.spectral_dim_slope = slope
        self.spectral_dim_intercept = intercept

    def make_mde_embedding(
            self,
            n_neighbors: int = 7,
            repulsive_fraction: int = 5,
            attractive_penalty: pymde.functions.function.Function = pymde.penalties.Log1p,
            repulsive_penalty: pymde.functions.function.Function = pymde.penalties.InvPower,
            device: torch.device = torch.device("cpu"),
            max_iter: int = 500,
            verbose: bool = True,
            **kwargs
        ):
        
        mde = pymde.preserve_neighbors(
            self.z_qp.T,  # we are embedding the prompts (perturbations)
            device=device,
            verbose=verbose,
            n_neighbors=n_neighbors,
            repulsive_fraction=repulsive_fraction,
            attractive_penalty=attractive_penalty,
            repulsive_penalty=repulsive_penalty,
            **kwargs)

        self.embedding_p2 = mde.embed(verbose=verbose, max_iter=max_iter).cpu().numpy()

    def plot_mde_embedding(
            self,
            marker_size: int = 2,
            highlight_marker_size: int = 4,
            width: int = 800,
            height: int = 800,
            highlight_gene_sets: dict[str, t.Tuple[list[str], list[str], str]] | None = None,
        ) -> go.Figure:

        assert self.embedding_p2 is not None, "Must compute MDE embedding first"
        assert self.leiden_membership is not None, "Must compute Leiden communities first"

        plot_title = f"""{self.cell_type}<br>{self.tissue}<br>{self.disease}"""
        
        # Create a color map for the memberships
        memberships_p = self.leiden_membership
        unique_memberships = np.unique(memberships_p)

        # Create the color map with string keys
        colormap = {str(label): cc.glasbey[i % len(cc.glasbey)] for i, label in enumerate(unique_memberships)}

        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'x': self.embedding_p2[:, 0],
            'y': self.embedding_p2[:, 1],
            'label': self.prompt_gene_symbols,
            'membership': memberships_p.astype(str)  # Convert to string
        })

        # Create the scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            hover_name='label',
            title=plot_title,
            color='membership',
            color_discrete_map=colormap
        )

        # Update marker size
        fig.update_traces(marker=dict(size=marker_size))  # Adjust the size as needed

        if highlight_gene_sets is not None:
            for gene_set_name, (gene_ids, gene_symbols, color) in highlight_gene_sets.items():
                prompt_gene_indices = [self.prompt_gene_id_to_idx_map[gene_id] for gene_id in gene_ids]

                # show a scatter plot and color the markers in red
                fig.add_scatter(
                    x=self.embedding_p2[prompt_gene_indices, 0],
                    y=self.embedding_p2[prompt_gene_indices, 1],
                    mode='markers',
                    marker=dict(color=color, size=highlight_marker_size),
                    text=gene_symbols,
                    showlegend=True,
                    name=gene_set_name
                )

        # Update layout to decrease the width of the plot
        fig.update_layout(
            width=width,  # Adjust the width as needed
            height=height,
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                title='MDE_1'
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                title='MDE_2'
            )
        )

        return fig

    def plot_spectral_dimension(self, ax: plt.Axes) -> None:
        assert self.eigs is not None, "Must compute spectral dimension first"
        ax.scatter(self.log_eigs_asymptotic, self.log_n_lambda_asymptotic)
        ax.plot(
            self.log_eigs_asymptotic,
            self.spectral_dim_slope * self.log_eigs_asymptotic + self.spectral_dim_intercept,
            color='red',
            label=f"$d_S$ = {self.spectral_dim:.2f}")
        ax.set_xlabel("ln $\lambda$")
        ax.set_ylabel("ln N($\lambda$)")
        ax.set_title(self.cell_type)
        ax.legend()


class JacobianContext(GeneNetworkAnalysisBase):
    def __init__(
            self,
            adata_obs: pd.DataFrame,
            gene_info_tsv_path: str,
            jacobian_point: str,
            query_var_names: list[str],
            prompt_var_names: list[str],
            jacobian_qp: np.ndarray,
            prompt_empirical_mean_p: np.ndarray,
            query_empirical_mean_q: np.ndarray,
            prompt_marginal_mean_p: np.ndarray,
            prompt_marginal_std_p: np.ndarray,
            query_marginal_mean_q: np.ndarray,
            query_marginal_std_q: np.ndarray,
            verbose: bool = True):

        super().__init__(
            adata_obs=adata_obs,
            gene_info_tsv_path=gene_info_tsv_path,
            query_var_names=query_var_names,
            prompt_var_names=prompt_var_names,
            response_qp=jacobian_qp,
            prompt_marginal_mean_p=prompt_marginal_mean_p,
            prompt_marginal_std_p=prompt_marginal_std_p,
            query_marginal_mean_q=query_marginal_mean_q,
            query_marginal_std_q=query_marginal_std_q,
            verbose=verbose)
        
        n_query_vars = len(query_var_names)
        n_prompt_vars = len(prompt_var_names)

        assert prompt_empirical_mean_p.shape == (n_prompt_vars,)
        assert query_empirical_mean_q.shape == (n_query_vars,)

        self.jacobian_point = jacobian_point
        self.prompt_empirical_mean_p = prompt_empirical_mean_p
        self.query_empirical_mean_q = query_empirical_mean_q
    
    @staticmethod
    def from_old_jacobian_pt_dump(
            jacobian_pt_path: str,
            adata_path: str,
            gene_info_tsv_path: str) -> 'JacobianContext':
        
        # suppres FutureWarning in a context manager
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            adata = sc.read_h5ad(adata_path)
            old_jac_dict = torch.load(jacobian_pt_path)

        # make a metacell
        X_meta_g = np.asarray(adata.X.sum(0))

        # set total mrna umis to the mean of the dataset
        target_total_mrna_umis = adata.obs['total_mrna_umis'].mean()
        X_meta_g = X_meta_g * target_total_mrna_umis / X_meta_g.sum()

        # make a metacell anndata
        adata_meta = adata[0, :].copy()
        adata_meta.X = X_meta_g
        adata_meta.obs['total_mrna_umis'] = [target_total_mrna_umis]

        prompt_empirical_mean_p = adata_meta[0, old_jac_dict['prompt_var_names']].X.flatten()
        query_empirical_mean_q = adata_meta[0, old_jac_dict['query_var_names']].X.flatten()

        return JacobianContext(
            adata_obs=adata_meta.obs,
            gene_info_tsv_path=gene_info_tsv_path,
            jacobian_point=old_jac_dict['jacobian_point'],
            query_var_names=old_jac_dict['query_var_names'],
            prompt_var_names=old_jac_dict['prompt_var_names'],
            jacobian_qp=old_jac_dict['jacobian_qg'].cpu().numpy(),
            prompt_empirical_mean_p=prompt_empirical_mean_p,
            query_empirical_mean_q=query_empirical_mean_q,
            prompt_marginal_mean_p=old_jac_dict['prompt_marginal_dict']['gene_marginal_means_q'].cpu().numpy(),
            prompt_marginal_std_p=old_jac_dict['prompt_marginal_dict']['gene_marginal_std_q'].cpu().numpy(),
            query_marginal_mean_q=old_jac_dict['query_marginal_dict']['gene_marginal_means_q'].cpu().numpy(),
            query_marginal_std_q=old_jac_dict['query_marginal_dict']['gene_marginal_std_q'].cpu().numpy(),
        )

    def __str__(self) -> str:
        return (
            f"JacobianContext({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__
