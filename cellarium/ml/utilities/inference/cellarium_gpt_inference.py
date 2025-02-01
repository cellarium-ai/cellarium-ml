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
from cellarium.ml import CellariumModule, CellariumPipeline
from cellarium.ml.models.cellarium_gpt import PredictTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc

from scipy.stats import linregress
from scipy.linalg import eigh


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a handler
handler = logging.StreamHandler()

# Create and set a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


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
            attention_backend: str = "mem_efficient"):
        
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
            context_indices[f'query_{metadata_key}'] = n_query_vars + n_prompt_vars + offset
            offset += 1

        # return gene_tokens_dict, metadata_tokens_dict
        tokenizer_output = self.predict_tokenizer(
            metadata_tokens_n=metadata_tokens_dict,
            metadata_prompt_masks_n=expanded_metadata_prompt_masks_dict,
            gene_tokens_nc=gene_tokens_dict,
            gene_prompt_mask_nc=gene_prompt_mask_nc,
        )

        return tokenizer_output, context_indices

    def generate_gene_tokens_by_metadata(
        self,
        assay: str,
        suspension_type: str,
        prompt_metadata_dict: dict,
        total_mrna_umis: int,
        query_gene_ids: list[list],
        perturb_gene_ids: list[str] | None,
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

        if perturb_gene_ids is None:
            n_cells = 1
        else:
            n_cells = len(perturb_gene_ids) + 1

        # Generate a placeholder AnnData. There is always only gene at the prompt, which will
        # be replaced with the gene to be perturbed. If no perturbation is required, the gene
        # will be set to the first gene in the gene ID dictionary.
        obs_df = pd.DataFrame({key: [value] * n_cells for key, value in metadata_dict.items()})
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
        tokens_dict['prompt_mask_nc'][0, 0] = False
        tokens_dict['gene_tokens_nc']['gene_value'][0, context_indices['prompt_genes'], 1] = 1  # Queried

        # In subsequent cells, prompt genes are set to 0 sequentially
        if perturb_gene_ids is not None:
            assert all(gene_id in self.var_name_to_index_map for gene_id in perturb_gene_ids)
            tokens_dict['gene_tokens_nc']['gene_id'] = tokens_dict['gene_tokens_nc']['gene_id'].clone()
            tokens_dict['gene_tokens_nc']['gene_id'][1:, 0] = torch.tensor([
                self.var_name_to_index_map[var_name] for var_name in perturb_gene_ids])

        return tokens_dict, context_indices, pert_adata, metadata_prompt_masks_dict

    def get_marginal_mean_std(
            self,
            adata: sc.AnnData,
            query_var_names: list[str],
            query_total_mrna_umis: float | None = None,
            prompt_gene_values_g: torch.Tensor | None = None,
            metadata_prompt_masks_dict: dict[str, bool] | None = None
        ) -> t.Tuple[torch.Tensor, torch.Tensor]:
    
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

    def get_marginal_mean_std_from_tokens(
            self,
            tokens_dict: dict,
            context_indices: dict,
            use_logsumexp: bool = True,
            max_counts: int | None = None,
            verbose: bool = False,
        ) -> t.Tuple[torch.Tensor, torch.Tensor]:
    
        # convert to cuda
        if verbose:
            logger.info("Transferring the tokens to the device ...")

        tokens_dict = self.gpt_pipeline.transfer_batch_to_device(tokens_dict, self.device, 0)

        if verbose:
            logger.info("Done.")
        
        # get model predictions
        if verbose:
            logger.info("Predicting ...")

        logits_dict = self.gpt_pipeline.model.predict(
            gene_tokens_nc=tokens_dict["gene_tokens_nc"],
            metadata_tokens_n=tokens_dict["metadata_tokens_n"],
            prompt_mask_nc=tokens_dict["prompt_mask_nc"],
            predict_keys=['gene_value']
        )

        if verbose:
            logger.info("Done.")

        # note: we use `q` to denote query genes
        if verbose:
            logger.info("Calculating marginal mean and std ...")
        
        if verbose:
            logger.info("Obtaining gene logits ...")

        query_gene_indices = torch.tensor(context_indices['query_genes'], device=self.device, dtype=torch.int64)

        if max_counts is None:
            gene_logits_nqk = logits_dict['gene_value'][:, query_gene_indices, :]
            max_counts = gene_logits_nqk.shape[-1]
        else:
            assert max_counts > 0
            gene_logits_nqk = logits_dict['gene_value'][:, query_gene_indices, :max_counts]

        if verbose:
            logger.info("Done.")

        if verbose:
            logger.info("Normalizing gene logits ...")

        gene_logits_nqk = gene_logits_nqk - torch.logsumexp(gene_logits_nqk, dim=-1, keepdim=True)

        if verbose:
            logger.info("Done.")

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

        if verbose:
            logger.info("Done.")

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

        def yield_chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        query_var_names_chunks = list(yield_chunks(query_gene_ids, query_chunk_size))
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
    
    def __str__(self) -> str:
        return (
            f"GeneNetworkAnalysisBase({self.cell_type}, {self.tissue}, {self.disease}, {self.development_stage}, "
            f"n_umi={self.total_mrna_umis:.2f})"
        )

    __repr__ = __str__

    def process(
            self,
            response_normalization_strategy: t.Literal["mean", "std", "none"] = "mean",
            feature_normalization_strategy: t.Literal["l2", "query_z_score", "prompt_z_score"] = "query_z_score",
            min_prompt_gene_tpm: float = 10.,
            min_query_gene_tpm: float = 10.,
            query_response_amp_min_pct: float | None = None,
            feature_max_value: float = 10.,
            eps: float = 1e-8,
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
        self.z_qp = self.response_qp * (z_p[None, :] + eps) / (z_q[:, None] + eps)

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
            
        # apply the mask to everything else
        self.prompt_var_names = [self.prompt_var_names[i] for i in range(len(self.prompt_var_names)) if self.mask_p[i]]
        # self.prompt_empirical_mean_p = self.prompt_empirical_mean_p[self.mask_p]
        self.prompt_marginal_mean_p = self.prompt_marginal_mean_p[self.mask_p]
        self.prompt_marginal_std_p = self.prompt_marginal_std_p[self.mask_p]

        self.query_var_names = [self.query_var_names[i] for i in range(len(self.query_var_names)) if self.mask_q[i]]
        # self.query_empirical_mean_q = self.query_empirical_mean_q[self.mask_q]
        self.query_marginal_mean_q = self.query_marginal_mean_q[self.mask_q]
        self.query_marginal_std_q = self.query_marginal_std_q[self.mask_q]

        # apply the mask to z_qp
        self.z_qp = self.z_qp[self.mask_q, :][:, self.mask_p]

        if feature_normalization_strategy == "prompt_z_score":
            self.z_qp = (self.z_qp - np.mean(self.z_qp, axis=0, keepdims=True)) / (
                np.sqrt(self.z_qp.shape[0]) * (eps + np.std(self.z_qp, axis=0, keepdims=True)))
        elif feature_normalization_strategy == "query_z_score":
            self.z_qp = (self.z_qp - np.mean(self.z_qp, axis=1, keepdims=True)) / (
                np.sqrt(self.z_qp.shape[1]) * (eps + np.std(self.z_qp, axis=1, keepdims=True)))
        elif feature_normalization_strategy == "l2":
            self.z_qp = self.z_qp / (eps + np.linalg.norm(self.z_qp, axis=-1, keepdims=True))
        else:
            raise ValueError("Invalid feature normalization strategy")

        # clip features
        self.z_qp[np.isnan(self.z_qp)] = 0.
        self.z_qp[np.isinf(self.z_qp)] = 0.
        self.z_qp = np.clip(self.z_qp, -feature_max_value, feature_max_value)

        self.processed = True

        # adj
        self.a_qq = None
        
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

        if adjacency_strategy == "shifted_correlation":
            assert "beta" in kwargs, "Must provide beta for shifted correlation"
            beta = kwargs["beta"]
            a_qq = np.power(0.5 * (1 + np.dot(self.z_qp, self.z_qp.T)), beta)
        elif adjacency_strategy == "unsigned_correlation":
            assert "beta" in kwargs, "Must provide beta for unsigned correlation"
            beta = kwargs["beta"]
            a_qq = np.power(np.abs(np.dot(self.z_qp, self.z_qp.T)), beta)
        elif adjacency_strategy == "positive_correlation":
            assert "beta" in kwargs, "Must provide beta for positive correlation"
            beta = kwargs["beta"]
            a_qq = np.power(np.maximum(0, np.dot(self.z_qp, self.z_qp.T)), beta)
        elif adjacency_strategy == "positive_correlation_binary":
            assert n_neighbors is None, "n_neighbors must be None for binary adjacency"
            assert "tau" in kwargs, "Must provide correlation threshold for binary adjacency"
            tau = kwargs["tau"]
            a_qq = (np.maximum(0, np.dot(self.z_qp, self.z_qp.T)) > tau).astype(float)
        else:
            raise ValueError("Invalid adjacency strategy")

        assert np.isclose(a_qq, a_qq.T).all(), "Adjacency matrix must be symmetric -- something is wrong!"

        if n_neighbors is not None:
            assert n_neighbors > 0, "n_neighbors must be positive"
            t_qq = np.argsort(a_qq, axis=-1)[:, -n_neighbors:]  # take the top n_neighbors

            # make a mask for the top n_neighbors
            _a_qq = np.zeros_like(a_qq)
            for q in range(a_qq.shape[0]):
                _a_qq[q, t_qq[q]] = a_qq[q, t_qq[q]]
                _a_qq[t_qq[q], q] = a_qq[t_qq[q], q]
        else:
            _a_qq = a_qq
        a_qq = _a_qq
        
        if not self_loop:
            np.fill_diagonal(a_qq, 0)

        self.a_qq = a_qq

    def _compute_igraph_from_adjacency(self, directed: bool = False) -> ig.Graph:
        assert self.a_qq is not None, "Must compute adjacency matrix first"
        sources, targets = self.a_qq.nonzero()
        weights = self.a_qq[sources, targets]
        g = ig.Graph(directed=directed)
        g.add_vertices(self.a_qq.shape[0])  # this adds adjacency.shape[0] vertices
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
        assert self.a_qq is not None, "Must compute adjacency matrix first"
        
        # calculate normalized laplacian and its eigenvalues
        norm_q = 1. / (1e-9 + np.sqrt(self.a_qq.sum(0)))
        lap_qq = np.eye(self.a_qq.shape[0]) - norm_q[:, None] * norm_q[None, :] * self.a_qq
        eigs = eigh(lap_qq.astype(np.float64), eigvals_only=True)
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
            self.z_qp,
            device=device,
            verbose=verbose,
            n_neighbors=n_neighbors,
            repulsive_fraction=repulsive_fraction,
            attractive_penalty=attractive_penalty,
            repulsive_penalty=repulsive_penalty,
            **kwargs)

        self.embedding_q2 = mde.embed(verbose=verbose, max_iter=max_iter).cpu().numpy()

    def plot_mde_embedding(
            self,
            marker_size: int = 2,
            highlight_marker_size: int = 4,
            width: int = 800,
            height: int = 800,
            highlight_gene_sets: dict[str, t.Tuple[list[str], list[str], str]] | None = None,
        ) -> go.Figure:

        assert self.embedding_q2 is not None, "Must compute MDE embedding first"
        assert self.leiden_membership is not None, "Must compute Leiden communities first"

        plot_title = f"""{self.cell_type}<br>{self.tissue}<br>{self.disease}"""
        
        # Create a color map for the memberships
        memberships_q = self.leiden_membership
        unique_memberships = np.unique(memberships_q)

        # Convert memberships to strings for categorical mapping
        unique_memberships_str = unique_memberships.astype(str)

        # Create the color map with string keys
        colormap = {str(label): cc.glasbey[i % len(cc.glasbey)] for i, label in enumerate(unique_memberships)}

        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'x': self.embedding_q2[:, 0],
            'y': self.embedding_q2[:, 1],
            'label': self.query_gene_symbols,
            'membership': memberships_q.astype(str)  # Convert to string
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
                query_gene_indices = [self.query_gene_id_to_idx_map[gene_id] for gene_id in gene_ids]

                # show a scatter plot and color the markers in red
                fig.add_scatter(
                    x=self.embedding_q2[query_gene_indices, 0],
                    y=self.embedding_q2[query_gene_indices, 1],
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
