# %%
import os
import re
import argparse
import math
import torch
import random
import pickle
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from braceexpand import braceexpand
from tqdm import tqdm
import multiprocessing as mp
from copy import copy
from scipy.sparse import csr_matrix
from datetime import datetime

# for flex attention
import torch._dynamo
import torch.multiprocessing as mp 
torch._dynamo.config.suppress_errors = True

sc.set_figure_params(figsize=(4, 4))

from cellarium.ml.utilities.inference.cellarium_gpt_inference import \
    CellariumGPTInferenceContext, \
    GeneNetworkAnalysisBase

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--target_assay', type=str, required=True)
    #args = parser.parse_args()
    target_assay = '10x Chromium (v3)'

# %%
    ROOT_PATH = "/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm"

    ADATA_FP = '/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm/data/pbmc_adata.h5ad'
    REF_ADATA_FP = '/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm/data/extract_0.h5ad'
    OUT_ADATA_DIR = '/work/hdd/bbjr/mallina1/data/human_cellariumgpt_v2/assay_conversion2'

    GENE_INFO_PATH = os.path.join(ROOT_PATH, "gene_info", "gene_info.tsv")
    CHECKPOINT_PATH = "/work/hdd/bbjr/mallina1/cellarium/models/compute_optimal_checkpoints/epoch=6-step=63560.ckpt"
    DEVICE = 'cuda'

    assay_label_map = {
        "10x Chromium (v2) A": "10x 3' v2",
        "10x Chromium (v2) B": "10x 3' v2",
        "10x Chromium (v3)": "10x 3' v3",
        "Drop-seq": "Drop-seq",
        "Seq-Well": "Seq-Well",
        "inDrops": "inDrop"
    }

    assay_ontology_term_id_map = {
        "Seq-Well": "EFO:0008919",
        "10x 3' v3": "EFO:0009922",
        "Drop-seq": "EFO:0008722",
        "inDrop": "EFO:0008780",
        "10x 3' v2": "EFO:0009899"
    }

    methods = list(assay_label_map.keys())

    # %%
    ref_adata = sc.read_h5ad(REF_ADATA_FP)
    val_adata = sc.read_h5ad(ADATA_FP)

    # Obtain Cellarium var_names for subsampling later as well
    ref_var_names = set(ref_adata.var_names)

    # First subset the validation gene counts to those that show up in Cellarium training
    val_adata = val_adata[:, val_adata.var_names.isin(ref_var_names)]

    # Next subset by assays that we have relevant data for in Cellarium based on the assay_label_map
    val_adata = val_adata[val_adata.obs.Method.isin(list(assay_label_map.keys()))]


    # %%
    # pre-process val_adata for conversion
    processed_val_adata = val_adata.copy()

    # Get random values for cell_type, tissue, disease, sex, and development_stage from ref_adata
    # since we are going to mask all of this anyway, just needed to have a valid category in the
    # get_tokens_from_adata function.
    cell_type_ontology_term_id = [ref_adata.obs.cell_type_ontology_term_id.unique()[0]] * val_adata.shape[0]
    tissue_ontology_term_id = [ref_adata.obs.tissue_ontology_term_id.unique()[0]] * val_adata.shape[0]
    disease_ontology_term_id = [ref_adata.obs.disease_ontology_term_id.unique()[0]] * val_adata.shape[0]
    sex_ontology_term_id = [ref_adata.obs.sex_ontology_term_id.unique()[0]] * val_adata.shape[0]
    development_stage_ontology_term_id = [ref_adata.obs.development_stage_ontology_term_id.unique()[0]] * val_adata.shape[0]

    assay = []
    suspension_type = []
    assay_ontology_term_id = []
    total_mrna_umis = []
    for val_obs_idx in tqdm(range(val_adata.shape[0])):
        val_obs_row = val_adata.obs.iloc[val_obs_idx]
        
        # PBMC adata fields
        pbmc_method = val_obs_row.Method

        # (converted) cellarium adata fields
        suspension_type.append('cell')
        assay.append(assay_label_map[pbmc_method])
        assay_ontology_term_id.append(assay_ontology_term_id_map[assay_label_map[pbmc_method]])
        total_mrna_umis.append(val_obs_row.nUMI)

    processed_val_adata.obs['assay'] = assay
    processed_val_adata.obs['suspension_type'] = suspension_type
    processed_val_adata.obs['assay_ontology_term_id'] = assay_ontology_term_id
    processed_val_adata.obs['total_mrna_umis'] = np.array(total_mrna_umis, dtype=np.int64)
    processed_val_adata.obs['cell_type_ontology_term_id'] = cell_type_ontology_term_id
    processed_val_adata.obs['tissue_ontology_term_id'] = tissue_ontology_term_id
    processed_val_adata.obs['disease_ontology_term_id'] = disease_ontology_term_id
    processed_val_adata.obs['sex_ontology_term_id'] = sex_ontology_term_id
    processed_val_adata.obs['development_stage_ontology_term_id'] = development_stage_ontology_term_id


    # %%
    ctx = CellariumGPTInferenceContext(
        cellarium_gpt_ckpt_path=CHECKPOINT_PATH,
        ref_adata_path=REF_ADATA_FP,
        gene_info_tsv_path=GENE_INFO_PATH,
        device=DEVICE,
        attention_backend="mem_efficient"
    )

    # %% [markdown]
    # ## Batched

    # %%
    # processed_val_adata = processed_val_adata[:8000]

    rng = torch.Generator(device='cpu')

    metadata_prompt_dict = {
        "cell_type": False,
        "tissue": False,
        "disease": False,
        "sex": False,
        "development_stage": False
    }

    # target_assay = '10x Chromium (v2) A'
    #target_assay = args.target_assay
    n_fixed_query_genes = 4096

    var_names = list(set(processed_val_adata.var_names))
    sample_indices = random.sample(range(len(var_names)), n_fixed_query_genes)
    sample_indices.sort()
    fixed_query_genes = [var_names[i] for i in sample_indices]
    # fixed_query_genes = np.random.choice(var_names, size=n_fixed_query_genes, replace=False)

    adata_fixed_genes_original = processed_val_adata[:, np.array(fixed_query_genes)]
    # adata_fixed_genes_original = processed_val_adata[:, processed_val_adata.var_names.isin(fixed_query_genes)].copy()
    adata_fixed_genes_converted = adata_fixed_genes_original.copy()

    batch_size = 32
    skipped_row = []
    X_lil = adata_fixed_genes_converted.X.tolil()
    for batch_idx in tqdm(range(0, adata_fixed_genes_original.shape[0], batch_size)):
        batch_query_total_mrna_umis = []

        obs_idx = []
        for val_obs_idx in range(batch_idx, batch_idx + batch_size):
            if val_obs_idx >= adata_fixed_genes_original.shape[0]:
                continue

            pbmc_cell_type = adata_fixed_genes_original.obs.iloc[val_obs_idx].CellType

            # UMIs by cell type and target assay in order to sample from in conversion
            pbmc_umis = val_adata[val_adata.obs.CellType == pbmc_cell_type]
            pbmc_umis = pbmc_umis[pbmc_umis.obs.Method == target_assay]

            # Use the global nUMI in the adata to sample from
            pbmc_umis = pbmc_umis.obs.nUMI.to_numpy()

            # Some cell types don't have any data for a given target_assay, skip these and flag it for now
            if len(pbmc_umis) == 0:
                skipped_row.append(True)
                continue

            # Use the empirical nUMI subset by the random sampled genes in the adata to sample from
            # pbmc_umis = np.array(pbmc_umis[:, pbmc_umis.var_names.isin(fixed_query_genes)].X.sum(-1)).squeeze()

            # for all query genes we sample the UMIs based on  subsetting to (cell_type, target_assay)
            # allow replacement because n_fixed_query_genes might be > len(pbmc_umis) from this dataset
            query_total_mrna_umis = np.random.choice(pbmc_umis, size=(n_fixed_query_genes,), replace=True)
            query_total_mrna_umis = np.array(query_total_mrna_umis, dtype=np.int64)
            batch_query_total_mrna_umis.append(query_total_mrna_umis[None, :])
            obs_idx.append(val_obs_idx)
            skipped_row.append(False)
        
        batch_query_total_mrna_umis = np.concatenate(batch_query_total_mrna_umis, axis=0)

        query_assay = assay_label_map[target_assay]
        query_assay_ontology_term_id = assay_ontology_term_id_map[query_assay]

        with torch.no_grad():
            tokens_dict, context_indices = ctx.generate_tokens_from_adata(adata_fixed_genes_original, 
                                                                        obs_index=obs_idx, 
                                                                        query_var_names=fixed_query_genes,
                                                                        metadata_prompt_masks_dict=metadata_prompt_dict,
                                                                        query_total_mrna_umis=batch_query_total_mrna_umis,
                                                                        query_assay_ontology_term_id=query_assay_ontology_term_id)
            
            gene_logits_nqk = ctx.get_gene_value_logits_from_tokens(tokens_dict,
                                                                    context_indices,
                                                                    max_counts=None)
            gene_logits_nqk = gene_logits_nqk.cpu()

            for idx, val_obs_idx in enumerate(obs_idx):
                dist = torch.distributions.categorical.Categorical(logits = gene_logits_nqk[idx].squeeze())
                row_sample = dist.sample().numpy()
                X_lil[val_obs_idx,:] = row_sample

    adata_fixed_genes_converted.X = X_lil.tocsr()

    adata_fixed_genes_original.obs['skipped_row'] = skipped_row
    adata_fixed_genes_converted.obs['skipped_row'] = skipped_row

    # %%
    formatted = datetime.now().strftime("%Y_%m_%d_%H_%M")

    filename = f'{formatted}_convert_to_{"_".join(target_assay.split())}'
    clean_filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)

    target_assay_out_dir = os.path.join(OUT_ADATA_DIR, clean_filename)
    os.makedirs(target_assay_out_dir)

    results_fp1 = os.path.join(target_assay_out_dir, f'original.h5ad')
    results_fp2 = os.path.join(target_assay_out_dir, f'converted.h5ad')

    sc.write(results_fp1, adata_fixed_genes_original)
    sc.write(results_fp2, adata_fixed_genes_converted)

# %%

if __name__=='__main__':
    main()

