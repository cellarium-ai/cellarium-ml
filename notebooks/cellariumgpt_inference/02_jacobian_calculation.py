#!/usr/bin/env python
# coding: utf-8

# #### Jacobian calculation

# In[1]:


import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from cellarium.ml import CellariumModule, CellariumPipeline

import torch._dynamo
torch._dynamo.config.suppress_errors = True

DEVICE = torch.device('cuda')


# In[2]:


ROOT_PATH = "/mnt/cellariumgpt-xfer/mb-ml-dev-vm"
CHECKPOINTS_PATH = "/mnt/cellariumgpt-xfer/100M_long_run/run_001/lightning_logs/version_0/checkpoints"


# ### Load an AnnData Extract
# 
# We will use it for category mappings ...

# In[3]:


# Load an AnnData extract
adata_path = os.path.join(ROOT_PATH, "data", "extract_0.h5ad")
adata = sc.read_h5ad(adata_path)


# In[4]:


gene_ontology_infos = dict()

ref_obs = adata.obs

gene_ontology_infos["assay_ontology_term_id"] = dict()
gene_ontology_infos["assay_ontology_term_id"]["names"] = list(ref_obs['assay_ontology_term_id'].cat.categories)  
gene_ontology_infos["assay_ontology_term_id"]["labels"] = list(ref_obs['assay_ontology_term_id'].cat.categories) # just because I am lazy

gene_ontology_infos["suspension_type"] = dict()
gene_ontology_infos["suspension_type"]["names"] = list(ref_obs['suspension_type'].cat.categories)  # for uniformity -- this variable does not have an ontology (does it?)
gene_ontology_infos["suspension_type"]["labels"] = list(ref_obs['suspension_type'].cat.categories)


# In[5]:


# gene IDs, gene symbols, useful maps
model_var_names = np.asarray(adata.var_names)
model_var_names_set = set(model_var_names)
var_name_to_index_map = {var_name: i for i, var_name in enumerate(model_var_names)}

gene_info_tsv_path = os.path.join(ROOT_PATH, "gene_info", "gene_info.tsv")
gene_info_df = pd.read_csv(gene_info_tsv_path, sep="\t")

gene_symbol_to_gene_id_map = dict()
for gene_symbol, gene_id in zip(gene_info_df['Gene Symbol'], gene_info_df['ENSEMBL Gene ID']):
    if gene_symbol != float('nan'):
        gene_symbol_to_gene_id_map[gene_symbol] = gene_id


# ### Load a CellariumGPT checkpoint

# In[6]:


ckpt_path = os.path.join(CHECKPOINTS_PATH, "epoch=2-step=180000.ckpt")
gpt_model = CellariumModule.load_from_checkpoint(ckpt_path, map_location=DEVICE)

# Inject gene categories
gpt_model.model.gene_categories = np.asarray(adata.var_names)


# In[7]:


# Get the ontology infos from the TrainTokenizer, which is the first step of the pipeline
metadata_ontology_infos = gpt_model.pipeline[0].ontology_infos
print(type(metadata_ontology_infos))


# ### Helper functions

# In[8]:


from cellarium.ml.models.cellarium_gpt import PredictTokenizer

# Rewire the pipeline with a PredictTokenizer
predict_tokenizer = PredictTokenizer(
    max_total_mrna_umis=100_000,
    gene_vocab_sizes={
        "assay": 19,
        "gene_id": 36601,
        "gene_value": 2001,
        "suspension_type": 2,
    },
    metadata_vocab_sizes={
        "cell_type": 890,
        "development_stage": 191,
        "disease": 350,
        "sex": 2,
        "tissue": 822,
    },
    ontology_infos=metadata_ontology_infos,
)

gpt_model.pipeline = CellariumPipeline([
    predict_tokenizer,
    gpt_model.model,
])


# In[9]:


import pandas as pd

from cellarium.ml.models.cellarium_gpt import CellariumGPT


def generate_tokens_from_adata(
        adata: sc.AnnData,
        gene_ontology_infos: dict[str, dict[str, list[str]]],
        metadata_ontology_infos: dict[str, dict[str, list[str]]],
        model_gene_categories: list[str],
        obs_index: int | list[int] | None,
        query_var_index: list[int],
        query_total_mrna_umis: float | None,
        metadata_prompt_masks_dict: dict[str, bool],
        tokenizer: PredictTokenizer,
        device: torch.device = torch.device("cpu"),
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
    model_var_name_to_index_map = {var_name: var_index for var_index, var_name in enumerate(model_gene_categories)}
    assert all([var_name in model_var_name_to_index_map for var_name in adata_var_names])
    prompt_var_index = [model_var_name_to_index_map[var_name] for var_name in adata_var_names]
    n_prompt_vars = len(prompt_var_index)
    n_query_vars = len(query_var_index)
    n_total_vars = n_prompt_vars + n_query_vars
    
    # gene id
    gene_ids_nc = torch.tensor(
        prompt_var_index + query_var_index,
        dtype=torch.int64, device=device)[None, :].expand(n_cells, n_total_vars)
    
    # gene prompt mask
    gene_prompt_mask_nc = torch.tensor(
        [1] * n_prompt_vars + [0] * n_query_vars,
        dtype=torch.bool, device=device)[None, :].expand(n_cells, n_total_vars)
    
    # gene value
    try:
        prompt_X_ng = np.asarray(adata.X.todense())
    except AttributeError:
        prompt_X_ng = adata.X
    prompt_gene_value_nc = torch.tensor(prompt_X_ng, dtype=torch.float32, device=device)
    query_gene_value_nc = torch.zeros(n_cells, n_query_vars, dtype=torch.float32, device=device)
    gene_value_nc = torch.cat([prompt_gene_value_nc, query_gene_value_nc], dim=1)

    # total mrna umis
    prompt_total_mrna_umis_nc = torch.tensor(
        adata.obs["total_mrna_umis"].values,
        dtype=torch.float32, device=device)[:, None].expand(n_cells, n_prompt_vars)
    if query_total_mrna_umis is None:
        # the same as prompt
        query_total_mrna_umis_nc = torch.tensor(
            adata.obs["total_mrna_umis"].values,
            dtype=torch.float32, device=device)[:, None].expand(n_cells, n_query_vars)
    else:
        query_total_mrna_umis_nc = torch.tensor(
            [query_total_mrna_umis] * n_cells,
            dtype=torch.float32, device=device)[:, None].expand(n_cells, n_query_vars)
    total_mrna_umis_nc = torch.cat([prompt_total_mrna_umis_nc, query_total_mrna_umis_nc], dim=1)

    # convert assay and suspension_type to codes
    assay_nc = torch.tensor(
        pd.Categorical(
            adata.obs["assay_ontology_term_id"].values,
            categories=gene_ontology_infos["assay_ontology_term_id"]["names"]).codes,
        dtype=torch.int64, device=device)[:, None].expand(n_cells, n_total_vars)
    suspension_type_nc = torch.tensor(
        pd.Categorical(
            adata.obs["suspension_type"].values,
            categories=gene_ontology_infos["suspension_type"]["names"]).codes,
        dtype=torch.int64, device=device)[:, None].expand(n_cells, n_total_vars)

    gene_tokens_dict = {
        "assay": assay_nc,  # categorical
        "suspension_type": suspension_type_nc,  # categorical
        "gene_id": gene_ids_nc,  # categorical
        "gene_value": gene_value_nc,  # continuous
        "total_mrna_umis": total_mrna_umis_nc,  # continuous
    }

    # metadata prompt masks
    expanded_metadata_prompt_masks_dict = dict()
    for key in metadata_ontology_infos.keys():  # note: key order is important ...
        expanded_metadata_prompt_masks_dict[key] = torch.tensor(
            [metadata_prompt_masks_dict[key]] * n_cells, dtype=torch.bool, device=device)
    
    # generate metadata tokens dicts; `PredictTokenizer` will convert these to codes
    metadata_tokens_dict = {
        "cell_type": adata.obs["cell_type_ontology_term_id"].values,  # categorical
        "development_stage": adata.obs["development_stage_ontology_term_id"].values,  # categorical
        "disease": adata.obs["disease_ontology_term_id"].values,  # categorical
        "sex": adata.obs["sex_ontology_term_id"].values,  # categorical
        "tissue": adata.obs["tissue_ontology_term_id"].values,  # categorical
    }

    # where to find each thing in the context?
    context_indices = dict()
    context_indices['prompt_genes'] = np.arange(0, n_prompt_vars).tolist()
    context_indices['query_genes'] = np.arange(n_prompt_vars, n_query_vars + n_prompt_vars).tolist()
    offset = 0
    for metadata_key in metadata_ontology_infos.keys():
        context_indices[f'query_{metadata_key}'] = n_query_vars + n_prompt_vars + offset
        offset += 1

    # return gene_tokens_dict, metadata_tokens_dict
    tokenizer_output = tokenizer(
        metadata_tokens_n=metadata_tokens_dict,
        metadata_prompt_masks_n=expanded_metadata_prompt_masks_dict,
        gene_tokens_nc=gene_tokens_dict,
        gene_prompt_mask_nc=gene_prompt_mask_nc,
    )

    return tokenizer_output, context_indices


# In[10]:


def snap_to_marginal_mean_manifold(
        adata: sc.AnnData,
        gene_ontology_infos: dict[str, dict[str, list[str]]],
        metadata_ontology_infos: dict[str, dict[str, list[str]]],
        query_var_names: list[str],
        query_total_mrna_umis: float | None,
        predict_tokenizer: PredictTokenizer,
        cellarium_gpt_module: CellariumModule,
        prompt_gene_values_g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    
    assert len(adata) == 1, "Only a single cell is allowed"
    
    model_gene_categories = cellarium_gpt_module.model.gene_categories
    var_name_to_index_map = {var_name: i for i, var_name in enumerate(model_gene_categories)}
    query_var_index = [var_name_to_index_map[var_name] for var_name in query_var_names]

    metadata_prompt_masks_dict = {
        "cell_type": False,
        "development_stage": False,
        "disease": False,
        "sex": False,
        "tissue": False,
    }

    tokens_dict, context_indices = generate_tokens_from_adata(
        adata=adata,
        gene_ontology_infos=gene_ontology_infos,
        metadata_ontology_infos=metadata_ontology_infos,
        model_gene_categories=model_var_names,
        obs_index=None,
        query_var_index=query_var_index,
        query_total_mrna_umis=query_total_mrna_umis,
        metadata_prompt_masks_dict=metadata_prompt_masks_dict,
        tokenizer=predict_tokenizer,
        device=torch.device("cpu"),
    )

    # convert to cuda
    tokens_dict = cellarium_gpt_module.transfer_batch_to_device(tokens_dict, cellarium_gpt_module.device, 0)
    
    # get a reference to prompt gene values
    FIRST_CELL_DIM = 0
    GENE_VALUE_DIM = 0
    prompt_gene_log1p_values_g = tokens_dict['gene_tokens_nc']['gene_value'][
        FIRST_CELL_DIM, context_indices['prompt_genes'], GENE_VALUE_DIM]
    
    # this is the "source"
    if prompt_gene_values_g is None:
        prompt_gene_values_g = torch.expm1(prompt_gene_log1p_values_g).clone()
    
    # inject back to tokens_dict to re-establish the reference for Jacobian calculation
    tokens_dict['gene_tokens_nc']['gene_value'][
        FIRST_CELL_DIM, context_indices['prompt_genes'], GENE_VALUE_DIM] = torch.log1p(prompt_gene_values_g)

    # get model predictions
    logits_dict = cellarium_gpt_module.model.predict(
        gene_tokens_nc=tokens_dict["gene_tokens_nc"],
        metadata_tokens_n=tokens_dict["metadata_tokens_n"],
        prompt_mask_nc=tokens_dict["prompt_mask_nc"],
    )

    # note: we use `q` to denote query genes
    gene_logits_qk = logits_dict['gene_value'][FIRST_CELL_DIM, context_indices['query_genes'], :]
    gene_logits_qk = gene_logits_qk - torch.logsumexp(gene_logits_qk, dim=-1, keepdim=True)
    log_counts_k = torch.arange(0, gene_logits_qk.shape[-1], device=gene_logits_qk.device).log()
    gene_marginal_means_q = torch.logsumexp(gene_logits_qk + log_counts_k[None, :], dim=-1).exp()

    return prompt_gene_values_g, gene_marginal_means_q


# #### Jacobian calculation

# In[44]:


# load a test anndata
dataset_names = [
    "luca_CD8_ex_LUAD",
    "luca_CD8_act_LUAD",
    "luca_CD8_naive_LUAD",
    "luca_CD8_em_LUAD",
    "luca_Treg_LUAD",
    "luca_CD8_ex_normal",
    "luca_CD8_act_normal",
    "luca_CD8_naive_normal",
    "luca_CD8_em_normal",
    "luca_Treg_normal",
]

jacobian_points = [
    "actual",
    "marginal_mean"
]

target_total_mrna_umis = 2078.0732
query_chunk_size = 200
max_query_genes = None


# In[47]:


import argparse
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(description="Process two integers: jacobian_point_index and dataset_name_index.")
    parser.add_argument("jacobian_point_index", type=int, help="Index of the Jacobian point")
    parser.add_argument("dataset_name_index", type=int, help="Index of the dataset name")

    args = parser.parse_args()

    print(f"Jacobian Point Index: {args.jacobian_point_index}")
    print(f"Dataset Name Index: {args.dataset_name_index}")

    jacobian_point = jacobian_points[int(args.jacobian_point_index)]
    dataset_name = dataset_names[int(args.dataset_name_index)]

    test_adata_path = os.path.join(ROOT_PATH, "cellariumgpt_pd1_lung", "output", f"{dataset_name}.h5ad")
    test_adata = sc.read_h5ad(test_adata_path)

    # add total mNRA umis
    test_adata.obs['total_mrna_umis'] = test_adata.X.sum(axis=1)

    # make a metacell
    X_meta_g = np.asarray(test_adata.X.sum(0))

    # set total mrna umis to the mean of the dataset
    X_meta_g = X_meta_g * target_total_mrna_umis / X_meta_g.sum()

    # make a metacell anndata
    adata_meta = test_adata[0, :].copy()
    adata_meta.X = X_meta_g
    adata_meta.obs['total_mrna_umis'] = [target_total_mrna_umis]

    # choose top genes
    top_genes_path = os.path.join(ROOT_PATH, "cellariumgpt_pd1_lung", "output", "immune_top_5000_genes.csv")
    top_genes_df = pd.read_csv(top_genes_path)
    top_gene_ids = top_genes_df['gene_id'].values

    # restrict top_gene_ids to those in the model categories
    top_gene_ids = [gene_id for gene_id in top_gene_ids if gene_id in model_var_names_set]
    prompt_gene_ids = top_gene_ids

    adata_meta = adata_meta[:, prompt_gene_ids]

    # query var names
    query_var_names = prompt_gene_ids
    if max_query_genes is not None:
        query_var_names = query_var_names[:max_query_genes]

    # jacobian point
    if jacobian_point == "actual":
        print("Using the actual metacell counts as the point to calculate the Jacobian on ...")

    elif jacobian_point == "marginal_mean":
        with torch.inference_mode():
            # get the mean marginal
            print("Calculating marginal mean ...")
            prompt_gene_values_g, gene_marginal_means_q = snap_to_marginal_mean_manifold(
                adata=adata_meta,
                gene_ontology_infos=gene_ontology_infos,
                metadata_ontology_infos=metadata_ontology_infos,
                query_var_names=prompt_gene_ids,
                query_total_mrna_umis=None,
                predict_tokenizer=predict_tokenizer,
                cellarium_gpt_module=gpt_model,
            )

            # inject into the adata
            adata_meta.X = gene_marginal_means_q.detach().cpu().numpy()[None, :]
    else:
        raise ValueError

    def yield_chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    query_var_names_chunks = list(yield_chunks(query_var_names, query_chunk_size))
    jacobian_chunks = []

    for query_var_names_chunk in tqdm(query_var_names_chunks):

        prompt_gene_values_g = torch.tensor(
            adata_meta.X[0].toarray().flatten(),
            device=gpt_model.device,
            dtype=torch.float32)

        def _wrapped_snap_to_marginal_mean_manifold(
                prompt_gene_values_g: torch.Tensor) -> torch.Tensor:
            
            return snap_to_marginal_mean_manifold(
                adata=adata_meta,
                gene_ontology_infos=gene_ontology_infos,
                metadata_ontology_infos=metadata_ontology_infos,
                query_var_names=query_var_names_chunk,
                query_total_mrna_umis=None,
                predict_tokenizer=predict_tokenizer,
                cellarium_gpt_module=gpt_model,
                prompt_gene_values_g=prompt_gene_values_g,
            )[1]

        chunk_jacobian_qg = torch.autograd.functional.jacobian(
            func=_wrapped_snap_to_marginal_mean_manifold, 
            inputs=prompt_gene_values_g,
            create_graph=False,
            vectorize=False,
        )

        jacobian_chunks.append(chunk_jacobian_qg)

    jacobian_qg = torch.cat(jacobian_chunks, dim=0)

    result_dict = {
        'dataset_name': dataset_name,
        'jacobian_point': jacobian_point,
        'query_var_names': query_var_names,
        'prompt_var_names': prompt_gene_ids,
        'jacobian_qg': jacobian_qg,
        'prompt_gene_values_g': adata_meta.X[0].toarray().flatten(),
    }

    torch.save(result_dict, f"./output/jacobian__{dataset_name}__{jacobian_point}.pt")


if __name__ == "__main__":
    main()
    


# In[ ]:




