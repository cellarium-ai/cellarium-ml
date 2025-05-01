# %%
import os
import re
import csv
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from scipy.stats import pearsonr
from anndata import AnnData
import anndata

# for flex attention
import torch._dynamo
import torch.multiprocessing as mp 
torch._dynamo.config.suppress_errors = True

sc.set_figure_params(figsize=(4, 4))

from cellarium.ml.utilities.inference.cellarium_gpt_inference import \
    CellariumGPTInferenceContext

# %%
data_root = '/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm/data/GSE153807/tsvs'

fnames = ['GSM4654467_Nuc-RM101-2.raw.tsv', 
          'GSM4654468_Nuc-RM102-1.raw.tsv', 'GSM4654469_Nuc-RM102-2.raw.tsv', 
          'GSM4654470_Nuc-RM77-1.raw.tsv', 'GSM4654471_Nuc-RM77-2.raw.tsv',
          'GSM4654472_Nuc-RM95-1.raw.tsv', 'GSM4654473_Nuc-RM95-2.raw.tsv']

sex_per_fname = ['Female', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female']
sex_ontology_type_id_per_fname = ['PATO:0000383', 'PATO:0000383', 'PATO:0000383', 'PATO:0000384', 'PATO:0000384', 'PATO:0000383', 'PATO:0000383']

gene_info_path = '/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm/gene_info/gene_info.tsv'
ontology_infos_path = '/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm/ontology_infos.pt'

idx_to_run = 0

# %%
ontology_infos = torch.load(ontology_infos_path)
gene_info_df = pd.read_csv(gene_info_path, sep='\t', index_col=0)
gene_info_df

# %%
gene_symb_to_gene_id = gene_info_df['Gene Symbol'].reset_index().set_index('Gene Symbol')['ENSEMBL Gene ID'].to_dict()

# %%
# gene_info_df[gene_info_df.index == 'ENSG00000267104']


# %%
ontology_genes = set(gene_symb_to_gene_id.keys())

# %%
gene_symb_to_gene_id

# %%
def load_df_into_anndata(fname):
    df = pd.read_csv(os.path.join(data_root, fname), sep='\t', index_col=0)
    df = df[df.index.isin(ontology_genes)]

    original_symbols = df.index.to_series(name='ENSEMBL Gene ID')
    mapped_ids = original_symbols.map(lambda s: gene_symb_to_gene_id.get(s))

    df.index = mapped_ids

    data = {
        'sample_id': fname,
        'suspension_type': ['nucleus'] * len(df.columns),
        'total_mrna_umis': df.sum(axis=0),
        'assay_ontology_term_id': ['EFO:0009899'] * len(df.columns),
        'assay': ["10x 3' v2"] * len(df.columns),
        'sex': [sex_per_fname[idx_to_run]] * len(df.columns),
        'sex_ontology_term_id': [sex_ontology_type_id_per_fname[idx_to_run]] * len(df.columns)
    }

    obs = pd.DataFrame(index=df.columns, data=data)
    var = pd.DataFrame(index=df.index)        # one row per gene ID
    var['gene_symbol'] = original_symbols.tolist()     # store the original symbol

    adata = AnnData(X=df.values.T, obs=obs, var=var)

    return adata

def unify_adatas(adatas):
    adatas_reindexed = []
    
    dup_var_names = [var for ad in adatas for var in ad.var_names]
    dup_gene_symbol = [gene_symb for adata in adatas for gene_symb in adata.var['gene_symbol']]

    seen = set({})
    all_var_names, all_gene_symbol = [], []
    for x, y in zip(dup_var_names, dup_gene_symbol):
        if (x, y) not in seen:
            seen.add((x, y))
            all_var_names.append(x)
            all_gene_symbol.append(y)

    sort_idx = np.argsort(all_var_names)
    all_vars = np.array(all_var_names)[sort_idx]
    all_genes = np.array(all_gene_symbol)[sort_idx]

    for adata in adatas:
        df = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )
        df = df.reindex(columns=all_vars, fill_value=0)

        var = pd.DataFrame(index=all_vars)
        var['gene_symbol'] = list(all_genes)

        ad_re = AnnData(
            X=df.values,
            obs=adata.obs.copy(),
            var=var,
            uns = adata.uns.copy(),
            layers={k: v.copy() for k, v in adata.layers.items()}
        )
        # preserve any uns, layers, etc, if you want:
        # ad_re.layers = ad.layers  
        adatas_reindexed.append(ad_re)

    combined = anndata.concat(
        adatas_reindexed,
        axis=0,               # stack cells (obs) on top of each other
        join="inner",         # now that all have the same var, an “inner” join is equivalent to union
        label="batch",        # optionally records which sample each cell came from
        fill_value=0          # should already be zeros, but safe to specify
    )

    combined.var['gene_symbol'] = all_genes

    return combined

# %%
adatas = []
for fname in fnames:
    adatas.append(load_df_into_anndata(fname))

# %%
combined = unify_adatas(adatas)

# %%
del adatas

# %%
ROOT_PATH = "/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm"

REF_ADATA_FP = '/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm/data/extract_0.h5ad'
OUT_ADATA_DIR = '/work/hdd/bbjr/mallina1/data/human_cellariumgpt_v2/suspension_type_conversion'

GENE_INFO_PATH = os.path.join(ROOT_PATH, "gene_info", "gene_info.tsv")

# CHECKPOINT_PATH = "/work/hdd/bbjr/mallina1/cellarium/models/compute_optimal_checkpoints/epoch=1-step=28244.ckpt"
# CHECKPOINT_PATH = "/work/hdd/bbjr/mallina1/cellarium/models/compute_optimal_checkpoints/epoch=6-step=63560.ckpt"
CHECKPOINT_PATH = "/work/hdd/bbjr/mallina1/cellarium/models/compute_optimal_checkpoints/epoch=10-step=78917.ckpt"

DEVICE = 'cuda'

ref_adata = sc.read_h5ad(REF_ADATA_FP)
ref_var_names = set(ref_adata.var_names)

# %%
genes_to_keep = ["AC004448.2","AC010894.3","AC011468.3","AC011586.2","AC016708.1","AC022217.3","AC024230.1",
                 "AC044781.1","AC072062.1","AC245014.3","ACTB","AIF1","AL136454.1","ALOX5AP","AMBRA1","APOC1",
                 "APOE","APOO","ARMC9","ATP5F1E","ATP5MC2","ATP6V0B","ATP6V0E1","B2M","BAIAP2L1","BDNF-AS",
                 "BTF3","BTG2","C1QA","C1QB","C1QC","CARMIL1","CCDC200","CCL2","CCL3","CCL3L1","CCL4","CCL4L2",
                 "CD14","CD37","CD63","CD68","CD74","CEBPB","CEBPD","CFL1","CHCHD3","COMMD6","CORO1A","COX4I1",
                 "CST3","CTSB","CYBA","DAPK1","DDIT4","DNAJB1","DUSP1","EEF1A1","EEF1B2","EEF1D","EEF2","EFCAB3",
                 "EIF1","FAU","FCER1G","FCGRT","FOLR2","FOS","FP700111.1","FTH1","FTL","GADD45B","GGACT",
                 "GPR183","GPX4","GRN","GSTP1","H3F3B","HCST","HERPUD1","HLA-A","HLA-B","HLA-C","HLA-DPA1",
                 "HLA-DPB1","HLA-DRA","HLA-DRB1","HLA-DRB5","HLA-E","HMOX1","HNRNPA1","HSP90AA1","HSPA1A",
                 "HSPA1B","HSPB1","IER2","IER3","ITM2B","JUN","JUNB","KIZ-AS1","LAMTOR4","LAPTM4A","LAPTM5",
                 "LINC01500","LINC01736","LINGO1","LTC4S","MAMDC2","MARCKS","MECOM","MT-ATP6","MT-CO1","MT-CO2",
                 "MT-CO3","MT-CYB","MT-ND2","MT-ND3","MT-ND4","MYL6","NACA","NACA2","NBEAL1","NFKBIA","NHSL2",
                 "NINJ1","NPC2","OLFML3","OOEP","OTULINL","PDK4","PFDN5","PFN1","PLD4","PLEKHA6","PLEKHA7",
                 "PNRC1","PSAP","PTMA","PYCARD","RAC1","RACK1","RGS1","RGS10","RHOB","RHOG","RNASE6","RPL10",
                 "RPL10A","RPL11","RPL12","RPL13","RPL13A","RPL14","RPL15","RPL18","RPL18A","RPL19","RPL21",
                 "RPL23","RPL23A","RPL24","RPL27","RPL27A","RPL28","RPL29","RPL3","RPL30","RPL31","RPL32","RPL34",
                 "RPL35","RPL35A","RPL36","RPL36AL","RPL37","RPL37A","RPL38","RPL39","RPL4","RPL41","RPL5","RPL6",
                 "RPL7","RPL7A","RPL8","RPLP0","RPLP1","RPLP2","RPS11","RPS12","RPS13","RPS14","RPS15","RPS15A",
                 "RPS16","RPS17","RPS18","RPS19","RPS2","RPS20","RPS23","RPS24","RPS25","RPS26","RPS27","RPS27A",
                 "RPS28","RPS29","RPS3","RPS3A","RPS4X","RPS5","RPS6","RPS7","RPS8","RPS9","RPSA","S100A11","SAT1",
                 "SERF2","SIK3","SLC25A6","SLC27A4","SLC47A1","SPP1","SRGN","TEX14","TMSB10","TMSB4X","TOMM7","TPT1",
                 "TREM2","TSPO","TUBA1B","TXNRD1","TYROBP","UBA52","UBC","VSIR","XPO5","YBX1","ZFP36","ZFP36L1",
                 "ZFP36L2","ZNF90"]

print(len(genes_to_keep))

gene_ids_to_keep = []
for x in genes_to_keep:
    if x in gene_symb_to_gene_id and gene_symb_to_gene_id[x] in ref_var_names:
        gene_ids_to_keep.append(gene_symb_to_gene_id[x])

print(len(gene_ids_to_keep))

n_fixed_query_genes = 4096

# %%
_adata = combined[:, combined.var_names.isin(ref_var_names)].copy() 

# %%
sc.pp.highly_variable_genes(_adata, flavor='seurat_v3', n_top_genes=n_fixed_query_genes)
_adata.var['highly_variable']

# %%
temp_subset = _adata[:, _adata.var['highly_variable']].copy()
final_gene_list = list(set(gene_ids_to_keep).union(set(temp_subset.var_names)))

final_subset = _adata[:, _adata.var_names.isin(final_gene_list)]
final_subset = _adata[:, np.array(final_gene_list)].copy()

# %%
ctx = CellariumGPTInferenceContext(
    cellarium_gpt_ckpt_path=CHECKPOINT_PATH,
    ref_adata_path=REF_ADATA_FP,
    gene_info_tsv_path=GENE_INFO_PATH,
    device=DEVICE,
    attention_backend="mem_efficient"
)

# %%
not_present = [var_name in ctx.var_name_to_index_map for var_name in final_gene_list]
print(np.where(np.array(not_present)==False))
final_gene_list[36]

# %%
final_subset.obs.total_mrna_umis.mean()

# %%
final_subset.obs['cell_type_ontology_term_id'] = None
final_subset.obs['tissue_ontology_term_id'] = None
final_subset.obs['disease_ontology_term_id'] = None
# final_subset.obs['sex_ontology_term_id'] = None
final_subset.obs['development_stage_ontology_term_id'] = None

# %%
metadata_prompt_dict = {
    "cell_type": False,
    "tissue": False,
    "disease": False,
    "sex": True,
    "development_stage": False
}

# %%
batch_size = 32

## run metacell first
pbar = tqdm(total=final_subset.shape[0])

cell_samples = []
nucleus_samples = []

for val_obs_idx in range(0, final_subset.shape[0], batch_size):
    obs_idx = np.arange(val_obs_idx, min(val_obs_idx + batch_size, final_subset.shape[0]))

    tokens_dict, context_indices = ctx.generate_tokens_from_adata(final_subset, 
                                                                    obs_index=obs_idx, 
                                                                    query_var_names=final_gene_list,
                                                                    metadata_prompt_masks_dict=metadata_prompt_dict,
                                                                    query_total_mrna_umis=4900,
                                                                    query_suspension_type='cell')

    with torch.no_grad():
        gene_logits_nqk = ctx.get_gene_value_logits_from_tokens(tokens_dict,
                                                                context_indices,
                                                                max_counts=None)

        # gene_marginal_mean_nq, _ = ctx.calculate_gene_mean_std_from_logits(gene_logits_nqk,
        #                                                                     gene_logits_nqk.shape[-1],
        #                                                                     use_logsumexp=True)

        dist = torch.distributions.categorical.Categorical(logits = gene_logits_nqk)
        sampled_counts = dist.sample().cpu()

        cell_samples.append(sampled_counts)

    tokens_dict, context_indices = ctx.generate_tokens_from_adata(final_subset, 
                                                                    obs_index=obs_idx, 
                                                                    query_var_names=final_gene_list,
                                                                    metadata_prompt_masks_dict=metadata_prompt_dict,
                                                                    query_total_mrna_umis=4900,
                                                                    query_suspension_type='nucleus')

    with torch.no_grad():
        gene_logits_nqk = ctx.get_gene_value_logits_from_tokens(tokens_dict,
                                                                context_indices,
                                                                max_counts=None)

        # gene_marginal_mean_nq, _ = ctx.calculate_gene_mean_std_from_logits(gene_logits_nqk,
        #                                                                     gene_logits_nqk.shape[-1],
        #                                                                     use_logsumexp=True)

        dist = torch.distributions.categorical.Categorical(logits = gene_logits_nqk)
        sampled_counts = dist.sample().cpu()

        nucleus_samples.append(sampled_counts) 
    
    pbar.update(len(obs_idx))

    # if val_obs_idx == 96:
    #     break

# %%
cell_samples = torch.cat(cell_samples, dim=0)
nucleus_samples = torch.cat(nucleus_samples, dim=0)

# %%
cell_samples.shape, nucleus_samples.shape

# %%
new_X = np.vstack([cell_samples.numpy(), nucleus_samples.numpy()])

cell_obs = final_subset.obs.copy()
cell_obs.suspension_type = 'cell'
cell_obs.total_mrna_umis = 4900

nucleus_obs = final_subset.obs.copy()
nucleus_obs.total_mrna_umis = 4900

new_obs = pd.concat([cell_obs, nucleus_obs], axis=0)

output_adata = AnnData(
    X = new_X,
    obs = new_obs,
    var = final_subset.var.copy()
)

output_adata.obs.suspension_type = output_adata.obs.suspension_type.astype('category')

# %%
output_adata

# %%
del output_adata.obs['cell_type_ontology_term_id']
del output_adata.obs['tissue_ontology_term_id']
del output_adata.obs['disease_ontology_term_id']
# del output_adata.obs['sex_ontology_term_id']
del output_adata.obs['development_stage_ontology_term_id']
sc.write('/work/hdd/bbjr/mallina1/data/human_cellariumgpt_v2/suspension_type_conversion/final_resampled_nucleus.h5ad', output_adata)

# %%
new_X = np.vstack([cell_samples.numpy(), final_subset.X])

cell_obs = final_subset.obs.copy()
cell_obs.suspension_type = 'cell'
cell_obs.total_mrna_umis = 4900

nucleus_obs = final_subset.obs.copy()

new_obs = pd.concat([cell_obs, nucleus_obs], axis=0)

output_adata = AnnData(
    X = new_X,
    obs = new_obs,
    var = final_subset.var.copy()
)

output_adata.obs.suspension_type = output_adata.obs.suspension_type.astype('category')

# %%
output_adata

# %%
del output_adata.obs['cell_type_ontology_term_id']
del output_adata.obs['tissue_ontology_term_id']
del output_adata.obs['disease_ontology_term_id']
# del output_adata.obs['sex_ontology_term_id']
del output_adata.obs['development_stage_ontology_term_id']
sc.write('/work/hdd/bbjr/mallina1/data/human_cellariumgpt_v2/suspension_type_conversion/final_original_nucleus.h5ad', output_adata)


