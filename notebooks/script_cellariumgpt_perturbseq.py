#!/bin/python

import scanpy as sc
import pandas as pd
import numpy as np
import torch
import os
import tqdm

from cellarium.ml.downstream.noise_prompting import in_silico_perturbation
from cellarium.ml.downstream.cellarium_utils import get_pretrained_model_as_pipeline

import warnings
warnings.filterwarnings("ignore")

n_control_cells_per_batch = 10
n_batches = 10

pipeline = get_pretrained_model_as_pipeline(device="cuda" if torch.cuda.is_available() else "cpu")

# the perturbseq data from the RPE1 cell line
adata = sc.read_h5ad('/home/sfleming/data/ReplogleWeissman2022_rpe1.h5ad')
adata.var = adata.var.reset_index()
adata.var.index = adata.var['ensembl_id'].copy()
print(adata)

# find control cells
adata_control = adata[adata.obs['perturbation'] == 'control'].copy()
adata_control.obs['total_mrna_umis'] = adata_control.obs['UMI_count'].copy().astype(np.float32)
print('control cells:')
print(adata_control)
print(f'{n_control_cells_per_batch} control cells per batch, {n_batches} batches:')
vc = adata_control.obs['batch'].value_counts()
top_batches = vc.index[:n_batches]
inds = (
    adata_control.obs
    [adata_control.obs['batch'].isin(top_batches)]
    [['batch']]
    .reset_index()
    .groupby('batch', observed=True)
    .apply(lambda x: x.sample(n_control_cells_per_batch))
    ['cell_barcode']
    .values
)
print(len(inds))
adata_control = adata_control[inds].copy()
adata_control.layers['count'] = adata_control.X.copy()
print(adata_control)

# no perturbation
print('running control cells to get baseline ...')
adata_out_nopert = in_silico_perturbation(
    adata_control,
    pipeline=pipeline,
    prompt_gene_inds=torch.arange(adata_control.shape[1]),
    perturbation={},
    measured_count_layer_key='count',
    output_layer_key='perturbed_gpt',
)

# all the perturbations

# dfs = []
gids = adata.obs['gene_id'].dropna().unique()
measured_gids = set(adata.var['ensembl_id'].values)
gids = [gid for gid in gids if ((gid in measured_gids) and (not os.path.exists(f'/home/sfleming/data/perturbseq_lfc_{gid}.csv')))]

for gid in tqdm.tqdm(gids):

    out_path = f'/home/sfleming/data/perturbseq_lfc_{gid}.csv'

    # if os.exists(out_path):
    #     print(f'... skipping {gid} which already exists')
    #     continue

    # if not gid in adata.var['ensembl_id'].values:
    #     print(f'... skipping {gid} which is not in adata.var["ensembl_id"]')
    #     continue

    print(f'... working on {gid} =====================================')

    adata_out = in_silico_perturbation(
        adata_control,
        pipeline=pipeline,
        prompt_gene_inds=torch.arange(adata_control.shape[1]),
        perturbation={gid: 0.0},
        measured_count_layer_key='count',
        output_layer_key='perturbed_gpt',
    )
    lfc_df = pd.DataFrame(
        np.log2(adata_out.layers['perturbed_gpt'].mean(axis=0) + 1e-10) - np.log2(adata_out_nopert.layers['perturbed_gpt'].mean(axis=0) + 1e-10) ,
        index=adata_out.var['gene_name'],
        columns=[gid],
    ).transpose()

    lfc_df.to_csv(out_path)
    print(f'wrote {out_path}')

#     dfs.append(lfc_df)

# pd.concat(dfs, axis=0).to_csv('/home/sfleming/data/perturbseq_lfc.csv')
