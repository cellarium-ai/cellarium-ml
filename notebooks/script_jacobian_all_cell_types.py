#!/home/sfleming/miniforge3/envs/cellarium/bin/python

from cellarium.ml.downstream.cellarium_utils import get_pretrained_model_as_pipeline, harmonize_anndata_with_model
from cellarium.ml.downstream.noise_prompting import compute_jacobian
import anndata
import torch
import numpy as np
import pandas as pd
import glob
import time
import os

suffix = "_jacobian_20240717.csv"

pipeline = get_pretrained_model_as_pipeline(device="cuda" if torch.cuda.is_available() else "cpu")

# adata_1m = anndata.read_h5ad('/home/sfleming/cellarium-ml/notebooks/outputs/three_prime_whole_cell_1M.h5ad')

# run this as a pipeline

# files = glob.glob("/home/sfleming/cellarium-ml/notebooks/cell_selection/random_cells/*.h5ad")
files = glob.glob("/home/sfleming/cellarium-ml/notebooks/cell_selection/hematopoiesis_means/*.h5ad")
files = [f for f in files if ('noise_prompt' not in f)]
files = [f for f in files if not os.path.exists(f[:-5] + suffix)]

print('working on files:')
print(files)

if len(files) == 0:
    raise ValueError('no files found at /home/sfleming/cellarium-ml/notebooks/cell_selection/hematopoiesis_means/*.h5ad')

for i, file in enumerate(files):
    print(f"Working on {file} ({i + 1}/{len(files)})")
    adata = anndata.read_h5ad(file)
    adata.X = adata.layers['count'].copy()
    means_in_celltype = adata.var[['mean']].copy()
    print("... harmonizing")
    adata_cell = harmonize_anndata_with_model(adata, pipeline)
    adata_cell.var['gpt_include'] = False
    adata_cell.var['gpt_include'] = adata.var['gpt_include'].copy().astype(bool)
    adata_cell.var.loc[pd.isnull(adata_cell.var['gpt_include']), 'gpt_include'] = False
    adata_cell.var['gpt_include'] = adata_cell.var['gpt_include'].astype(bool)
    print(adata_cell.var['gpt_include'].value_counts(dropna=False))
    adata_cell.layers['count'] = adata_cell.X.copy()
    adata_cell.var['mean'] = 0
    adata_cell.var['mean'] = means_in_celltype['mean'].copy().astype(float)
    adata_cell.var.loc[pd.isnull(adata_cell.var['mean']), 'mean'] = 0
    adata_cell.var['mean'] = adata_cell.var['mean'].astype(float)

    # limit to at most 2500 genes with max mean expression
    if adata_cell.var['gpt_include'].sum() > 2500:
        keepers = adata_cell.var['mean'][adata_cell.var['gpt_include']].sort_values(ascending=False).index[:2500]
        adata_cell.var['gpt_include'] = False
        adata_cell.var.loc[keepers, 'gpt_include'] = True
    
    var_inclusion_key = 'gpt_include'
    print(f"... {adata_cell.var[var_inclusion_key].sum()} genes included")

    print("... computing jacobian")
    t = time.time()
    jacobian_df = compute_jacobian(
        adata_cell,
        pipeline=pipeline,
        var_key_include_genes=var_inclusion_key,
        summarize='mean',
        layer='count',
        var_key_gene_name='gene_name',
    )
    print(f"... done in {(time.time() - t) / 60:.2f} mins")

    output_file = file[:-5] + suffix
    jacobian_df.to_csv(output_file, index=True)
    print(f"Saved {output_file}\n")
