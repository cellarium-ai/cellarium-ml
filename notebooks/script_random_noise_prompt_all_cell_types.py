#!/home/sfleming/miniforge3/envs/cellarium/bin/python

from cellarium.ml.downstream.cellarium_utils import get_pretrained_model_as_pipeline, harmonize_anndata_with_model
from cellarium.ml.downstream.gene_set_utils import GeneSetRecords
from cellarium.ml.downstream.noise_prompting import noise_prompt_random
import anndata
import torch
import umap
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sklearn
import glob
import os

suffix = "_noise_prompted_20240709_gaussian_noise.h5ad"

pipeline = get_pretrained_model_as_pipeline(device="cuda" if torch.cuda.is_available() else "cpu")

# adata_1m = anndata.read_h5ad('/home/sfleming/cellarium-ml/notebooks/outputs/three_prime_whole_cell_1M.h5ad')

# run this as a pipeline

n_perturbations = 1000
perturbation_scale = 0.1
min_factors = 5
gene_scale_eps = 1e-4

assert n_perturbations > 100, 'using a 100-dim umap, so use more than 100 perturbations'

# files = glob.glob("/home/sfleming/cellarium-ml/notebooks/cell_selection/random_cells/*.h5ad")
files = glob.glob("/home/sfleming/cellarium-ml/notebooks/cell_selection/hematopoiesis_means/*.h5ad")
files = [f for f in files if 'noise_prompted' not in f]
files = [f for f in files if not os.path.exists(f[:-5] + suffix)]

if len(files) == 0:
    raise ValueError('no files found at /home/sfleming/cellarium-ml/notebooks/cell_selection/hematopoiesis_means/*.h5ad')

for i, file in enumerate(files):
    print(f"Working on {file} ({i + 1}/{len(files)})")
    adata = anndata.read_h5ad(file)
    adata.X = adata.layers['count'].copy()
    means_in_celltype = adata.var[['mean']].copy()
    # means_in_celltype = np.array(
    #     adata_1m[adata_1m.obs['cell_type'] == adata.obs['cell_type'].item()].X.mean(axis=0)
    # ).squeeze()
    print("... harmonizing")
    adata_cell = harmonize_anndata_with_model(adata, pipeline)
    adata_cell.var['gpt_include'] = False
    adata_cell.var['gpt_include'] = adata.var['gpt_include'].copy().astype(bool)
    adata_cell.var.loc[pd.isnull(adata_cell.var['gpt_include']), 'gpt_include'] = False
    adata_cell.var['gpt_include'] = adata_cell.var['gpt_include'].astype(bool)
    print(adata_cell.var['gpt_include'].value_counts(dropna=False))
    # import sys
    # sys.exit(1)
    # adata_cell.var['gpt_include'][pd.isnull(adata_cell.var['gpt_include'])] = False
    adata_cell.layers['count'] = adata_cell.X.copy()
    adata_cell.var['mean'] = 0
    adata_cell.var['mean'] = means_in_celltype['mean'].copy().astype(float)
    adata_cell.var.loc[pd.isnull(adata_cell.var['mean']), 'mean'] = 0
    adata_cell.var['mean'] = adata_cell.var['mean'].astype(float)
    
    # adata_cell.var['gpt_include2'] = adata_cell.var['mean'] > np.percentile(adata_cell.var['mean'], q=60)
    # var_inclusion_key = 'gpt_include2'
    var_inclusion_key = 'gpt_include'
    print(f"... {adata_cell.var[var_inclusion_key].sum()} genes included")

    adata_cell.var['flat'] = 1.0

    print("... noise prompting")
    adata_out = noise_prompt_random(
        adata_cell, 
        pipeline=pipeline,
        n_perturbations=n_perturbations, 
        perturbation_scale=perturbation_scale,
        var_key_include_genes=var_inclusion_key,
        var_key_gene_scale='measured_gpt_1',  # make method use output of measured_gpt
        gene_scale_eps=gene_scale_eps,
        perturbations_poisson=False,
        seed=0,
        n_pcs=10,
        n_ics=10,
    )

    print("... computing logFCs")
    logfc = np.log2(adata_out.layers['perturbed_gpt'].A) - np.log2(adata_out.layers['measured_gpt'].A)
    adata_out.layers['logfc'] = logfc.copy()

    print("... estimating manifold dimension using UMAP + PCA")
    um_model = umap.UMAP(n_components=100, n_epochs=5000)
    um = um_model.fit_transform(logfc)
    pca = PCA(n_components=50)
    pcs = pca.fit_transform(um)
    target_cumulative_variance_explained = 0.9
    n_components = (np.cumsum(pca.explained_variance_ratio_) < target_cumulative_variance_explained).sum() + 1
    n_components = max(min_factors, n_components)
    print(f"... using {n_components} factors to explain "
          f"{pca.explained_variance_ratio_[:n_components].sum():.0%} of variance in the UMAP")

    print("... using dictionary learning to compute factors")
    dl = sklearn.decomposition.DictionaryLearning(
        n_components=n_components, 
        transform_n_nonzero_coefs=10, 
        positive_dict=True,
        n_jobs=1,
    )
    dl.fit(logfc)
    adata_out.varm['dictionary_learning'] = dl.components_.T

    output_file = file[:-5] + suffix
    adata_out.write_h5ad(output_file)
    print(f"Saved {output_file}\n")
