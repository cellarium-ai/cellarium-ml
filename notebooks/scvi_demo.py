from notebooks_functions import get_dataset_from_anndata, embed
import torch
import os
import yaml
import subprocess
from cellarium.ml.core import CellariumPipeline, CellariumModule
import os
import scanpy as sc
config_file = "../example_configs/scvi_pbmc_config.yaml"

#subprocess.call(["/opt/conda/bin/python","../cellarium/ml/cli.py","scvi","fit","-c",config_file])

checkpoint_file = 'lightning_logs/version_0/checkpoints/epoch=49-step=3150.ckpt'

# load the trained model
scvi_model = CellariumModule.load_from_checkpoint(checkpoint_file).model

# move the model to the correct device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scvi_model.to(device)
scvi_model.eval()

# construct the pipeline
pipeline = CellariumPipeline([scvi_model])

# get the location of the dataset
with open(config_file, "r") as file:
    config_dict = yaml.safe_load(file)
data_path = config_dict['data']['dadc']['init_args']['filenames']
print(f'Data is coming from {data_path}')

# get a dataset object
dataset = get_dataset_from_anndata(
    data_path,
    batch_size=128,
    shuffle=False,
    seed=0,
    drop_last=False,
)
adata = embed(dataset, pipeline,device= device)
sc.set_figure_params(fontsize=14, vector_friendly=True)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=20, n_neighbors=15, metric='euclidean', method='umap')
sc.tl.umap(adata)
adata.obsm['X_raw_umap'] = adata.obsm['X_umap'].copy()

sc.pl.embedding(adata, basis='raw_umap', color=['final_annotation', 'batch'], ncols=1)

sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=15, metric='euclidean', method='umap')
sc.tl.umap(adata)
adata.obsm['X_scvi_umap'] = adata.obsm['X_umap'].copy()

sc.pl.embedding(adata, basis='scvi_umap', color=['final_annotation', 'batch'], ncols=1)