import concurrent
import gc
import itertools
import time
import types
import warnings
import datetime
import re
import pandas as pd
import seaborn as sns
import os, sys
local_repository= True
if local_repository:
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /opt/project/cellarium-ml/ or /home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml
    sys.path.insert(1,module_dir)
    # Set the PYTHONPATH environment variable for subprocess to inherit
    env = os.environ.copy()
    env['PYTHONPATH'] = module_dir
else:
    # Set the PYTHONPATH environment variable
    env = os.environ.copy()
from cellarium.ml.core import CellariumPipeline, CellariumModule
from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.data.fileio import read_h5ad_file
from cellarium.ml.utilities.data import AnnDataField, densify, categories_to_codes
import multiprocessing
from multiprocessing.pool import ThreadPool
from functools import partial
import torch
import numpy as np
import anndata
import scanpy as sc
import scipy.sparse as sp
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import os,shutil
import scvi
import yaml
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import rc_context
from sklearn.decomposition import NMF
import dataframe_image as dfi
import rapids_singlecell as rsc
from collections import defaultdict, Counter
from google.cloud import storage
from anndata import AnnData, read_h5ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

#mpl.use('agg')

def subset_adata_by_var(adata,nsamples,filename,var="study"):

    unique_studies = adata.obs[var].value_counts()
    data_unobserved_list = []
    nobs = adata.obs.shape[0]

    for partition, counts in unique_studies.items():
        data_unobserved_i = adata[(adata.obs["study"] == partition)]
        percentage = (counts*100)/nobs
        new_counts = int(percentage*nsamples/100)
        data_unobserved_i = sc.pp.subsample(data_unobserved_i, fraction=1, n_obs=new_counts,copy=True)
        data_unobserved_list.append(data_unobserved_i)

    adata = data_unobserved_list[0].concatenate(data_unobserved_list[1:])

    adata.write(filename)

def divide_train_test(adata:anndata.AnnData,adata_file:str,ntrain:float=0.8):
    """:param adata
        :param adata_file: Path to
       :param ntrain: percentage of train datapoints
        """

    ndata = adata.X.shape[0]
    ntrain = int(ndata*ntrain)
    idx_all = np.arange(0,ndata)
    train_idx_int = np.random.choice(idx_all,ntrain,replace=False)
    train_idx = (idx_all[...,None] == train_idx_int).any(-1)

    adata_train = adata[train_idx]
    adata_test = adata[~train_idx]
    adata_train.obs["subset"] = "Train"
    adata_test.obs["subset"] = "Test"

    adata_train.write(adata_file.replace(".h5ad","_train.h5ad"))
    adata_test.write(adata_file.replace(".h5ad","_test.h5ad"))

    return adata_train,adata_test

def folders(folder_name,basepath,overwrite=True):
    """ Creates a folder at the indicated location. It rewrites folders with the same name
    :param str folder_name: name of the folder
    :param str basepath: indicates the place where to create the folder
    """
    #basepath = os.getcwd()
    if not basepath:
        newpath = folder_name
    else:
        newpath = basepath + "/%s" % folder_name
    if not os.path.exists(newpath):
        try:
            original_umask = os.umask(0)
            os.makedirs(newpath, 0o777)
        finally:
            os.umask(original_umask)
    else:
        if overwrite:
            print("Removing folder and subdirectories. Review that this is the desired behaviour or set overwrite to False)") #if this is reached is because you are running the folders function twice with the same folder name
            shutil.rmtree(newpath)  # removes all the subdirectories!
            os.makedirs(newpath,0o777)
        else:
            pass


def matching_file(datapath:str,filename:str):
    """Finds any file starting with <filename>"""
    pattern = re.compile(filename)
    matched = [ file if pattern.match(file) else None for file in os.listdir(datapath)]
    matched = [i for i in matched if i is not None]
    if len(matched) > 1:
        matched = [match for match in matched if "_maxfreqcell" not in match]
        #Return the smallest? kind of works
        matched = list(sorted(matched,key=len))
        filepath = os.path.join(datapath, matched[0])
    elif len(matched) == 1:
        filepath = os.path.join(datapath, matched[0])
    else:
        filepath = f"{datapath}/{filename}"

    return matched,filepath


def setup_model(checkpoint_file):
    # load the trained model
    scvi_model = CellariumModule.load_from_checkpoint(checkpoint_file).model
    # move the model to the correct device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scvi_model.to(device)
    scvi_model.eval()
    # construct the pipeline
    pipeline = CellariumPipeline([scvi_model])

    return pipeline, device


def download_predict(config_file,gene_names,filepath,pipeline,device,matched,filename,overwrite):
    # get the location of the dataset
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    if not matched or overwrite:
        data_path = config_dict['data']['dadc']['init_args']['filenames']
        print(f'Data is coming from {data_path}')
        # get a dataset object
        dataset = get_dataset_from_anndata(
            data_path,
            batch_size=128,
            shard_size=config_dict["data"]["dadc"]["init_args"]["shard_size"],
            shuffle=False,
            seed=0,
            obs_batch_key = config_dict["data"]["batch_keys"]["batch_index_n"]["key"],#recall we can also change the obs key
            drop_last=False,
        )
        filepath = filepath + ".h5ad" if not filepath.endswith(".h5ad") else filepath
        adata = embed(dataset, pipeline,device= device,filepath=filepath)
        # Highlight: Reconstruct de-noised/de-batched data
        for label in range(pipeline[-1].n_batch):
            print("Label: {}".format(label))
            adata_tmp = reconstruct_debatched(dataset, pipeline, transform_to_batch_label=label,layer_key_added=f'scvi_reconstructed_{label}', device=device)
            adata.layers[f'scvi_reconstructed_{label}'] = adata_tmp.layers[f'scvi_reconstructed_{label}']
            break


        if gene_names: #TODO: Warning: Not all datasets have gone through these conditions, in case it fails ...
            if gene_names == "var.index":
                adata.var_names = adata.var.index.map(str.upper)
            else:
                adata.var_names = adata.var[gene_names]
                adata.var_names = adata.var_names.map(str.upper)
                adata.var_names.name = "gene_names" #hope this helps with the name crashing that was occuring
        adata.write(filepath)
        return adata
    else:
        print(f"File found : {filename}")
        print("Reading file : {}".format(filepath))
        adata = sc.read(filepath)

        return adata

class AutosizedDistributedAnnDataCollection(DistributedAnnDataCollection):

    def __init__(self, *args, **kwargs):
        # I'm being lazy here and doing something real ugly
        # I want it to take the shard_size from the first file
        try:
            # this allows super to find the list of filenames
            super().__init__(*args, **kwargs)
        except AssertionError:
            try:
                # this allows super to create the cache
                kwargs.pop("shard_size")
                kwargs = kwargs | {"shard_size": 10000}
                super().__init__(*args, **kwargs)
            except AssertionError:
                pass
            # load first file and cache it
            adata0 = self.cache[self.filenames[0]] = read_h5ad_file(self.filenames[0])
            # pull shard_size from that file
            kwargs.pop("shard_size")
            kwargs = kwargs | {"shard_size": len(adata0)}
            # finally initialize for real
            super().__init__(*args, **kwargs)


def get_dataset_from_anndata(
    adata: anndata.AnnData | str,
    batch_size: int = 128,
    shard_size: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    obs_batch_key: str = 'batch',
    drop_last: bool = False,
):
    """
    Get IterableDistributedAnnDataCollectionDataset from an AnnData object or h5ad file specifier.

    Args:
        adata: AnnData object or h5ad file, allowing brace notation for several files.
        batch_size: Batch size.
        shard_size: Shard size.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        IterableDistributedAnnDataCollectionDataset.
    """

    if isinstance(adata, anndata.AnnData):
        tmpfile = tempfile.mkstemp(suffix='.h5ad')
        adata.write(tmpfile[1])
        file = tmpfile[1]
    else:
        file = adata

    dadc = AutosizedDistributedAnnDataCollection(
        file,
        shard_size=shard_size,
        max_cache_size=1,
        #limits = [1000] #31774 -> leave unassigned to have the entire dataset
    )

    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "var_names_g": AnnDataField(attr="var_names"),
            "batch_index_n": AnnDataField(attr="obs", key=obs_batch_key, convert_fn=categories_to_codes),
        },
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    return dataset

def embed(
    dataset: IterableDistributedAnnDataCollectionDataset,
    pipeline: CellariumPipeline,
    maximum_anndata_files_to_download: int = 5,
    obsm_key_added: str = 'X_scvi',
    device : str = "cpu",
    filepath: str = ""
) -> anndata.AnnData:
    """
    Embed the dataset using the pipeline.

    Args:
        dataset: Dataset.
        pipeline: Pipeline.
        maximum_anndata_files_to_download: Maximum number of anndata files to download.

    Returns:
        AnnData with scVI embeddings in adata.obsm[obsm_key_added]
    """

    # get the anndata object
    adatas = [dataset.dadc.adatas[i].adata for i in range(min(maximum_anndata_files_to_download, len(dataset.dadc.adatas)))]
    adata = anndata.concat(adatas, axis=0, merge="same")

    # get the latent space dimension
    latent_space_dim = pipeline[-1].z_encoder.mean_encoder.out_features

    # run the pipeline
    i = 0
    adata.obsm[obsm_key_added] = np.zeros((len(adata), latent_space_dim), dtype=np.float32)
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            batch['x_ng'] = torch.from_numpy(batch['x_ng'].copy()).to(device)
            batch['batch_index_n'] = torch.from_numpy(batch['batch_index_n']).to(device)
            out = pipeline.predict(batch)
            z_mean_nk = out['qz'].mean
            adata.obsm[obsm_key_added][i:(i + len(z_mean_nk)), :] = z_mean_nk.cpu().numpy()
            i += len(z_mean_nk)
    adata.write(filepath)
    return adata


def reconstruct_debatched(
    dataset: IterableDistributedAnnDataCollectionDataset,
    pipeline: CellariumPipeline,
    transform_to_batch_label: int,
    maximum_anndata_files_to_download: int = 5,
    layer_key_added: str = 'scvi_reconstructed',
    device: str = "cpu",
) -> anndata.AnnData:
    """
    Reconstruct the dataset using the pipeline.

    Args:
        dataset: Dataset.
        pipeline: Pipeline.
        transform_to_batch_label: batch label to reconstruct as
        maximum_anndata_files_to_download: Maximum number of anndata files to download.
        layer_key_added: Output counts will be stored in adata.layers[layer_key_added]

    Returns:
        AnnData with scVI reconstruction in adata.layers[layer_key_added]
    """

    # get the anndata object
    adatas = [dataset.dadc.adatas[i].adata for i in range(min(maximum_anndata_files_to_download, len(dataset.dadc.adatas)))]
    adata = anndata.concat(adatas, axis=0, merge="same")

    scvi_model = pipeline[-1]

    # run the pipeline
    sparse_coos = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            batch['x_ng'] = torch.from_numpy(batch['x_ng']).to(device)
            batch['batch_index_n'] = torch.from_numpy(batch['batch_index_n']).to(device)
            # out = pipeline.predict(batch)
            counts_ng = scvi_model.reconstruct(
                x_ng=batch['x_ng'] ,
                var_names_g=batch['var_names_g'],
                batch_index_n=batch['batch_index_n'],
                transform_batch=transform_to_batch_label,
                sample=False, #sample False to return the mean, if sample True it returns 1 sample
            )
            sparse_coos.append(sp.coo_matrix(counts_ng.cpu().numpy()))

    # make the final sparse matrix and keep it as a layer
    csr = sp.vstack(sparse_coos).tocsr()
    adata.layers[layer_key_added] = csr


    return adata

#
# def plot_umap(adata: anndata.AnnData,filepath: str,figpath:str,figname,basis,rep,color_keys = ['final_annotation', 'batch']):
#     print(f"UMAP of {basis} ...")
#
#
#     sc.set_figure_params(fontsize=14, vector_friendly=True)
#     sc.pp.neighbors(adata, use_rep=rep, n_neighbors=15, metric='euclidean',knn=True,method="umap") #X_scvi #X_raw
#     sc.pp.pca(adata,svd_solver="auto")
#
#     sc.tl.umap(adata)
#     adata.obsm[f'X_{basis}'] = adata.obsm['X_umap'].copy()
#
#     palette = list(matplotlib.colors.CSS4_COLORS.values()) if len(adata.obs[color_keys[0]].value_counts().keys()) > 20 else "tab20b"
#
#     sc.pl.embedding(adata, basis=f'{basis}',color=color_keys,
#                     palette=palette,
#                     ncols=1,show=False)
#     plt.savefig(f"{figpath}/{figname}.jpg", bbox_inches="tight")
#     plt.clf()
#     plt.close()
#     if basis == "raw_umap":
#         adata.obsm['X_raw_umap'] = adata.obsm['X_umap'].copy()
#     else:
#
#         adata.layers['raw'] = adata.X.copy()
#
#     adata.write(filepath)
#
#     return adata
#
#
# def plot_umap_cuda(adata: anndata.AnnData,filepath: str,figpath:str,figname,basis,rep,color_keys = ['final_annotation', 'batch']):
#     print(f"UMAP of {basis} GPU-accelerated ...")
#     rsc.get.anndata_to_GPU(adata=adata,convert_all=True)
#     rsc.pp.filter_genes(adata, min_count=1)
#
#     sc.set_figure_params(fontsize=14, vector_friendly=True)
#     rsc.pp.neighbors(adata, use_rep=rep, n_neighbors=15, metric='euclidean') #X_scvi #X_raw
#     rsc.pp.pca(adata,svd_solver="auto")
#     rsc.tl.umap(adata)
#     adata_transformed = rsc.get.anndata_to_CPU(adata,convert_all=True,copy=True)
#
#     adata_transformed.obsm[f'X_{basis}'] = adata_transformed.obsm['X_umap'].copy()
#     palette = list(matplotlib.colors.CSS4_COLORS.values()) if len(adata.obs[color_keys[0]].value_counts().keys()) > 20 else "tab20b"
#
#     sc.pl.embedding(adata_transformed, basis=f'{basis}',color=color_keys,
#                     palette=palette,
#                     ncols=1,show=False)
#     plt.savefig(f"{figpath}/{figname}_GPU.jpg", bbox_inches="tight")
#     plt.clf()
#     plt.close()
#     if basis == "raw_umap":
#         adata_transformed.obsm['X_raw_umap'] = adata.obsm['X_umap'].copy()
#     else:
#
#         adata_transformed.layers['raw'] = adata.X.copy()
#
#     adata_transformed.write(filepath)
#
#     return adata_transformed

class ComputeUMAP():
    def __init__(self,adata: anndata.AnnData,filepath: str,figpath:str,figname,basis,rep,use_cuda, overwrite, color_keys = ['final_annotation', 'batch']):
        self.adata = adata
        self.filepath = filepath
        self.figpath = figpath
        self.figname = figname
        self.basis = basis
        self.rep = rep
        self.color_keys = color_keys
        self.use_cuda = use_cuda
        self.overwrite = overwrite

    def plot_umap_cuda(self):
        adata = self.adata

        print(f"UMAP of {self.basis} GPU-accelerated ...")
        rsc.get.anndata_to_GPU(adata=adata, convert_all=True)
        rsc.pp.filter_genes(adata, min_count=1)
        sc.set_figure_params(fontsize=14, vector_friendly=True)
        rsc.pp.pca(adata, svd_solver="auto")
        rsc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15, metric='euclidean')  # X_scvi #X_raw
        rsc.tl.umap(adata)
        adata_transformed = rsc.get.anndata_to_CPU(adata, convert_all=True, copy=True)

        adata_transformed.obsm[f'X_{self.basis}'] = adata_transformed.obsm['X_umap'].copy()
        palette = list(matplotlib.colors.CSS4_COLORS.values()) if len(adata.obs[self.color_keys[0]].value_counts().keys()) > 20 else "tab20b"

        sc.pl.embedding(adata_transformed, basis=f'{self.basis}', color=self.color_keys,
                        palette=palette,
                        ncols=1, show=False)
        plt.savefig(f"{self.figpath}/{self.figname}_GPU.jpg", bbox_inches="tight")
        plt.clf()
        plt.close()
        if self.basis == "raw_umap":
            adata_transformed.obsm['X_raw_umap'] = adata.obsm['X_umap'].copy()
        else:

            adata_transformed.layers['raw'] = adata.X.copy()


        plt.show()

        #adata_transformed.write(self.filepath)

        return adata_transformed

    def plot_umap(self):
        adata = self.adata
        print(f"UMAP of {self.basis} ...")

        sc.set_figure_params(fontsize=14, vector_friendly=True,figsize=(5,8))
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.pca(adata, svd_solver="auto")
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15, metric='euclidean', knn=True,method="umap")  # X_scvi #X_raw
        sc.tl.umap(adata)

        adata.obsm[f'X_{self.basis}'] = adata.obsm['X_umap'].copy()


        palette = list(matplotlib.colors.CSS4_COLORS.values()) if len(
            adata.obs[self.color_keys[0]].value_counts().keys()) > 20 else "tab20b"

        plot = sc.pl.embedding(adata, basis=f'{self.basis}', color=self.color_keys,
                        palette=palette, ncols=1, show=False, wspace=1, hspace=1)
        plt.savefig(f"{self.figpath}/{self.figname}.jpg", bbox_inches="tight")
        plt.clf()
        plt.close()
        if self.basis == "raw_umap":
            adata.obsm['X_raw_umap'] = adata.obsm['X_umap'].copy()
        else:
            adata.layers['raw'] = adata.X.copy()

        #adata.write(self.filepath) #TODO: fix?

        return adata

    def plot_umap_plotly(self):
        """

        :return:
        """

        adata = self.adata
        print(f"UMAP of {self.basis} ...")

        sc.set_figure_params(fontsize=14, vector_friendly=True)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.pca(adata, svd_solver="auto")
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15, metric='euclidean', knn=True,method="umap")  # X_scvi #X_raw
        sc.tl.umap(adata)

        adata.obsm[f'X_{self.basis}'] = adata.obsm['X_umap'].copy()


        # df = pd.DataFrame({
        #                    "UMAP_x": adata.obsm['X_umap'][:,0],
        #                    "UMAP_y": adata.obsm['X_umap'][:,1],
        #                    "tissue_labels": adata.obs['tissue_label'],
        #                    "cell_type_label": adata.obs['cell_type_label'],
        #                    })
        #
        # df.to_csv(f"{self.figpath}/umap_labels_only_glyco.tsv",sep="\t",index=False)



        tissue_labels = list(adata.obs[self.color_keys[0]].value_counts().keys())
        cell_labels = list(adata.obs[self.color_keys[1]].value_counts().keys())

        palette = list(matplotlib.colors.CSS4_COLORS.values()) if len(tissue_labels) > 20 else "tab20b"

        print("Making interactive plotly")
        # Create figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Tissue", "Cell types"])

        # Add separate traces for each label
        for label in tissue_labels:

            adata_subset = adata[adata.obs[self.color_keys[0]] == label]

            umap_x = adata_subset.obsm["X_umap"][:,0]
            umap_y = adata_subset.obsm["X_umap"][:,1]

            fig.add_trace(go.Scatter(
                x=umap_x, y=umap_y,
                #mode='markers+text',
                mode='markers',
                text=[label] * len(umap_x),  # Show label on points
                textposition="top center",
                #name=f'Label {label}',  # Toggle each label using legend
                marker=dict(size=10)
            ), row=1, col=1)

            del adata_subset

        # Add separate traces for each label
        for label in cell_labels:

            adata_subset = adata[adata.obs[self.color_keys[1]] == label]

            umap_x = adata_subset.obsm["X_umap"][:,0]
            umap_y = adata_subset.obsm["X_umap"][:,1]

            fig.add_trace(go.Scatter(
                x=umap_x, y=umap_y,
                mode='markers',
                #mode='markers+text',
                text=[label] * len(umap_x),  # Show label on points
                textposition="top center",
                #name=f'Label {label}',  # Toggle each label using legend
                marker=dict(size=10)
            ), row=1, col=2)

        # Create dropdown menus for each plot
        dropdowns = [
            {
                "buttons": [
                    {"label": "Show All", "method": "update",
                     "args": [{"visible": [True] * (len(tissue_labels) + len(cell_labels))}]},
                    *[
                        {"label": f"Show {label}", "method": "update",
                         "args": [{"visible": [(label == lbl) if i < len(tissue_labels) else True for i, lbl in
                                               enumerate(tissue_labels + cell_labels)]}]}
                        for label in tissue_labels
                    ]
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.2, "y": 1.15  # Position dropdown
            },
            {
                "buttons": [
                    {"label": "Show All", "method": "update",
                     "args": [{"visible": [True] * (len(tissue_labels) + len(cell_labels))}]},
                    *[
                        {"label": f"Show {label}", "method": "update",
                         "args": [{"visible": [(True if i < len(tissue_labels) else label == lbl) for i, lbl in
                                               enumerate(tissue_labels + cell_labels)]}]}
                        for label in cell_labels
                    ]
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.8, "y": 1.15  # Position dropdown
            }
        ]

        # Apply dropdown menus
        fig.update_layout(updatemenus=dropdowns, title_text="Interactive scatter Plots")

        fig.write_html(f"{self.figpath}/UMAP_plotly_only_glyco.html")

    def run(self):
        if f"X_{self.basis}" not in list(self.adata.obsm.keys()) or self.overwrite:
            if self.use_cuda:
                return self.plot_umap_cuda()
            else:
                return self.plot_umap()
                #return self.plot_umap_plotly()
        else:
            print(f"Key 'X_{self.basis}' found, continue")

            return self.adata


def reconstruct_debatched(
    dataset: IterableDistributedAnnDataCollectionDataset,
    pipeline: CellariumPipeline,
    transform_to_batch_label: int,
    maximum_anndata_files_to_download: int = 5,
    layer_key_added: str = 'scvi_reconstructed',
    device:bool = "cuda",
) -> anndata.AnnData:
    """
    Reconstruct the dataset using the pipeline.

    Args:
        dataset: Dataset.
        pipeline: Pipeline.
        transform_to_batch_label: batch label to reconstruct as
        maximum_anndata_files_to_download: Maximum number of anndata files to download.
        layer_key_added: Output counts will be stored in adata.layers[layer_key_added]

    Returns:
        AnnData with scVI reconstruction in adata.layers[layer_key_added]
    """


    # get the anndata object
    adatas = [dataset.dadc.adatas[i].adata for i in range(min(maximum_anndata_files_to_download, len(dataset.dadc.adatas)))]
    adata = anndata.concat(adatas, axis=0, merge="same")

    scvi_model = pipeline[-1]

    # run the pipeline
    sparse_coos = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            batch['x_ng'] = torch.from_numpy(batch['x_ng']).to(device)
            batch['batch_index_n'] = torch.from_numpy(batch['batch_index_n']).to(device)
            # out = pipeline.predict(batch)
            counts_ng = scvi_model.reconstruct(
                x_ng=batch['x_ng'] ,
                var_names_g=batch['var_names_g'],
                batch_index_n=batch['batch_index_n'],
                transform_batch=transform_to_batch_label,
                sample=True,
            )
            sparse_coos.append(sp.coo_matrix(counts_ng.cpu().numpy()))

    # make the final sparse matrix and keep it as a layer
    csr = sp.vstack(sparse_coos).tocsr()
    adata.layers[layer_key_added] = csr

    return adata


def define_gene_expressions(adata,gene_set,filepath,gene_names,overwrite):

    if "genes_of_interest" not in list(adata.var.keys()) or overwrite:
        print("Key 'genes_of_interest' not found, computing")

        if gene_names:
            adata.var["genes_of_interest"] = adata.var[gene_names].isin(gene_set)
            adata_gene_set = adata[:, adata.var[gene_names].isin(gene_set)]
            print(adata_gene_set.var[gene_names].tolist())
        elif not gene_names:
            adata.var['genes_of_interest'] = adata.var_names.isin(gene_set)  # 23 glycogenes found only adata[:,adata.var_names.isin(gene_set)]
            adata_gene_set = adata[:, adata.var_names.isin(gene_set)]  # only used for counting
            print(adata_gene_set.var_names.tolist())
        else:
            warnings.warn("Please check for the name of the column/key for the HGCN gene names in your dataset and try again")

        #numpy.matrix
        # aggregate umi-count expression values
        adata.var['expr'] = np.array(adata.layers['raw'].sum(axis=0).getA()).squeeze()
        adata_gene_set.var['expr'] = np.array(adata_gene_set.layers['raw'].sum(axis=0).getA()).squeeze()
        high_expressed_genes = adata.var.sort_values(by='expr').index[-200:] # highly expressed among all genes
        top20_high_expressed_genes = adata.var.sort_values(by='expr').index[-20:] # top 20 highly expressed among all genes
        high_gene_set = adata_gene_set.var.sort_values(by='expr').index[-20:] #glyco high expressed
        low_expressed_genes = adata.var.sort_values(by='expr').index[:500]
        low_gene_set = adata_gene_set.var.sort_values(by='expr').index[:20]
        adata.var['low_exp_genes_of_interest'] = adata.var_names.isin(low_gene_set)
        adata.var['low_exp_genes'] = adata.var_names.isin(low_expressed_genes)
        adata.var['high_exp_genes_of_interest'] = adata.var_names.isin(high_gene_set)
        adata.var['high_exp_genes'] = adata.var_names.isin(high_expressed_genes)
        adata.var['top20_high_exp_genes'] = adata.var_names.isin(top20_high_expressed_genes)
        adata.write(filepath)
        return adata
    else:
        print("Key 'genes_of_interest' found, continue")
        return adata

def retrieve_genes():

    # gene_set_dict = {"all":['B4GALNT2', 'HS6ST2', 'MGAT4EP', 'ST6GALNAC4', 'ALG13', 'ABO', 'STT3B', 'PIGZ', 'UST', 'CHPF', 'ALG9', 'B4GALT4', 'HS2ST1', 'PIGG', 'ALG10B', 'ST6GALNAC3', 'DSEL', 'FUT4', 'GCNT2', 'XYLT1', 'PGAP4', 'GGTA1P', 'GLT8D2', 'UGGT1', 'GALNT14', 'EXTL3', 'POGLUT1', 'B4GAT1', 'MGAT1', 'NDST3', 'POFUT1', 'PYGM', 'ALG3', 'NDST2', 'UGT1A4', 'GAL3ST1', 'XXYLT1', 'UGT1A7', 'GAL3ST2', 'UGT2B7', 'GXYLT2', 'UGT8', 'B4GALNT3', 'CHST12', 'CHST13', 'POGLUT2', 'PIGN', 'FUT9', 'GLT8D1', 'RXYLT1', 'ST6GALNAC2', 'B4GALT5', 'CHST1', 'ALG14', 'GTDC1', 'GALNT5', 'TMTC1', 'ST8SIA5', 'EXTL2', 'GXYLT1', 'B3GNT4', 'B4GALT1', 'HS3ST2', 'PIGV', 'ST3GAL2', 'UGT2B17', 'ST8SIA1', 'B4GALT3', 'FUT1', 'GALNTL5', 'EXTL1', 'UGT2B28', 'PIGL', 'B4GALT6', 'GAL3ST4', 'GALNT13', 'ALG1L2', 'PYGB', 'ST8SIA6', 'DPY19L3', 'MAN1A2', 'GALNT1', 'UGT3A1', 'MGEA5', 'B3GALT4', 'DPY19L1', 'UGT2B4', 'CHST14', 'POMT1', 'DPM1', 'HAS3', 'GALNT16', 'ALG1', 'B3GNT5', 'PIGO', 'MAN1C1', 'ST3GAL4', 'PLOD3', 'FUT11', 'GALNT3', 'GYG2', 'B4GALNT4', 'MGAT4B', 'UGT1A9', 'CMAS', 'CHSY1', 'GALNT8', 'PYGL', 'B3GNT9', 'LARGE1', 'GLT1D1', 'TMTC4', 'MGAT2', 'DPY19L4', 'GLCE', 'CHST10', 'B3GALNT2', 'CHST11', 'UGT1A6', 'CHPF2', 'XYLT2', 'POGLUT3', 'GALNT2', 'ST8SIA4', 'MAN1B1', 'UGT2B15', 'ST3GAL6', 'ALG6', 'POMGNT1', 'UGT2A3', 'COLGALT1', 'UGT2B11', 'MFNG', 'DPAGT1', 'MAN1A1', 'OGT', 'GALNT15', 'HAS1', 'FUT7', 'CHST15', 'GYS2', 'FKTN', 'B3GALT5', 'ALG2', 'CHST7', 'HS3ST6', 'HS3ST1', 'ST3GAL5', 'ST6GAL2', 'LFNG', 'ALG1L', 'A4GALT', 'FUT10', 'B3GNT2', 'GALNT6', 'GALNT7', 'GCNT7', 'HS3ST5', 'POMGNT2', 'LARGE2', 'ST3GAL1', 'GALNT9', 'GALNTL6', 'FUT2', 'UGT1A1', 'GCNT1', 'CERCAM', 'UGT1A5', 'UGGT2', 'POMK', 'GALNT18', 'A3GALT2', 'GALNT10', 'CHST5', 'ALG12', 'PIGA', 'MGAT3', 'MGAT4D', 'GYS1', 'B3GALT2', 'RFNG', 'FUT3', 'B3GAT3', 'HAS2', 'B3GLCT', 'TMTC2', 'GLT6D1', 'B4GALT2', 'DSE', 'MGAT4A', 'UGT1A8', 'COLGALT2', 'GALNT11', 'ST3GAL3', 'ST6GALNAC5', 'C1GALT1', 'GALNT17', 'B3GAT2', 'ST8SIA3', 'CSGALNACT2', 'GALNT12', 'MGAT5', 'PIGM', 'UGT3A2', 'GCNT4', 'B3GNT8', 'GCNT3', 'TMTC3', 'HS3ST3B1', 'C1GALT1C1', 'HS6ST1', 'B3GNT7', 'B3GALT1', 'B3GALT6', 'A4GNT', 'B3GNT3', 'FKRP', 'GAL3ST3', 'FUT5', 'POFUT2', 'B3GALNT1', 'CHST6', 'UGT2A1', 'FUT6', 'UGT2B10', 'GYG1', 'HS6ST3', 'POMT2', 'ST6GALNAC6', 'GCNT2P', 'STT3A', 'B3GNT6', 'B4GALT7', 'EOGT', 'ALG11', 'WSCD1', 'ST8SIA2', 'UGCG', 'FUT8', 'ALG10', 'HS3ST3A1', 'CHST8', 'PIGB', 'ALG8', 'CHSY3', 'B3GAT1', 'B3GNTL1', 'WSCD2', 'CASD1', 'NDST4', 'MGAT4C', 'UGT1A3', 'GALNT4', 'ALG5', 'GBGT1', 'NDST1', 'ST6GALNAC1', 'CHST4', 'ST6GAL1', 'HS3ST4', 'CHST3', 'CHST2', 'CHST9', 'EXT2', 'B4GALNT1', 'MGAT5B', 'EXT1', 'CSGALNACT1', 'PIGW', 'DPY19L2']}
    # 'B4GALNT2', 'HS6ST2', 'MGAT4EP', 'ST6GALNAC4', 'ALG13', 'ABO', 'STT3B', 'PIGZ', 'UST', 'CHPF', 'ALG9', 'B4GALT4', 'HS2ST1', 'PIGG', 'ALG10B', 'ST6GALNAC3', 'DSEL', 'FUT4', 'GCNT2', 'XYLT1', 'PGAP4', 'GGTA1P', 'GLT8D2', 'UGGT1', 'GALNT14', 'EXTL3', 'POGLUT1', 'B4GAT1', 'MGAT1', 'NDST3', 'POFUT1', 'PYGM', 'ALG3', 'NDST2', 'UGT1A4', 'GAL3ST1', 'XXYLT1', 'UGT1A7', 'GAL3ST2', 'UGT2B7', 'GXYLT2', 'UGT8', 'B4GALNT3', 'CHST12', 'CHST13', 'POGLUT2', 'PIGN', 'FUT9', 'GLT8D1', 'RXYLT1', 'ST6GALNAC2', 'B4GALT5', 'CHST1', 'ALG14', 'GTDC1', 'GALNT5', 'TMTC1', 'ST8SIA5', 'EXTL2', 'GXYLT1', 'B3GNT4', 'B4GALT1', 'HS3ST2', 'PIGV', 'ST3GAL2', 'UGT2B17', 'ST8SIA1', 'B4GALT3', 'FUT1', 'GALNTL5', 'EXTL1', 'UGT2B28', 'PIGL', 'B4GALT6', 'GAL3ST4', 'GALNT13', 'ALG1L2', 'PYGB', 'ST8SIA6', 'DPY19L3', 'MAN1A2', 'GALNT1', 'UGT3A1', 'MGEA5', 'B3GALT4', 'DPY19L1', 'UGT2B4', 'CHST14', 'POMT1', 'DPM1', 'HAS3', 'GALNT16', 'ALG1', 'B3GNT5', 'PIGO', 'MAN1C1', 'ST3GAL4', 'PLOD3', 'FUT11', 'GALNT3', 'GYG2', 'B4GALNT4', 'MGAT4B', 'UGT1A9', 'CMAS', 'CHSY1', 'GALNT8', 'PYGL', 'B3GNT9', 'LARGE1', 'GLT1D1', 'TMTC4', 'MGAT2', 'DPY19L4', 'GLCE', 'CHST10', 'B3GALNT2', 'CHST11', 'UGT1A6', 'CHPF2', 'XYLT2', 'POGLUT3', 'GALNT2', 'ST8SIA4', 'MAN1B1', 'UGT2B15', 'ST3GAL6', 'ALG6', 'POMGNT1', 'UGT2A3', 'COLGALT1', 'UGT2B11', 'MFNG', 'DPAGT1', 'MAN1A1', 'OGT', 'GALNT15', 'HAS1', 'FUT7', 'CHST15', 'GYS2', 'FKTN', 'B3GALT5', 'ALG2', 'CHST7', 'HS3ST6', 'HS3ST1', 'ST3GAL5', 'ST6GAL2', 'LFNG', 'ALG1L', 'A4GALT', 'FUT10', 'B3GNT2', 'GALNT6', 'GALNT7', 'GCNT7', 'HS3ST5', 'POMGNT2', 'LARGE2', 'ST3GAL1', 'GALNT9', 'GALNTL6', 'FUT2', 'UGT1A1', 'GCNT1', 'CERCAM', 'UGT1A5', 'UGGT2', 'POMK', 'GALNT18', 'A3GALT2', 'GALNT10', 'CHST5', 'ALG12', 'PIGA', 'MGAT3', 'MGAT4D', 'GYS1', 'B3GALT2', 'RFNG', 'FUT3', 'B3GAT3', 'HAS2', 'B3GLCT', 'TMTC2', 'GLT6D1', 'B4GALT2', 'DSE', 'MGAT4A', 'UGT1A8', 'COLGALT2', 'GALNT11', 'ST3GAL3', 'ST6GALNAC5', 'C1GALT1', 'GALNT17', 'B3GAT2', 'ST8SIA3', 'CSGALNACT2', 'GALNT12', 'MGAT5', 'PIGM', 'UGT3A2', 'GCNT4', 'B3GNT8', 'GCNT3', 'TMTC3', 'HS3ST3B1', 'C1GALT1C1', 'HS6ST1', 'B3GNT7', 'B3GALT1', 'B3GALT6', 'A4GNT', 'B3GNT3', 'FKRP', 'GAL3ST3', 'FUT5', 'POFUT2', 'B3GALNT1', 'CHST6', 'UGT2A1', 'FUT6', 'UGT2B10', 'GYG1', 'HS6ST3', 'POMT2', 'ST6GALNAC6', 'GCNT2P', 'STT3A', 'B3GNT6', 'B4GALT7', 'EOGT', 'ALG11', 'WSCD1', 'ST8SIA2', 'UGCG', 'FUT8', 'ALG10', 'HS3ST3A1', 'CHST8', 'PIGB', 'ALG8', 'CHSY3', 'B3GAT1', 'B3GNTL1', 'WSCD2', 'CASD1', 'NDST4', 'MGAT4C', 'UGT1A3', 'GALNT4', 'ALG5', 'GBGT1', 'NDST1', 'ST6GALNAC1', 'CHST4', 'ST6GAL1', 'HS3ST4', 'CHST3', 'CHST2', 'CHST9', 'EXT2', 'B4GALNT1', 'MGAT5B', 'EXT1', 'CSGALNACT1', 'PIGW', 'DPY19L2'

    gene_set_dict = {
        'dpy19l': ['DPY19L1', 'DPY19L2', 'DPY19L3', 'DPY19L4'],
        'piga': ['PGAP4', 'PIGA', 'PIGB', 'PIGM', 'PIGV', 'PIGZ'],
        'ugcg': ['A4GALT', 'B3GALNT1', 'B3GALT4', 'B3GNT5', 'B4GALNT1', 'B4GALT5', 'B4GALT6', 'UGCG'],
        'ugt8': ['UGT8'],
        'ost': ['OST', 'STT3A/B', 'ALG10', 'ALG10B', 'ALG11', 'ALG12', 'ALG13', 'ALG14', 'ALG2', 'ALG3', 'ALG6', 'ALG8',
                'ALG9', 'DPAGT1', 'FUT8', 'MGAT1', 'MGAT2', 'MGAT3', 'MGAT4A', 'MGAT4B', 'MGAT4C', 'MGAT4D', 'MGAT5',
                'STT3A', 'UGGT1', 'UGGT2'],
        'ogt': ['OGT'],
        'colgalt': ['COLGALT1', 'COLGALT2'],
        'eogt': ['EOGT'],
        'galnt': ['B3GNT6', 'C1GALT1', 'C1GALT1C1', 'GALNT1', 'GALNT10', 'GALNT11', 'GALNT12', 'GALNT13', 'GALNT14',
                  'GALNT15', 'GALNT16', 'GALNT17', 'GALNT18', 'GALNT2', 'GALNT3', 'GALNT4', 'GALNT5', 'GALNT6',
                  'GALNT7', 'GALNT8', 'GALNT9', 'GCNT1', 'GCNT3', 'GCNT4'],
        'pofut1': ['LFNG', 'MFNG', 'POFUT1', 'RFNG'],
        'pofut2': ['B3GLCT', 'POFUT2'],
        'poglut': ['GXYLT1', 'GXYLT2', 'POGLUT1', 'POGLUT2', 'POGLUT3', 'XXYLT1'],
        'pomt': ['B3GALNT2', 'B4GAT1', 'FKRP', 'FKTN', 'LARGE1', 'LARGE2', 'MGAT5B', 'POMGNT1', 'POMGNT2', 'POMK',
                 'POMT1', 'POMT2', 'RXYLT1'],
        'tmtc': ['TMTC1', 'TMTC2', 'TMTC3', 'TMTC4', 'TMEM260'],
        'xylt1/2': ['B3GALT6', 'B3GAT3', 'B4GALT7', 'CHPF', 'CHPF2', 'CHSY1', 'CHSY3', 'CSGALNACT1', 'CSGALNACT2',
                    'EXT1', 'EXT2', 'EXTL1', 'EXTL2', 'EXTL3', 'XYLT1', 'XYLT2']
    }

    # gene_set_dict = {
    # "A4GALT" : ['A4GALT'],
    # "B3GALNT1" : ['B3GALNT1'],
    # "B3GNT5" : ['B3GNT5'],
    # "B4GALNT1" : ['B4GALNT1'],
    # "B3GALT4" : ['B3GALT4'],
    # "B4GALT5" : ['B4GALT5'],
    # "B4GALT6" : ['B4GALT6'],
    # "UGCG" : ['UGCG']
    # }
    #
    # gene_set_dict = {
    #     "Phase_S":['ABCA8','ANAPC2', 'ARFGAP1', 'BCAP29', 'BCL3', 'BZW1', 'C17orf99', 'CCBL1', 'CD2BP2', 'CDC16', 'CDC23', 'CDC6', 'CDT1', 'CTNNBL1', 'DAD1', 'EEF2', 'EIF5', 'FRMD1', 'GINS2', 'GLIS1', 'GON4L', 'HIST1H2AC', 'HRSP12', 'IFT122', 'LCE3D', 'LRRC49', 'MCCD1', 'MEX3C', 'MGRN1', 'MRPS26', 'MYO1F', 'MYO3B', 'NFKBIB', 'NKAIN1', 'PFN4', 'PLA1A', 'PODXL2', 'POLA1', 'POLE2', 'POLL', 'RAB11A', 'RAMP2', 'REG3G', 'RNF183', 'RPS2', 'RPS24', 'RRM1', 'RRM2', 'SENP6', 'SLBP', 'SMAD1', 'SMARCD1', 'TCF7L1', 'TFDP1', 'TMEM158', 'TMEM38A', 'TMEM92', 'U2AF2', 'WEE1', 'ZDHHC12', 'ZNF22'],
    #     "Phase_GM2":['ACADS','ACP2', 'ADAD2', 'ADAM29', 'ADCY7', 'ADPRHL1', 'ADRBK1', 'AEBP2', 'AKR1B10', 'AKR7A3', 'ALDH3B2', 'ALKBH6', 'ALPP', 'ALS2', 'ALS2CL', 'ANKMY1', 'ANLN', 'ANXA5', 'AP3B2', 'APC2', 'APH1A', 'API5', 'APOBEC3F', 'APOC1', 'APOC2', 'APOL1', 'AQP4', 'ARFGAP2', 'ARHGAP24', 'ARHGAP44', 'ARHGEF40', 'ARHGEF7', 'ARL10', 'ARRDC4', 'ARSG', 'ARX', 'ASIC2', 'ASPA', 'ATOX1', 'ATP4B', 'AURKB', 'BAIAP2', 'BAIAP3', 'BARD1', 'BCAM', 'BIRC5', 'BMP4', 'C11orf16', 'C12orf39', 'C14orf2', 'C16orf11', 'C17orf80', 'C19orf53', 'C1orf173', 'C1orf61', 'C3orf33', 'C4orf45', 'C7orf10', 'C7orf62', 'C8orf58', 'C9orf139', 'C9orf173', 'CA13', 'CA3', 'CA4', 'CAB39', 'CABYR', 'CACNA1H', 'CAMTA2', 'CAP1', 'CBX6', 'CCDC101', 'CCDC120', 'CCDC130','CCDC151', 'CCL11', 'CCL13', 'CCL20', 'CCZ1B', 'CD27', 'CDC25B', 'CDC40', 'CDC5L', 'CDCA3', 'CDCA4', 'CDCA5', 'CDCA8', 'CDHR3', 'CDK1', 'CDK12', 'CDK18', 'CDK2', 'CDKL2', 'CDKN1A', 'CDKN1B', 'CDKN2B', 'CDKN3', 'CDX1', 'CDX2', 'CEP170', 'CES1', 'CFLAR', 'CHMP4A', 'CHMP6', 'CHRM2', 'CHRM5', 'CHSY1', 'CITED2', 'CLCN1', 'CLEC2B', 'CLIC4', 'CMKLR1', 'CNEP1R1', 'COL9A1', 'COMMD3', 'COPS6', 'COX18', 'COX6A1', 'CPSF3L', 'CPT1A', 'CRYBB1', 'CRYBB3', 'CTXN1', 'CUL1', 'CUL5', 'CX3CR1', 'CYB561D1', 'CYP1A2', 'CYP2F1', 'DACT2', 'DAGLA', 'DDB1', 'DDX54', 'DENND2D', 'DGCR14', 'DHODH', 'DIO1', 'DLX5', 'DNAAF1', 'DNAI1', 'DNAJB2', 'DNAJC19', 'DPP3', 'DUSP10', 'DUSP9', 'DVL3', 'DYNLRB2', 'EBAG9', 'EFCAB12', 'EFCC1', 'EHD3', 'EIF2AK1', 'EIF3C', 'EML3', 'ENKD1', 'ENPP7', 'EPC1', 'EPHB2', 'EPN3', 'ESAM', 'ESPL1', 'ESRP2', 'ETFB', 'ETV2', 'EVA1B', 'EWSR1', 'F8A1', 'FAM124A', 'FAM131A', 'FAM175B', 'FAM178B', 'FAM219A', 'FAM65B', 'FAM81B', 'FAM83E', 'FAM92A1', 'FBXL16', 'FBXL2', 'FBXO33', 'FBXO5', 'FFAR1', 'FGF22', 'FGR', 'FIBCD1', 'FN3KRP', 'FOXA2', 'FRMD8', 'FSTL4', 'FUS', 'FZR1', 'GABRA2', 'GADL1', 'GAK', 'GAL3ST3', 'GALNT14', 'GAS2L2', 'GCH1', 'GFOD2', 'GFRA4', 'GIMAP8', 'GINS4', 'GJA8', 'GKN2', 'GLYCTK', 'GMPPA', 'GNRH2', 'GORASP1', 'GPBAR1', 'GPM6B', 'GPR101', 'GPR153', 'GPR50', 'GRIK4', 'GRM4', 'H2AFJ', 'H2AFY2', 'HAUS2', 'HAUS4', 'HDAC10', 'HECA', 'HES2', 'HIF3A', 'HIGD1A', 'HIST2H2BE', 'HK3', 'HMCES', 'HOMER1', 'HOPX', 'HRK', 'HSPA6', 'HSPB2', 'HSPB3', 'HSPBP1', 'IFI30', 'IKBKB', 'IL16', 'IL4I1', 'INCENP', 'ING2', 'INPP5K', 'IRAK1', 'IRF2BP2', 'IRF7', 'IRS1', 'ITFG3', 'ITGA7', 'ITM2C', 'JMJD6', 'JTB', 'KATNB1', 'KCNE1', 'KCNJ2', 'KCNN2', 'KCNS1', 'KCTD17', 'KDM2B', 'KIAA0895', 'KIF11', 'KIF12', 'KIF13B', 'KIF23', 'KIF5A', 'KIF6', 'KIFC2', 'KIFC3', 'KLC3', 'KLHL13', 'KMT2B', 'KRT5', 'LIMK2', 'LIN54', 'LOXL3', 'LTB', 'MAF', 'MAGED2', 'MAPK15', 'MAPKAPK3', 'MAS1L', 'MAST4', 'MBD2', 'MCM10', 'MDH2', 'MED9', 'MEF2C', 'MGST1', 'MIEN1', 'MIIP', 'MITD1', 'MKL1', 'MMP2', 'MOSPD3', 'MPDU1', 'MPHOSPH8', 'MPZL1', 'MRPL13', 'MRPL34', 'MRPL55', 'MRPS10', 'MRPS34', 'MSH6', 'MST4', 'MYBL2', 'MYCBPAP', 'MYCN', 'MYH3', 'MYOF', 'MYOZ1', 'NCOR2', 'NCSTN', 'NDUFB1', 'NELFB', 'NFKBIA', 'NFKBIE', 'NFKBIZ', 'NOL9', 'NPHP1', 'NR0B2', 'NR4A2', 'NSF', 'NSMCE1', 'NSMCE4A', 'NUDT8', 'NUTM1', 'ORC3', 'OXSM', 'P2RX2', 'PACSIN2', 'PAGE2', 'PAK6', 'PARP14', 'PARP9', 'PDE1B', 'PDGFA', 'PDZD2', 'PEPD', 'PFDN1', 'PFN1', 'PGA5', 'PGAP2', 'PGBD4', 'PHACTR2', 'PHLDB1', 'PIF1', 'PIFO', 'PLA2G3', 'PLCB3', 'PLCE1', 'PLEKHS1', 'PLIN3', 'PLK1', 'PLXNA1', 'PNLIPRP3', 'PNMT', 'POLR2E', 'POMC', 'POMK', 'PON1', 'POPDC2', 'PPFIA3', 'PPFIBP1', 'PPIH', 'PPP1CB', 'PPP1R14A', 'PPP1R32', 'PPP1R3F', 'PRC1', 'PRDM12', 'PRKCB', 'PRKD2', 'PRKG1', 'PROSER2', 'PRSS50', 'PRSS8', 'PSKH1', 'PSMA3', 'PSMB7', 'PTGR1', 'PTMA', 'PTOV1', 'PTPN11', 'PVALB', 'RAB28', 'RAB3A', 'RABGAP1L', 'RACGAP1', 'RAD23A', 'RANBP3L', 'RBBP8NL', 'REEP2', 'REEP4', 'RELT', 'RFC2', 'RGL2', 'RGS14', 'RGS19', 'RIMKLA', 'RIPK1', 'RNF113A', 'RNF13', 'RNF185', 'RNF19B', 'ROPN1', 'RPS13', 'RPS20', 'RPS6KA4', 'RRAS2', 'RRNAD1', 'RSL1D1', 'RTBDN', 'S100A14', 'SAMD15', 'SCARF2', 'SCN8A', 'SEL1L3', 'SELP', 'SEPN1', 'SERPINF2', 'SHPK', 'SHROOM2', 'SIGIRR', 'SKP2', 'SLA2', 'SLC13A1', 'SLC22A18', 'SLC25A19', 'SLC28A1', 'SLC38A5', 'SLC6A4', 'SLC7A2', 'SLC9A3R2', 'SMARCA4', 'SNRPB', 'SNX3', 'SOCS1', 'SORBS3', 'SPNS1', 'SRP72', 'SRSF3', 'ST8SIA2', 'STAC3', 'STAT6', 'STK19', 'STX10', 'SULF1', 'SYT5', 'TAC1', 'TACR1', 'TBCK', 'TCEAL3', 'TCF15', 'TEAD2', 'TEKT4', 'TET2', 'TFAP2A', 'THAP3', 'THAP4', 'THAP8', 'THEG', 'TIMM13', 'TLE1', 'TMEM144', 'TMEM160', 'TMEM175', 'TMEM184B', 'TMEM56', 'TMEM86A', 'TNFAIP3', 'TNIP1', 'TNNI1', 'TONSL', 'TOP3B', 'TOR3A', 'TPX2', 'TRAF3IP1', 'TSPAN9', 'TTC7A', 'TTYH1', 'TUBA3C', 'TUBA3E', 'TUBB4A', 'TUBB6', 'TVP23C', 'UAP1L1', 'UBC', 'UBE2U', 'UBXN1', 'UCHL1', 'UFL1', 'UGT3A2', 'UNC5C', 'UPF1', 'UPK3B', 'UQCC1', 'USH1G', 'UTP6', 'VPS16', 'WDR74', 'WFDC1', 'WFDC12', 'WIPI1', 'XIRP1', 'XKR9', 'XYLT2', 'YIPF3', 'ZBTB12', 'ZDHHC21', 'ZFP64', 'ZNF576', 'ZNF764', 'ZSCAN2'],
    #     "Phase_S_GM2":['ACSL1','AKR1C2', 'AZI1', 'BCCIP', 'C1QTNF1', 'CHAF1A', 'COX4I2', 'DMRTA1', 'DUSP23', 'FKBP5', 'FXYD1', 'GBAS', 'GDPD5', 'GPIHBP1', 'GSTA2', 'GTPBP10', 'IRX3', 'KRT1', 'LPIN1', 'LRRC59', 'LY6H', 'MAPK8IP1', 'MED29', 'NAA38', 'NCALD', 'NOSIP', 'NR1H2', 'PAOX', 'PARVG', 'PDE4B', 'PITPNM1', 'POU2F1', 'PPP2CA', 'PRKAR1A', 'PTGES3L-AARSD1', 'RNF31', 'RNPS1', 'RPA1', 'RPS27L', 'SEC61A1', 'SLC16A3', 'SLC25A39', 'SLC7A6OS', 'SLX4', 'SMPD1', 'SP6', 'SS18L2', 'SYNJ2', 'TARS', 'TMCO5A', 'TMED2', 'TMEM154', 'ZNF440'],
    #     "Phase_G1": ['ACP5','ACTRT2', 'ADCY4', 'ADCY8', 'AFAP1L2', 'AGBL5', 'AGPAT6', 'AGXT2', 'AK3', 'ANKZF1', 'APRT', 'ARPP19', 'ASIC3', 'ASNA1', 'ATG7', 'ATP1A1', 'ATP1B2', 'ATP6AP1L', 'ATP6V1B2', 'AVPR1A', 'B3GNT1', 'BBC3', 'BCL11B', 'BTN2A2', 'BUB1', 'C5orf42', 'C5orf64', 'CA14', 'CACNA1A', 'CACNG2', 'CASP8', 'CCNC', 'CCND3', 'CCNYL1', 'CDKN2D', 'CEBPB', 'CEP55', 'CHRM3', 'CHST10', 'CISH', 'CLEC1A', 'CLK1', 'CMTM4', 'COX6A2', 'CRMP1', 'CTLA4', 'CTSK', 'CUL2', 'CXCR3', 'CYP27B1', 'DCLRE1B', 'DCTN2', 'DDX19B', 'DENND3', 'DGKA', 'DHX30', 'DISC1', 'DNMBP', 'DYX1C1', 'EIF4G3', 'EML5', 'EMR2', 'EPS8L3', 'ERP44', 'ESR1', 'ESYT2', 'EYA1', 'FAH', 'FAM173B', 'FAU', 'FCRL4', 'FGFBP1', 'FGFR4', 'FHL2', 'FKBP6', 'FMNL3', 'FPR1', 'GALR3', 'GIT2', 'GJA3', 'GNAI1', 'GPR115', 'GPX7', 'GRB7', 'HCFC1', 'HCN1', 'HMGCLL1', 'HS3ST3A1', 'IER5L', 'IFRD1', 'IFT81', 'IL20RA', 'INA', 'IQGAP1', 'ITGA5', 'ITGA6', 'ITPK1', 'JAK2', 'KBTBD3', 'KCNH5', 'KHDRBS3', 'KIAA1324L', 'KLHL28', 'KLRB1', 'KRBOX4', 'LBX1', 'LCA5', 'LCP1', 'LPCAT4', 'LRFN5', 'LRGUK', 'MAP3K7', 'MATR3', 'METAP2', 'MLF1', 'MPP2', 'MRPL42', 'MTFR2', 'MTMR1', 'MYO18B', 'N6AMT1', 'NCOA3', 'NF1', 'NFAT5', 'NFIL3', 'NKAPL', 'NLRP5', 'NOP9', 'NRGN', 'NTRK3', 'NUDT6', 'NXPH3', 'OBP2B', 'OVGP1', 'P2RX5', 'PAK2', 'PANK3', 'PF4V1', 'PIGF', 'PLSCR1', 'PRRX1', 'PTGDR', 'PTPN13', 'RAD17', 'RANBP9', 'RASGEF1B', 'RBBP7', 'RBM11', 'RBM47', 'REC8', 'RELB', 'RETSAT', 'RFESD', 'RNF150', 'RORC', 'RPRD1B', 'SEC11A', 'SEC13', 'SIRT7', 'SLC25A40', 'SLC35B1', 'SLC5A11', 'SLC5A9', 'SLC7A6', 'SMURF1', 'SMYD3', 'SPOPL', 'SPTLC2', 'ST6GAL2', 'ST6GALNAC5', 'STK35', 'STK38L', 'STX3', 'TARBP1', 'TBCA', 'TCEB3', 'TEX30', 'TFDP2', 'THBD', 'TMED1', 'TMEM163', 'TNFRSF19', 'TOP1MT', 'TRH', 'TRIM11', 'TRMT2A', 'TTLL6', 'U2AF1L4', 'UGT1A5', 'UNC5B', 'VIPR1', 'YME1L1', 'ZFP91', 'ZNF184', 'ZNF226', 'ZNF45', 'ZNF596', 'ZNF641', 'ZNF683', 'ZNF76']
    # }
    # gene_set_dict={
    #     "Growth_enhancing":['ACSL1','ACTRT2', 'ADAM29', 'ADCY7', 'ADCY8', 'AGXT2', 'AK3', 'AKR1B10', 'ALPP', 'ALS2', 'ANKMY1', 'ANKZF1', 'ANXA5', 'APOBEC3F', 'APOC2', 'AQP4', 'ARHGAP24', 'ARHGAP44', 'ARHGEF7', 'ARL10', 'ARPP19', 'ARRDC4', 'ARX', 'ASIC3', 'ASPA', 'ATOX1', 'ATP4B', 'ATP6AP1L', 'AVPR1A', 'BCAP29', 'BCL11B', 'BCL3', 'BMP4', 'BTN2A2', 'BZW1', 'C12orf39', 'C14orf2', 'C17orf80', 'C17orf99', 'C1orf173', 'C1QTNF1', 'C3orf33', 'C4orf45', 'C5orf64', 'C7orf62', 'CABYR', 'CASP8', 'CBX6', 'CCBL1', 'CCL11', 'CCL13', 'CCL20', 'CCNYL1', 'CCZ1B', 'CDKL2', 'CDKN1B', 'CDKN2D', 'CDKN3', 'CDX1', 'CEP170', 'CES1', 'CHRM3', 'CHRM5', 'CHSY1', 'CLEC1A', 'CLIC4', 'CLK1', 'CMTM4', 'COL9A1', 'COX6A2', 'CRYBB1', 'CTLA4', 'CTXN1', 'CUL5', 'CXCR3', 'CYB561D1', 'DGKA', 'DIO1', 'DISC1', 'DLX5', 'DNAAF1', 'DUSP23', 'DUSP9', 'DYNLRB2', 'EBAG9', 'EIF2AK1', 'EML5', 'EMR2', 'EPC1', 'EPHB2', 'ERP44', 'ESAM', 'ESYT2', 'EVA1B', 'EYA1', 'FAM131A', 'FAM178B', 'FAM81B', 'FBXL16', 'FBXL2', 'FFAR1', 'FGFBP1', 'FGFR4', 'FKBP5', 'FOXA2', 'FPR1', 'FRMD1', 'FSTL4', 'GADL1', 'GALNT14', 'GALR3', 'GCH1', 'GFRA4', 'GIMAP8', 'GJA8', 'GKN2', 'GLYCTK', 'GMPPA', 'GNAI1', 'GPM6B', 'GPR101', 'GPR115', 'GPR50', 'GPX7', 'HCN1', 'HES2', 'HMGCLL1', 'HOMER1', 'HOPX', 'HRK', 'HSPA6', 'HSPB3', 'IFRD1', 'IFT81', 'IL20RA', 'INPP5K', 'IRF7', 'IRX3', 'ITM2C', 'JAK2', 'KBTBD3', 'KCNE1', 'KCNH5', 'KCNJ2', 'KCNN2', 'KHDRBS3', 'KIAA0895', 'KIAA1324L', 'KIF12', 'KIF6', 'KLHL13', 'KLHL28', 'KLRB1', 'KRT1', 'LBX1', 'LCA5', 'LCE3D', 'LPCAT4', 'LPIN1', 'LRFN5', 'LY6H', 'MAF', 'MAGED2', 'MAS1L', 'MDH2', 'MGST1', 'MLF1', 'MMP2', 'MPHOSPH8', 'MPZL1', 'MST4', 'MTFR2', 'MTMR1', 'NCALD', 'NFIL3', 'NFKBIA', 'NLRP5', 'NR1H2', 'NRGN', 'NUDT6', 'OBP2B', 'PACSIN2', 'PARP14', 'PARVG', 'PDGFA', 'PEPD', 'PF4V1', 'PFN4', 'PGA5', 'PGAP2', 'PGBD4', 'PIFO', 'PLEKHS1', 'PLSCR1', 'PLXNA1', 'PNLIPRP3', 'PPFIA3', 'PPFIBP1', 'PPP1R32', 'PRDM12', 'PROSER2', 'PRRX1', 'PTGR1', 'RAB28', 'RAB3A', 'RANBP3L', 'RASGEF1B', 'RBM11', 'RBM47', 'REEP2', 'RELB', 'RFESD', 'RGS19', 'RIMKLA', 'RNF13', 'RNF150', 'RNF185', 'RNF19B', 'RORC', 'RRAS2', 'RTBDN', 'SCARF2', 'SEL1L3', 'SELP', 'SKP2', 'SLA2', 'SLC13A1', 'SLC5A11', 'SMAD1', 'SMPD1', 'SMURF1', 'SNX3', 'SOCS1', 'SPOPL', 'ST6GALNAC5', 'ST8SIA2', 'STX10', 'SYT5', 'TAC1', 'TARBP1', 'TCEAL3', 'TCEB3', 'TCF15', 'TEX30', 'TFAP2A', 'THAP3', 'THBD', 'TLE1', 'TMED1', 'TMEM144', 'TMEM154', 'TMEM158', 'TMEM160', 'TMEM175', 'TMEM38A', 'TMEM56', 'TNFRSF19', 'TRH', 'TUBA3E', 'TUBB6', 'UAP1L1', 'UBE2U', 'UGT3A2', 'VIPR1', 'WFDC1', 'WFDC12', 'WIPI1', 'ZDHHC21', 'ZNF184', 'ZNF440', 'ZNF45', 'ZNF683', 'ZNF764'],
    #     "Growth_restricting":['ABCA8','ACADS', 'ACP2', 'ACP5', 'ADAD2', 'ADCY4', 'ADPRHL1', 'ADRBK1', 'AEBP2', 'AFAP1L2', 'AGBL5', 'AGPAT6', 'AKR1C2', 'AKR7A3', 'ALDH3B2', 'ALKBH6', 'ALS2CL', 'ANAPC2', 'ANLN', 'AP3B2', 'APC2', 'APH1A', 'API5', 'APOC1', 'APOL1', 'APRT', 'ARFGAP1', 'ARFGAP2', 'ARHGEF40', 'ARSG', 'ASIC2', 'ASNA1', 'ATG7', 'ATP1A1', 'ATP1B2', 'ATP6V1B2', 'AURKB', 'AZI1', 'B3GNT1', 'BAIAP2', 'BAIAP3', 'BARD1', 'BBC3', 'BCAM', 'BCCIP', 'BIRC5', 'BUB1', 'C11orf16', 'C16orf11', 'C19orf53', 'C1orf61', 'C5orf42', 'C7orf10', 'C8orf58', 'C9orf139', 'C9orf173', 'CA13', 'CA14', 'CA3', 'CA4', 'CAB39', 'CACNA1A', 'CACNA1H', 'CACNG2', 'CAMTA2', 'CAP1', 'CCDC101', 'CCDC120', 'CCDC130', 'CCDC151', 'CCNC', 'CCND3', 'CD27', 'CD2BP2', 'CDC16', 'CDC23', 'CDC25B', 'CDC40', 'CDC5L', 'CDC6', 'CDCA3', 'CDCA4', 'CDCA5', 'CDCA8', 'CDHR3', 'CDK1', 'CDK12', 'CDK18', 'CDK2', 'CDKN1A', 'CDKN2B', 'CDT1', 'CDX2', 'CEBPB', 'CEP55', 'CFLAR', 'CHAF1A', 'CHMP4A', 'CHMP6', 'CHRM2', 'CHST10', 'CISH', 'CITED2', 'CLCN1', 'CLEC2B', 'CMKLR1', 'CNEP1R1', 'COMMD3', 'COPS6', 'COX18', 'COX4I2', 'COX6A1', 'CPSF3L', 'CPT1A', 'CRMP1', 'CRYBB3', 'CTNNBL1', 'CTSK', 'CUL1', 'CUL2', 'CX3CR1', 'CYP1A2', 'CYP27B1', 'CYP2F1', 'DACT2', 'DAD1', 'DAGLA', 'DCLRE1B', 'DCTN2', 'DDB1', 'DDX19B', 'DDX54', 'DENND2D', 'DENND3', 'DGCR14', 'DHODH', 'DHX30', 'DMRTA1', 'DNAI1', 'DNAJB2', 'DNAJC19', 'DNMBP', 'DPP3', 'DUSP10', 'DVL3', 'DYX1C1', 'EEF2', 'EFCAB12', 'EFCC1', 'EHD3', 'EIF3C', 'EIF4G3', 'EIF5', 'EML3', 'ENKD1', 'ENPP7', 'EPN3', 'EPS8L3', 'ESPL1', 'ESR1', 'ESRP2', 'ETFB', 'ETV2', 'EWSR1', 'F8A1', 'FAH', 'FAM124A', 'FAM173B', 'FAM175B', 'FAM219A', 'FAM65B', 'FAM83E', 'FAM92A1', 'FAU', 'FBXO33', 'FBXO5', 'FCRL4', 'FGF22', 'FGR', 'FHL2', 'FIBCD1', 'FKBP6', 'FMNL3', 'FN3KRP', 'FRMD8', 'FUS', 'FXYD1', 'FZR1', 'GABRA2', 'GAK', 'GAL3ST3', 'GAS2L2', 'GBAS', 'GDPD5', 'GFOD2', 'GINS2', 'GINS4', 'GIT2', 'GJA3', 'GLIS1', 'GNRH2', 'GON4L', 'GORASP1', 'GPBAR1', 'GPIHBP1', 'GPR153', 'GRB7', 'GRIK4', 'GRM4', 'GSTA2', 'GTPBP10', 'H2AFJ', 'H2AFY2', 'HAUS2', 'HAUS4', 'HCFC1', 'HDAC10', 'HECA', 'HIF3A', 'HIGD1A', 'HIST1H2AC', 'HIST2H2BE', 'HK3', 'HMCES', 'HRSP12', 'HS3ST3A1', 'HSPB2', 'HSPBP1', 'IER5L', 'IFI30', 'IFT122', 'IKBKB', 'IL16', 'IL4I1', 'INA', 'INCENP', 'ING2', 'IQGAP1', 'IRAK1', 'IRF2BP2', 'IRS1', 'ITFG3', 'ITGA5', 'ITGA6', 'ITGA7', 'ITPK1', 'JMJD6', 'JTB', 'KATNB1', 'KCNS1', 'KCTD17', 'KDM2B', 'KIF11', 'KIF13B', 'KIF23', 'KIF5A', 'KIFC2', 'KIFC3', 'KLC3', 'KMT2B', 'KRBOX4', 'KRT5', 'LCP1', 'LIMK2', 'LIN54', 'LOXL3', 'LRGUK', 'LRRC49', 'LRRC59', 'LTB', 'MAP3K7', 'MAPK15', 'MAPK8IP1', 'MAPKAPK3', 'MAST4', 'MATR3', 'MBD2', 'MCCD1', 'MCM10', 'MED29', 'MED9', 'MEF2C', 'METAP2', 'MEX3C', 'MGRN1', 'MIEN1', 'MIIP', 'MITD1', 'MKL1', 'MOSPD3', 'MPDU1', 'MPP2', 'MRPL13', 'MRPL34', 'MRPL42', 'MRPL55', 'MRPS10', 'MRPS26', 'MRPS34', 'MSH6', 'MYBL2', 'MYCBPAP', 'MYCN', 'MYH3', 'MYO18B', 'MYO1F', 'MYO3B', 'MYOF', 'MYOZ1', 'N6AMT1', 'NAA38', 'NCOA3', 'NCOR2', 'NCSTN', 'NDUFB1', 'NELFB', 'NF1', 'NFAT5', 'NFKBIB', 'NFKBIE', 'NFKBIZ', 'NKAIN1', 'NKAPL', 'NOL9', 'NOP9', 'NOSIP', 'NPHP1', 'NR0B2', 'NR4A2', 'NSF', 'NSMCE1', 'NSMCE4A', 'NTRK3', 'NUDT8', 'NUTM1', 'NXPH3', 'ORC3', 'OVGP1', 'OXSM', 'P2RX2', 'P2RX5', 'PAGE2', 'PAK2', 'PAK6', 'PANK3', 'PAOX', 'PARP9', 'PDE1B', 'PDE4B', 'PDZD2', 'PFDN1', 'PFN1', 'PHACTR2', 'PHLDB1', 'PIF1', 'PIGF', 'PITPNM1', 'PLA1A', 'PLA2G3', 'PLCB3', 'PLCE1', 'PLIN3', 'PLK1', 'PNMT', 'PODXL2', 'POLA1', 'POLE2', 'POLL', 'POLR2E', 'POMC', 'POMK', 'PON1', 'POPDC2', 'POU2F1', 'PPIH', 'PPP1CB', 'PPP1R14A', 'PPP1R3F', 'PPP2CA', 'PRC1', 'PRKAR1A', 'PRKCB', 'PRKD2', 'PRKG1', 'PRSS50', 'PRSS8', 'PSKH1', 'PSMA3', 'PSMB7', 'PTGDR', 'PTGES3L-AARSD1', 'PTMA', 'PTOV1', 'PTPN11', 'PTPN13', 'PVALB', 'RAB11A', 'RABGAP1L', 'RACGAP1', 'RAD17', 'RAD23A', 'RAMP2', 'RANBP9', 'RBBP7', 'RBBP8NL', 'REC8', 'REEP4', 'RELT', 'RETSAT', 'RFC2', 'RGL2', 'RGS14', 'RIPK1', 'RNF113A', 'RNF183', 'RNF31', 'RNPS1', 'ROPN1', 'RPA1', 'RPRD1B', 'RPS13', 'RPS2', 'RPS20', 'RPS24', 'RPS27L', 'RPS6KA4', 'RRM1', 'RRM2', 'RRNAD1', 'RSL1D1', 'S100A14', 'SAMD15', 'SCN8A', 'SEC11A', 'SEC13', 'SEC61A1', 'SENP6', 'SEPN1', 'SERPINF2', 'SHPK', 'SHROOM2', 'SIGIRR', 'SIRT7', 'SLBP', 'SLC16A3', 'SLC22A18', 'SLC25A19', 'SLC25A39', 'SLC25A40', 'SLC28A1', 'SLC35B1', 'SLC38A5', 'SLC5A9', 'SLC6A4', 'SLC7A2', 'SLC7A6', 'SLC7A6OS', 'SLC9A3R2', 'SLX4', 'SMARCA4', 'SMARCD1', 'SMYD3', 'SNRPB', 'SORBS3', 'SP6', 'SPNS1', 'SPTLC2', 'SRP72', 'SRSF3', 'SS18L2', 'ST6GAL2', 'STAC3', 'STAT6', 'STK19', 'STK35', 'STK38L', 'STX3', 'SULF1', 'SYNJ2', 'TACR1', 'TARS', 'TBCA', 'TBCK', 'TCF7L1', 'TEAD2', 'TEKT4', 'TET2', 'TFDP1', 'TFDP2', 'THAP4', 'THAP8', 'THEG', 'TIMM13', 'TMCO5A', 'TMED2', 'TMEM163', 'TMEM184B', 'TMEM86A', 'TMEM92', 'TNFAIP3', 'TNIP1', 'TNNI1', 'TONSL', 'TOP1MT', 'TOP3B', 'TOR3A', 'TPX2', 'TP53', 'TRAF3IP1', 'TRIM11', 'TRMT2A', 'TSPAN9', 'TTC7A', 'TTLL6', 'TTYH1', 'TUBA3C', 'TUBB4A', 'TVP23C', 'U2AF1L4', 'U2AF2', 'UBC', 'UBXN1', 'UCHL1', 'UFL1', 'UGT1A5', 'UNC5B', 'UNC5C', 'UPF1', 'UPK3B', 'UQCC1', 'USH1G', 'UTP6', 'VPS16', 'WDR74', 'WEE1', 'XIRP1', 'XKR9', 'XYLT2', 'YIPF3', 'YME1L1', 'ZBTB12', 'ZDHHC12', 'ZFP64', 'ZFP91', 'ZNF22', 'ZNF226', 'ZNF576', 'ZNF596', 'ZNF641', 'ZNF76', 'ZSCAN2']
    # }
    #

    # gene_set  = "MIR892B,MIR873,USH1C,CTDSP2,RAD50,FEM1B,CDK2,CDK3,CDK4,PSME3,CDK5,CDK6,CTDSPL,CDK7,CDKN1A,CDK2AP2,CDKN1B,AKAP8,CDKN1C,CDKN2A,CDKN2B,CCNO,CDKN2C,CDKN2D,CDKN3,NPM2,BTN2A2,NDC80,GPNMB,MAD2L2,TACC3,UBD,CENPE,CENPF,KHDRBS1,NES,PLK2,ARPP19,NEK6,DBF4,CCNI,RCC1,PIM2,UBE2C,TOPBP1,CHEK1,FOXN3,ZWINT,FAM107A,CHEK2,TREX1,CDCA5,ECD,PHB2,PABIR1,CKS1B,CKS2,PRAP1,DCUN1D3,PLK3,PLK5,IQGAP3,CHMP4B,TRIM71,TPRA1,LSM11,HUS1B,NACC2,ATF2,CRY1,MAPK14,CACUL1,E2F7,RAD9B,EME1,SPC24,DTX3L,NEK10,CYP1A1,SASS6,SDE2,DDB1,DDX3X,CDC14C,DLG1,DNA2,DNM2,DUSP1,E2F1,E2F2,E2F3,E2F4,E2F6,EGFR,ARID2,EME2,EIF4E,EIF4EBP1,EIF4G1,AIF1,ENSA,EPS8,ERCC2,AKT1,ERCC3,ERCC6,ESRRB,EZH2,FANCD2,STOX1,CCNY,FGF10,MAPK15,FHL1,ATF5,SIRT2,PAXIP1,MYO16,FOXM1,CLASP2,PHF8,SMC5,FBXL7,PLCB1,KLHL18,CLASP1,UFL1,BRD4,SPDYA,STXBP4,MBLAC1,FBXO7,ZNF324,INTS7,ANAPC15,SIN3A,SYF2,KANK2,HINFP,ANKRD17,GIGYF2,APPL1,TIPRL,FBXO6,FBXO5,FBXO4,LATS2,GFI1,AKAP8L,MTBP,KCNH5,VPS4A,CHMP2A,UBE2S,PRPF19,GLI1,GML,CCDC57,NSMCE2,RGCC,BABAM1,BRD7,GSPT1,TMOD3,ANAPC2,GPR132,RPA4,ANAPC4,DONSON,NOP53,ANXA1,H2AX,APBB1,APBB2,APC,PRMT2,HSPA2,BIRC5,HUS1,HYAL1,ID2,ID4,GEN1,APP,IK,INCENP,INHBA,ITGB1,KCNA5,USP17L2,GPR15LG,NANOGP8,MIR10A,MIR133A1,MIR137,MIR15A,MIR15B,MIR16-1,MIR193A,MIR195,MIR19B1,MIR208A,MIR214,MIR221,MIR222,MIR26A1,MIR29A,MIR29B1,MIR29C,MIR30C2,MAD2L1,MDM2,MECP2,MLF1,MAP3K11,FOXO4,MN1,MNAT1,MOS,MRE11,EIF2AK4,MIR133B,MIR372,MSH2,MUC1,MYC,NBN,ATM,NEUROG1,NFIA,NFIB,NPAT,NPM1,DDR2,ATP2B4,CRNN,ORC1,OVOL1,RRM2B,PBX1,RPS27L,DYNC1LI1,ING4,MRNIP,CDK16,TFDP3,CDK17,CDK18,WAC,DACT1,FZR1,TAOK3,MBTPS2,CRLF3,PPME1,ACTL6B,ANAPC5,ANAPC7,LCMT1,TRIAP1,GTSE1,DTL,ANAPC11,SIRT7,METTL13,CPSF3,UIMC1,MAP3K20,CDK14,ABCB1,PKD1,PKD2,PLCG2,PLK1,PLRG1,PML,POLE,ANLN,ETAA1,ATR,INO80,CCNJ,PAF1,NSUN2,SPDL1,TIPIN,PINX1,USP47,ZWILCH,PPP1R10,CDCA8,PPP2CA,RFWD3,PBRM1,APPL2,PHF10,FBXW7,PPP3CA,PIDD1,PPP6C,AMBRA1,PKIA,CHFR,CDK5RAP2,RIOK2,PCID2,CENPJ,PPP2R2D,KMT2E,PRKDC,RCC2,TEX14,SUSD2,MEPCE,PROX1,TRIM39,TCIM,PSMG2,KNL1,AVEN,BACH1,GJC2,PSME1,PSME2,PTEN,MIR362,SPC25,MIR451A,MIR495,MIR515-1,MIR520A,MIR519D,MIR520H,MIR503,ARID1B,MTA3,HECW2,RPTOR,TAOK1,USP28,CAMSAP3,USP29,USP37,PTPN6,CCAR2,PTPN11,PTPRC,BARD1,RAD1,CTDSP1,RHOU,INIP,BCAT1,RAD9A,RAD17,RAD21,RAD51,RAD51C,RAD51B,RB1,RBBP8,RBL1,RBL2,CCND1,BCL2,RDX,UPF1,DPF2,RFPL1,ACTB,BCL7A,RINT1,MIIP,RPA2,RPL24,RPL26,RPS6,RPS6KB1,RRM1,RRM2,CCL2,BID,CLSPN,BLM,SETMAR,CCNI2,ANAPC1,NABP1,STIL,SIX3,SKP2,CDK15,INTS3,SMARCA2,SMARCA4,STK33,SMARCB1,SMARCC1,DDRGK1,SMARCC2,SMARCD1,SMARCD2,SMARCD3,SMARCE1,SOX2,SPAST,BRCA1,BRCA2,ZFP36L1,ZFP36L2,AURKA,ADAM17,TAF1,TAF2,TAF10,TBX2,MIR638,BUB1,BUB1B,TERT,TFAP4,TFDP1,TP53,TP53BP1,TPD52L1,TPR,TTK,UBE2A,UBE2E2,UBE2L3,WEE1,WNT10B,XPC,XRCC3,ZNF16,ZNF207,CACNB4,ZNF655,NABP2,FBXL15,BRCC3,DDX39B,PAGR1,CDC73,CCNJL,RNASEH2B,FBXO31,KDM8,NEK11,ATAD5,CCNP,JADE1,WDR76,CALM1,CTC1,DBF4B,TTI2,MUS81,CEP63,CDK5RAP3,CALM2,CUL5,CALM3,DPF3,CAMK2A,FAM83D,CDT1,TMEM14B,DPF1,ARID1A,CDC7,CDC45,GFI1B,CASP2,NUF2,PARP9,RHNO1,MAD1L1,USP26,HASPIN,BRIP1,HORMAD1,USP44,ATRIP,ABRAXAS1,DYRK3,DOT1L,BRSK1,CUL4B,CUL4A,CUL3,CUL2,CUL1,KLF11,PPP1R9B,KLHL22,PPM1D,MASTL,ZFYVE19,LSM10,CCNB3,PIAS1,CDC14B,CDC14A,CDK10,THOC5,ACTL6A,CDC23,MBTPS1,CRADD,RAB11A,IER3,CDC16,ZPR1,NAE1,CCNA2,CCNA1,CCNB1,TIMELESS,PHOX2B,CCND2,CCND3,CCNE1,CCNF,ACVR1,CCNG1,CCNG2,CCNH,BRSK2,TM4SF5,TICRR,PKMYT1,ACVR1B,LATS1,CCNB2,CCNE2,CTDP1,ZNF830,SLFN11,CHMP7,ZW10,BUB3,CCNQ,AURKB,CHMP4C,BCL7C,BCL7B,KLF4,TRIP13,TAOK2,ADAMTS1,VPS4B,MACROH2A1,CLOCK,BABAM2,MAD2L1BP,MDC1,TTI1,ESPL1,KNTC1,DLGAP5,CDK1,MELK,CDC5L,TELO2,CDC6,CDC20,KIF14,CDC25A,CDC25B,CDC25C,CDC27,CDC34,THOC1".split(',')
    gene_set = []
    list(map(gene_set.extend, list(gene_set_dict.values())))

    return gene_set_dict, gene_set

def umap_group_genes(adata: anndata.AnnData, filepath: str):
    # Before SCVI
    for layer in adata.layers.keys():
        if not layer.startswith('scvi_reconstructed'):
            continue
        print(f'working on {layer}')
        adata.X = adata.layers[layer].copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_pcs=20, n_neighbors=15, metric='euclidean', method='umap')
        sc.tl.umap(adata)
        adata.obsm[f'X_{layer}_umap'] = adata.obsm['X_umap'].copy()

    adata.write(filepath)

    return adata

def umap_group_genes_cuda(adata: anndata.AnnData,filepath: str):
    print("UMAP per reconstructed layer with GPU acceleration")
    # Before SCVI
    rsc.get.anndata_to_GPU(adata=adata, convert_all=True)
    for layer in adata.layers.keys():
        if not layer.startswith('scvi_reconstructed'):
            continue
        print(f'working on {layer}')
        adata.X = adata.layers[layer].copy()
        rsc.pp.normalize_total(adata)
        rsc.pp.log1p(adata)
        rsc.pp.filter_genes(adata, min_count=1)
        rsc.pp.pca(adata)
        rsc.pp.neighbors(adata, n_pcs=20, n_neighbors=15, metric='euclidean')
        rsc.tl.umap(adata)
        adata_transformed = rsc.get.anndata_to_CPU(adata, convert_all=True, copy=True)
        adata_transformed.obsm[f'X_{layer}_umap'] = adata.obsm['X_umap'].copy()

    adata_transformed.write(filepath)

    return adata_transformed

class SeabornFig2Grid():
    """Class from https://stackoverflow.com/questions/47535866/how-to-iteratively-populate-matplotlib-gridspec-with-a-multipart-seaborn-plot/47624348#47624348"""
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig

        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        elif isinstance(self.sg, sns.matrix.ClusterGrid):#https://github1s.com/mwaskom/seaborn/blob/master/seaborn/matrix.py#L696
            # print(dir(self.sg))
            # print(dir(self.sg.figure))
            self._moveclustergrid()
        elif isinstance(self.sg,matplotlib.figure.Figure):
            print("reached")
            # print(self.sg)
            # print(type(self.sg))
            # print(vars(self.sg))
            self.sg = self.sg.axes[0]


            self._moveaxes(self.sg,self.subplot)

        else:
            print("what am i")


        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveclustergrid(self):
        """Move cluster grid"""
        r = len(self.sg.figure.axes)
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r, r + 10, subplot_spec=self.subplot)
        subplots_axes = self.sg.figure.axes
        self._resize()
        self._moveaxes(subplots_axes[0], self.subgrid[1:, 0:3]) #left cladogram #ax_row_dendrogram
        self._moveaxes(subplots_axes[1], self.subgrid[0, 4:-2]) #top cladogram #ax_col_dendrogram
        self._moveaxes(subplots_axes[2], self.subgrid[1:, 3]) #labels bar
        self._moveaxes(subplots_axes[3], self.subgrid[1:, 4:-2]) #heatmap #ax_heatmap
        self._moveaxes(subplots_axes[4], self.subgrid[1:, -1]) #colorbar


    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        if hasattr(self.sg,"fig"):
            plt.close(self.sg.fig)
        elif hasattr(self.sg,"figure"):
            plt.close(self.sg.figure)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def plot_settings_helper(adata, gene_key, gene_values, cmap="pink_r"):

    if isinstance(gene_values, list):
        adata_subset = adata[:, adata.var_names.isin(gene_values)]
        if adata_subset.n_vars > 0:
            expression_gene = np.asarray(adata_subset.X.mean(axis=1)).squeeze(-1)
        else:
            expression_gene = None
    else:  # englobe the single gene as a list
        adata_subset = adata[:, adata.var_names.isin([gene_values])]
        if adata.n_vars > 0:
            expression_gene = np.asarray(adata_subset.X[:,0])
        else:
            expression_gene = None
    if expression_gene is not None:

        adata.obs[f"{gene_key}"] = expression_gene.copy()
        expression_gene_unique = np.unique(expression_gene).tolist()
        if 0 not in expression_gene_unique:
            expression_gene_unique = [0] + expression_gene_unique  # add background color
        # colormap_expression = matplotlib.cm.get_cmap(cmap, len(expression_gene_unique))
        colormap_expression = matplotlib.colormaps[cmap].resampled(len(expression_gene_unique))
        colormap_expression_array = np.array([colormap_expression(i) for i in range(colormap_expression.N)])
        colors_dict = dict(zip(expression_gene_unique, colormap_expression_array))

        return {"expression_gene": expression_gene,
                "expression_gene_unique": expression_gene_unique,
                "colormap_expression": colormap_expression,
                "colormap_expression_array": colormap_expression_array,
                "colors_dict": colors_dict,
                "gene_hgnc_values": gene_values,
                "gene_hgnc_key":gene_key,
                "adata":adata
                }
    else:
        return None

def plot_by_expression(adata,adata_proj,gene_set_values,gene_set_key,cmap="OrRd",alpha=0.7,size=5,fontsize=15):
    """"""

    settings_plot = plot_settings_helper(adata, gene_set_values, cmap)

    if settings_plot is not None:

        dataframe = pd.DataFrame({"adata_proj_x": adata_proj[:, 0],
                                  "adata_proj_y": adata_proj[:, 1],
                                  f"expression_{gene_set_key}": settings_plot["expression_gene"],
                                  })

        g = sns.FacetGrid(dataframe, hue=f"expression_{gene_set_key}",
                           subplot_kws={"fc": "white"},
                           palette=settings_plot["colormap_expression_array"],
                           hue_order=settings_plot["expression_gene_unique"]
                           )
        g.set(yticks=[])
        g.set(xticks=[])
        g_axes = g.axes.flatten()
        g_axes[0].set_title("{} genes".format(gene_set_key.replace("significant_genes","").replace("_"," ")), fontsize=fontsize)
        g.map(plt.scatter, "adata_proj_x", "adata_proj_y", alpha=alpha, s=size)
        g_axes[0].set_xlabel("")
        g_axes[0].set_ylabel("")

        return g,settings_plot
    else:
        return None,None

def calculate_dimensions_plot(g_plots_list,max_rows=3,include_colorbars=True):
    nelements = len(g_plots_list)
    print("nelements: {}".format(nelements))
    nplots = nelements if (int(nelements) % 2 == 0) else int(nelements) + 1
    print("Calculated nplots: {}".format(nplots))
    nrows = int(nplots / 2) if int(nplots / 2) <= max_rows else max_rows
    ncols = int(nplots / nrows) if nplots / nrows % 2 == 0 else int(nplots / nrows) + 1
    print("Calculated nrows {} ncols {}".format(nrows, ncols))
    if nrows * ncols > nelements and nrows * (ncols - 1) >= nelements:
        print("Dropping 1 column")
        ncols = ncols - 1
    print("Creating nrows {} ncols {}".format(nrows, ncols))
    if include_colorbars:
        ncols *= 2  # we duplicate the number of columns to fit the color bars
    width_ratios = [6] * ncols  # we duplicate the number of columns to fit the color bars
    if include_colorbars:
        width_ratios[1::2] = [0.3] * int(ncols / 2) #set the width of the color bars smaller
    #print(width_ratios)

    nrows_idx = np.repeat(list(range(nrows)),ncols)
    ncols_idx = np.tile(list(range(ncols)),nrows)
    # print(nrows_idx)
    # print(ncols_idx)
    if nrows * ncols > nelements * 2 and nrows * (ncols - 1) >= nelements * 2:
        print("Dropping 1 row")
        nrows = nrows - 1

    return nrows,ncols,nrows_idx,ncols_idx,width_ratios

def plot_avg_expression(adata:anndata.AnnData,basis:str,gene_set_dict:dict,figpath:str,figname:str,color_keys:list):
    """Plots the average glyco expression per glyco pathway (averages the expression of all the genes involved in that pathway)"""
    print("Plotting average glyco expression per glyco pathway")
    genes_list = []
    for gene_key, gene_set in gene_set_dict.items():
        gene_key = gene_key.lower()
        settings_plot = plot_settings_helper(adata, gene_key,gene_set, cmap="OrRd")
        if settings_plot is not None:
            genes_list.append(gene_key)
            adata = settings_plot["adata"]

    nrows,ncols,nrows_idx,ncols_idx,width_ratios = calculate_dimensions_plot(genes_list,max_rows=4,include_colorbars=False)

    color_by = [color_keys[0]] + genes_list if len(adata.obs[color_keys[0]].value_counts().keys()) <= 20 else genes_list
    sc.pl.embedding(adata,
                    basis=basis,
                    color=color_by,
                    #color=genes_list,
                    projection="2d",
                    cmap = "magma_r",
                    ncols=nrows,
                    wspace=0.7,
                    show=False)


    plt.savefig(f"{figpath}/{figname}.jpg",dpi=600, bbox_inches="tight")

def plot_settings_cluster_helper(adata_cluster,cluster, present_gene_set,gene_key,gene_set):
    # pool.starmap(partial(plot_settings_cluster_helper, adata_cluster,cluster,present_gene_set,gene_set_dict),zip(list(gene_set_dict.keys()),list(gene_set_dict.values())))
    gene_set = [gene_set] if not isinstance(gene_set, list) else gene_set
    if len(set(present_gene_set).intersection(gene_set)) >= 1:
        adata_cluster = adata_cluster[adata_cluster.obs["clusters"].isin([cluster])]
        adata_gene_cluster = adata_cluster[:, adata_cluster.var_names.isin(gene_set)]
        gene_key = gene_key.lower()
        if adata_gene_cluster.X.size != 0:
            settings_plot = plot_settings_helper(adata_gene_cluster, f"cluster_{cluster}_{gene_key}", gene_set,cmap="OrRd")
            average_expression = settings_plot["expression_gene"].mean()
            del settings_plot

            return (gene_key,cluster,average_expression)

def plot_avg_expression_cluster(adata:anndata.AnnData,basis:str,gene_set_dict:dict,cluster_counts_dict:dict,figpath:str,figname:str,color_keys:list,by_pathway:bool=False):
    """Plots the average glyco expression per glyco pathway (averages the expression of all the genes involved in that pathway)"""
    print("Plotting average glyco expression per leiden cluster")

    if not by_pathway: #plot all genes separately
        all_genes = sum(gene_set_dict.values(), [])
        gene_set_dict = dict(zip(all_genes,all_genes))
    present_gene_set = adata.var_names.tolist()
    start = time.time()
    def parallel_nested_loops(present_gene_set):
        """Parallelization of the computation of the average expression per gene per Leiden cluster"""
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            # Submit all combinations of i, j to the ThreadPool
            for cluster in cluster_counts_dict.keys():  # Outer loop
                adata_cluster = adata[adata.obs["clusters"].isin([cluster])]
                for gene_key,gene_set in gene_set_dict.items(): #inner loop
                    futures.append(executor.submit(partial(plot_settings_cluster_helper, adata_cluster,cluster,present_gene_set), gene_key, gene_set))
            # Collecting the results as they complete
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results = list(filter(lambda v: v is not None, results))  # clear None values
        genes_cluster_dict = {k: {sub_k: v for _, sub_k, v in group} for k, group in itertools.groupby(sorted(results), key=lambda x: x[0])} #itertools groups by the elements in the first position of all tuples. data = [("a", "3", "5"), ("a", "4", "6"), ("b", "7", "9"), ("b", "8", "10")] -> (a : [("a", "3", "5"),("a", "4", "6")], b: [("b", "7", "9"),("b", "8", "10")])
        return genes_cluster_dict
    end = time.time()

    genes_cluster_dict = parallel_nested_loops(present_gene_set)
    print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))

    nrows,ncols,nrows_idx,ncols_idx,width_ratios = calculate_dimensions_plot(list(genes_cluster_dict.keys()),max_rows=4,include_colorbars=False)

    def plot_cluster_size_avg(cluster_counts_dict,ax,gene_name,cluster_values):
        """"""

        cluster_sizes = [cluster_counts_dict[cluster] for cluster in cluster_values.keys()] #TODO: Are they same all the time, I think so
        df = pd.DataFrame({"sizes":cluster_sizes,"expression":cluster_values.values()})
        df.sort_values(by="sizes",inplace=True)
        df.plot(x="sizes",y="expression",marker="o",ax=ax,legend=False)
        ax.set_title(gene_name)
        ax.set_ylim(0,1)
        ax.set_xscale('log')
        del df

    fig, axs = plt.subplots(ncols = ncols,nrows=nrows,figsize=(30,20))
    axs = axs.ravel()
    with ThreadPool(multiprocessing.cpu_count() - 1) as pool:
         pool.starmap(partial(plot_cluster_size_avg,cluster_counts_dict),zip(axs,genes_cluster_dict.keys(),genes_cluster_dict.values()))

    fig.savefig(f"{figpath}/{figname}.jpg",dpi=600, bbox_inches="tight")
    plt.subplots_adjust(wspace=0.2,hspace=0.4)
    plt.close()
    plt.clf()

def differential_gene_expression(adata,gene_set,figpath,figname):

    print("Plotting differential gene expression")
    # Step 3: Subset the data for the gene sets
    expression_1 = adata[:, adata.var['high_exp_genes'] ].X
    expression_2 = adata[:, adata.var['high_exp_genes_of_interest']].X

    # Convert to DataFrame for easier manipulation
    df_1 = pd.DataFrame(expression_1.toarray(), columns=adata.var_names[adata.var['high_exp_genes'] ])
    df_2 = pd.DataFrame(expression_2.toarray(), columns= adata.var_names[adata.var['high_exp_genes_of_interest']])

    # Step 4: Calculate summary statistics or visualize the data

    # Example: Plotting violin plots for each gene in both sets
    df_1_melted = df_1.melt(var_name='Gene', value_name='Expression')
    df_2_melted = df_2.melt(var_name='Gene', value_name='Expression')

    # Adding a column to distinguish between the two sets
    df_1_melted['Set'] = 'Top 20 expressed all genes'
    df_2_melted['Set'] = 'Top 20 expressed glyco'

    # Combine the dataframes
    combined_df = pd.concat([df_1_melted, df_2_melted])

    # Plotting
    plt.figure(figsize=(10, 6))
    #sns.violinplot(x='Gene', y='Expression', hue='Set', data=combined_df, split=True)
    sns.kdeplot( x='Expression', hue='Set', data=combined_df)
    plt.title('Expression Distributions')
    plt.savefig("{}/{}.jpg".format(figpath,figname))

def split_into_chunks(input_list,chunks=2):
    """Divides list into even splits
    :param list input_list
    :param int chunks: Number of divisions"""
    quot, rem = divmod(len(input_list), chunks)
    divpt = lambda i: i * quot + min(i, rem)
    return [input_list[divpt(i):divpt(i + 1)] for i in range(chunks)]

def plot_violin_expression_distribution(adata:anndata.AnnData,genes_list:list,figpath:str,figname:str,layer_name:str):
    """"""
    genes_list = list(sorted(genes_list))
    genes_list_chunks = split_into_chunks(genes_list,chunks=12)

    nrows, ncols, nrows_idx, ncols_idx, width_ratios = calculate_dimensions_plot(genes_list_chunks,include_colorbars=False)
    ylim = int(adata.layers[layer_name].max()) + 1
    fig,axs = plt.subplots(nrows, ncols,tight_layout=True)
    axs = axs.ravel()
    #TODO: Switch to threadpool
    for idx,gene_set in enumerate(genes_list_chunks):
        print(gene_set)
        sc.pl.violin(adata,
                     gene_set,
                     layer=layer_name,
                     use_raw=False,
                     ax=axs[idx],
                     rotation=90,show=False,size=0.2,stripplot=False,jitter=False,scale="count",**{"color":"red","fill":True,"linewidth":0.05,"fill":True,"inner":None})


        # axs[idx].set_xticks(list(range(0,len(gene_set)*2,2)))
        # axs[idx].set_xticklabels(gene_set,rotation=90)
        axs[idx].set_ylim(0,ylim)
        axs[idx].tick_params(axis="x",labelsize=5)

    plt.savefig(f"{figpath}/{figname}.jpg", dpi=700, bbox_inches="tight")

def plot_density(iterables,adata,axs,layer_name):
    gene_name,idx = iterables
    #print(gene_name)
    adata = adata[:,gene_name].layers[layer_name].todense().A1
    adata = adata[adata.nonzero()]
    data = pd.DataFrame({f"{gene_name}":adata})

    sns.kdeplot(data=data, x=f"{gene_name}",ax=axs[idx],common_norm=True,linewidth=1,fill=True,color="red") #,bw_method=0.5)

    #axs[idx].plot(list(range(5)),list(range(5)))
    axs[idx].set_ylabel("")
    axs[idx].set_xlabel(f"{gene_name}",fontsize=25)

def plot_density_expression(adata:anndata.AnnData,genes_list:list,layer_name:str,figpath:str,figname:str):
    print("Plotting density expression")

    genes_list = sorted(genes_list)
    #genes_list = genes_list[:30]
    nrows, ncols, nrows_idx, ncols_idx, width_ratios = calculate_dimensions_plot(genes_list,max_rows=7,include_colorbars=False)
    fig,axs = plt.subplots(nrows, ncols,figsize=(60,60))
    axs = axs.ravel()

    #sc.pp.filter_genes(adata, inplace=True,min_cells=10)
    sc.pp.filter_cells(adata, inplace=True,min_counts=5)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    iterables = zip(genes_list,list(range(len(genes_list))))

    with ThreadPool(multiprocessing.cpu_count() - 1) as pool:
         pool.map(partial(plot_density, adata=adata,axs=axs,layer_name=layer_name),iterables)


    fig.suptitle("UMI counts density",y=0.9,fontsize=25)
    plt.savefig(f"{figpath}/{figname}.jpg", dpi=700, bbox_inches="tight")
    plt.close(fig)
    plt.clf()

def calculate_non_zero_expression(gene_name,adata,layer_name=None):

    gene_expression = adata[:,gene_name].layers[layer_name].todense().A1
    gene_expression = gene_expression[gene_expression.nonzero()]
    if gene_expression.size == 0:
        mean_expr = 0
        var_expr = 0
    else:
        mean_expr = gene_expression.mean()
        var_expr = gene_expression.std()

    return (mean_expr,var_expr)

def plot_rank_expression(adata:anndata.AnnData,genes_list:list,layer_name:str,figpath:str,figname:str):
    """Plot rank expression per cell/tissue"""
    print("Plotting rank expression...")


    genes_list = sorted(genes_list)
    #genes_list = genes_list[:30]

    #sc.pp.filter_genes(adata, inplace=True,min_cells=10)
    sc.pp.filter_cells(adata, inplace=True,min_counts=5)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    #TODO: This or masked mean
    with ThreadPool(multiprocessing.cpu_count() - 1) as pool:
        results = pool.map(partial(calculate_non_zero_expression, adata=adata,layer_name=layer_name),genes_list)

    results = list(zip(*results))
    average = results[0]
    std = results[1]

    gene_expression = pd.DataFrame({"average":average,"std":std},index=genes_list)
    gene_expression = gene_expression.sort_values(by="average",ascending=False)


    gene_expression =  gene_expression.style.background_gradient(axis=None,  cmap='YlOrRd') #.hide(axis='index')

    dfi.export( gene_expression, '{}/{}.jpg'.format(figpath,figname), max_cols=-1,
               max_rows=-1,
                #dpi=600,
               table_conversion="matplotlib",
               )

def analysis_nmf(adata,genes_list:list,filepath_subset:str,filepath:str,gene_group_name:str,genes_slice):
    """"""
    # #Restrict only to the glyco genes
    adata.var[f'gene_subset_{gene_group_name}'] = adata.var_names.isin(genes_list)
    #adata.var['gene_subset'].sum()
    # TODO: Remove rows with expression values 0?
    # sc.pp.filter_cells(adata, inplace=True,min_counts=10)
    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    #
    # sc.pp.filter_cells(adata_subset, inplace=True,min_counts=10)
    # sc.pp.normalize_total(adata_subset)
    # sc.pp.log1p(adata_subset)


    #sc.pp.pca(adata_subset)

    for layer in adata.layers.keys():
        if not layer.startswith('scvi_reconstructed'): #ignore raw dataset
            continue

        print(f'working on {layer}')
        adata.X = adata.layers[layer].copy()

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

        if adata.X.size != 0:
            sc.pp.pca(adata, mask_var=f'gene_subset_{gene_group_name}') #before glyco_gene #n_comps=5
            sc.pp.neighbors(adata, n_neighbors=15, metric='euclidean', method='umap') #does not have mask_var
            sc.tl.umap(adata) #does not have mask_var
            adata.obsm[f'X_{layer}_{gene_group_name}_geneset_umap'] = adata.obsm['X_umap'].copy()
            break

    adata_subset = adata[:, genes_slice]
    if adata_subset.X.size != 0:

        #Analyze PCA components
        num_pcs = adata_subset.varm['PCs'].shape[1]
        for i in range(num_pcs):
            adata_subset.var[f'pc{i}_{gene_group_name}'] = adata_subset.varm['PCs'][:, i].squeeze()

        #Examing the first PC capturing the highest variance in the dataset. Then, sort the values of the first PC to find those features weighted higher
        adata_subset.var.sort_values(by=f'pc0_{gene_group_name}', ascending=False)


        nmf = NMF(n_components=5, #"auto"
                  init='nndsvdar', # 'random' 'nndsvd' 'nndsvda' (zeros filled with the average of x), 'nndsvdar' (zeros filled with small random values)
                  random_state=0,
                  beta_loss='frobenius', #'frobenius','kullback-leibler', 'itakura-saito' (input matrix cannot contain zeroes)
                  max_iter=200)
        W = nmf.fit_transform(adata_subset.layers['scvi_reconstructed_0']) # n_obs x rank (p)
        #H = nmf.components_ # rank(p) x n_genes

        inferred_rank = nmf.components_.shape[0]
        for i in range(inferred_rank): #for i in range (inferred_rank)
            adata_subset.var[f'{gene_group_name}_nmf{i}'] = nmf.components_[i, :].squeeze() #index the component from the H  matrix that reflects all the genes
        nmf_cols = adata_subset.var.columns[(adata_subset.var.columns.str.startswith(f"{gene_group_name}_nmf"))]
        for col in nmf_cols:
            top_genes_in_set = adata_subset.var.sort_values(by=col, ascending=False).head(20).index #highest values first
            sc.tl.score_genes(adata_subset, gene_list=top_genes_in_set, score_name=f'{col}_{gene_group_name}_score',use_raw=False) #average expression of a set of genes subtracted with the average expression of a reference set of genes

        adata_subset.write(filepath_subset)

    else:
        warnings.warn("Skipping NMF analysis, adata_subset shape is : {}, no genes of interest found".format(adata_subset.shape))
    adata.write(filepath)

    return adata,adata_subset

def analysis_nmf_cuda(adata,genes_list:list,filepath_subset:str,filepath:str,gene_group_name:str,genes_slice):
    """"""
    # #Restrict only to the glyco genes
    adata.var[f'gene_subset_{gene_group_name}'] = adata.var_names.isin(genes_list)
    #adata.var['gene_subset'].sum()
    # TODO: Remove rows with expression values 0?
    # sc.pp.filter_cells(adata, inplace=True,min_counts=10)
    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    #
    # sc.pp.filter_cells(adata_subset, inplace=True,min_counts=10)
    # sc.pp.normalize_total(adata_subset)
    # sc.pp.log1p(adata_subset)


    #sc.pp.pca(adata_subset)
    rsc.get.anndata_to_GPU(adata=adata, convert_all=True)
    for layer in adata.layers.keys():
        if not layer.startswith('scvi_reconstructed'): #ignore raw dataset
            continue

        print(f'working on {layer}')
        adata.X = adata.layers[layer].copy()

        rsc.pp.normalize_total(adata)
        rsc.pp.log1p(adata)
        rsc.pp.filter_genes(adata,min_count=1)

        if adata.X.size != 0:
            rsc.pp.pca(adata, mask_var=f'gene_subset_{gene_group_name}') #before glyco_gene #n_comps=5
            rsc.pp.neighbors(adata, n_neighbors=15, metric='euclidean') #does not have mask_var
            rsc.tl.umap(adata) #does not have mask_var
            adata.obsm[f'X_{layer}_{gene_group_name}_geneset_umap'] = adata.obsm['X_umap'].copy()
            break

    adata_subset = adata[:, genes_slice]

    if adata_subset.X.size != 0:
        # rsc.get.anndata_to_GPU(adata=adata_subset, convert_all=True)
        # rsc.pp.filter_genes(adata_subset, min_count=1)

        #Analyze PCA components
        num_pcs = adata_subset.varm['PCs'].shape[1]
        for i in range(num_pcs):
            adata_subset.var[f'pc{i}_{gene_group_name}'] = adata_subset.varm['PCs'][:, i].squeeze()

        #Examing the first PC capturing the highest variance in the dataset. Then, sort the values of the first PC to find those features weighted higher
        adata_subset.var.sort_values(by=f'pc0_{gene_group_name}', ascending=False)

        print("Starting NMF")
        nmf = NMF(n_components=5, #"auto"
                  init='nndsvdar', # 'random' 'nndsvd' 'nndsvda' (zeros filled with the average of x), 'nndsvdar' (zeros filled with small random values)
                  random_state=0,
                  beta_loss='frobenius', #'frobenius','kullback-leibler', 'itakura-saito' (input matrix cannot contain zeroes)
                  max_iter=200)

        rsc.get.anndata_to_CPU(adata, convert_all=True)
        rsc.get.anndata_to_CPU(adata_subset, convert_all=True)
        W = nmf.fit_transform(adata_subset.layers['scvi_reconstructed_0']) # n_obs x rank (p)
        #H = nmf.components_ # rank(p) x n_genes

        inferred_rank = nmf.components_.shape[0]
        for i in range(inferred_rank): #for i in range (inferred_rank)
            adata_subset.var[f'{gene_group_name}_nmf{i}'] = nmf.components_[i, :].squeeze() #index the component from the H  matrix that reflects all the genes
        nmf_cols = adata_subset.var.columns[(adata_subset.var.columns.str.startswith(f"{gene_group_name}_nmf"))]
        for col in nmf_cols:
            top_genes_in_set = adata_subset.var.sort_values(by=col, ascending=False).head(20).index #highest values first
            sc.tl.score_genes(adata_subset, gene_list=top_genes_in_set, score_name=f'{col}_{gene_group_name}_score') #average expression of a set of genes subtracted with the average expression of a reference set of genes
        adata_subset.write(filepath_subset)

    else:
        warnings.warn("Skipping NMF analysis, adata_subset shape is : {}, no genes of interest found".format(adata_subset.shape))


    adata_transformed = rsc.get.anndata_to_CPU(adata, convert_all=True, copy=True)
    adata_transformed.write(filepath)

    return adata,adata_subset

def plot_nmf(adata_subset,color_keys,figpath,gene_group_name=""):

    fig,ax = plt.subplots(figsize=(30,30))
    sc.pl.embedding(adata_subset,
                    basis=f'scvi_reconstructed_0_{gene_group_name}_geneset_umap',
                    color=[c for c in adata_subset.obs.columns if c.startswith(f'{gene_group_name}_nmf') and c.endswith(f'{gene_group_name}_score')] + color_keys,
                    cmap='Oranges', vmin=-1, vmax=3, ncols=2,show=False)
    plt.savefig(f"{figpath}/nmf_components_score_{gene_group_name}.jpg", dpi=600, bbox_inches="tight")
    plt.close()
    plt.clf()

    #TODO: also color by cell type

class Compute_Leiden_clusters():

    def __init__(self,adata,gene_set_dict,figpath,color_keys,filepath,overwrite,plot_all,use_cuda):
        self.adata = adata
        self.gene_set_dict = gene_set_dict
        self.figpath = figpath
        self.color_keys = color_keys
        self.filepath = filepath
        self.overwrite = overwrite
        self.plot_all = plot_all
        self.plot_all2 = False
        self.use_cuda = use_cuda



    def run(self):
        if self.use_cuda:
            datasets_dict = self.plot_neighbour_leiden_clusters_cuda()
            return datasets_dict["all"]
        else:
            datasets_dict = self.plot_neighbour_leiden_clusters()
            return datasets_dict["all"]


    def compute_leiden_plots(self,dataset,name):
        with rc_context({"figure.figsize": (15, 15)}):
            sc.pl.umap(
                dataset,
                layer="X_scvi_reconstructed_0_umap",
                use_raw=False,
                color=[self.color_keys[0], "clusters"],
                add_outline=True,
                legend_loc="on data",
                legend_fontsize=12,
                legend_fontoutline=2,
                frameon=False,
                title="clustering of cells",
                palette="Set1",
                show=False,
                # legend_fontsize=20,
            )
            plt.savefig(f"{self.figpath}/leiden_clusters{name}.jpg", dpi=600, bbox_inches="tight")
            plt.close()
            plt.clf()

        # TODO: dotplot cannot handle many clusters, if there are too many, the gridspec will complain, perhaps with a larger figsize
        # dataset= dataset[(dataset.obs["clusters"] == "0") | (dataset.obs["clusters"] == "1")]

        cluster_counts_dict = dataset.obs["clusters"].value_counts()
        dataset_topk = dataset[dataset.obs["clusters"].isin(cluster_counts_dict.keys()[:20])]  # pick only the top 20 clusters with more members

        singlemember_clusters = [key for key, val in cluster_counts_dict.items() if val == 1]
        if singlemember_clusters:
            print("Removing clusters with a single element")
            dataset_topk = dataset_topk[
                ~dataset_topk.obs["clusters"].isin(singlemember_clusters)]  # remove clusters with a single element

        gene_set = dataset_topk.var_names[dataset_topk.var['genes_of_interest']]
        if self.plot_all2:
            if name == "_maxfreqcell":
                plot_avg_expression(dataset_topk, "X_scvi_reconstructed_0_umap", self.gene_set_dict, self.figpath,"glyco_expression_maxfreqcell", self.color_keys)
            plot_avg_expression_cluster(dataset_topk, "scvi_reconstructed_0", self.gene_set_dict, cluster_counts_dict,self.figpath, f"leiden_cluster_size_glyco_expression{name}", self.color_keys)
        print("Plotting dotplot clusters {}".format(name))
        sc.pp.log1p(dataset_topk, layer="scvi_reconstructed_0")
        sc.tl.dendrogram(dataset_topk, groupby="clusters")  # need to re-run because
        sc.pl.dotplot(dataset_topk,
                      gene_set,
                      layer="scvi_reconstructed_0",
                      use_raw=False,
                      groupby="clusters",
                      figsize=(20, 20),
                      dendrogram=True,
                      show=False,
                      dot_max=1)
        plt.savefig(f"{self.figpath}/leiden_dotplot{name}.jpg", dpi=600, bbox_inches="tight")
        plt.close()
        plt.clf()
        # with rc_context({"figure.figsize": (4.5, 3)}):
        #     sc.pl.violin(adata, gene_set, groupby="clusters",ncols=5,save="violin-glyco",show=False)
        sc.tl.dendrogram(dataset_topk, groupby="clusters")
        fig, axs = plt.subplots(nrows=2, figsize=(25, 15))
        print("Plotting stacked violin {}".format(name))
        if len(gene_set) > 10:
            batch = int(len(gene_set) / 2)
            sc.pl.stacked_violin(dataset_topk, {"Glyco_1": gene_set[:batch]}, groupby="clusters",
                                 layer="scvi_reconstructed_0",
                                 swap_axes=False,
                                 dendrogram=True,
                                 show=False, use_raw=False, ax=axs[0])
            sc.pl.stacked_violin(dataset_topk, {"Glyco_2": gene_set[batch:]}, groupby="clusters",
                                 layer="scvi_reconstructed_0",
                                 swap_axes=False,
                                 dendrogram=True,
                                 show=False, use_raw=False, ax=axs[1])
        else:

            sc.pl.stacked_violin(dataset_topk, {"Glyco": gene_set}, groupby="clusters",
                                 layer="scvi_reconstructed_0",
                                 swap_axes=False,
                                 dendrogram=True,
                                 show=False, use_raw=False)
        plt.savefig(f"{self.figpath}/leiden_stacked_violin{name}.jpg", dpi=600, bbox_inches="tight")
        plt.close()
        plt.clf()
        print("Plotting rank genes {}".format(name))
        sc.tl.rank_genes_groups(dataset_topk,
                                layer="scvi_reconstructed_0",
                                use_raw=False,
                                groupby="clusters",
                                method="wilcoxon",
                                corr_method="benjamini-hochberg",
                                mask_var="genes_of_interest")
        sc.pl.rank_genes_groups(dataset_topk, n_genes=25, sharey=False, show=False)
        plt.savefig(f"{self.figpath}/leiden_rank_genes{name}.jpg", dpi=600, bbox_inches="tight")
        plt.close()
        plt.clf()

    def plot_neighbour_leiden_clusters(self):
        """

        NOTES:
            https://scanpy.readthedocs.io/en/stable/tutorials/plotting/core.html
            https://chethanpatel.medium.com/community-detection-with-the-louvain-algorithm-a-beginners-guide-02df85f8da65
            https://i11www.iti.kit.edu/_media/teaching/theses/ba-nguyen-21.pdf
            https://www.ultipa.com/document/ultipa-graph-analytics-algorithms/leiden/v4.3
            Resolution profile: https://leidenalg.readthedocs.io/en/stable/advanced.html
            Benchmarking atlas-level integration single cell data: https://github.com/theislab/scib

        TODO:
            -Silhouette score: https://github.com/scverse/scanpy/issues/222

        """
        adata = self.adata

        if not os.path.exists(self.filepath.replace(".h5ad", "maxfreqcell.h5ad")):
            # cluster_assignments = adata.obs["clusters"].array
            cell_type = self.color_keys[0]
            cell_counts = adata.obs[cell_type].value_counts()  # .index[0]
            # print("cell counts : {}".format(cell_counts))
            maxfreq_cell = cell_counts.index[0]
            maxfreq = cell_counts.loc[maxfreq_cell]
            print(f"Most frequent cell found is {maxfreq_cell} with {maxfreq} members---------------------- ")
            adata_maxfreqcell = adata[adata.obs[cell_type].isin([maxfreq_cell])].copy()
        else:
            adata_maxfreqcell = sc.read(self.filepath.replace(".h5ad", "_maxfreqcell.h5ad"))
        datasets_dict = {"maxfreqcell": [adata_maxfreqcell, "_maxfreqcell"],
                         "all": [adata, ""],
                         }

        for dataset_info in list(datasets_dict.values()):
            dataset, name = dataset_info
            if "clusters" not in list(dataset.obs.keys()) or self.overwrite:
                print("Computing neighbour clusters using Leiden hierarchical clustering")

                # compute clusters using the leiden method and store the results with the name `clusters` > sc.pp.neighbours already run before
                if name == "_maxfreqcell":
                    sc.pp.neighbors(dataset, n_neighbors=5)
                sc.tl.leiden(
                    dataset,
                    key_added="clusters",
                    resolution=0.5,
                    # higher values more clusters, increases the weight over the coarseness of the clustering.
                    n_iterations=5,
                    flavor="igraph",
                    directed=False,
                )

                dataset.write(self.filepath.replace(".h5ad", f"{name}.h5ad"))
            else:
                print("Precomputed clusters found, continue")

            if self.plot_all:
                self.compute_leiden_plots(dataset, name)

            datasets_dict[name].append(dataset)
            dataset.write(self.filepath.replace(".h5ad", f"{name}.h5ad"))
            del dataset
            gc.collect()

        return datasets_dict

    def plot_neighbour_leiden_clusters_cuda(self):
        """

        NOTES:
            https://scanpy.readthedocs.io/en/stable/tutorials/plotting/core.html
            https://chethanpatel.medium.com/community-detection-with-the-louvain-algorithm-a-beginners-guide-02df85f8da65
            https://i11www.iti.kit.edu/_media/teaching/theses/ba-nguyen-21.pdf
            https://www.ultipa.com/document/ultipa-graph-analytics-algorithms/leiden/v4.3
            Resolution profile: https://leidenalg.readthedocs.io/en/stable/advanced.html
            Benchmarking atlas-level integration single cell data: https://github.com/theislab/scib

        TODO:
            -Silhouette score: https://github.com/scverse/scanpy/issues/222

        """
        adata = self.adata
        if not os.path.exists(self.filepath.replace(".h5ad", "maxfreqcell.h5ad")):
            # cluster_assignments = adata.obs["clusters"].array
            cell_type = self.color_keys[0]
            cell_counts = adata.obs[cell_type].value_counts()  # .index[0]
            # print("cell counts : {}".format(cell_counts))
            maxfreq_cell = cell_counts.index[0]
            maxfreq = cell_counts.loc[maxfreq_cell]
            print(f"Most frequent cell found is {maxfreq_cell} with {maxfreq} members---------------------- ")
            adata_maxfreqcell = adata[adata.obs[cell_type].isin([maxfreq_cell])].copy()
        else:
            adata_maxfreqcell = sc.read(self.filepath.replace(".h5ad", "_maxfreqcell.h5ad"))
        datasets_dict = {"maxfreqcell": [adata_maxfreqcell, "_maxfreqcell"],
                         "all": [adata, ""],
                         # "maxfreqcell": [adata_maxfreqcell, "_maxfreqcell"]
                         }

        for dataset_info in list(datasets_dict.values()):
            dataset, name = dataset_info
            rsc.get.anndata_to_GPU(adata=dataset, convert_all=True)
            rsc.pp.filter_genes(dataset, min_count=1)
            if "clusters" not in list(dataset.obs.keys()) or self.overwrite:
                print("Computing neighbour clusters using Leiden hierarchical clustering")
                # compute clusters using the leiden method and store the results with the name `clusters` > sc.pp.neighbours already run before

                dataset.obs_names_make_unique(join='-')
                if name == "_maxfreqcell":
                    rsc.pp.neighbors(dataset, n_neighbors=5)  # we need to re-run this
                rsc.tl.leiden(
                    dataset,
                    key_added="clusters",
                    resolution=0.5,
                    # higher values more clusters, increases the weight over the coarseness of the clustering.
                    n_iterations=100,
                )
                dataset = rsc.get.anndata_to_CPU(dataset, convert_all=True, copy=True)
                dataset.write(self.filepath.replace(".h5ad", f"{name}.h5ad"))
            else:
                print("Precomputed clusters found, continue")
                dataset = rsc.get.anndata_to_CPU(dataset, convert_all=True, copy=True)

            if self.plot_all:
                self.compute_leiden_plots(dataset,name)

            dataset.write(self.filepath.replace(".h5ad", f"{name}.h5ad"))
            del dataset
            gc.collect()
        return datasets_dict

def predict_cell_query():
    """Use to transform the dataset: https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.ingest.html"""

def scanpy_scvi(adata_file):

    adata = sc.read_h5ad(adata_file)
    #print(adata)

    scvi.settings.seed = 0
    scvi.settings.dl_num_workers = 4
    scvi.settings.batch_size = 1000 #128
    #scvi.settings.num_threads = 2

    # scvi.dataloaders.num_workers=4
    # scvi.dataloaders.pin_memory = True
    model_cls = scvi.model.SCVI
    model_cls.device = "cuda"


    model_cls.setup_anndata(adata, batch_key="batch_key",labels_key="cell_type")  # unlabeled_category="Unknown"
    model = model_cls(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train(max_epochs=50)
    latent = model.get_latent_representation()
    adata.obsm["og_scvi_latent"] = latent

def get_cell_ontology_label(term_id, ontology_file_path):
    """
    Retrieve the label for a given Cell Ontology (CL) term using RDFLib.

    Parameters:
    term_id (str): The Cell Ontology term ID (e.g., 'CL:4023064')
    ontology_file_path (str): Full path to the Cell Ontology .owl file

    Returns:
    str: The label of the term, or None if not found
    """
    import rdflib

    try:
        # Create a new graph
        graph = rdflib.Graph()

        # Load the ontology file
        graph.parse(ontology_file_path, format='xml')

        # Define namespaces
        rdf = rdflib.namespace.RDF
        rdfs = rdflib.namespace.RDFS
        owl = rdflib.namespace.OWL
        oboInOwl = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')

        # Construct the full URI for the term
        term_uri = rdflib.URIRef(f'http://purl.obolibrary.org/obo/{term_id.replace(":", "_")}')

        # Query for the label
        labels = list(graph.objects(term_uri, rdfs.label))
        print("Labels: {}".format(labels))

        # Return the first label if found
        return labels[0] if labels else None

    except Exception as e:
        print(f"Error retrieving label: {e}")
        return None

def search_cell_ontology_rdflib(owl_graph, term_id, return_children=False):
    """

    Args:
        owl_graph: Parsed owl file using  rdflib.Graph().parse(owl_file_path, format="xml")
        term_id: UBERON ontology term such UBERON:0000955
        return_children: whether to resturn the children terms associated to the term_id

    Returns:
        label: str or dict of human readable terms, i.e brain

    """
    import rdflib
    from rdflib.namespace import RDFS
    print("Finding readable cell ontology term")

    # Load the UBERON ontology file:
    # In RDF, a graph is constructed from triples, each of which represents an RDF statement that has at least three components:
    # subject: the entity being annotated
    # predicate: a relation between the subject and the object
    # object: another entity or a literal value

    # g = rdflib.Graph()
    # g.parse(owl_file, format="xml")
    # # Define Cell ontology namespace, pulling URL
    # uri = 'http://www.geneontology.org/formats/oboInOwl#'
    # CellOntologies = rdflib.Namespace(uri)
    # print("Done loading graph")

    # Construct the full URI for the term
    ontology_uri = rdflib.URIRef(f'http://purl.obolibrary.org/obo/{term_id.replace(":", "_")}')

    # Function to get the label of a CELL ONTOLOGY term
    def get_cell_ontology_label(cellonto_id):
        """
        rdfs:label a rdf:Property ;
        rdfs:isDefinedBy <http://www.w3.org/2000/01/rdf-schema#> ;
        rdfs:label "label" ;
        rdfs:comment "A human-readable name for the subject." ;
        rdfs:domain rdfs:Resource ;
        rdfs:range rdfs:Literal .
	"""
        #term = term_uri[cellonto_id]  # just adds the digit part of the uberon term http://purl.obolibrary.org/obo/UBERON_0002107
        label = list(owl_graph.objects(ontology_uri, RDFS.label))  # [rdflib.term.Literal('liver')]
        parents = list(owl_graph.objects(ontology_uri, RDFS.subClassOf))  # [rdflib.term.Literal('liver')]
        if label:
            return label[0].toPython()
        else:
            return None

    # Example queries

    # Get the label of the UBERON term for "heart"

    label = get_cell_ontology_label(term_id.replace("CL:", ""))

    print(f"Found label for {term_id}: {label}")
    if return_children:
        # Get all subclasses of the "organ" class
        organ_class = ontology_uri[term_id.replace("CL:", "")]  # UBERON ID for "organ" 0000062
        print("Example : {}".format(organ_class))
        organ_class = ontology_uri[term_id.replace("CL:", "")]  # UBERON ID for "organ" 0000062
        print("Given : {}".format(organ_class))
        organ_subclasses = list(owl_graph.subjects(RDFS.subClassOf, organ_class))
        organ_labels = [get_cell_ontology_label(str(s).split("_")[-1]) for s in organ_subclasses]
        return {"main": label, "children": organ_labels}
    else:
        return label

def search_uberon_rdflib(owl_graph,term_id,return_children=False):
    """

    Args:
        owl_graph: Parsed owl file using  rdflib.Graph().parse(owl_file_path, format="xml")
        term_id: UBERON ontology term such UBERON:0000955
        return_children: whether to resturn the children terms associated to the term_id

    Returns:
        label: str or dict of human readable terms, i.e brain

    """
    import rdflib
    from rdflib.namespace import RDFS
    print("Finding readable Uberon term")

    # Load the UBERON ontology file:

    # g = rdflib.Graph()
    # g.parse(owl_file, format="xml")

    # Define UBERON namespace, pulling URL
    uri = "http://purl.obolibrary.org/obo/UBERON_"
    UBERON = rdflib.Namespace(uri)
    print("Done loading graph")
    # Function to get the label of a UBERON term
    def get_uberon_label(uberon_id):
        """
        rdfs:label a rdf:Property ;
        rdfs:isDefinedBy <http://www.w3.org/2000/01/rdf-schema#> ;
        rdfs:label "label" ;
        rdfs:comment "A human-readable name for the subject." ;
        rdfs:domain rdfs:Resource ;
        rdfs:range rdfs:Literal .
	"""
        term = UBERON[uberon_id] #just adds the digit part of the uberon term http://purl.obolibrary.org/obo/UBERON_0002107
        label = list(owl_graph.objects(term, RDFS.label)) #[rdflib.term.Literal('liver')]
        parents = list(owl_graph.objects(term, RDFS.subClassOf)) #[rdflib.term.Literal('liver')]
        parents = [UBERON[uberon_id]]
        if label:
            return label[0].toPython()
        else:
            return None

    # Example queries

    # Get the label of the UBERON term for "heart"

    label = get_uberon_label(term_id.replace("UBERON:",""))

    print(f"Found label for {term_id}: {label}")
    if return_children:
        # Get all subclasses of the "organ" class
        organ_class = UBERON[term_id.replace("UBERON:","")]  # UBERON ID for "organ" 0000062
        print("Example : {}".format(organ_class))
        organ_class = UBERON[term_id.replace("UBERON:","")]  # UBERON ID for "organ" 0000062
        print("Given : {}".format(organ_class))
        organ_subclasses = list(owl_graph.subjects(RDFS.subClassOf, organ_class))
        organ_labels = [get_uberon_label(str(s).split("_")[-1]) for s in organ_subclasses]
        return {"main":label,"children":organ_labels}
    else:
        return label

def build_coarsened_metadata(cellxgene_census:types.ModuleType,script_dir:str, method="broad",overwrite=False) -> pd.DataFrame:
    """
    https://console.cloud.google.com/storage/browser/cellarium-human-primary-data/curriculum/human_all_primary/extract_files;tab=objects?authuser=1&prefix=&forceOnObjectsSortingFiltering=false
    Args:
        cellxgene_census:
        script_dir:

    Returns:

    """

    filepath_coarsened = f"{script_dir}/data/pseudobulk/cell_metadata_coarsened_{method}.pkl"
    if not os.path.exists(filepath_coarsened) or overwrite:
        print(f"Building coarsened cell metadata method {method}...")
        census = cellxgene_census.open_soma(census_version="2024-07-01")
        overwrite = False
        filepath = "{}/data/pseudobulk/cell_metadata.pkl".format(script_dir)
        if not os.path.exists(filepath) or overwrite:
            print("File {} notfound or overwrite is set to True".format(filepath))
            # Reads SOMADataFrame as a slice
            cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
                value_filter=f"""(is_primary_data == True)""",
                column_names=["assay", "cell_type", "cell_type_ontology_term_id",
                              "tissue", "tissue_ontology_term_id", "dataset_id", "donor_id",
                              "tissue_general", "suspension_type",
                              "disease", "disease_ontology_term_id",
                              "development_stage", "development_stage_ontology_term_id",
                              "raw_sum",
                              "sex", "soma_joinid"]
            ).concat().to_pandas()
            cell_metadata.to_pickle(filepath)
        else:
            print("Reading {}".format(filepath))
            cell_metadata = pd.read_pickle(filepath)


        #cell_metadata[['cell_type', 'cell_type_ontology_term_id']].drop_duplicates(inplace=True) #TODO: Apparently we might not use this, was just for analyzing
        cell_metadata['cell_type_unknown'] = cell_metadata['cell_type'] == 'unknown'
        cell_metadata['cell_type_unknown'].sum() / len(cell_metadata)

        if method == "broad":
            # filepath = "{}/data/pseudobulk/info.pkl".format(script_dir)
            # overwrite = False
            # if not os.path.exists(filepath) or overwrite:
            #     print("File {} notfound or overwrite is set to True".format(filepath))
            #     info = census["census_info"]["summary"].read().concat().to_pandas()  # read SOMA object, convert to pyarrow object (concat) and then to pandas
            #     info.to_pickle(filepath)
            # else:
            #     print("Reading {}".format(filepath))
            #     info = pd.read_pickle(filepath)
            #
            # filepath = "{}/data/pseudobulk/dataset_df.pkl".format(script_dir)
            # overwrite = False
            # if not os.path.exists(filepath) or overwrite:
            #     print("File {} notfound or overwrite is set to True".format(filepath))
            #     dataset_df = census["census_info"]["datasets"].read().concat().to_pandas()  # read SOMA object, convert to pyarrow object (concat) and then to pandas
            #     dataset_df.to_pickle(filepath)
            # else:
            #     print("Reading {}".format(filepath))
            #     dataset_df = pd.read_pickle(filepath)

            donor_cell_type_df1 = (
                cell_metadata[['dataset_id', 'donor_id', 'cell_type']]
                .groupby(['dataset_id', 'donor_id'],
                         observed=True)  # observed= True only show observed values for categorical groupers
                .agg(lambda s: s.nunique())  # counts the number of rows with the same ["dataset_id","donor_id"] combo
            ).rename(columns={'cell_type': 'n_unique_cell_types'})

            donor_cell_type_df2 = (
                cell_metadata[['dataset_id', 'donor_id', 'cell_type_unknown']]
                .groupby(['dataset_id', 'donor_id'], observed=True)
                .mean()
            ).rename(columns={'cell_type_unknown': 'cell_type_unknown_fraction'})
            donor_cell_type_df = pd.concat([donor_cell_type_df1, donor_cell_type_df2], axis=1)

            num_cell_df = (
                cell_metadata[['dataset_id', 'donor_id', 'cell_type']]
                .groupby(['dataset_id', 'donor_id'], observed=True)
                .count()
            ).rename(columns={'cell_type': 'n_cells'})

            devstage_coarsener = np.load(f'{script_dir}/data/pseudobulk/devstage_coarsen_map.npy',
                                         allow_pickle=True).item()
            devstage_name_lookup = np.load(f'{script_dir}/data/pseudobulk/devstage_name_lookup.npy',
                                           allow_pickle=True).item()

            cell_metadata['coarse_development_stage_ontology_id'] = cell_metadata[
                'development_stage_ontology_term_id'].str.replace(':', '_').map(devstage_coarsener)
            cell_metadata['coarse_development_stage'] = cell_metadata['coarse_development_stage_ontology_id'].map(
                devstage_name_lookup)

            development_df = (
                cell_metadata[['dataset_id', 'donor_id', 'development_stage', 'development_stage_ontology_term_id',
                               'coarse_development_stage', 'coarse_development_stage_ontology_id']]
                .groupby(['dataset_id', 'donor_id'], observed=True)
                .agg(lambda s: s.unique().tolist()[0])
            )

            disease_df = (
                cell_metadata[['dataset_id', 'donor_id', 'disease', 'disease_ontology_term_id']]
                .groupby(['dataset_id', 'donor_id'], observed=True)
                .agg(lambda s: s.unique().tolist()[0])
            )

            sex_df = (
                cell_metadata[['dataset_id', 'donor_id', 'sex']]
                .groupby(['dataset_id', 'donor_id'], observed=True)
                .agg(lambda s: s.unique().tolist()[0])
            )


            # Highlight: cell development stages

            devstage_coarsener = np.load(f'{script_dir}/data/pseudobulk/devstage_coarsen_map.npy',
                                         allow_pickle=True).item()
            devstage_name_lookup = np.load(f'{script_dir}/data/pseudobulk/devstage_name_lookup.npy',
                                           allow_pickle=True).item()

            cell_metadata['coarse_development_stage_ontology_id'] = cell_metadata[
                'development_stage_ontology_term_id'].str.replace(':', '_').map(devstage_coarsener)
            cell_metadata['coarse_development_stage'] = cell_metadata['coarse_development_stage_ontology_id'].map(devstage_name_lookup)

            tissue_coarsener = np.load(f'{script_dir}/data/pseudobulk/tissue_coarsen_map.npy', allow_pickle=True).item()
            tissue_dist = np.load(f'{script_dir}/data/pseudobulk/tissue_distances_from_root.npy', allow_pickle=True).item()
            tissue_name_lookup = np.load(f'{script_dir}/data/pseudobulk/tissue_name_lookup.npy', allow_pickle=True).item()

            cell_metadata['tissue_coarse_ontology_id'] = cell_metadata['tissue_ontology_term_id'].str.replace(':', '_').map(tissue_coarsener)
            cell_metadata['tissue_coarse'] = cell_metadata['tissue_coarse_ontology_id'].map(tissue_name_lookup)
            cell_metadata['tissue_name_ont_coarsename_coarseont'] = list(zip(
                cell_metadata['tissue'],
                cell_metadata['tissue_ontology_term_id'],
                cell_metadata['tissue_coarse'],
                cell_metadata['tissue_coarse_ontology_id'],
            ))

            tissue_df = (
                cell_metadata[['dataset_id', 'donor_id', 'tissue_name_ont_coarsename_coarseont']]
                .groupby(['dataset_id', 'donor_id'], observed=True)
                .agg(lambda s: s.unique().tolist())
            )

            dfs = [donor_cell_type_df, num_cell_df, tissue_df, development_df, disease_df, sex_df]

            df = pd.concat(dfs, axis=1)
            df.reset_index()
            df.to_pickle(filepath_coarsened)

        elif method == "ku":
            tissue_coarsener = pd.read_csv(f"{script_dir}/data/common_files/uberon_ontology_map.tsv",sep="\t",skiprows=1)
            tissue_coarsener_dict = dict(zip(tissue_coarsener.iloc[:,0],tissue_coarsener.iloc[:,1]))
            #tissue_coarsener_dict_reverse = dict(zip(tissue_coarsener.iloc[:,1],tissue_coarsener.iloc[:,0]))
            cell_metadata['tissue_coarse_ontology_id'] = cell_metadata['tissue_ontology_term_id'].map(tissue_coarsener_dict)
            #cell_metadata.loc[cell_metadata['tissue_coarse_ontology_id'].isna(),"tissue_coarse_ontology_id"] = cell_metadata.loc[cell_metadata['tissue_coarse_ontology_id'].isna(),"tissue_ontology_term_id"].map(tissue_coarsener_dict_reverse)

            cell_coarsener = pd.read_csv(f"{script_dir}/data/common_files/cell_ontology_map.tsv",sep="\t",skiprows=1)
            cell_coarsener_dict = dict(zip(cell_coarsener.iloc[:, 0], cell_coarsener.iloc[:, 1]))

            cell_metadata['cell_type_coarse_ontology_term_id'] = cell_metadata['cell_type_ontology_term_id'].map(cell_coarsener_dict).replace(['^UBERON','^BFO','^PR'], np.nan, regex=True)

            # tissue_df = (
            #     cell_metadata[['dataset_id', 'donor_id', 'tissue_coarse_ontology_id']]
            #     .groupby(['dataset_id', 'donor_id'], observed=True,dropna=True)
            #     #.agg(lambda s: s.unique().tolist())
            #     .agg(lambda s: [x for x in s.tolist() if not np.isnan(x)])
            # )
            #
            # cell_type_df = (
            #     cell_metadata[['dataset_id', 'donor_id', 'cell_type_coarse_ontology_term_id']]
            #     .groupby(['dataset_id', 'donor_id'], observed=True,dropna=True)
            #     # .agg(lambda s: s.unique().tolist())
            #     .agg(lambda s: [x for x in s.tolist() if not np.isnan(x)]) #TODO: Fix
            # )

            # df = (
            #     cell_metadata[['dataset_id', 'donor_id', 'cell_type_coarse_ontology_term_id','tissue_coarse_ontology_id','sex','disease', 'disease_ontology_term_id']]
            #     .groupby(['cell_type_coarse_ontology_term_id','tissue_coarse_ontology_id','sex',"disease_ontology_term_id"], observed=True,dropna=True,as_index=False)
            #     .agg(lambda s: s.unique().tolist())
            #     #.agg(lambda s: [x for x in s.tolist() if not np.isnan(x)])
            # )

            df = cell_metadata #TODO: Delete if the groupby are not used
            df.to_pickle(filepath_coarsened)
        census.close()

    else:
        print(f"Reading {filepath_coarsened}")
        df = pd.read_pickle(filepath_coarsened)

    return df

def read_h5ad_gcs(filename: str, storage_client: storage.Client | None = None) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the Google Cloud Storage.

    Example::

        >>> adata = read_h5ad_gcs("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename: Path to the data file in Cloud Storage.
    """
    if not filename.startswith("gs:"):
        raise ValueError("The filename must start with 'gs:' protocol name.")
    # parse bucket and blob names from the filename

    filename = re.sub(r"^gs://?", "", filename)
    bucket_name, blob_name = filename.split("/", 1)

    if storage_client is None:
        print("Initializing storage")
        storage_client = storage.Client()

    print("Initializing bucket")
    bucket = storage_client.bucket(bucket_name)
    print("Initializing blob")
    blob = bucket.blob(blob_name)


    with blob.open("rb") as f:
        return sc.read_h5ad(f,backed=None) #backed = "r" / "r+"

        #return read_h5ad(f)

    # with fsspec.open(filename, mode='rb') as f: #SLOWER
    #     adata = read_h5ad(f)
    #     return adata

def plot_histogram(coarsened_metadata,category,type="Barplot",bucket_size=1000):

    print("Starting bar plot ....")
    title_dict = {
        "cell_type_coarse_ontology_term_id":["cell ontology",1,(24,24)],
        "tissue_coarse_ontology_id":["tissue ontology",2,(24,24)],
        "sex":["sex",1,(10,10)],
        "combinations_all":["cell+tissue+sex+disease combinations",10,(30,24)],
        "combinations_cell_tissue":["cell+tissue combinations",1,(30,24)],
        "combinations_cell_tissue_disease":["cell+tissue+disease combinations",1,(30,24)],
             }
    title,step,size = title_dict[category]
    value_counts = coarsened_metadata[category].value_counts()  # .to_dict()
    # Highlight: Finding diversity of cell types within the combos with less than 250 cell counts
    df = value_counts.rename_axis(category).reset_index(name='counts')
    df = df[df["counts"] < 250]

    numtissuesonts = len(pd.unique(coarsened_metadata["tissue_coarse_ontology_id"])) # 61 tissues with same ontology term and 232 tissues in cellxgene
    numcellstypesonts = len(pd.unique(coarsened_metadata["cell_type_coarse_ontology_term_id"])) #135 with the sama cell type ontology term and 229 in annotated cell types

    tissue_uberon_dict = dict(zip(coarsened_metadata["tissue_coarse_ontology_id"].values.tolist(),coarsened_metadata["tissue_general"].values.tolist()))
    celltype_uberon_dict = dict(zip(coarsened_metadata["cell_type_coarse_ontology_term_id"].values.tolist(),coarsened_metadata["cell_type"].values.tolist()))


    if category == "combinations_cell_tissue":
        df[['cell_type_ont', 'tissue_ont']] = df[category].str.split(',', n=1, expand=True)
        df = df.groupby("tissue_ont",as_index=False)["cell_type_ont"].apply(np.unique)
        df["tissue"] = df["tissue_ont"].map(tissue_uberon_dict)
        df["num_cell_types_ont"] = df.cell_type_ont.map(len)

        df = df.sort_values("num_cell_types_ont",ascending=False)
        df = df.explode("cell_type_ont")

        df["cell_type"] = df["cell_type_ont"].map(celltype_uberon_dict)

        #df.drop("cell_type_ont",axis=1,inplace=True)
        df.to_csv("Diversity_df_{}.tsv".format(category),index=False,sep="\t")

    else:

        coarsened_metadata = coarsened_metadata[coarsened_metadata[category].isin(df[category])]

        if category == "tissue_coarse_ontology_id":
            columns = [category,"tissue","cell_type","cell_type_coarse_ontology_term_id"]
        else:
            columns = [category, "tissue", "cell_type", "tissue_coarse_ontology_id"]

        df = pd.merge(coarsened_metadata[columns],df,on=category,how="right")
        df = df.groupby("tissue_coarse_ontology_id",as_index=False)["cell_type_coarse_ontology_term_id"].apply(np.unique)
        df["num_cell_types_ont"] = df.cell_type_coarse_ontology_term_id.map(len)
        df = df.sort_values("num_cell_types_ont",ascending=False)
        df.drop("cell_type_coarse_ontology_term_id",axis=1,inplace=True)
        df.to_csv("Diversity_df_{}.tsv".format(category),index=False,sep="\t")


    value_counts = value_counts.to_dict()

    fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(nrows=2,ncols=2,figsize=size)
    cell_counts = list(value_counts.values())
    maxval = np.max(cell_counts)
    minval = np.min(cell_counts)

    #Highlight: Bucket resizing at the beginning
    start_buckets = [0,250,500,750]
    end_buckets = [250,500,750,1000]
    minval=1000

    start_buckets = start_buckets + list(range(minval, maxval, bucket_size)) #TODO: Remove start_buckets
    end_buckets = end_buckets + list(range(minval + bucket_size, maxval, bucket_size))

    intervals = list(zip(start_buckets, end_buckets))

    # Init dict

    intervals_cumsums = defaultdict(int)
    intervals_numcombos = defaultdict(int)
    intervals_tissues = defaultdict(list)
    intervals_cells = defaultdict(list)

    # cumsum cell counts
    for key, value in value_counts.items():
        split = key.split(",")
        if len(split) > 1:
            cell, tissue = key.split(",")
        else:
            cell = tissue = key
        for interval in intervals:
            if interval[0] < value <= interval[1]:
                intervals_cumsums[interval] += value
                intervals_numcombos[interval] += 1
                if interval[1] <= 250:
                    intervals_tissues[interval].append(tissue)
                    intervals_cells[interval].append(cell)


    intervals_cumsums = {str(key):val for key,val in intervals_cumsums.items() if val != 0}
    intervals_numcombos = {str(key):val for key,val in intervals_numcombos.items() if val != 0}
    intervals_tissues = {str(key): Counter(val) for key, val in intervals_tissues.items() if val}
    intervals_cells = {str(key): Counter(val) for key, val in intervals_cells.items() if val}


    positions = list(range(0, len(intervals_cumsums) * step, step)) #step controls the distance between bars

    ax0.bar(positions,intervals_cumsums.values())
    ax0.set_xticks(positions)
    ax0.set_xticklabels(intervals_cumsums.keys(),rotation=65)
    ax0.set_xlabel(title)
    ax0.set_ylabel("cumsum cell counts")
    ax0.set_title("Number of cells per interval")

    total_numcombos = sum(intervals_numcombos.values())

    numcombos_perinterval = [ (numcombo/ total_numcombos)*100 for numcombo in intervals_numcombos.values()]

    ax1.bar(positions, numcombos_perinterval)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(intervals_numcombos.keys(),rotation=65)
    ax1.set_xlabel(title)
    ax1.set_ylabel("cumsum num combinations")
    ax1.set_title("Number of cell-tissue combos per interval")


    #Highlight: Tissue and cell counts within the lowest interval
    def plot_axes_(intervals_dict,ax,subtitle,ntotal):
        categories = []
        subcategories = []
        percentage = []
        counts = []
        colors = []
        np.random.seed(13)
        for i, (intervals, counter) in enumerate(intervals_dict.items()):
            for subcategory, count in counter.items():
                r = np.round(np.random.rand(), 1)
                g = np.round(np.random.rand(), 1)
                b = np.round(np.random.rand(), 1)

                categories.append(intervals)
                colors.append([r, g, b])
                subcategories.append(subcategory)
                percentage.append((float(count)/ntotal)*100)
                counts.append(count)
                i += 1

        ax.bar(subcategories, counts, color=colors)
        ax.set_xticklabels(subcategories, rotation=65)
        ax.set_title(f"{subtitle}")

    if category in ["combinations_cell_tissue","tissue_coarse_ontology_id"]:
        plot_axes_(intervals_tissues,ax2,f"Tissues lost in smallest interval (0,250). Total num tissues ontologies: {numtissuesonts}",numtissuesonts)
    if category in ["combinations_cell_tissue","cell_type_coarse_ontology_term_id"]:
        plot_axes_(intervals_cells,ax3,f"Cell types lost in smallest interval (0,250). Total num cells types: {numcellstypesonts}",numcellstypesonts)

    fig.suptitle("Cellxgene coarsed dataset by -{}-. Bucket size: {}. Number combinations: {}".format(category,bucket_size,len(value_counts)),fontsize=20)
    plt.savefig("{}_{}_bucketsize{}.jpg".format(type,category,bucket_size))

def coarsed_cell_types_diversity(metadata,build_uberon_map, script_dir,type="Barplot",bucket_size=1000):

    print("Starting bar plot ....")

    #Highlight: Mapping to coarsed ontology term
    tissue_coarsener = pd.read_csv(f"{script_dir}/data/common_files/uberon_ontology_map.tsv", sep="\t", skiprows=0, index_col=None)
    tissue_coarsener_dict = dict(zip(tissue_coarsener.iloc[:, 0], tissue_coarsener.iloc[:, 1]))
    metadata['tissue_coarse_ontology_id'] = metadata['tissue_ontology_term_id'].map(tissue_coarsener_dict)
    cell_coarsener = pd.read_csv(f"{script_dir}/data/common_files/cell_ontology_map.tsv", sep="\t", skiprows=0,index_col=None)


    cell_coarsener_dict = dict(zip(cell_coarsener.iloc[:, 0], cell_coarsener.iloc[:, 1]))
    metadata['cell_type_coarse_ontology_term_id'] = metadata['cell_type_ontology_term_id'].map(cell_coarsener_dict).replace(['^UBERON', '^BFO', '^PR'], np.nan, regex=True)

    # Highlight: Build or load previously found uberon-labels #TODO: Remember to re-make with the new data, right now it only contains intestine stuff
    uberon_map_path = f"{script_dir}/data/coarsed/merged/uberon_map_dict.json"
    uberon_map_dict = build_uberon_map(uberon_map_path,
                                       uberon_list=metadata["tissue_coarse_ontology_id"].unique().astype(str),
                                       overwrite=False)
    # Highlight: Build or load previously found cell ontology labels
    cell_ontology_map_path = f"{script_dir}/data/coarsed/merged/cell_ontology_map_dict.json"
    cell_ontology_map_dict = build_uberon_map(cell_ontology_map_path,
                                              uberon_list=metadata["cell_type_coarse_ontology_term_id"].unique().astype(str),
                                              type="cellontology",
                                              overwrite=False)

    #Highlight: Map the og cell types to the coarsened ones

    metadata["cell_type_coarsed"] = metadata["cell_type_coarse_ontology_term_id"].map(cell_ontology_map_dict)
    metadata["tissue_type_coarsed"] = metadata["tissue_coarse_ontology_id"].map(uberon_map_dict)
    category = "cell_type_coarsed"

    value_counts = metadata[category].value_counts()
    df = value_counts.rename_axis(category).reset_index(name='counts')

    grouped_metadata = metadata.groupby(category, as_index=False)["cell_type"].apply(np.unique)
    grouped_metadata["diversity"] = grouped_metadata["cell_type"].apply(len)
    diversity_dict = grouped_metadata.set_index(category)['diversity'].to_dict()

    df["cell_type_diversity"] = df[category].map(diversity_dict)

    df.to_csv(f"{script_dir}/data/common_files/coarsedned_cell_types_diversity.tsv", sep = "\t",index=False)















