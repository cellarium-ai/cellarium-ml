import gc
import warnings

import pandas as pd
import seaborn as sns
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
import umap
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import os,shutil
import yaml
import scvi
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.colorbar import Colorbar
from matplotlib.pyplot import rc_context
from sklearn.decomposition import NMF
import dataframe_image as dfi
#mpl.use('agg')



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


def plot_umap(adata: anndata.AnnData,filepath: str,figpath:str,figname,basis,rep,color_keys = ['final_annotation', 'batch']):
    print(f"UMAp of {basis} ...")
    sc.set_figure_params(fontsize=14, vector_friendly=True)
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=15, metric='euclidean',knn=True,method="umap") #X_scvi #X_raw
    sc.pp.pca(adata,svd_solver="auto")
    sc.tl.umap(adata)
    adata.obsm[f'X_{basis}'] = adata.obsm['X_umap'].copy()

    palette = list(matplotlib.colors.CSS4_COLORS.values()) if len(adata.obs[color_keys[0]].value_counts().keys()) > 20 else "tab20b"

    sc.pl.embedding(adata, basis=f'{basis}',color=color_keys,
                    palette=palette,
                    ncols=1,show=False)
    plt.savefig(f"{figpath}/{figname}.jpg", bbox_inches="tight")
    plt.clf()
    plt.close()
    if basis == "raw_umap":
        adata.obsm['X_raw_umap'] = adata.obsm['X_umap'].copy()
    else:

        adata.layers['raw'] = adata.X.copy()
    adata.write(filepath)

    return adata


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


def define_gene_expressions(adata,gene_set,foldername,filepath,gene_names):

    if gene_names:
        adata.var["genes_of_interest"] = adata.var[gene_names].isin(gene_set)
        adata_gene_set = adata[:, adata.var[gene_names].isin(gene_set)]
        print(adata_gene_set.var[gene_names].tolist())
    elif not gene_names:
        adata.var['genes_of_interest'] = adata.var_names.isin(gene_set)  # 23 glycogenes found only adata[:,adata.var_names.isin(gene_set)]
        adata_gene_set = adata[:, adata.var_names.isin(gene_set)]  # only used for counting
        print(adata_gene_set.var_names.tolist())
    else:
        warnings.warn("Please check for the name of the column/key for the genes in your dataset and try again")

    # aggregate umi-count expression values
    adata.var['expr'] = np.array(adata.layers['raw'].sum(axis=0)).squeeze()
    adata_gene_set.var['expr'] = np.array(adata_gene_set.layers['raw'].sum(axis=0)).squeeze()
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

def umap_group_genes(adata: anndata.AnnData,filepath: str):
    print("UMAP per reconstructed layer")
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

        adata.obs[f"{gene_key}"] = expression_gene
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

def plot_avg_expression_old(adata,adata_proj,gene_set_dict,filename):

    g_plots_list = []
    settings_plot_list = []


    for gene_key, gene_set in gene_set_dict.items():
        print(gene_key)
        g,settings_plot = plot_by_expression(adata, adata_proj, gene_set, gene_key, cmap="OrRd", alpha=0.7, size=5, fontsize=15)
        if g is not None:
            print("Gene set found")
            g_plots_list.append(g)
            settings_plot_list.append(settings_plot)
        else:
            print("Gene set not found")



    if g_plots_list:

        nrows,ncols,nrows_idx,ncols_idx,width_ratios = calculate_dimensions_plot(g_plots_list)
        fig = plt.figure(figsize=(25, 40))
        gs = gridspec.GridSpec(nrows, ncols,figure=fig,width_ratios=width_ratios)
        g_plots_list = np.repeat(g_plots_list, 2) #we need to repeat it cuz the colorbar
        settings_plot_list = np.repeat(settings_plot_list, 2) #also we need to repeat it cuz the colorbar

        #g_plots_list.insert(0, [("final_annotation", "Cell annotation"),("final_annotation", "Cell annotation")])
        #TODO: Finish
        titles_dict = {"final_annotation":"Cell annotation"}
        #g_plots_list = np.insert(g_plots_list,0,["final_annotation","final_annotation"])
        #settings_plot_list = np.insert(settings_plot_list,0,[None,None])

        for idx_row,idx_col,g_plot,settings_plot in zip(nrows_idx,ncols_idx,g_plots_list,settings_plot_list):

            if idx_col%2 == 0:
                SeabornFig2Grid(g_plot, fig, gs[idx_row, idx_col])
            else:
                if settings_plot is not None:
                    print("Adding colorbar to position {},{}".format(idx_row, idx_col))
                    cbax = plt.subplot(gs[idx_row, idx_col])
                    Colorbar(ax=cbax,mappable=plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=settings_plot["colormap_expression"]))
                    print("------------------------------------------")

        plt.savefig(f"figures/avg_expression_coloured_UMAP_{filename}.jpg")

    else:
        print("Genes not found. Cannot make the plot")

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

def analysis_nmf(adata,adata_subset,genes_list:list,filepath_subset:str,filepath:str,gene_group_name:str):
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
            sc.tl.score_genes(adata_subset, gene_list=top_genes_in_set, score_name=f'{col}_{gene_group_name}_score') #average expression of a set of genes subtracted with the average expression of a reference set of genes
        adata_subset.write(filepath_subset)

    else:
        warnings.warn("Skipping NMF analysis, adata_subset shape is : {}, no genes of interest found".format(adata_subset.shape))
    adata.write(filepath)

    return adata,adata_subset

def plot_nmf(adata,adata_subset,color_keys,figpath,gene_group_name=""):


    fig,ax = plt.subplots(figsize=(30,30))
    sc.pl.embedding(adata_subset,
                    basis=f'scvi_reconstructed_0_{gene_group_name}_geneset_umap',
                    color=[c for c in adata_subset.obs.columns if c.startswith(f'{gene_group_name}_nmf') and c.endswith(f'{gene_group_name}_score')] + color_keys,
                    cmap='Oranges', vmin=-1, vmax=3, ncols=2,show=False)
    plt.savefig(f"{figpath}/nmf_components_score_{gene_group_name}.jpg", dpi=600, bbox_inches="tight")
    plt.close()
    plt.clf()

    #TODO: also color by cell type

def plot_scree_plot(adata):

    # Perform a scree plot: Analysis of eigen values magnitudes

    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')

def plot_neighbour_leiden_clusters(adata,gene_set_dict,figpath,color_keys,filepath,overwrite):
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

    if not os.path.exists(filepath.replace(".h5ad","maxfreqcell.h5ad")):
        #cluster_assignments = adata.obs["clusters"].array
        cell_type = color_keys[0]
        cell_counts = adata.obs[cell_type].value_counts()  # .index[0]
        # print("cell counts : {}".format(cell_counts))
        maxfreq_cell = cell_counts.index[0]
        maxfreq = cell_counts.loc[maxfreq_cell]
        print(f"Most frequent cell found is {maxfreq_cell} with {maxfreq} members---------------------- ")
        adata_maxfreqcell = adata[adata.obs[cell_type].isin([maxfreq_cell])].copy()
    else:
        adata_maxfreqcell = sc.read(filepath.replace(".h5ad","_maxfreqcell.h5ad"))
    datasets_dict = {"maxfreqcell":[adata_maxfreqcell,"_maxfreqcell"],
                      "all":[adata,""],
                     #"maxfreqcell": [adata_maxfreqcell, "_maxfreqcell"]

                     }
    overwrite=False
    for dataset_info in list(datasets_dict.values()):
        dataset,name = dataset_info
        if "clusters" not in list(dataset.obs.keys()) or overwrite:
            print("Computing neighbour clusters using Leiden hierarchical clustering")
            #sc.pp.neighbors(dataset,n_neighbors=15,)
            # compute clusters using the leiden method and store the results with the name `clusters` > sc.pp.neighbours already run before
            sc.tl.leiden(
                dataset,
                key_added="clusters",
                resolution=0.5, #higher values more clusters, increases the weight over the coarseness of the clustering.
                n_iterations=5,
                flavor="igraph",
                directed=False,
            )

            dataset.write(filepath.replace(".h5ad",f"{name}.h5ad"))
        else:
            print("Precomputed clusters found, continue")

        with rc_context({"figure.figsize": (15, 15)}):
            sc.pl.umap(
                dataset,
                layer = "X_scvi_reconstructed_0_umap",
                use_raw=False,
                color=[color_keys[0],"clusters"],
                add_outline=True,
                legend_loc="on data",
                legend_fontsize=12,
                legend_fontoutline=2,
                frameon=False,
                title="clustering of cells",
                palette="Set1",
                show=False,
            )
            plt.savefig(f"{figpath}/leiden_clusters{name}.jpg",dpi=600, bbox_inches="tight")
            plt.close()
            plt.clf()

        # TODO: dotplot cannot handle many clusters, if there are too many, the gridspec will complain, perhaps with a larger figsize
        #dataset= dataset[(dataset.obs["clusters"] == "0") | (dataset.obs["clusters"] == "1")]

        cluster_counts_dict = dataset.obs["clusters"].value_counts()
        dataset = dataset[dataset.obs["clusters"].isin(cluster_counts_dict.keys()[:20])] #pick only the top 20 clusters with more members

        singlemember_clusters = [key for key,val in  cluster_counts_dict.items() if val == 1]
        if singlemember_clusters:
            print("Removing clusters with a single element")
            dataset = dataset[~dataset.obs["clusters"].isin(singlemember_clusters)] #remove clusters with a single element

        # print("Before")
        # print(cluster_counts_dict)
        #
        # print("After")
        # print(dataset.obs["clusters"].value_counts())

        gene_set =  dataset.var_names[dataset.var['genes_of_interest']]

        if name == "_maxfreqcell":
            plot_avg_expression(dataset, "X_scvi_reconstructed_0_umap", gene_set_dict, figpath, "glyco_expression_maxfreqcell" , color_keys)

        sc.pp.log1p(dataset)
        sc.tl.dendrogram(dataset,groupby="clusters") #need to re-run because
        sc.pl.dotplot(dataset,
                      gene_set,
                      layer="scvi_reconstructed_0",
                      use_raw=False,
                      groupby="clusters",
                      figsize=(20,20),
                      dendrogram=True,
                      show=False,
                      dot_max=1)
        plt.savefig(f"{figpath}/leiden_dotplot{name}.jpg", dpi=600, bbox_inches="tight")
        plt.close()
        plt.clf()
        # with rc_context({"figure.figsize": (4.5, 3)}):
        #     sc.pl.violin(adata, gene_set, groupby="clusters",ncols=5,save="violin-glyco",show=False)
        sc.tl.dendrogram(dataset, groupby="clusters")
        sc.pl.stacked_violin(dataset, {"Glyco":gene_set}, groupby="clusters",
                             layer="scvi_reconstructed_0",
                             swap_axes=False,
                             dendrogram=True,
                             show=False,use_raw=False)
        plt.savefig(f"{figpath}/leiden_stacked_violin{name}.jpg",dpi=600, bbox_inches="tight")
        plt.close()
        plt.clf()

        sc.tl.rank_genes_groups(dataset,
                                layer="scvi_reconstructed_0",
                                use_raw=False,
                                groupby="clusters",
                                method="wilcoxon",
                                mask_var="genes_of_interest")
        sc.pl.rank_genes_groups(dataset, n_genes=25, sharey=False,show=False)
        plt.savefig(f"{figpath}/leiden_rank_genes{name}.jpg", dpi=600, bbox_inches="tight")
        plt.close()
        plt.clf()

        dataset.write(filepath.replace(".h5ad",f"{name}.h5ad"))
        del dataset
        gc.collect()


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






