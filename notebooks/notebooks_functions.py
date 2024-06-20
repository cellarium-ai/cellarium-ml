import pandas as pd
import seaborn as sns
from cellarium.ml.core import CellariumPipeline, CellariumModule
from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.data.fileio import read_h5ad_file
from cellarium.ml.utilities.data import AnnDataField, densify, categories_to_codes

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
            "batch_index_n": AnnDataField(attr="obs", key="batch", convert_fn=categories_to_codes),
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
                sample=True,
            )
            sparse_coos.append(sp.coo_matrix(counts_ng.cpu().numpy()))

    # make the final sparse matrix and keep it as a layer
    csr = sp.vstack(sparse_coos).tocsr()
    adata.layers[layer_key_added] = csr


    return adata

def plot_raw_data(adata: anndata.AnnData,filepath: str,figpath: str,color_keys = ['final_annotation', 'batch'] ):

    #sc.set_figure_params(fontsize=14, vector_friendly=True)
    sc.set_figure_params(fontsize=14, vector_friendly=True)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    #https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pl.embedding.html#scanpy.pl.embedding
    sc.pp.pca(adata) #pca projection initializes the umap use_rep?
    sc.pp.neighbors(adata, n_pcs=20, n_neighbors=15, metric='euclidean', method='umap')
    #sc.rapids_singlecell.pp.neighbors(adata, n_pcs=20, n_neighbors=15, metric='euclidean', method='umap')
    sc.tl.umap(adata)
    #sc.rapids_singlecell.tl.umap(adata)
    adata.obsm['X_raw_umap'] = adata.obsm['X_umap'].copy()


    try:
        sc.pl.embedding(adata, basis='raw_umap', color=color_keys,
                               ncols=1,show=False)
        plt.savefig(f"{figpath}/umap-raw.pdf", bbox_inches="tight")
        plt.close()
        plt.clf()
    except:
        pass

    adata.write(filepath)
    return adata

def plot_latent_representation(adata: anndata.AnnData,filepath: str,figpath:str,color_keys = ['final_annotation', 'batch']):
    sc.set_figure_params(fontsize=14, vector_friendly=True)
    sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=15, metric='euclidean', method='umap')
    sc.tl.umap(adata)
    adata.obsm['X_scvi_umap'] = adata.obsm['X_umap'].copy()
    try:
        sc.pl.embedding(adata, basis='scvi_umap',color=color_keys, ncols=1,show=False) #,save=".pdf")
        plt.savefig(f"{figpath}/scvi-latent-umap.pdf", bbox_inches="tight")
        plt.clf()
        plt.close()
    except:
        pass
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


def umap_group_genes(adata: anndata.AnnData,filepath: str):
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

def calculate_dimensions_plot(g_plots_list):
    nelements = len(g_plots_list)
    max_rows = 3
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
    ncols *= 2  # we duplicate the number of columns to fit the color bars
    ncols = 2 * ncols
    width_ratios = [6] * ncols  # we duplicate the number of columns to fit the color bars
    width_ratios[1::2] = [0.3] * int(ncols / 2)
    print(width_ratios)
    nrows_idx = np.repeat(list(range(nrows)),ncols)
    ncols_idx = np.tile(list(range(ncols)),nrows)
    print(nrows_idx)
    print(ncols_idx)
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
                # print("Adding plot to position {},{}".format(idx_row,idx_col))
                # if isinstance(g_plot,str):
                #     print("Here1")
                #     print(g_plot)
                #     print(titles_dict[g_plot])
                #
                #     #
                #     # SeabornFig2Grid(ax, fig, gs[idx_row, idx_col])
                #
                # else:
                #     print("Here2")
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

def plot_avg_expression(adata,basis,gene_set_dict,figpath,figkey,figname):

    genes_list = []
    for gene_key, gene_set in gene_set_dict.items():
        gene_key = gene_key.lower()
        settings_plot = plot_settings_helper(adata, gene_key,gene_set, cmap="OrRd")
        if settings_plot is not None:
            genes_list.append(gene_key)
            adata = settings_plot["adata"]

    nrows,ncols,nrows_idx,ncols_idx,width_ratios = calculate_dimensions_plot(genes_list)

    sc.pl.embedding(adata,
                    basis=basis,
                    color=['final_annotation', 'batch'] + genes_list,
                    projection="2d",
                    cmap = "magma_r",
                    ncols=nrows,
                    wspace=0.7,
                    show=False)
                    #save=".pdf")
    plt.savefig(f"{figpath}/{figname}", bbox_inches="tight")

def plot_neighbour_clusters(adata,gene_set,figpath):
    """

    NOTES:
        https://scanpy.readthedocs.io/en/stable/tutorials/plotting/core.html
        https://chethanpatel.medium.com/community-detection-with-the-louvain-algorithm-a-beginners-guide-02df85f8da65
        https://i11www.iti.kit.edu/_media/teaching/theses/ba-nguyen-21.pdf
        https://www.ultipa.com/document/ultipa-graph-analytics-algorithms/leiden/v4.3

    """
    print("Computing neighbour clusters using Lediden hierarchical clustering")
    # compute clusters using the leiden method and store the results with the name `clusters`
    sc.tl.leiden(
        adata,
        key_added="clusters",
        resolution=2, #higher values more clusters, control the coarseness of the clustering.
        n_iterations=20,
        flavor="igraph",
        directed=False,
    )

    #cluster_assignments = adata.obs["clusters"].array

    with rc_context({"figure.figsize": (15, 15)}):
        sc.pl.umap(
            adata,
            color=['final_annotation',"clusters"],
            add_outline=True,
            legend_loc="on data",
            legend_fontsize=12,
            legend_fontoutline=2,
            frameon=False,
            title="clustering of cells",
            palette="Set1",
            show=False,
            #save = "-leiden-clusters.pdf"
        )
        plt.savefig(f"{figpath}/leiden-clusters.pdf", bbox_inches="tight")
        plt.close()
        plt.clf()
    gene_set =  adata.var_names[adata.var['genes_of_interest']]
    sc.pl.dotplot(adata, gene_set, "clusters", dendrogram=True, show=False)#,save="clusters-glyco")
    plt.savefig(f"{figpath}/dotplot-leiden-clusters.pdf", bbox_inches="tight")
    plt.close()
    plt.clf()
    # with rc_context({"figure.figsize": (4.5, 3)}):
    #     sc.pl.violin(adata, gene_set, groupby="clusters",ncols=5,save="violin-glyco",show=False)

    ax = sc.pl.stacked_violin(
        adata, {"Glyco":gene_set}, groupby="clusters", swap_axes=False, dendrogram=True,show=False
    )
    plt.savefig(f"{figpath}/stacked-violin-leiden-clusters.pdf", bbox_inches="tight")
    plt.close()
    plt.clf()

    return adata


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






