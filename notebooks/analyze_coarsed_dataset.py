import collections
import json
import pickle
from itertools import cycle
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import os ,sys
import signal

from dask.array import histogram
from joblib import Parallel, delayed
import functools
import argparse
from argparse import RawTextHelpFormatter
os.path.realpath("__file__")
fx_dir =os.path.dirname(os.path.abspath("__file__"))
script_dir = os.path.dirname(fx_dir)
sys.path.extend(["{}".format(fx_dir)])
import notebooks_functions as NF
import matplotlib
from collections import defaultdict
from coarse_cellxgene import glyco_map, glyco_pathway_map
from matplotlib.pyplot import rc_context
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
import seaborn as sns
Attributes = namedtuple("attributes",["colormap148","colormap20","max_xval","max_yval"])


def analyze_census_cellxgene_metadata(build=False,plot=False):
    #Highlight: Read metadata anc coarse according to ontologies
    # c = pd.read_csv(f"{script_dir}/data/common_files/cellontology_mapping_result.tsv",sep="\t")
    # ontologies_list = c["cell_type_coarse_ontology_term_id"].tolist()
    # cell_ontology_map_path = f"{script_dir}/data/common_files/cell_ontology_map_dict.json"
    # cell_ontology_map_dict = build_uberon_map(cell_ontology_map_path, uberon_list=ontologies_list, type="cellontology",
    #                                           overwrite=False)

    if build:
        import cellxgene_census #expensive operation
        coarsened_metadata = NF.build_coarsened_metadata(cellxgene_census,script_dir,method="ku",overwrite=True)
        coarsened_metadata = coarsened_metadata.dropna( subset=["cell_type_coarse_ontology_term_id", "tissue_coarse_ontology_id"], how="any")

        coarsened_metadata["combinations_all"] = coarsened_metadata["cell_type_coarse_ontology_term_id"].astype(str) + "," + coarsened_metadata["tissue_coarse_ontology_id"].astype(str) + "," + coarsened_metadata["sex"].astype(str) + "," + coarsened_metadata["disease_ontology_term_id"].astype(str)
        coarsened_metadata["combinations_cell_tissue"] = coarsened_metadata["cell_type_coarse_ontology_term_id"].astype(str) + "," + coarsened_metadata["tissue_coarse_ontology_id"].astype(str)
        coarsened_metadata["combinations_cell_tissue_disease"] = coarsened_metadata["cell_type_coarse_ontology_term_id"].astype(str) + "," + coarsened_metadata["tissue_coarse_ontology_id"].astype(str) + "," + coarsened_metadata["disease_ontology_term_id"].astype(str)

        # df = coarsened_metadata[["tissue_coarse_ontology_id", "tissue_ontology_term_id", "tissue", ]].drop_duplicates()
        # uberon_map_dict_path = "/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/common_files/uberon_map_dict.json"
        # uberon_map_dict = build_uberon_map(uberon_map_dict_path,overwrite=False)
        # df["tissue_coarsed"] = df["tissue_coarse_ontology_id"].map(uberon_map_dict)
        #df.to_csv(f"{script_dir}/data/common_files/uberon_mapping_result.tsv",sep="\t",index=False)

        #Highlight: Save old and new ontologies
        df2 = coarsened_metadata[["cell_type_coarse_ontology_term_id","cell_type_ontology_term_id","cell_type"]].drop_duplicates()
        cell_ontology_map_path = f"{script_dir}/data/common_files/cell_ontology_map_dict.json"
        cell_ontology_map_dict = build_uberon_map(cell_ontology_map_path, uberon_list=None, type="cellontology", overwrite=False)
        df2["cell_coarsed"] = df2["cell_type_coarse_ontology_term_id"].map(cell_ontology_map_dict)
        df2.to_csv(f"{script_dir}/data/common_files/cellontology_mapping_result.tsv",sep="\t",index=False)
        exit()

        # unique_cell_onts = coarsened_metadata["cell_type_coarse_ontology_term_id"].drop_duplicates()
        # unique_cell_onts.to_csv("Cell_ontologies_coarsed.tsv",sep="\t",index=False)
        #
        # unique_cell_onts = coarsened_metadata["cell_type_ontology_term_id"].drop_duplicates()
        # unique_cell_onts.to_csv("Cell_ontologies.tsv",sep="\t",index=False)

        if plot :
            NF.plot_histogram(coarsened_metadata,"combinations_cell_tissue","Barplot",250)
            NF.plot_histogram(coarsened_metadata,"tissue_coarse_ontology_id","Barplot",1000)
            NF.plot_histogram(coarsened_metadata,"cell_type_coarse_ontology_term_id","Barplot",1000)

def retrieve_ontology(owl_graph,term_id,type="uberon"):
    """"""
    #Highlight : Alt + L for single line execution
    if type == "uberon":
        print("Retrieving uberon ontologies")
        # owl_file = "{}/cellarium-ml/data/pseudobulk/uberon.owl".format(fx_dir)
        # owl_path = "http://purl.obolibrary.org/obo/uberon/releases/2024-09-03/uberon.owl"
        result = NF.search_uberon_rdflib(owl_graph, term_id, return_children=False)

    elif type == "cellontology":
        print("Retrieving cell ontologies")
        # owl_path = "http://purl.obolibrary.org/obo/cl/releases/2024-09-26/cl.owl"
        # owl_path = "http://purl.obolibrary.org/obo/cl.owl"
        result = NF.search_cell_ontology_rdflib(owl_graph, term_id, return_children=False)
        #result = NF.get_cell_ontology_label(term_id, owl_path)

    return result

def timeout_handler(signum, frame):
    raise TimeoutError("Time limit reached!")

def build_uberon_map(uberon_map_path,uberon_list=None,type="uberon",overwrite=False):
    """Uberon or cell ontology mapping"""
    import rdflib
    if type == "uberon":
        print("Preparing Uberon graph")
        owl_path = "http://purl.obolibrary.org/obo/uberon/releases/2024-09-03/uberon.owl"
    elif type == "cellontology":
        print("Retrieving cell ontologies")
        #owl_path = "http://purl.obolibrary.org/obo/cl/releases/2024-09-26/cl.owl"#TODO: Probably works
        owl_path = "http://purl.obolibrary.org/obo/cl.owl"

    if not os.path.exists(uberon_map_path) or overwrite:
        owl_graph = rdflib.Graph()
        owl_graph.parse(owl_path, format="xml")
        # Define Cell ontology namespace, pulling URL
        # uri = 'http://www.geneontology.org/formats/oboInOwl#'
        # CellOntologies = rdflib.Namespace(uri)
        print("Done loading graph")
        if uberon_list is not None:
            print("Uberon map dict not found or overwrite is set to True, building it ")

            d = defaultdict()
            for uberon in uberon_list:
                # Define a timeout handler
                time_limit = 60
                # Register the timeout handler
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(time_limit)
                try:
                    print(f"Looking for label for {uberon}")
                    label = retrieve_ontology(owl_graph,uberon,type)
                    print("Process completed.")
                except TimeoutError as e:
                    print(e)
                    label = "not found"
                finally:
                    # Cancel the alarm if the code completes before the timeout
                    signal.alarm(0)
                d[uberon] = label
                #pickle.dump(d, open(uberon_map_path, "wb"))
                json.dump(d, open(uberon_map_path, "w"),indent=1)

            return  d

        else:
            raise ValueError("Please provide an list of Uberon/Cell ontologies terms to map")
    else:
        print("Loading pre-existing Uberon_map_file")
        #uberon_map_dict = pickle.load(open(uberon_map_path,"rb"))
        uberon_map_dict = json.load(open(uberon_map_path,"r"))
        return uberon_map_dict

def split_into_chunks_anndata(input_adata,axis=0,chunks=2):
    """Divides adata into even splits
    :param adata input_adata
    :param int chunks: Number of divisions"""
    quot, rem = divmod(input_adata.shape[axis], chunks)
    divpt = lambda i: i * quot + min(i, rem)
    if axis == 1:
        return [input_adata[:,divpt(i):divpt(i + 1)] for i in range(chunks)]
    elif axis==0:
        return [input_adata[divpt(i):divpt(i + 1)] for i in range(chunks)]

def read_adata(filepath):
    a = sc.read_h5ad(filepath)
    # sc.pp.normalize_total(a) #cmp normalization #TODO: check
    return a

def load_counts(counts_path,sorter):
    "Reads in a tsv file with cell counts and the -combo- column "
    counts_combos = pd.read_csv(counts_path, sep="\t")
    counts_combos["combo"] = counts_combos["combo"].str.replace(".h5ad", "")

    counts_combos = counts_combos[counts_combos["combo"].isin(sorter)]
    counts_combos["combo_sorted"] = pd.Categorical(counts_combos["combo"], categories=sorter, ordered=True)
    counts_combos = counts_combos.sort_values("combo_sorted")
    counts_combos.drop("combo_sorted",axis=1,inplace=True)
    return counts_combos

def extract_glyco(adata,glyco_df):
    glyco_genes = glyco_df["ensembl_gene_id"].tolist()
    adata = adata[:, adata.var_names.isin(glyco_df["ensembl_gene_id"].tolist())]
    adata = adata[:, glyco_genes]  # reorder
    return adata

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_correlations(output_path, summary_chunk, suffix,analysis_dimension,title,attributes):

    colormap148 = attributes.colormap148
    colormap20 = attributes.colormap20
    max_xval= attributes.max_xval
    max_yval = attributes.max_yval
    markers = [
        'o',  # Circle
        '^',  # Upward-pointing triangle
        'v',  # Downward-pointing triangle
        '<',  # Left-pointing triangle
        '>',  # Right-pointing triangle
        's',  # Square
        'p',  # Pentagon
        '*',  # Star
        'h',  # Hexagon-1
        'H',  # Hexagon-2
        '+',  # Plus
        'x',  # Cross
        'X',  # Crossed diamonds
        'D',  # Diamond
        'd',  # Thin diamond
        '|',  # Vertical line
        '_',  # Horizontal line
    ]
    unique_genes = summary_chunk["ensembl"].unique().astype(str).tolist() #unique glyco ensembl
    unique_tissues = summary_chunk["tissue_ontology"].unique().astype(str)
    n_genes = len(unique_genes)
    n_tissues = len(unique_tissues)
    if analysis_dimension == "genes":
        unique_vals =unique_genes
    elif analysis_dimension == "tissues":
        unique_vals = unique_tissues
    markers = [item for item, idx in zip(cycle(markers), range(len(unique_vals)))]
    if len(unique_vals) > 20:
        color_dict = dict(zip(unique_vals, colormap148))
    else:
        color_dict = dict(zip(unique_vals, colormap20))


    fig, ax = plt.subplots(1, 1, figsize=(30, 15), layout="constrained")

    if analysis_dimension ==  "genes":
        for i,gene_ensembl in enumerate(unique_vals):  # for each gene
            gene_chunk = summary_chunk[summary_chunk["ensembl"] == gene_ensembl]
            mean_vals = gene_chunk["mean"].values
            idx_nonzeros = np.where(mean_vals != 0)[0]
            mean_vals = mean_vals[idx_nonzeros]
            gene_chunk = gene_chunk.iloc[idx_nonzeros]
            std_vals = gene_chunk["std"].values
            if len(mean_vals) != 0:
                #std_vals = std_vals[idx_nonzeros]
                gene_hgnc = gene_chunk["hgnc"].iloc[0]


                color = color_dict[gene_ensembl]
                ax.scatter(mean_vals, std_vals, color=color, label=gene_hgnc,
                           #marker=markers[i],
                           s=gene_chunk["ratio_cells"].tolist())
                confidence_ellipse(mean_vals, std_vals, ax, edgecolor=color, alpha=1)
                ax.scatter(mean_vals.mean(), std_vals.mean(), color=color, s=15)

    else:  # galnt15, we color by tissue cell combos
        for i,uberon_tissue in enumerate(unique_vals):

            tissue_gene_chunk = summary_chunk[summary_chunk["tissue_ontology"] == uberon_tissue]
            mean_vals = tissue_gene_chunk["mean"].values
            idx_nonzeros = np.where(mean_vals != 0)[0]
            mean_vals = mean_vals[idx_nonzeros]
            tissue_gene_chunk = tissue_gene_chunk.iloc[idx_nonzeros]
            std_vals = tissue_gene_chunk["std"].values
            tissue_label = tissue_gene_chunk["tissue_label"].iloc[0]
            color = color_dict[uberon_tissue]

            ax.scatter(mean_vals, std_vals,
                       color=color,
                       label=tissue_label,
                       s=np.rint(tissue_gene_chunk["ratio_cells"].tolist())
                       ) #, marker=markers[i])
            confidence_ellipse(mean_vals, std_vals, ax, edgecolor=color, alpha=1)
            ax.scatter(mean_vals.mean(), std_vals.mean(), color=color, s=15)

    # ax.hlines(y=2*adata_chunk_std.X.mean(axis=0),xmin=0,xmax=max_xval, color=color,linewidth=4)
    # ax.hlines(y=-2*adata_chunk_std.X.mean(axis=0),xmin=0,xmax=max_xval ,color=color,linewidth=4,linestyle='dotted')
    marker_sizes = np.rint(sorted(summary_chunk["ratio_cells"].tolist()))

    marker_sizes_dict = {"max-size":max(marker_sizes),
                         "median-size":np.median(marker_sizes),
                         "average-size":np.mean(marker_sizes),
                         "min-size-":min(marker_sizes),
                         }

    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=np.sqrt(size), label=f"{key}:{int(np.exp(size/10))}")
        for key,size in marker_sizes_dict.items()
    ]

    # Add legend for marker sizes
    markers_sizes_legends = ax.legend(handles=size_handles, title="Marker Sizes", loc="upper right",bbox_to_anchor=(0.75, 0.8, 0.4, 0.2), fontsize=25)
    ax.add_artist(markers_sizes_legends)

    ax.set_ylabel("Std expression", fontsize=20)
    ax.set_xlabel("Mean expression", fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.axis([-1, max_xval, -1, max_yval])
    ncols = 2 if n_genes > 18 else 1
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', ncols=ncols,
               bbox_to_anchor=(0.8, 0.5, 0.4, 0.2), fontsize=25)  # x,y,width,height
    plt.title(r"{}".format(title),
              fontsize=25)
    plt.savefig("{}/{}_correlations_mean_std_glyco.jpg".format(output_path, suffix))
    plt.clf()
    plt.close()


def umap_plot(mean,mean_path,figpath,overwrite=True,use_cuda=False):

    color_keys = ["tissue_label","cell_type_label"]  # TODO: Should have done mode for rest of the obs values before the mean
    NF.ComputeUMAP(mean, mean_path, figpath, "raw_umap", "raw_umap", "X_umap", use_cuda, overwrite,color_keys).run()


def correlations_plots(summary_df,figpath):
    #Highlight: Plot mean vs std for all genes per tissue-cell combo
    colormap148 = list(matplotlib.colors.CSS4_COLORS.values())  # 148
    colormap20 = matplotlib.colormaps["tab20"].colors


    #glyco_pathway_dict_ensembl = {"galnt_alone":"ENSG00000131386","galnt":glyco_pathway_dict_ensembl["galnt"]}
    #glyco_pathway_dict_ensembl = {**{"galnt_alone":"ENSG00000131386"},**glyco_pathway_dict_ensembl}

    glyco_file = f"{script_dir}/data/coarsed_glyco/analysis/glyco_pathway_df.p"
    glyco_pathway_dict = glyco_pathway_map(glyco_file, save=False, overwrite=False)

    glycopathways_subgroups = {initiator:summary_df[summary_df["glycopathway"] == initiator] for initiator in glyco_pathway_dict}

    max_xval = summary_df["mean"].max()*0.1
    max_yval = summary_df["std"].max()*0.01
    # max_xval = summary_df["mean"].max()
    # max_yval = summary_df["std"]max()

    attributes = Attributes(colormap148=colormap148, colormap20=colormap20, max_xval=max_xval, max_yval=max_yval)
    for key,summary_chunk in glycopathways_subgroups.items() :
        plot_correlations(figpath,
                          summary_chunk,
                          "mean_std_{}_zoomed".format(key.replace("/","_")),
                          "genes",
                          "Correlations between $\mu$ and $\sigma$ for every gene per combo (covariance confidence ellipsis)",
                          attributes)

    exit()
    glyco_pathway_dict_ensembl = {"galnt_alone": "ENSG00000131386"}

    gene_of_interest = "EOGT" #"ENSG00000163378"
    #gene_of_interest = "GALNT15" #"ENSG00000131386"
    n_chunks = 10
    summary_gene_df = summary_df[summary_df["hgnc"] == gene_of_interest]

    summary_gene_chunks = dict(zip(list(map('{:0}'.format, range(n_chunks))),split_into_chunks_anndata(summary_gene_df,axis=0,chunks=n_chunks)))

    # for key,summary_gene_chunk in summary_gene_chunks.items() :
    #     plot_correlations(figpath,
    #                       summary_gene_chunk,
    #                       "{}_part_{}_zoomed".format(gene_of_interest,key.replace("/","_")),
    #                       "tissues",
    #                       f"Correlations between $\mu$ and $\sigma$ for {gene_of_interest.upper()} per tissue-cell combo",
    #                       attributes)
    # exit()


def histogram_plot(summary_df,figpath):


    gene_names = summary_df["hgnc"].unique()


    nrows, ncols, nrows_idx, ncols_idx, width_ratios = NF.calculate_dimensions_plot(gene_names, max_rows=6,include_colorbars=False)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(70, 20), layout="constrained")
    max_val,min_val = summary_df["mean"].max(), summary_df["mean"].min()
    axs = axs.ravel()

    def plot_parallel(ax,gene,summary_df):

    #for ax, gene in zip(axs, gene_names):
        expr_mean = summary_df[summary_df["hgnc"] == gene]
        expr_mean =expr_mean["mean"].values
        sns.histplot(expr_mean, bins=10, ax = ax, kde=True, color="blue")
        ax.set_xlim((min_val,max_val))
        ax.set_title(gene)

    results = Parallel(n_jobs=-2)(
        delayed(functools.partial(plot_parallel, summary_df=summary_df))((ax, gene_name)) for ax, gene_name in
        zip(axs, gene_names))
    print(f"saved at {figpath}/Histograms/")
    plt.savefig(f"{figpath}/Histograms/histogram_expression.jpg")

def build_coarsed_summary_df(overwrite=False):

    folder_suffix = "_intestine" #TODO: Automatize
    #folder_suffix = ""
    summary_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_summary.tsv"
    if overwrite or not os.path.exists(summary_path):
        print("Creating summary dataframe")
        #Highlight: Load mean and std files
        mean_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_mean.h5ad"
        mean_full = read_adata(mean_path)
        std_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_std.h5ad"
        std_full = read_adata(std_path)

        #galant15 = mean_full[:,mean_full.var_names.isin(["ENSG00000131386"])]

        #Highlight: Load the conversion between glyco ensemble and hgnc
        glyco_file = f"{script_dir}/data/coarsed_glyco/analysis/glyco_df.tsv"
        glyco_df = glyco_map(glyco_file)

        glyco_file = f"{script_dir}/data/coarsed_glyco/analysis/glyco_pathway_df.p"
        glyco_pathway_dict = glyco_pathway_map(glyco_file,save=False,overwrite=False)
        glyco_pathway_df = pd.DataFrame([(k, v) for k, vals in glyco_pathway_dict.items() for v in vals], columns=["initiator", "gene"])
        glyco_pathway_dict = dict(zip(glyco_pathway_df.gene,glyco_pathway_df.initiator))

        glyco_dict1 = dict(zip(glyco_df.ensembl_gene_id,glyco_df.hgnc_symbol))
        glyco_dict2 = dict(zip(glyco_df.hgnc_symbol,glyco_df.ensembl_gene_id))

        glyco_pathway_dict_ensembl = defaultdict()
        for key, vals in glyco_pathway_dict.items():
            vals = [glyco_dict2[val.upper()] for val in vals if val.upper() in glyco_dict2.keys()]
            glyco_pathway_dict_ensembl[key] = vals

        #Highlight: Restrict the dataset only to glyco genes
        # mean_full.obs["combo_ontology"] = mean_full.obs["combo_ontology"].str.replace("/opt/project/cellxgene/coarsed/","").str.replace(r"\.$", "", regex=True) #TODO: Might need not be needed anymore
        mean_full.obs[['tissue_ontology', 'cell_ontology']] = mean_full.obs['combo_ontology'].str.split('_', n=1,expand=True)
        std_full.obs[['tissue_ontology', 'cell_ontology']] = std_full.obs['combo_ontology'].str.split('_', n=1,expand=True)
        mean_glyco = extract_glyco(mean_full,glyco_df)
        std_glyco = extract_glyco(std_full,glyco_df)



        #accidental_duplicates = [item for i,(item, count) in enumerate(collections.Counter(sorter).items()) if count > 1]
        idx_duplicates = mean_glyco.obs['combo_ontology'].duplicated()
        mean_glyco = mean_glyco[~idx_duplicates]
        std_glyco = std_glyco[~idx_duplicates]
        sorter = mean_glyco.obs["combo_ontology"].values.tolist()

        #Highlight: Load cell counts to filter those combos with lower amounts of cells. Make sure to sort it so everything is in the same order ready to merge

        counts_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/counts_combos.tsv"
        counts_combos = load_counts(counts_path, sorter)

        #Highlight: Load non-zero cell counts
        non_zero_counts_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/non_zero_cell_counts_combos.tsv"
        non_zero_counts_combos = load_counts(non_zero_counts_path,sorter)
        non_zero_counts_combos = non_zero_counts_combos[["combo"] + glyco_df["ensembl_gene_id"].tolist()] #slice only the glyco ones
        #non_zero_counts_combos[["tissue_ontology","cell_ontology"]] = non_zero_counts_combos["combo"].str.split("_",n=1,expand=True)
        non_zero_counts_combos_dict = non_zero_counts_combos.set_index("combo").to_dict(orient="index")

        #Highlight: Build or load previously found uberon-labels #TODO: Remember to re-make with the new data, right now it only contains intestine stuff
        uberon_map_path = f"{script_dir}/data/common_files/uberon_map_dict.json"
        uberon_map_dict = build_uberon_map(uberon_map_path, uberon_list=mean_glyco.obs["tissue_ontology"].unique().astype(str), overwrite=False)
        #Highlight: Build or load previously found cell ontology labels
        cell_ontology_map_path = f"{script_dir}/data/common_files/cell_ontology_map_dict.json"
        cell_ontology_map_dict = build_uberon_map(cell_ontology_map_path, uberon_list=mean_glyco.obs["cell_ontology"].unique().astype(str), type="cellontology", overwrite=False)


        mean_glyco.obs["cell_type_label"] = mean_glyco.obs["cell_ontology"].map(cell_ontology_map_dict)
        mean_glyco.obs["tissue_label"] = mean_glyco.obs["tissue_ontology"].map(uberon_map_dict)
        std_glyco.obs["cell_type_label"] = std_glyco.obs["cell_ontology"].map(cell_ontology_map_dict)
        std_glyco.obs["tissue_label"] = std_glyco.obs["tissue_ontology"].map(uberon_map_dict)


        mean_full.obs["cell_type_label"] = mean_full.obs["cell_ontology"].map(cell_ontology_map_dict)
        mean_full.obs["tissue_label"] = mean_full.obs["tissue_ontology"].map(uberon_map_dict)
        std_full.obs["cell_type_label"] = std_full.obs["cell_ontology"].map(cell_ontology_map_dict)
        std_full.obs["tissue_label"] = std_full.obs["tissue_ontology"].map(uberon_map_dict)


        sc.write(f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_mean_glyco.h5ad",mean_glyco)
        sc.write(f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_std_glyco.h5ad",std_glyco)




        sc.write(f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_mean.h5ad",mean_full)
        sc.write(f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_std.h5ad",std_full)


        n_combos = mean_glyco.X.shape[0]
        n_genes = mean_glyco.X.shape[1]




        summary_df = pd.DataFrame({"mean":mean_glyco.X.flatten(),
                                   "std":std_glyco.X.flatten(),
                                   "combos_ontology":np.array(sorter).repeat(n_genes),
                                   "cell_ontology":np.array(mean_glyco.obs["cell_ontology"]).repeat(n_genes),
                                   "tissue_ontology":np.array(mean_glyco.obs["tissue_ontology"]).repeat(n_genes),
                                   "total_num_cells": counts_combos["counts"].values.repeat(n_genes),
                                   "ensembl": np.tile(np.array(mean_glyco.var_names),n_combos)
                                   })


        summary_df["non_zero_cell_counts"] = summary_df.apply(lambda row: non_zero_counts_combos_dict[row["combos_ontology"]][row["ensembl"]], axis=1)


        summary_df["tissue_label"] = summary_df["tissue_ontology"].map(uberon_map_dict)
        summary_df["cell_type_label"] = summary_df["cell_ontology"].map(cell_ontology_map_dict)
        summary_df["hgnc"] = summary_df["ensembl"].map(glyco_dict1)
        summary_df["glycopathway"] = summary_df["hgnc"].map(glyco_pathway_dict)
        #summary_df.dropna(axis=0)

        summary_df.to_csv(summary_path,sep="\t",index=False)


    else:
        print("Loading summary df")
        summary_df = pd.read_csv(summary_path,sep="\t")

        mean_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_mean.h5ad"
        mean_full = read_adata(mean_path)

        std_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_std.h5ad"
        std_full = read_adata(std_path)

        mean_glyco_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_mean_glyco.h5ad"
        mean_glyco = read_adata(mean_glyco_path)

        std_glyco_path = f"/{script_dir}/data/coarsed{folder_suffix}/merged/cellxgene_coarsed_std_glyco.h5ad"
        std_glyco = read_adata(std_glyco_path)



    results = namedtuple('Summary', ['summary_df', 'mean_full','mean_glyco','std_full','std_glyco'])(summary_df,mean_full,mean_glyco,std_full,std_glyco)
    return results

def cell_counts_vs_mean_std_plots(summary_df,figpath):
    """"""
    gene_names = summary_df["hgnc"].unique()
    n_genes = len(gene_names)
    nrows, ncols, nrows_idx, ncols_idx, width_ratios = NF.calculate_dimensions_plot(gene_names,max_rows=6, include_colorbars=False)
    fig, axs = plt.subplots(nrows, ncols, figsize=(30, 15), layout="constrained")
    axs = axs.ravel()
    xmax = summary_df["num_cells"].max()
    ymax = summary_df["mean"].max()
    left = 0
    for iterables in zip(axs,gene_names):
    #def plot_parallel(iterables,summary_df):

        ax,gene_name = iterables
        print(gene_name)
        gene_extract = summary_df[summary_df["hgnc"] == gene_name]
        means = gene_extract["mean"].values
        stds = gene_extract["std"].values
        num_cells = gene_extract["num_cells"].values
        ax.plot(num_cells,means)
        #ax.fill_between(np.arange(len(means)), means - stds, means + stds)
        ax.set_title(gene_name)
        ax.tick_params(left=False, bottom=False)  # Remove ticks
        ax.tick_params(labelleft=False, labelbottom=False)  # Remove tick labels
        ax.set_xlim(0,xmax)
        ax.set_ylim(0,ymax)
        left +=1

    for i in range(left,len(axs)):#remove leftover axes
        axs[i].tick_params(left=False, bottom=False)
        axs[i].tick_params(labelleft=False, labelbottom=False)
        axs[i].axis("off")

    #results = Parallel(n_jobs=-2)(delayed(functools.partial(plot_parallel,summary_df=summary_df))((ax,gene_name)) for ax, gene_name in zip(axs,gene_names)) #n_jobs=-2, all cpus minus one


    #plt.xlabel("Mean", fontsize=20)
    #plt.ylabel("Cell counts", fontsize=20)


    plt.show()

def correlation_matrix(adata,figpath):
    #TODO. put als hgnc and so on in the anndata


    sc.pp.pca(adata,n_comps=2)
    sc.tl.dendrogram(adata)
    print(adata)
    ax = sc.pl.correlation_matrix(adata, groupby="combo",figsize=(5, 3.5))

    plt.show()

    exit()

def gene_expression_comparison(adata1path,adata2path,outdir):
    """"""

    glyco_file = f"{script_dir}/data/coarsed_glyco/analysis/glyco_df.tsv"
    glyco_df = glyco_map(glyco_file, save=False)
    glyco_dict = glyco_df.set_index('ensembl_gene_id').to_dict()['hgnc_symbol']

    adata1name = os.path.basename(adata1path)
    adata2name = os.path.basename(adata2path)
    adata1 = sc.read_h5ad(adata1path)

    adata1 = adata1[~adata1.obs_names.isin(["mean","std"])]
    adata2 = sc.read_h5ad(adata2path)
    adata2 = adata2[~adata2.obs_names.isin(["mean", "std"])]
    adata = sc.concat([adata1,adata2],join="outer")
    var_names_hgnc = [glyco_dict[name] for name in adata.var_names]
    adata.var_names = var_names_hgnc


    with rc_context({"figure.figsize": (15, 15)}):
        sc.tl.rank_genes_groups(adata, groupby='combo', method='wilcoxon',corr_method="benjamini-hochberg")
        dataframe_scores = sc.get.rank_genes_groups_df(adata, group=None)
        print(dataframe_scores)
        exit()
        sc.pl.rank_genes_groups(adata)

    #TODO: Somehow store a p-value or something

    plt.savefig(f"{outdir}/wilcoxon_{adata1name}_{adata2name}.jpg", dpi=600, bbox_inches="tight")
    plt.close()
    plt.clf()


def analysis_mean_coarsed_adata(plot_type=""):
    """
    NOTES:
        SC Analysis: https://biocellgen-public.svi.edu.au/mig_2019_scrnaseq-workshop/comparing-and-combining-scrna-seq-datasets.html
        Gene lineage expression: https://www.nature.com/articles/s41467-024-47158-y
        Gene expression differentiation: https://nbisweden.github.io/workshop-scRNAseq/labs/scanpy/scanpy_05_dge.html
        Ellipses: https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    Returns:

    """

    results = build_coarsed_summary_df(overwrite=False)


    summary_df = results.summary_df
    summary_df = summary_df[summary_df["total_num_cells"] > 250]
    summary_df["ratio_cells"] = np.log(summary_df["total_num_cells"])*10 #neperian logarithn
    #summary_df["ratio_cells"] = summary_df["num_cells"]

    figpath=f"{script_dir}/notebooks/figures/cellxgene10x"

    if plot_type == "cell_counts_vs_mean_std_plots":
        cell_counts_vs_mean_std_plots(summary_df,figpath)
    elif plot_type == "correlations":
        correlations_plots(summary_df, figpath)
    elif plot_type == "umap":
        umap_plot(results.mean_full,"",figpath)
    elif plot_type == "correlation_matrix":
        correlation_matrix(results.mean_glyco,figpath)
    elif plot_type == "histogram":
        histogram_plot(results.summary_df,figpath)

    #TODO: gene expression /correlations



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="laedingr args",formatter_class=RawTextHelpFormatter)
    #
    # analyze_census_cellxgene_metadata(build=True, plot=False)
    #analysis_mean_coarsed_adata(plot_type="correlation_matrix")
    analysis_mean_coarsed_adata(plot_type="umap")
    exit()

    adata1path = "/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed_glyco/UBERON:0000002_CL:0008007.h5ad"
    adata2path = "/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed_glyco/UBERON:0000004_CL:0000000.h5ad"
    gene_expression_comparison(adata1path,adata2path,"/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/notebooks/figures/cellxgene10x/differential_expression")