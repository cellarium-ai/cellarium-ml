import re
import os
import scanpy as sc
import sys
import torch
import subprocess
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
import notebooks_functions as NF



#TODO: https://support.parsebiosciences.com/hc/en-us/articles/7704577188500-How-to-analyze-a-1-million-cell-data-set-using-Scanpy-and-Harmony


"""Glyco genes per dataset
{"pmbc_results": ['B4GALT5', 'CHPF', 'B3GALT4', 'UGCG', 'CSGALNACT2', 'POFUT2', 'MGAT4A', 'GALNT3', 'GALNT10', 'LFNG', 'EXTL3', 'GALNT12', 'GXYLT1', 'TMTC2', 'CHSY1', 'XYLT2', 'GALNT1', 'GCNT4', 'CSGALNACT1', 'GALNT2', 'MGAT1', 'C1GALT1', 'OGT'],
"tucker_human_heart_atlas": ['MGAT2', 'EXTL1', 'ALG10B', 'RFNG', 'B3GNT6', 'UGGT2', 'STT3A'],
"human_heart_atlas": ['MGAT2', 'EXTL1', 'ALG10B', 'RFNG', 'B3GNT6', 'UGGT2', 'STT3A'],

"""

"""
pbmc_count:  31774 × 4000
tucker_human_heart_atlas: 276940 × 2889 #only 1 batch type
human_heart_atlas: 665234 × 2889
gut_cell_atlas_normed: 428469 × 30535
cellxgene_liver: 481828 × 60664
human_pancreas_rna: 14700 × 34363
Kidney_muto_rna: 19985 × 33234
mouse_Liver_rna: 9714 × 16706
cellxgene_lung: 584944 × 27957

"""

#TODO: Other datasets:
# https://collections.cellatlas.io/liver-development
# Lung: https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0


foldername_dict = {
                    0: ["pmbc_results",'lightning_logs/version_58/checkpoints/epoch=49-step=3150.ckpt',"../example_configs/scvi_config_pbmc.yaml","../data/pbmc_count.h5ad",['final_annotation', 'batch'],""],
                    1: ["pmbc_results_train",'lightning_logs/version_70/checkpoints/epoch=49-step=2500.ckpt',"../example_configs/scvi_config_pbmc_train.yaml","../data/pbmc_count_train.h5ad",['final_annotation', 'batch'],""],
                    2: ["pmbc_results_test",'',"../example_configs/scvi_config_pbmc_test.yaml","../data/pbmc_count_test.h5ad",['final_annotation', 'batch'],""],
                    3: ["pmbc_results_subset_genes",'',"../example_configs/scvi_config_pbmc_subset_genes.yaml","../data/pbmc_count.h5ad",['final_annotation', 'batch'],""],
                    4 : ["tucker_human_heart_atlas","lightning_logs/version_52/checkpoints/epoch=29-step=16230.ckpt","../example_configs/scvi_config_tucker_heart_atlas.yaml","../data/tucker_human_heart_atlas.h5ad",["Cluster","batch"],"gene_names"],
                    5 : ["human_heart_atlas","lightning_logs/version_53/checkpoints/epoch=39-step=26640.ckpt","../example_configs/scvi_config_human_heart_atlas.yaml","../data/human_heart_atlas.h5ad",["cell_type","batch"],"gene_name-new"], #10 GB
                    6 : ["gut_cell_atlas_raw","lightning_logs/version_51/checkpoints/epoch=39-step=17160.ckpt","../example_configs/scvi_config_gut_cell_atlas_raw.yaml","../data/gut_cell_atlas_raw.h5ad",["category"],""],
                    7 : ["cellxgene_liver","lightning_logs/version_57/checkpoints/epoch=49-step=47100.ckpt","../example_configs/scvi_config_cellxgene_liver.yaml","../data/cellxgene_liver.h5ad",["cell_type"],"feature_name"],
                    8: ["cellxgene_lung","lightning_logs/version_66/checkpoints/epoch=49-step=57150.ckpt","../example_configs/scvi_config_cellxgene_lung.yaml","../data/single_cell_lung_atlas.h5ad",["cell_type","study"],"feature_name"], #too big
                    9: ["cellxgene_lung_subset","lightning_logs/version_67/checkpoints/epoch=49-step=34200.ckpt","../example_configs/scvi_config_cellxgene_lung_subset.yaml","../data/single_cell_lung_atlas_subset.h5ad",["cell_type","study"],"feature_name"],#fails because it has floats, not integers
                    10: ["human_pancreas_rna","","../example_configs/scvi_config_human_pancreas_rna.yaml","../data/human_pancreas_rna.h5ad",["cell_type","study"],"var.index"], #no raw data available?
                    11: ["Kidney_muto_rna","lightning_logs/version_71/checkpoints/epoch=49-step=2000.ckpt","../example_configs/scvi_config_Kidney_muto_rna.yaml","../data/Kidney_muto_rna.h5ad",["cell_type_category","author_cell_type"],"feature_name"],
                    12: ["mouse_Liver_rna","","../example_configs/scvi_config_mouse_Liver_rna.yaml","../data/mouse_Liver_rna.h5ad",["cell_type","study"],"var.index"], #no raw data available?
                    13: ["extract0"]
                    }

#TODO: Warning: perhaps n_cats_per_cov is a problem or the categorical_covariate_index
foldername,checkpoint_file,config_file, adata_file,color_keys,gene_names = foldername_dict[0]
use_test = False
if use_test:
    foldername_test,checkpoint_file_test,config_file_test, adata_file_test,color_keys_test,gene_names_test = foldername_dict[2]
    color_keys += ["subset"]

# adata = sc.read(adata_file)
# NF.divide_train_test(adata,adata_file)

#NF.subset_adata(adata,350000,"../data/single_cell_lung_atlas_subset.h5ad")

#NF.scanpy_scvi(adata_file) #too slow to handle
subprocess.call([f"{sys.executable}","../cellarium/ml/cli.py","scvi","fit","-c",config_file],env=env) #/opt/conda/bin/python

exit()



NF.folders(foldername,"figures",overwrite=False)
NF.folders(foldername,"tmp_data",overwrite=False)


pipeline, device = NF.setup_model(checkpoint_file) #scvi_model

filename_suffix = ""
figpath = f"figures/{foldername}"
datapath = f"tmp_data/{foldername}"
overwrite = False
use_cuda = False




filename = f"adata_processed{filename_suffix}"
matched,filepath = NF.matching_file(datapath,filename)

if use_test:
    filename_test = filename + "_test"
    filename_train = filename + "_train"
    matched_train, filepath_train = NF.matching_file(datapath, filename_train)
    matched_test, filepath_test = NF.matching_file(datapath, filename_test)


if use_test:
    adata_train = NF.download_predict(config_file, gene_names, filepath_train, pipeline, device, matched_train, filename_train, overwrite)
    adata_test = NF.download_predict(config_file_test, gene_names_test, filepath_test, pipeline, device,matched_test,filename_test,overwrite)
    adata = sc.concat([adata_train,adata_test])
    filepath = filepath + ".h5ad" if not filepath.endswith(".h5ad") else filepath
    adata.write(filepath)

else:
    adata = NF.download_predict(config_file, gene_names, filepath, pipeline, device, matched, filename, overwrite)

# adata = adata[:1000]

# if "X_raw_umap" not in list(adata.obsm.keys()) or overwrite:
#     print("Key 'X_raw_umap' not found or overwrite is True, computing")
#     if use_cuda:
#         adata = NF.plot_umap_cuda(adata,filepath,figpath,"raw_umap","raw_umap",None,color_keys)
#     else:
#         adata = NF.plot_umap(adata, filepath, figpath, "raw_umap", "raw_umap", None, color_keys)
#
#
# else:
#     print("Key 'X_raw_umap' found, continue")

NF.ComputeUMAP(adata, filepath, figpath, "raw_umap", "raw_umap", None, use_cuda, overwrite,color_keys).run()

NF.ComputeUMAP(adata, filepath, figpath, "scvi_latent_umap", "scvi_umap", "X_scvi" ,use_cuda, overwrite,color_keys).run()

# if "X_scvi_umap" not in list(adata.obsm.keys()) or overwrite:
#     print("Key 'X_scvi_umap' not found  or overwrite is True, computing")
#     if use_cuda:
#         adata = NF.plot_umap_cuda(adata, filepath, figpath, "scvi_latent_umap", "scvi_umap", "X_scvi", color_keys)
#     else:
#         adata = NF.plot_umap(adata,filepath, figpath,"scvi_latent_umap","scvi_umap","X_scvi",color_keys)
# else:
#     print("Key 'X_scvi_umap' found, continue")


gene_set_dict,gene_set = NF.retrieve_genes()

#adata = adata[:1000]


adata = NF.define_gene_expressions(adata,gene_set,filepath,gene_names,overwrite)


adata_glyco = adata[:, adata.var_names.isin(gene_set)]

if "X_scvi_reconstructed_0_umap" not in list(adata.obsm.keys()) or overwrite: #TODO:Re-make for lung because we have overlooked the error
    print("Key 'X_scvi_reconstructed_0_umap' not found, computing")
    if use_cuda:
        adata = NF.umap_group_genes_cuda(adata, filepath)
    else:
        adata = NF.umap_group_genes(adata,filepath)
else:
    print("Key 'X_scvi_reconstructed_0_umap' found, continue")

#Highlight: Density differential expression

# figname = "expression_density_difference"
# NF.differential_gene_expression(adata,gene_set,figpath,figname)
# #
# figname = "glyco_expression_RAW"
# # # # figname = "cell_cycle_expression_RAW"
# NF.plot_avg_expression(adata,"X_raw_umap",gene_set_dict,figpath,figname,color_keys)
#
# figname = "glyco_expression_RECONSTRUCTED"
# # # # figname = "cell_cycle_expression_RECONSTRUCTED"
# NF.plot_avg_expression(adata,"X_scvi_reconstructed_0_umap",gene_set_dict,figpath,figname,color_keys)

#Highlight: Leiden clustering

# if use_cuda:
#     NF.plot_neighbour_leiden_clusters_cuda(adata, gene_set_dict, figpath, color_keys, filepath, overwrite=True,plot_all=False)
# else:
#     NF.plot_neighbour_leiden_clusters(adata,gene_set_dict,figpath,color_keys,filepath,overwrite=True,plot_all=False)
#


#Highlight: Glyco genes KDE expression

# if adata_glyco.X.size != 0:
#     # figname = f"adata_subset_scvi_reconstructed_KDE{filename_suffix}"
#     # NF.plot_density_expression(adata_glyco,present_gene_set,"scvi_reconstructed_0",figpath,figname)
#
#     figname = f"adata_subset_scvi_raw_KDE{filename_suffix}"
#     NF.plot_density_expression(adata_glyco,present_gene_set,"raw",figpath,figname)
#
#     figname = f"raw_RANK_dataframe{filename_suffix}"
#     NF.plot_rank_expression(adata,present_gene_set,"raw",figpath,figname)
#
#     figname = f"scvi_reconstructed_RANK_dataframe{filename_suffix}"
#     NF.plot_rank_expression(adata,present_gene_set,"scvi_reconstructed_0",figpath,figname)



#Highlight: Glyco genes Violin expression

# figname = f"adata_subset_scvi_reconstructed_VIOLIN{filename_suffix}"
# NF.plot_violin_expression_distribution(adata_glyco,present_gene_set,figpath,figname,"scvi_reconstructed_0") #TODO: delete?


dict_subsets = {
                "glyco":[f"adata_subset_processed_glyco{filename_suffix}",adata.var_names.isin(gene_set)],
                "top20genes":[f"adata_subset_processed_top20genes{filename_suffix}",adata.var['top20_high_exp_genes']],
                "top200genes":[f"adata_subset_processed_top200genes{filename_suffix}",adata.var['high_exp_genes']]
                }

gene_group_names=["glyco","top20genes","top200genes"] #this here for debugging, otherwise just use .keys()


overwrite=True
if "gene_subset_glyco" not in list(adata.var.keys()) or overwrite:
    for gene_group_name in gene_group_names:
        filename_subset,genes_slice = dict_subsets[gene_group_name]
        print("NMF analysis not found for glyco genes, computing")
        filepath_subset = os.path.join(datapath, f"{filename_subset}.h5ad")
        adata_subset = adata[:, genes_slice]
        present_gene_set = adata_subset.var_names.tolist()
        if use_cuda:
            adata, adata_subset = NF.analysis_nmf_cuda(adata, present_gene_set, filepath_subset, filepath,gene_group_name,genes_slice)
        else:
            adata,adata_subset = NF.analysis_nmf(adata,present_gene_set,filepath_subset,filepath,gene_group_name,genes_slice)


for gene_group_name in gene_group_names:
    filename_subset,slice = dict_subsets[gene_group_name]
    filepath_subset = os.path.join(datapath, f"{filename_subset}.h5ad")
    print("Reading and plotting file : {}".format(filepath_subset))
    adata_subset = sc.read(filepath_subset)

    NF.plot_nmf(adata_subset,color_keys,figpath,gene_group_name)

