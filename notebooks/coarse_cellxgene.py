import functools
import gc
import itertools
import multiprocessing
import os
import sys
import time
from anndata import AnnData
from scipy import sparse
import anndata
import h5py
import anndata as ad
import scanpy as sc
from collections import Counter, defaultdict, namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
from argparse import RawTextHelpFormatter
os.path.realpath("__file__")
fx_dir =os.path.dirname(os.path.abspath("__file__"))
script_dir = os.path.dirname(fx_dir)
sys.path.extend(["{}".format(fx_dir)])
from notebooks_functions import build_coarsened_metadata, read_h5ad_gcs,search_ontology_rdflib,plot_histogram
from joblib import Parallel, delayed


def retrieve_ontology():
    """"""
    #Highlight : Alt + L for single line execution
    owl_file = "{}/cellarium-ml/data/pseudobulk/uberon.owl".format(fx_dir)
    # owl_path = "http://purl.obolibrary.org/obo/uberon/releases/2024-09-03/uberon.owl"
    # term_id = "UBERON:0000955"
    # term_id = "UBERON_0002107"
    # search_ontology_rdflib(owl_path,term_id)

def analyze_cellxgene_metadata(build=False,plot=False):
    #Highlight: Read metadata anc coarse according to ontologies
    if build:
        import cellxgene_census #expensive operation
        coarsened_metadata = build_coarsened_metadata(cellxgene_census,script_dir,method="ku",overwrite=False)
        coarsened_metadata = coarsened_metadata.dropna( subset=["cell_type_coarse_ontology_term_id", "tissue_coarse_ontology_id"], how="any")

        coarsened_metadata["combinations_all"] = coarsened_metadata["cell_type_coarse_ontology_term_id"].astype(str) + "," + coarsened_metadata["tissue_coarse_ontology_id"].astype(str) + "," + coarsened_metadata["sex"].astype(str) + "," + coarsened_metadata["disease_ontology_term_id"].astype(str)
        coarsened_metadata["combinations_cell_tissue"] = coarsened_metadata["cell_type_coarse_ontology_term_id"].astype(str) + "," + coarsened_metadata["tissue_coarse_ontology_id"].astype(str)
        coarsened_metadata["combinations_cell_tissue_disease"] = coarsened_metadata["cell_type_coarse_ontology_term_id"].astype(str) + "," + coarsened_metadata["tissue_coarse_ontology_id"].astype(str) + "," + coarsened_metadata["disease_ontology_term_id"].astype(str)

        # unique_cell_onts = coarsened_metadata["cell_type_coarse_ontology_term_id"].drop_duplicates()
        # unique_cell_onts.to_csv("Cell_ontologies_coarsed.tsv",sep="\t",index=False)
        #
        # unique_cell_onts = coarsened_metadata["cell_type_ontology_term_id"].drop_duplicates()
        # unique_cell_onts.to_csv("Cell_ontologies.tsv",sep="\t",index=False)

        if plot :
            plot_histogram(coarsened_metadata,"combinations_cell_tissue","Barplot",250)
            plot_histogram(coarsened_metadata,"tissue_coarse_ontology_id","Barplot",1000)
            plot_histogram(coarsened_metadata,"cell_type_coarse_ontology_term_id","Barplot",1000)

def mapping_ensembl_hgcn(adata):
    ensembl_hgcn_path = f"{script_dir}/data/pseudobulk/ensembl_hgcn_ids.tsv"
    if not os.path.exists(ensembl_hgcn_path):
        ensembl_list_xlsx = pd.DataFrame({"ensembl": adata.var_names.tolist()})
        ensembl_path = f"{script_dir}/data/pseudobulk/ensembl_ids.xlsx"
        ensembl_list_xlsx.to_excel(ensembl_path)
        command = 'Rscript'
        script_path = os.path.abspath(os.path.dirname(__file__))
        path2script = "{}/conversion.R".format(script_path)
        subprocess.check_call([command, path2script, ensembl_path, ensembl_hgcn_path])

    else:
        print("Loading BioMart results")

    ensembl_hgcn_mapping = pd.read_csv(ensembl_hgcn_path,sep="\t")

    return ensembl_hgcn_mapping

#Highlight: Divide the extract and append to the correct dataframe
def convert_columns_to_dtype(df,dtype):
    for col in df.columns:
        df[col] =df[col].astype(dtype)
    return df

def fix_and_average_adata(adata_path,save=True,calculate_mean=True):
    """Convert a h5py file to scanpy anndata h5ad format.
    Notes:
        - https://chanzuckerberg.github.io/cellxgene-census/notebooks/analysis_demo/comp_bio_explore_and_load_lung_data.html
    """
    #Highlight: Dirty solution with try and except
    try:
        adata = sc.read_h5ad(adata_path)
        adata.obs_names_make_unique()
        print("Found complete adata")
        print(adata)
        if "mean" not in adata.obs_names and calculate_mean:
            print("Mean not found, adding it it")
            adata_coarsed = AnnData(
                X=np.concatenate([np.mean(adata.X, axis=0)[None, :], np.std(adata.X, axis=0)[None, :]], axis=0),
                var=adata.var
            )
            adata_coarsed.var_names = adata.var.index.astype(str)
            adata_coarsed.obs_names = ["mean", "std"]
            adata = sc.concat([adata,adata_coarsed],join="outer")
            sc.write(adata_path, adata)

        adata.file.close()
        del adata
        gc.collect()
        #return adata
    except:
        print(" Found adata that needs a fix...")
        # we need to read the file twice and merge conveniently into the h5ad format
        print(adata_path)
        print("................")
        adata = sc.read_hdf(adata_path, key="X")  # cannot read the h5py format properly
        adata_hdf5 = h5py.File(adata_path, "r+") # has different format from the anndata

        obs_names = np.array(adata_hdf5.get("obs_names")).astype(str)
        var_names = np.array(adata_hdf5.get("var_names")).astype(str)
        obs_col_names = np.array(adata_hdf5.get("obs_col_names")).astype(str)

        var_array = np.array(adata_hdf5.get("var")).astype(str)

        var_df = pd.DataFrame(var_array, index=var_names)
        var_df = convert_columns_to_dtype(var_df,str)

        obs_array = np.array(adata_hdf5.get("obs")).astype(str)
        obs_df = pd.DataFrame(obs_array,
                              columns=obs_col_names,
                              index=obs_names)
        obs_df = convert_columns_to_dtype(obs_df,str)

        adata.obs = obs_df
        adata.var = var_df
        adata.obs_names = adata.obs.index.astype(str)
        adata.var_names = adata.var.index.astype(str)
        adata.obs.index = adata.obs.index.astype(str)
        adata.var.columns = adata.var.columns.astype(str)



        if "mean" not in adata.obs_names and calculate_mean:
            print("Mean not found, adding it it")
            adata_coarsed = AnnData(
                X=np.concatenate([np.mean(adata.X, axis=0)[None, :], np.std(adata.X, axis=0)[None, :]], axis=0),
                var=adata.var
            )
            adata_coarsed.var_names = adata.var.index.astype(str)
            adata_coarsed.obs_names = ["mean", "std"]
            adata = sc.concat([adata,adata_coarsed],join="outer")

        #Highlight: Remove duplicated cells
        adata.obs_names_make_unique()
        adata.file.close()
        adata_hdf5.file.close()

        #Highlight: Overwrite
        #os.remove(adata_path) #not necessary when using the scanpy write function, I think
        if save:
            sc.write(adata_path,adata)
        del adata
        gc.collect()
        #return adata

def merge_data_helper(root,files,folder_path):
    for file_name in files:  # there might be many extracts from different places, problem remains on how to append
        adata = fix_and_average_adata("{}/{}".format(folder_path, file_name),save=False)  # fix_adata and append probably can be merged, worth the time?
        append_to_adata("{}/{}".format(folder_path, root), adata) #this part cannot be parallelized
        #os.remove("{}/{}".format(folder_path, file_name))

def merge_data(folder_path:str,output_folder:str):
    """Merges the files with the same UBERON-CELLONTOLOGY COMBO originated from different extracts

    NOTES: concatenating h5py files: https://stackoverflow.com/questions/18492273/combining-hdf5-files
    """

    # groups_dict = defaultdict(lambda:[])
    # for file_name in os.listdir(folder_path):
    #     root_name = file_name.split("_e")[0]
    #     groups_dict[root_name].append(file_name)

    # for root, files in groups_dict.items(): #This can be parallelized
    #       for file_name in files: #there might be many extracts from different places, problem remains on how to append
    #           adata = fix_adata("{}/{}".format(folder_path,file_name),save=False) #fix_adata and append probably can be merged, worth the time?
    #           append_to_adata(root,adata)
    #     #     #os.remove(file_name)
    # exit()
    #Find groups of files with
    #results = Parallel(n_jobs=-2)(delayed(functools.partial(merge_data_helper,folder_path=folder_path))(root,files) for root, files in groups_dict.items())


    for file in os.listdir(folder_path):
        if file.endswith(".h5ad"):
            adata = sc.read("{}/{}".format(folder_path,file))
            file_name = file.replace("h5ad","")
            adata_mean = adata[adata.obs_names == "mean"]
            adata_std = adata[adata.obs_names == "std"]
            adata_mean.obs["combo_ontology"] = file_name
            append_to_adata("{}/cellxgene_coarsed_mean.h5ad".format(output_folder),adata_mean)
            append_to_adata("{}/cellxgene_coarsed_std.h5ad".format(output_folder),adata_std)

def append_to_adata(storage_ad_path:str,input_ad:anndata.AnnData,overwrite:bool=False):
    """

    NOTES: https://nyu-cds.github.io/python-mpi/05-collectives/
    Args:
        storage_ad_path:
        input_ad:
        max_genes:
        overwrite:

    Returns:

    """

    print("Starting to append to anndata file bucket")


    gene_names = input_ad.var_names.tolist()
    dataset_ids = input_ad.obs["dataset_id"].tolist()

    if isinstance(input_ad.X,np.ndarray):
        X_df = pd.DataFrame(input_ad.X,
                            index=dataset_ids,
                            columns=gene_names)
    else:

        X_df = pd.DataFrame(input_ad.X.todense(),
                            index=dataset_ids,
                            columns=gene_names)

    ###############################
    var_df = input_ad.var
    var_df = convert_columns_to_dtype(var_df,str)
    var_names = input_ad.var_names

    var_array = np.array(var_df.values, dtype=h5py.special_dtype(vlen=str))
    obs_df = input_ad.obs
    obs_names = input_ad.obs_names
    obs_col_names = input_ad.obs.columns.tolist()
    obs_array = np.array(obs_df.values.astype(str), dtype=h5py.special_dtype(vlen=str))

    # Highlight: Remove duplicated genes from the expression matrix by averaging their expression values
    X_df = X_df.T.groupby(by=X_df.columns).mean().T #may not be necessary this time
    if overwrite:
        os.remove(storage_ad_path)
    if not os.path.exists(storage_ad_path):
        print("File does not exists, creating file")
        storage_ad = h5py.File(storage_ad_path, mode="a")
        storage_ad.create_dataset('X',
                                  data=X_df.to_numpy(),
                                  compression="gzip",
                                  chunks=True,
                                  maxshape=(None,None))  # if there is a max fail might because it has download extra genes, set to None for no limits


        storage_ad.create_dataset('var',
                                  data=var_array,
                                  compression="gzip",
                                  chunks=True,
                                  maxshape=(None, None),
                                  dtype=h5py.special_dtype(vlen=str)
                                  )

        #print("var_names: {}".format(len(var_names)))



        storage_ad.create_dataset('var_names',
                                  data=var_names,
                                  compression="gzip",
                                  chunks=True,
                                  maxshape = (None,),
                                  dtype=h5py.special_dtype(vlen=str)
                                  )



        #print("obs: {}".format(obs_array.shape))
        storage_ad.create_dataset('obs',
                                  data=obs_array,
                                  compression="gzip",
                                  chunks=True,
                                  maxshape=(None, None),
                                  dtype=h5py.special_dtype(vlen=str)
                                  )


        storage_ad.create_dataset('obs_names',
                                  data=obs_names,
                                  compression="gzip",
                                  chunks=True,
                                  maxshape=(None,),
                                  dtype=h5py.special_dtype(vlen=str)
                                  )


        storage_ad.create_dataset('row_names',
                                           data=dataset_ids,
                                           compression="gzip",
                                           chunks=True,
                                           maxshape=(None,))

        #print("obs_cols_names : {}".format(len(obs_col_names)))

        storage_ad.create_dataset('obs_col_names', # cell_type, tissue, ...
                                           data=obs_col_names,
                                           compression="gzip",
                                           chunks=True,
                                           maxshape=(None,))





    else:
        print("File existing file-bucket and appending to it")
        storage_ad = h5py.File(storage_ad_path, mode="a")

        prev_size = storage_ad['X'].shape[0]
        n_x = X_df.shape[0] #added number of cells
        n_y = X_df.shape[1] #added number of genes

        # print("-----------------")
        # print(storage_ad['X'].shape[0])
        # print(storage_ad['X'].shape[1])
        # print(n_x)
        # print(n_y)
        # print("-----------------")

        #storage_ad['X'].resize((storage_ad['X'].shape[0] + n_x), axis=0)
        storage_ad['X'].resize(prev_size + n_x, axis=0)
        storage_ad['X'][-n_x:] = X_df.to_numpy()



        storage_ad['obs_names'].resize(prev_size + n_x,axis=0)
        storage_ad['obs_names'][-n_x:] = X_df.index.astype(str).tolist()


        storage_ad['row_names'].resize(prev_size + n_x, axis=0)
        storage_ad['row_names'][-n_x:] = dataset_ids


        # n_var = var_array.shape[0]
        # storage_ad['var'].resize((storage_ad['var'].shape[0] + n_var), axis=0)
        # storage_ad['var'][-n_var:] = var_array

        n_obs = obs_array.shape[0]
        storage_ad['obs'].resize(storage_ad['obs'].shape[0] + n_x, axis=0)
        storage_ad['obs'][-n_obs:] = obs_array



    storage_ad.close()

def coarse_adata_helper(combo:str,adata:anndata.AnnData,suffix:str,args:namedtuple):
    print(combo)
    ad = adata[adata.obs["combo"] == combo]
    storage_ad_path = f"{args.storage_dir}/{combo}{suffix}.h5ad"
    append_to_adata(storage_ad_path, ad, overwrite=False)  # do not fix until everything has been appended

def coarse_adata(adata_path:str,suffix:str,args:namedtuple):
    """Read in the anndata extract, select only the healthy cells. Assign tissue and cell ontologies. Create tissue-cell ontology combos, deliver them to the corresponding tissue-cell anndata files
    After calculate the mean and standard deviation for each tissue and cell type combos """
    adata = sc.read_h5ad(adata_path)
    #Highlight: keep only healthy data
    adata = adata[adata.obs["disease"] == "normal"]
    #Highlight: Mapping to coarsed ontology term
    tissue_coarsener = pd.read_csv(f"{script_dir}/data/pseudobulk/uberon_ontology_map.tsv", sep="\t", skiprows=1)
    tissue_coarsener_dict = dict(zip(tissue_coarsener.iloc[:, 0], tissue_coarsener.iloc[:, 1]))
    adata.obs['tissue_coarse_ontology_id'] = adata.obs['tissue_ontology_term_id'].map(tissue_coarsener_dict)

    cell_coarsener = pd.read_csv(f"{script_dir}/data/pseudobulk/cell_ontology_map.tsv", sep="\t", skiprows=1)
    cell_coarsener_dict = dict(zip(cell_coarsener.iloc[:, 0], cell_coarsener.iloc[:, 1]))

    adata.obs['cell_type_coarse_ontology_term_id'] = adata.obs['cell_type_ontology_term_id'].map(cell_coarsener_dict).replace(['^UBERON', '^BFO', '^PR'], np.nan, regex=True)

    #Highlight: Creating combo tissue-cell
    adata = adata[~adata.obs["tissue_coarse_ontology_id"].isin([np.nan,pd.NA])]
    adata = adata[~adata.obs["cell_type_coarse_ontology_term_id"].isin([np.nan,pd.NA])]
    adata.obs.loc[:,"combo"] =  adata.obs.loc[:,"tissue_coarse_ontology_id"].astype(str) + "_"+ adata.obs.loc[:,"cell_type_coarse_ontology_term_id"].astype(str)

    value_counts = adata.obs["combo"].value_counts()

    #Highlight: Create a file per combo tissue-cell
    #TODO: https://stackoverflow.com/questions/63188571/fastest-way-to-perform-multiprocessing-of-a-loop-in-a-function
    #TODO: https://medium.com/@patelharsh7458/efficient-python-multiprocessing-example-parallelizing-tasks-99b0a6b838d4

    #results = list(map(lambda combo: functools.partial(coarse_adata_helper,adata=adata)(combo), list(value_counts.keys())))
    results = Parallel(n_jobs=-2)(delayed(functools.partial(coarse_adata_helper,adata=adata,suffix=suffix,args=args))(combo) for combo in value_counts.keys()) #n_jobs=-2, all cpus minus one

    # for combo,counts in value_counts.items():
    #     ad = adata[adata.obs["combo"] == combo]
    #     storage_ad_path = f"{storage_dir}/{combo}.h5ad"
    #     append_to_adata(storage_ad_path,ad,overwrite=False) # do not fix until everything has been appended
    #     #result = fix_adata(storage_ad_path)
    #     #print(result)

def repair_adata(input_path:str,calculate_mean:bool=True):

    if os.path.isdir(input_path):
        # for file in os.listdir(input_path):
        #     if file.endswith(".h5ad"):
        #         fix_and_average_adata("{}/{}".format(input_path,file),save=True)
        # exit()
        results = Parallel(n_jobs=-2)(delayed(functools.partial(fix_and_average_adata,save=True,calculate_mean=calculate_mean))("{}/{}".format(input_path, file)) for file in os.listdir(input_path) if file.endswith(".h5ad"))
    else:
        fix_and_average_adata(input_path, save=True)



def read_cellxgene_toy(cellxgene_path):
    """
    Install MPI compilation support:
        https://stackoverflow.com/questions/77207467/how-to-build-h5py-with-mpi-support-against-parallel-hdf5-on-linux-2023
        https://docs.olcf.ornl.gov/software/python/parallel_h5py.html
        https://lento234.ch/blog/HPC/2021-04-01-h5py-parallel/

        https://www.hdfgroup.org/download/hdf5-1-14-5-zip/

    Get files to download: printf '%s\n' gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{1..50} > paths.txt
    #Highlight: Do not forget the . at the end
    gsutil -m cp gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{1..50} .
    Args:
        cellxgene_path:

    Returns:

    """

    # filename = "gs://cellarium-human-primary-data/curriculum/human_all_primary/extract_files/extract_0.h5ad"
    # new = "gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_0.h5ad"
    extract0 = f"{script_dir}/data/extracts/curriculum_human_all_primary_extract_files_extract_0.h5ad"
    extract1 = f"{script_dir}/data/extracts/curriculum_human_all_primary_extract_files_extract_1.h5ad"

    extract_dict = {0:["extract_0.h5ad",extract0],
                    1:["extract_1.h5ad",extract1]}
    #
    for i,file in enumerate([extract1,extract0]):
        suffix = "_{}".format(extract_dict[i][0].replace(".h5ad",""))
        #suffix=""
        coarse_adata(file,suffix)
    exit()
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "_{}".format(extract_dict[i][0].replace(".h5ad",""))) for i,file_name in enumerate([extract0,extract1]))  # n_jobs=-2, all cpus minus one
    results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "") for i,file_name in enumerate([extract0,extract1]))  # Highlight: Cannot be done
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name,"_{}".format(file_name.replace(".h5ad",""))) for file_name in [extract0,extract1])  # n_jobs=-2, all cpus minus one

    exit()

def read_cellxgene(args):
    """
    Install MPI compilation support:
        https://stackoverflow.com/questions/77207467/how-to-build-h5py-with-mpi-support-against-parallel-hdf5-on-linux-2023
        https://docs.olcf.ornl.gov/software/python/parallel_h5py.html
        https://lento234.ch/blog/HPC/2021-04-01-h5py-parallel/

        https://www.hdfgroup.org/download/hdf5-1-14-5-zip/

    Get files to download: printf '%s\n' gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{1..50} > paths.txt
    #Highlight: Do not forget the . at the end
    gsutil -m cp gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{1..50} .
    Args:
        cellxgene_path:

    Returns:

    """

    #
    for i,file in enumerate(os.listdir(args.cellxgene_path)):
        suffix=""
        file = "{}/{}".format(args.cellxgene_path,file)
        coarse_adata(file,suffix,args)
    exit()
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "_{}".format(extract_dict[i][0].replace(".h5ad",""))) for i,file_name in enumerate([extract0,extract1]))  # n_jobs=-2, all cpus minus one
    results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "") for i,file_name in enumerate([extract0,extract1]))  # Highlight: Cannot be done
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name,"_{}".format(file_name.replace(".h5ad",""))) for file_name in [extract0,extract1])  # n_jobs=-2, all cpus minus one


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="laedingr args",formatter_class=RawTextHelpFormatter)
    parser.add_argument('-cellxgene','--cellxgene_path', type=str, nargs='?',
                        default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/extracts",
                        help='Path to cellxgene .h5ad extracts')
    parser.add_argument('-storage','--storage_dir', type=str, nargs='?',
                        default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed",
                        help='Path where to output the coarsed files. Do not include the last "\"')

    parser.add_argument('-output','--output_dir', type=str, nargs='?',
                        default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed/merged",
                        help='Path where to output the coarsed AVERAGED files. Do not include the last "\"')
    parser.add_argument('-disease','--disease', type=str, nargs='?',
                        default="normal",
                        help='Disease type: \n'
                             'normal'
                             'other'
                             )
    args = parser.parse_args()

    #python coarse_cellxgene.py -cellxgene /opt/project/cellxgene/extracts -storage /opt/project/cellxgene/coarsed

    #1) Highlight: Read the ecellxgene extract and divide it by tissue-celltype combo,
    #read_cellxgene(args)
    #read_cellxgene_toy(args.cellxgene_path)
    #2) Highlight: Fix the h5py format to suit the anndata format properly and calculate the mean and standard deviation
    #repair_adata(args.storage_dir)
    #3) Highlight: Merge all uberon-cell ontology mean and std into 2 files (one for mean, one for std)
    merge_data(args.storage_dir,args.output_dir)
    repair_adata(args.output_dir,calculate_mean=False)
















