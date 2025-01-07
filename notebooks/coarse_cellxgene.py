import ast
import functools
import gc
import os
import sys
import pickle

import sparse
from anndata import AnnData
import anndata
import h5py
import scanpy as sc
from collections import Counter, defaultdict, namedtuple
import pandas as pd
import numpy as np
import subprocess
import argparse
from argparse import RawTextHelpFormatter
from scipy import sparse

os.path.realpath("__file__")
fx_dir =os.path.dirname(os.path.abspath("__file__"))
script_dir = os.path.dirname(fx_dir)

sys.path.extend(["{}".format(fx_dir)])

from joblib import Parallel, delayed


def mapping_ensembl_hgcn(adata=None):
    ensembl_hgcn_path = f"{script_dir}/data/pseudobulk/ensembl_hgcn_ids.tsv"
    if not os.path.exists(ensembl_hgcn_path) and adata is not None:
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

def calculate_average_adata(adata:anndata.AnnData,adata_path:str):

    if adata.obs_names.str.startswith(("mean", "std", "cell_counts")).sum() > 1:
        print("Overwritting mean, std and cell_counts")
        adata = adata[~adata.obs_names.isin(["mean", "std", "cell_counts"])]  # remove them
        adata = adata[~adata.obs_names.str.startswith(("mean", "std", "cell_counts"))]  # remove them deeper

    print("Calculating mean, std and cell counts")
    idx_zeros = adata.X.astype(bool)  # False = 0, True=not zero
    # adata_galnt15 = adata[:,adata.var_names.isin(["ENSG00000131386"])]
    # idx_zeros_galnt15 = adata_galnt15.X.astype(bool)
    # mean_galnt15 = np.ma.masked_array(adata_galnt15.X, mask=~idx_zeros_galnt15, fill_value=0.).mean(axis=0).data[None, :]
    # print(mean_galnt15)
    mean = np.ma.masked_array(adata.X, mask=~idx_zeros, fill_value=0.).mean(axis=0).data[None,:]  # True means mask and ignore, therefore we flip it
    # mean = np.mean(adata.X, axis=0)[None, :]
    std = np.ma.masked_array(adata.X, mask=~idx_zeros, fill_value=0.).std(axis=0).data[None, :]  # True means mask and ignore
    # std = np.std(adata.X, axis=0)[None, :]
    cell_counts = np.array([adata.X.shape[0]]).repeat(adata.X.shape[1])[None, :] #adata.X.shape[1] is the number of genes
    adata_coarsed = AnnData(
        X=np.concatenate([mean, std, cell_counts], axis=0),
        var=adata.var
    )

    adata_coarsed.var_names = adata.var.index.astype(str)
    adata_coarsed.obs_names = ["mean", "std", "cell_counts"]

    return adata, adata_coarsed

def fix_and_average_adata(adata_path,save=True,calculate_mean=False,overwrite_mean=False):
    """Convert a h5py file to a proper scanpy anndata h5ad format.
    Notes:
        - https://chanzuckerberg.github.io/cellxgene-census/notebooks/analysis_demo/comp_bio_explore_and_load_lung_data.html
    """
    ##Highlight: Dirty solution with try and except
    try:
        adata = sc.read_h5ad(adata_path)
        adata.obs_names_make_unique()
        print("Found complete adata")
        if overwrite_mean or ("mean" not in adata.obs_names and calculate_mean):
            adata, adata_coarsed = calculate_average_adata(adata,adata_path)
            adata = sc.concat([adata,adata_coarsed],join="outer") #WARNING: this step is making things (mean, std, cell counts) duplicated or so if not taken care carefully
            adata.file.close()
            os.remove(adata_path)
            sc.write(adata_path, adata)
        adata.file.close()
        del adata
        gc.collect()
        #return adata
    except:
        print(" Found adata that needs a fix...")
        # we need to read the file twice and merge conveniently into the h5ad format
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

        if overwrite_mean or ("mean" not in adata.obs_names and calculate_mean): #TODO: remove?
            print("Overwritiing mean")
            adata,adata_coarsed = calculate_average_adata(adata)
            adata = sc.concat([adata,adata_coarsed],join="outer")

        #Highlight: Remove duplicated cells
        adata.obs_names_make_unique()
        adata.file.close()
        adata_hdf5.file.close()

        #Highlight: Overwrite
        os.remove(adata_path) #not necessary when using the scanpy write function, I think
        if save:
            sc.write(adata_path,adata)
        del adata
        gc.collect()
        #return adata

def merge_helper(adata,file_path,output_folder):
    file_name = os.path.basename(file_path).replace(".h5ad", "")
    adata_mean = adata[adata.obs_names == "mean"]
    adata_std = adata[adata.obs_names == "std"]
    adata_cellcounts = adata[adata.obs_names == "cell_counts"].X[0][0]
    adata_expression = adata[~adata.obs_names.str.startswith(("mean","std","cell_counts"))]
    #adata_expression_galnt15 = adata_expression[:,adata.var_names.isin(["ENSG00000131386"])]

    idx_zeros = adata_expression.X.astype(bool).astype(int) #expression = True, non-expression = false

    cells_non_zeros_counts = idx_zeros.sum(axis=0)
    gene_names = adata.var_names

    adata_mean.obs["combo_ontology"] = file_name
    adata_std.obs["combo_ontology"] = file_name

    append_to_adata("{}/cellxgene_coarsed_mean.h5ad".format(output_folder), adata_mean)
    append_to_adata("{}/cellxgene_coarsed_std.h5ad".format(output_folder), adata_std)

    combos_cell_counter_file = "{}/counts_combos.tsv".format(output_folder) #TODO: delete, now I do this inside de .h5ad files, although it might be safer to do it separately
    combos_non_zero_cell_counter_file = "{}/non_zero_cell_counts_combos.tsv".format(output_folder)
    if os.path.exists(combos_cell_counter_file):
        print("Appending to existing cell counts file")
        counter_df = pd.read_csv(combos_cell_counter_file, sep="\t")
        new_row = pd.DataFrame({"combo": [file_name], "counts": [adata_cellcounts]})
        counter_df = pd.concat([counter_df, new_row], axis=0, ignore_index=True)
    else:
        print("File for cell counts not found creating it")
        counter_df = pd.DataFrame({"combo": [file_name], "counts": [adata_cellcounts]})

    #Highlight: store the non-zero counts per gene in each cell-tissue combo
    count_non_zero_genes_dict = dict(zip(gene_names, list(map(lambda x: [x], cells_non_zeros_counts))))
    total_counts_dict = {"combo": [file_name]}
    count_info_dict = {**total_counts_dict, **count_non_zero_genes_dict}
    if os.path.exists(combos_non_zero_cell_counter_file):
        non_zero_cell_counter_df = pd.read_csv(combos_non_zero_cell_counter_file, sep="\t")
        new_row2 = pd.DataFrame(count_info_dict)
        non_zero_cell_counter_df = pd.concat([non_zero_cell_counter_df, new_row2], axis=0, ignore_index=True)
    else:
        non_zero_cell_counter_df = pd.DataFrame(count_info_dict) #TODO: gene names dict + non zero cell counts

    counter_df.to_csv(combos_cell_counter_file, sep="\t", index=False)
    non_zero_cell_counter_df.to_csv(combos_non_zero_cell_counter_file,sep="\t",index=False)

def merge_averaged_data(input_path:str,output_folder:str):
    """Merges the files with the same UBERON-CELLONTOLOGY COMBO originated from different extracts into a mean.h5ad file

    NOTES: concatenating h5py files: https://stackoverflow.com/questions/18492273/combining-hdf5-files
    """

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith(".h5ad") and not file.startswith("cellxgene"):
                file_path = "{}/{}".format(input_path,file)
                adata = sc.read(file_path)
                merge_helper(adata,file_path,output_folder)
    else:
        if input_path.endswith(".h5ad") and not input_path.startswith("cellxgene"):
            adata = sc.read(input_path)
            merge_helper(adata,input_path,output_folder)

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

    if len(set(X_df.columns.tolist())) != len(X_df.columns): #seems to work
        print("Averaging duplicated genes")
        old_column_order = X_df.columns
        X_df = X_df.groupby(axis=1,level=0).mean() #level > 1 is for multiindex, this seems to rearrange alphabetically the genes
        X_df = X_df[old_column_order]

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

def coarse_adata_helper(combo:str,adata:anndata.AnnData,args:namedtuple):
    """Helper function to slice the cell-tissue combo and append it to its corresponding .h5py file"""
    print(combo)
    ad = adata[adata.obs["combo"] == combo]
    storage_ad_path = f"{args.output_dir}/{combo}.h5ad"
    append_to_adata(storage_ad_path, ad, overwrite=False)  # do not fix until everything has been appended

def coarse_adata(adata_path:str,args:namedtuple):
    """Read in the anndata extract, select only the healthy cells. Assign tissue and cell ontologies. Create tissue-cell ontology combos, deliver them to the corresponding tissue-cell anndata files
    After calculate the mean and standard deviation for each tissue and cell type combos

     NOTES:
             #TODO: https://stackoverflow.com/questions/63188571/fastest-way-to-perform-multiprocessing-of-a-loop-in-a-function
            #TODO: https://medium.com/@patelharsh7458/efficient-python-multiprocessing-example-parallelizing-tasks-99b0a6b838d4
     """
    adata = sc.read_h5ad(adata_path)
    #Highlight: keep only healthy data
    adata = adata[adata.obs["disease"] == args.disease_status]
    #Highlight: Mapping to coarsed ontology term
    tissue_coarsener = pd.read_csv(f"{script_dir}/data/common_files/uberon_ontology_map.tsv", sep="\t", skiprows=1)
    tissue_coarsener_dict = dict(zip(tissue_coarsener.iloc[:, 0], tissue_coarsener.iloc[:, 1]))
    adata.obs['tissue_coarse_ontology_id'] = adata.obs['tissue_ontology_term_id'].map(tissue_coarsener_dict)

    cell_coarsener = pd.read_csv(f"{script_dir}/data/common_files/cell_ontology_map.tsv", sep="\t", skiprows=1)
    cell_coarsener_dict = dict(zip(cell_coarsener.iloc[:, 0], cell_coarsener.iloc[:, 1]))

    adata.obs['cell_type_coarse_ontology_term_id'] = adata.obs['cell_type_ontology_term_id'].map(cell_coarsener_dict).replace(['^UBERON', '^BFO', '^PR'], np.nan, regex=True)

    #Highlight: Creating combo tissue-cell
    adata = adata[~adata.obs["tissue_coarse_ontology_id"].isin([np.nan,pd.NA])]
    adata = adata[~adata.obs["cell_type_coarse_ontology_term_id"].isin([np.nan,pd.NA])]
    adata.obs.loc[:,"combo"] =  adata.obs.loc[:,"tissue_coarse_ontology_id"].astype(str) + "_"+ adata.obs.loc[:,"cell_type_coarse_ontology_term_id"].astype(str)

    value_counts = adata.obs["combo"].value_counts()

    #Highlight: Create a file per combo tissue-cell
    #results = list(map(lambda combo: functools.partial(coarse_adata_helper,adata=adata)(combo), list(value_counts.keys())))
    results = Parallel(n_jobs=-10)(delayed(functools.partial(coarse_adata_helper,adata=adata,args=args))(combo) for combo in value_counts.keys()) #n_jobs=-2, all cpus minus one

    # for combo,counts in value_counts.items():
    #     ad = adata[adata.obs["combo"] == combo]
    #     storage_ad_path = f"{storage_dir}/{combo}.h5ad"
    #     append_to_adata(storage_ad_path,ad,overwrite=False) # do not fix until everything has been appended
    #     #result = fix_adata(storage_ad_path)
    #     #print(result)

def repair_adata(input_path:str,calculate_mean:bool=False,overwrite_mean:bool=False):


    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith(".h5ad"):
                print(file)
                fix_and_average_adata("{}/{}".format(input_path,file),save=True,calculate_mean=calculate_mean,overwrite_mean=overwrite_mean)

        exit()
        results = Parallel(n_jobs=-2)(delayed(functools.partial(fix_and_average_adata,save=True,calculate_mean=calculate_mean,overwrite_mean=overwrite_mean))("{}/{}".format(input_path, file)) for file in os.listdir(input_path) if file.endswith(".h5ad"))
    else:
        if input_path.endswith(".h5ad"):
            fix_and_average_adata(input_path, save=True,calculate_mean=calculate_mean,overwrite_mean=overwrite_mean)

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
    #Highlight: Cannot be parallelized
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "_{}".format(extract_dict[i][0].replace(".h5ad",""))) for i,file_name in enumerate([extract0,extract1]))  # n_jobs=-2, all cpus minus one
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "") for i,file_name in enumerate([extract0,extract1]))  # Highlight: Cannot be done
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name,"_{}".format(file_name.replace(".h5ad",""))) for file_name in [extract0,extract1])  # n_jobs=-2, all cpus minus one

def read_cellxgene(input_path):
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

    if os.path.isdir(input_path):
        for i,file in enumerate(os.listdir(input_path)):
            file = "{}/{}".format(input_path,file)
            coarse_adata(file,args)
    else:
        if input_path.endswith(".h5ad"):
            coarse_adata(input_path,args)

    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "_{}".format(extract_dict[i][0].replace(".h5ad",""))) for i,file_name in enumerate([extract0,extract1]))  # n_jobs=-2, all cpus minus one
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name, "") for i,file_name in enumerate([extract0,extract1]))  # Highlight: Cannot be done
    #results = Parallel(n_jobs=-2)(delayed(coarse_adata)(file_name,"_{}".format(file_name.replace(".h5ad",""))) for file_name in [extract0,extract1])  # n_jobs=-2, all cpus minus one

def return_obs_arr(adata):

    obs_arr = sc.read_h5ad(adata).obs
    obs_arr = obs_arr[~obs_arr.index.isin(["mean", "std"])]
    return obs_arr

def extract_metadata(input_path,output_path):
    """"""
    if os.path.isdir(input_path):#TODO: Does not work when doing it with all files
        print("Found folder")
        files = os.listdir(input_path)
        n = 20
        for k,i in enumerate(range(0, len(files), n)):
            batch = files[i:i + n]
            # results = Parallel(n_jobs=-2)(
            #     delayed(return_obs_arr)("{}/{}".format(input_path, file)) for file in batch if
            #     file.endswith(".h5ad"))
            results=[]
            for file in batch:
                if file.endswith(".h5ad"):
                    obs_arr = return_obs_arr("{}/{}".format(input_path, file))
                    results.append(obs_arr)
                    del obs_arr
            metadata = pd.concat(results, axis=0)
            metadata.to_csv("{}/metadata_{}.csv".format(output_path,k))
            del metadata,results
            gc.collect()

    else:
        filename = os.path.basename(input_path)
        print("Extracting metadata from file {}".format(filename))
        metadata = return_obs_arr(input_path)
        metadata.to_csv("{}/metadata_{}.csv".format(output_path, filename.replace(".h5ad","")))

def glyco_map(file,save=False):

    if not os.path.exists(file):
        print("Glyco file not found, creating it")
        genes_info = pd.read_excel("{}/data/common_files/Glycoenzymes.xlsx".format(script_dir))
        glycoproteins_hgnc_genes_list = sorted(genes_info["HGNC"].tolist())  # 379
        mapping_df = mapping_ensembl_hgcn()
        glyco_hgnc_df = mapping_df[mapping_df["hgnc_symbol"].isin(glycoproteins_hgnc_genes_list)]
        glyco_df = pd.DataFrame({"hgnc_symbol":glyco_hgnc_df["hgnc_symbol"].tolist(),"ensembl_gene_id":glyco_hgnc_df["ensembl_gene_id"].tolist()})
        glyco_df = glyco_df.drop_duplicates("ensembl_gene_id", inplace=False)  # remove duplicates
        if save:
            glyco_df.to_csv(file,index=False,sep="\t")
    else:
        print("Reading pre-existing glyco file")
        glyco_df = pd.read_csv(file,sep="\t")
    return glyco_df

def glyco_pathway_map(file,save=False,overwrite=False):

    if not os.path.exists(file) or overwrite:
        print("Glyco file not found or overwrite is True, creating it")

        glycogenes = pd.read_excel("{}/data/common_files/Glycoenzymes.xlsx".format(script_dir),sheet_name=8)

        #glycogenes = glycogenes.groupby(["pathway", "Initiator-enzyme"], as_index=False)[["HGNC"]].agg(lambda x: list(x))  # .groupby("hgnc_symbol",as_index=False) [["uniprot_combined"]].agg(lambda x: list(x))
        glycogenes = glycogenes.groupby(["Initiator-enzyme"], as_index=False)[["HGNC"]].agg(lambda x: list(x))  # .groupby("hgnc_symbol",as_index=False) [["uniprot_combined"]].agg(lambda x: list(x))
        initiator_enzymes = glycogenes["Initiator-enzyme"].str.lower().tolist()
        group_members = glycogenes["HGNC"].apply(lambda x: ast.literal_eval(str(x)))
        dict_glycopathways = dict(zip(initiator_enzymes, group_members))

        if save:
            pickle.dump(dict_glycopathways, open(file, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Reading pre-existing glyco file")
        dict_glycopathways =  pickle.load(open(file,"rb"))
    return dict_glycopathways

def grep_glyco(filename,input_path,glyco_df,output_dir):
    print("Working on {}".format(filename))
    adata = sc.read_h5ad("{}/{}".format(input_path,filename))
    glyco_genes = glyco_df["ensembl_gene_id"].tolist()
    adata = adata[:,adata.var_names.isin(glyco_df["ensembl_gene_id"].tolist())]
    adata = adata[:,glyco_genes] #reorder
    adata.write("{}/{}".format(output_dir,filename))

def extract_glyco(input_path,output_dir):

    glyco_info = "{}/glyco_df.tsv".format(output_dir)
    glyco_df = glyco_map(glyco_info,save=True)

    if os.path.isdir(input_path):
        # for file in os.listdir(input_path):
        #     if file.endswith(".h5ad"):
        #         grep_glyco(file,input_path,glyco_df, output_dir)
        #
        results = Parallel(n_jobs=-2)(
            delayed(functools.partial(grep_glyco,input_path = input_path,glyco_df=glyco_df,output_dir=output_dir))(file) for file in os.listdir(input_path) if
            file.endswith(".h5ad"))
    else:
        grep_glyco(os.path.basename(input_path), os.path.dirname(input_path), glyco_df, output_dir)

def cumsum_glyco(input_file, output_dir):
    """"""
    #Highlight: If it does not exist, create a dataframe where we will store the non-zero counts per glyco transferase
    glyco_counter_file = "{}/cumsum_counts_glyco.tsv".format(output_dir)
    glyco_counter_df = glyco_map(glyco_counter_file)

    #Highlight: Read the adata file
    adata = sc.read_h5ad(input_file)
    adata_glyco = adata[:,adata.var_names.isin(glyco_counter_df["ensembl_gene_id"])] #seems like we have all 263 glyco enzymes
    glyco_available = adata_glyco.var_names
    glyco_counter_df = glyco_counter_df[glyco_counter_df["ensembl_gene_id"].isin(glyco_available)] #subset only to available glyco enzymes
    adata_glyco = adata_glyco[:,glyco_counter_df["ensembl_gene_id"].tolist()] #rearrange column order
    assert (adata_glyco.var_names == glyco_counter_df["ensembl_gene_id"]).all(), "Do not contain the same glyco genes in the same order"
    nonzero_counts = adata_glyco.X.astype(bool).sum(axis=0)
    if "counts" not in glyco_counter_df.columns:
        glyco_counter_df["counts"] = nonzero_counts
    else:
        glyco_counter_df["counts"] += nonzero_counts

    glyco_counter_df.to_csv(glyco_counter_file,sep="\t",index=False)

    #Highlight: Cumsum counts

def count_combos_cells(input_file,output_dir):

    print("Counting the number of cells per tissue-cell combo")
    adata = h5py.File(input_file, "r+")
    cell_counts = len(adata["X"])
    combo_name =  os.path.basename(input_file)

    combos_cell_counter_file = "{}/counts_combos.tsv".format(output_dir)

    if os.path.exists(combos_cell_counter_file):
        print("Appending to existing file")
        counter_df = pd.read_csv(combos_cell_counter_file,sep="\t")
        new_row = pd.DataFrame({"combo": [combo_name], "counts": [cell_counts]})
        counter_df = pd.concat([counter_df, new_row], axis=0, ignore_index=True)
    else:
        print("File not found creating it")
        counter_df= pd.DataFrame({"combo":[combo_name],"counts":[cell_counts]})


    counter_df.to_csv(combos_cell_counter_file,sep="\t",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="laedingr args",formatter_class=RawTextHelpFormatter)
    parser.add_argument('-disease_status','--disease_status', type=str, nargs='?',
                        default="normal",
                        help=' Whether to keep -normal- or -diseased- cells"\"')

    parser.add_argument('-input','--input_dir', type=str, nargs='?',
                        #default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/extracts",
                        default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed_intestine",
                        #default="/home/dragon/drive/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed/bigfile/UBERON:0000955_CL:0000126.h5ad",
                        #default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed_glyco",
                        help='Input path from where the files come from. Do not include the last "\"')

    parser.add_argument('-output','--output_dir', type=str, nargs='?',
                        default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed_intestine/merged",
                        #default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed_intestine/merged",
                        #default="/home/dragon/drive/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed/metadata",
                        #default="/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/coarsed_glyco/analysis",
                        help='Output path where to save the files resulting from the computation. Do not include the last "\"')
    parser.add_argument('-optiona','--optiona', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='To set to True simply activate the flag without any arguments'
                             )
    parser.add_argument('-optionb','--optionb', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='To set to True simply activate the flag without any arguments'
                             )
    args = parser.parse_args()

    #TODO: put args here

    #1) Highlight: Read the cellxgene extract and divide it by tissue-celltype combo,
    #read_cellxgene(args.input_dir)
    #read_cellxgene_toy(args.input_dir)
    #2) Highlight: Fix the h5py format to suit the anndata format properly and calculate the mean and standard deviation
    #repair_adata(args.input_dir,calculate_mean=True,overwrite_mean=True)
    #3) Highlight: Merge all uberon-cell ontology mean and std into 2 files (one for mean, one for std)
    #merge_averaged_data(args.input_dir,args.output_dir) #Highlight: REMEMBER TO DELETE THE PREVIOUS FILES (OR IT WILL APPEND TO THEM)
    #repair_adata(args.output_dir,calculate_mean=False,overwrite_mean=False)
    exit()
    #4) Highlight: Extract the metadata for analysis
    if args.optiona:
        print("Option A")
        repair_adata(args.input_dir, calculate_mean=True,overwrite_mean=True) # TODO: the repaired big file should also be uploaded back
        #extract_glyco(args.input_dir,args.output_dir)
        #extract_metadata(args.input_dir,args.output_dir)

    #5) Highlight: Extract the non-zero counts from the glyco transferases
    else:
        print("Option B")
        #cumsum_glyco(args.input_dir, args.output_dir)
        # for file in os.listdir(args.input_dir):
        #     if file.endswith(".h5ad"):
        #         count_combos_cells(f"{args.input_dir}/{file}",args.output_dir)
        count_combos_cells(args.input_dir,args.output_dir)


# Problematic files: Recall that in the laptop there are some small versions of these datasets (not the full versions)
# UBERON:0000955_CL:4023008.h5ad # done and uploaded
# TODO:https://www.silx.org/doc/pyFAI/latest/usage/tutorial/Parallelization/Direct_chunk_read.html
# UBERON:0000955_CL:0002319.h5ad # 27GB # not possible with scanpy, perhaps with h5py
# UBERON:0000955_CL:4023069.h5ad # 3.7GB #done and uploaded
# UBERON:0000955_CL:0000126.h5ad # 2.7 # done and uploaded
# UBERON:0000955_CL:0000540.h5ad #7.1 -> some problem arised?


#TODO:
# 1) repair(/opt/project/cellxgene/coarsed)
# 2) merge_averaged_data()
# 3) repair(merged)


















