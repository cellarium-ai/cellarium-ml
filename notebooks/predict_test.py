import re
import os
import time

import numpy as np
import scanpy as sc
import sys
import torch
from sklearn.metrics.pairwise import cosine_similarity

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


import notebooks_functions as NF


train_database = {
                "1a":["pmbc_results_train",'lightning_logs/version_70/checkpoints/epoch=49-step=2500.ckpt',"../example_configs/scvi_config_pbmc_train.yaml","../data/pbmc_count_train.h5ad",['final_annotation', 'batch'],""],
                #11:["Kidney_muto_rna","lightning_logs/version_71/checkpoints/epoch=49-step=2000.ckpt","../example_configs/scvi_config_Kidney_muto_rna.yaml","../data/Kidney_muto_rna.h5ad",["cell_type_category","author_cell_type"],"feature_name"],


                  }

test_database = {
                #"2a": ["pmbc_results_test_raw",'',"../example_configs/scvi_config_pbmc_test.yaml","../data/pbmc_count_test.h5ad",['final_annotation', 'batch'],""],
                "2b": ["pmbc_results_test_masked",'',"../example_configs/scvi_config_pbmc_test_masked.yaml","../data/pbmc_count_test.h5ad",['final_annotation', 'batch'],""],
                "2a": ["pmbc_results_test_raw",'',"../example_configs/scvi_config_pbmc_test.yaml","../data/pbmc_count_test.h5ad",['final_annotation', 'batch'],""],
                 }

def cosine_similarity2(a,b,correlation_matrix=False,parallel=False): #TODO: import from utils?
    """Calculates the cosine similarity between 2 arrays.
    :param numpy array a: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param numpy array b: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param bool:Calculate matrix correlation(as in numpy coorcoef)"""

    n_a = a.shape[0]
    n_b = b.shape[0]
    diff_sizes = False
    if n_a != n_b:
        dummy_row = np.zeros((np.abs(n_a-n_b),) + a.shape[1:])
        diff_sizes = True
        if n_a < n_b:
            a = np.concatenate((a,dummy_row),axis=0)
        else:
            b = np.concatenate((b,dummy_row),axis=0)

    if np.ndim(a) == 1:
        num = np.dot(a,b)
        p1 = np.sqrt(np.sum(a**2)) #equivalent to p1 = np.linalg.norm(a)
        p2 = np.sqrt(np.sum(b**2))
        p1_p2 = p1*p2
        p1_p2 = np.where(p1_p2 == 0, np.finfo(float).eps, p1_p2) #avoid zero division issues
        #cosine_sim = num / (p1 * p2)
        cosine_sim = num / (p1_p2)
        return cosine_sim

    elif np.ndim(a) == 2:
        if correlation_matrix:
            b = b - b.mean(axis=1)[:, None]
            a = a - a.mean(axis=1)[:, None]

        num = np.dot(a, b.T) #[seq_len,21]@[21,seq_len] = [seq_len,seq_len]

        p1 =np.sqrt(np.sum(a**2,axis=1))[:,None] #+ 1e-10#[seq_len,1] #equivalent to np.linalg.norm(a,axis=1)
        #p1 = np.linalg.norm(a,axis=1)[:,None]
        p2 = np.sqrt(np.sum(b ** 2, axis=1))[None, :] #[1,seq_len]
        #p2 = np.linalg.norm(b,axis=1)[None,:]
        p1_p2 = p1*p2
        p1_p2 = np.where(p1_p2 == 0, np.finfo(float).eps, p1_p2) #avoid zero division issues
        #cosine_sim = num / (p1 * p2)
        cosine_sim = num / (p1_p2)

        if diff_sizes: #remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a-n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:,:-remove]
        if parallel:
            return cosine_sim[None,:]
        else:
            return cosine_sim
    else: #TODO: use elipsis for general approach?
        if correlation_matrix:
            b = b - b.mean(axis=2)[:, :, None]
            a = a - a.mean(axis=2)[:, :, None]
        num = np.matmul(a[:, None], np.transpose(b, (0, 2, 1))[None,:]) #[n,n,seq_len,seq_len]
        p1 = np.sqrt(np.sum(a ** 2, axis=2))[:, :, None] + 1e-10#Equivalent to np.linalg.norm(a,axis=2)[:,:,None]
        p2 = np.sqrt(np.sum(b ** 2, axis=2))[:, None, :] + 1e-10 #Equivalent to np.linalg.norm(b,axis=2)[:,None,:]

        cosine_sim = num / (p1[:,None]*p2[None,:])

        if diff_sizes: #remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a-n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:,:-remove]

        return cosine_sim

gene_set_dict,gene_set = NF.retrieve_genes()

overwrite =False
use_cuda=True
for train_model in train_database.values():

    foldername_train, checkpoint_file_train, config_file_train, adata_file_train, color_keys_train, gene_names_train = train_model

    print(f"Predicting test sequences using trained model with data : {foldername_train}")

    pipeline, device = NF.setup_model(checkpoint_file_train)

    figpath_train = f"figures/{foldername_train}"
    datapath_train = f"tmp_data/{foldername_train}"

    NF.folders(foldername_train, "figures", overwrite=False)
    NF.folders(foldername_train, "tmp_data", overwrite=False)

    filename_suffix_train = "_train_predictions"
    filename_train = f"adata_processed{filename_suffix_train}"

    matched_train, filepath_train = NF.matching_file(datapath_train, filename_train)

    adata_train = NF.download_predict(config_file_train, gene_names_train, filepath_train, pipeline, device, matched_train, filename_train, overwrite)
    adata_train = NF.ComputeUMAP(adata_train, filepath_train, figpath_train, "raw_umap", "raw_umap", None, use_cuda, overwrite, color_keys_train).run()
    adata_train = NF.ComputeUMAP(adata_train, filepath_train, figpath_train, "scvi_latent_umap", "scvi_umap", "X_scvi", use_cuda, overwrite,color_keys_train).run()
    #Find the top 200 more variables genes and make them compulsory or fix them
    adata_train = NF.define_gene_expressions(adata_train,gene_set,filepath_train,gene_names_train,overwrite)
    #Derive the leiden clusters
    train_dataset_dict = NF.Compute_Leiden_clusters(adata_train,gene_set_dict,figpath_train,color_keys_train,filepath_train,overwrite,False,use_cuda).run()

    for test_model in test_database.values():
        foldername_test, _, config_file_test, adata_file_test, color_keys_test, gene_names_test = test_model
        figpath_test = f"figures/{foldername_test}"
        datapath_test = f"tmp_data/{foldername_test}"
        NF.folders(foldername_test, "figures", overwrite=False)
        NF.folders(foldername_test, "tmp_data", overwrite=False)
        filename_suffix_test = f"_test_predictions_{foldername_test}"
        filename_test = f"adata_processed{filename_suffix_test}"

        matched_test, filepath_test = NF.matching_file(datapath_test, filename_test)
        adata_test = sc.read(adata_file_test)


        if "masked" in foldername_test:
            print("Here")
            #Highlight: Mask or set to 0 some of the genes in the test that are not in the highly expressed train dataset
            nomask_train = adata_train.var['high_exp_genes'].tolist()
            mask_test_idx = ~adata_test.var_names.isin(nomask_train)
            cols_to_mask_test = adata_test.var_names[mask_test_idx].tolist()
            random_mask_cols_test = np.random.choice(cols_to_mask_test, 2000, replace=False) #mask 200 genes
            print("Done")
            random_mask_cols_test_idx = adata_test.var_names.isin(random_mask_cols_test)
            #adata_test.X[:] = 0
            adata_test.X[:,random_mask_cols_test_idx] = 0

            adata_test.write(adata_file_test.replace(".h5ad","_masked.h5ad"))
            filename_suffix_test = f"_test_MASKED_predictions_{foldername_test}"
            filename_test = f"adata_processed{filename_suffix_test}"
            matched_test, filepath_test = NF.matching_file(datapath_test, filename_test)
            overwrite=True
            adata_test = NF.download_predict(config_file_test, gene_names_test, filepath_test, pipeline, device,
                                             matched_test, filename_test,
                                             overwrite)  # prediction of the test dataset with this model
        else:
            print("Not masked version")
            adata_test = NF.download_predict(config_file_test, gene_names_test, filepath_test, pipeline, device,
                                             matched_test, filename_test,
                                             overwrite)  # prediction of the test dataset with this model



        #TODO: Random subsampling of adata_train by cluster

        #TODO: Make this optional if it has already been done
        cossim_matrix = cosine_similarity(adata_train.obsm["X_scvi"],adata_test.obsm["X_scvi"]) #(ntrain,ntest)
        argmax_cos_train = cossim_matrix.argmax(axis=0) # there are duplicates ....
        adata_test.obs["argmax_cos_train"] = argmax_cos_train

        train_obs_idx = np.arange(adata_train.X.shape[0])
        adata_train.obs["cell_index"] = train_obs_idx
        adata_train_subset = adata_train.obs[["cell_index","clusters",color_keys_train[0]]]
        adata_train_subset.rename(columns={"clusters":"clusters_train",color_keys_train[0]:f"{color_keys_train[0]}_train"}, inplace=True)

        adata_test.obs = adata_test.obs.merge(adata_train_subset, left_on="argmax_cos_train", right_on="cell_index", how="inner" )

        #print(adata_test.obs[[color_keys_test[0],f"{color_keys_train[0]}_train"]])
        target = adata_test.obs[color_keys_test[0]].values.to_numpy()
        prediction = adata_test.obs[f"{color_keys_train[0]}_train"].values.to_numpy()
        accuracy = (target == prediction).sum()/len(target) * 100
        print("------------------------------Pairwise-accuracy--------------------------")
        print(accuracy)





#Find a way to compare the cluster assignations for the test datasets predicted with the same train dataset








































