import gc
import re

import matplotlib.pyplot as plt
import torch
import os
import yaml
import subprocess
import os
import scanpy as sc
import sys
import umap
import numpy as np
import inspect
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
tucker_human_heart_atlas: 276940 × 2889
human_heart_atlas: 665234 × 2889
gut_cell_atlas_normed: 428469 × 30535
cellxgene_liver: 481828 × 60664

"""

#TODO: Other datasets:
# https://collections.cellatlas.io/liver-development
# Lung: https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0


foldername_dict = { 0: ["pmbc_results",'lightning_logs/version_58/checkpoints/epoch=49-step=3150.ckpt',"../example_configs/scvi_config_pbmc.yaml","../data/pbmc_count.h5ad",['final_annotation', 'batch'],""],
                    #1:  ["cas_50m_homo_sapiens_no_cancer_extract_0",'lightning_logs/version_37/checkpoints/epoch=49-step=1000.ckpt',"../example_configs/scvi_config_cas_50m_homo_sapiens_no_cancer.yaml","../data/cas_50m_homo_sapiens_no_cancer_extract_extract_0.h5ad",['final_annotation', 'batch'],"" ],
                    2 : ["tucker_human_heart_atlas","lightning_logs/version_52/checkpoints/epoch=29-step=16230.ckpt","../example_configs/scvi_config_tucker_heart_atlas.yaml","../data/tucker_human_heart_atlas.h5ad",["Cluster","batch"],"gene_names"],
                    3 : ["human_heart_atlas","lightning_logs/version_53/checkpoints/epoch=39-step=26640.ckpt","../example_configs/scvi_config_human_heart_atlas.yaml","../data/human_heart_atlas.h5ad",["cell_type","batch"],"gene_name-new"], #10 GB
                    4 : ["gut_cell_atlas_raw","lightning_logs/version_51/checkpoints/epoch=39-step=17160.ckpt","../example_configs/scvi_config_gut_cell_atlas_raw.yaml","../data/gut_cell_atlas_raw.h5ad",["category"],""],
                    5 : ["cellxgene_liver","lightning_logs/version_57/checkpoints/epoch=49-step=47100.ckpt","../example_configs/scvi_config_cellxgene_liver.yaml","../data/cellxgene_liver.h5ad",["cell_type"],"feature_name"],
                    6: ["cellxgene_lung","","../example_configs/scvi_config_cellxgene_lung.yaml","../data/single_cell_lung_atlas.h5ad",[""],""], #too big
                    7: ["cellxgene_lung_subset","","../example_configs/scvi_config_cellxgene_lung_subset.yaml","../data/single_cell_lung_atlas_subset.h5ad",[""],""],
                    }

#Done
foldername,checkpoint_file,config_file, adata_file,color_keys,gene_names = foldername_dict[3]



#NF.scanpy_scvi(adata_file) #too slow to handle
#subprocess.call([f"{sys.executable}","../cellarium/ml/cli.py","scvi","fit","-c",config_file],env=env) #/opt/conda/bin/python



NF.folders(foldername,"figures",overwrite=False)
NF.folders(foldername,"tmp_data",overwrite=False)

# load the trained model
scvi_model = CellariumModule.load_from_checkpoint(checkpoint_file).model
# move the model to the correct device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scvi_model.to(device)
scvi_model.eval()
# construct the pipeline
pipeline = CellariumPipeline([scvi_model])

filename_suffix = ""
figpath = f"figures/{foldername}"
datapath = f"tmp_data/{foldername}"
overwrite = True


def matching_file(datapath:str,filename:str):
    """Finds any file starting with <filename>"""
    pattern = re.compile(filename)
    matched = [ file if pattern.match(file) else None for file in os.listdir(datapath)]
    matched = [i for i in matched if i is not None]
    filepath = os.path.join(datapath, matched[0]) if len(matched) > 0 else datapath

    if datapath == filepath:
        filepath = f"{filepath}/{filename}"
    return matched,filepath

filename = f"adata_processed{filename_suffix}"
matched,filepath = matching_file(datapath,filename)

if not matched or overwrite:
    #filepath = os.path.join(datapath, f"{filename}.h5ad") #TODO: i think it can be removed
    # get the location of the dataset
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    data_path = config_dict['data']['dadc']['init_args']['filenames']
    print(f'Data is coming from {data_path}')
    # get a dataset object
    dataset = NF.get_dataset_from_anndata(
        data_path,
        batch_size=128,
        shard_size=None,
        shuffle=False,
        seed=0,
        drop_last=False,
    )
    filepath = filepath + ".h5ad" if not filepath.endswith(".h5ad") else filepath
    adata = NF.embed(dataset, pipeline,device= device,filepath=filepath)
    # Highlight: Reconstruct de-noised/de-batched data
    for label in range(pipeline[-1].n_batch):
        print("Label: {}".format(label))
        adata_tmp = NF.reconstruct_debatched(dataset, pipeline, transform_to_batch_label=label,layer_key_added=f'scvi_reconstructed_{label}', device=device)
        adata.layers[f'scvi_reconstructed_{label}'] = adata_tmp.layers[f'scvi_reconstructed_{label}']
        break
    if gene_names:
        adata.var_names = adata.var[gene_names]
    adata.write(filepath)
else:
    print(f"File found : {filename}")
    print("Reading file : {}".format(filepath))
    adata = sc.read(filepath)
    if gene_names:
        adata.var_names = adata.var[gene_names]


# if 'scvi_reconstructed_0' not in list(adata.layers.keys()) or overwrite:
#     print("Key 'scvi_reconstructed_0' not found, computing")
#     # Highlight: Reconstruct de-noised/de-batched data
#     for label in range(pipeline[-1].n_batch):
#         print("Label: {}".format(label))
#         adata_tmp = NF.reconstruct_debatched(dataset, pipeline, transform_to_batch_label=label,layer_key_added=f'scvi_reconstructed_{label}', device=device)
#         adata.layers[f'scvi_reconstructed_{label}'] = adata_tmp.layers[f'scvi_reconstructed_{label}']
#         break
#     adata.write(filepath)
# else:
#     print("Key 'scvi_reconstructed_0' found, continue")
#
# exit()
#filename = f"adata_scvi_reconstructed_raw_umap{filename_suffix}"
#matched,filepath = matching_file(datapath,filename)


if "X_raw_umap" not in list(adata.obsm.keys()) or overwrite:
    print("Key 'X_raw_umap' not found, computing")
    adata = NF.plot_umap(adata,filepath,figpath,"raw_umap","raw_umap",None,color_keys)
else:
    print("Key 'X_raw_umap' found, continue")

if "X_scvi_umap" not in list(adata.obsm.keys()) or overwrite:
    print("Key 'X_scvi_umap' not found, computing")
    adata = NF.plot_umap(adata,filepath, figpath,"scvi_latent_umap","scvi_umap","X_scvi",color_keys)
else:
    print("Key 'X_scvi_umap' found, continue")


#gene_set_dict = {"all":['B4GALNT2', 'HS6ST2', 'MGAT4EP', 'ST6GALNAC4', 'ALG13', 'ABO', 'STT3B', 'PIGZ', 'UST', 'CHPF', 'ALG9', 'B4GALT4', 'HS2ST1', 'PIGG', 'ALG10B', 'ST6GALNAC3', 'DSEL', 'FUT4', 'GCNT2', 'XYLT1', 'PGAP4', 'GGTA1P', 'GLT8D2', 'UGGT1', 'GALNT14', 'EXTL3', 'POGLUT1', 'B4GAT1', 'MGAT1', 'NDST3', 'POFUT1', 'PYGM', 'ALG3', 'NDST2', 'UGT1A4', 'GAL3ST1', 'XXYLT1', 'UGT1A7', 'GAL3ST2', 'UGT2B7', 'GXYLT2', 'UGT8', 'B4GALNT3', 'CHST12', 'CHST13', 'POGLUT2', 'PIGN', 'FUT9', 'GLT8D1', 'RXYLT1', 'ST6GALNAC2', 'B4GALT5', 'CHST1', 'ALG14', 'GTDC1', 'GALNT5', 'TMTC1', 'ST8SIA5', 'EXTL2', 'GXYLT1', 'B3GNT4', 'B4GALT1', 'HS3ST2', 'PIGV', 'ST3GAL2', 'UGT2B17', 'ST8SIA1', 'B4GALT3', 'FUT1', 'GALNTL5', 'EXTL1', 'UGT2B28', 'PIGL', 'B4GALT6', 'GAL3ST4', 'GALNT13', 'ALG1L2', 'PYGB', 'ST8SIA6', 'DPY19L3', 'MAN1A2', 'GALNT1', 'UGT3A1', 'MGEA5', 'B3GALT4', 'DPY19L1', 'UGT2B4', 'CHST14', 'POMT1', 'DPM1', 'HAS3', 'GALNT16', 'ALG1', 'B3GNT5', 'PIGO', 'MAN1C1', 'ST3GAL4', 'PLOD3', 'FUT11', 'GALNT3', 'GYG2', 'B4GALNT4', 'MGAT4B', 'UGT1A9', 'CMAS', 'CHSY1', 'GALNT8', 'PYGL', 'B3GNT9', 'LARGE1', 'GLT1D1', 'TMTC4', 'MGAT2', 'DPY19L4', 'GLCE', 'CHST10', 'B3GALNT2', 'CHST11', 'UGT1A6', 'CHPF2', 'XYLT2', 'POGLUT3', 'GALNT2', 'ST8SIA4', 'MAN1B1', 'UGT2B15', 'ST3GAL6', 'ALG6', 'POMGNT1', 'UGT2A3', 'COLGALT1', 'UGT2B11', 'MFNG', 'DPAGT1', 'MAN1A1', 'OGT', 'GALNT15', 'HAS1', 'FUT7', 'CHST15', 'GYS2', 'FKTN', 'B3GALT5', 'ALG2', 'CHST7', 'HS3ST6', 'HS3ST1', 'ST3GAL5', 'ST6GAL2', 'LFNG', 'ALG1L', 'A4GALT', 'FUT10', 'B3GNT2', 'GALNT6', 'GALNT7', 'GCNT7', 'HS3ST5', 'POMGNT2', 'LARGE2', 'ST3GAL1', 'GALNT9', 'GALNTL6', 'FUT2', 'UGT1A1', 'GCNT1', 'CERCAM', 'UGT1A5', 'UGGT2', 'POMK', 'GALNT18', 'A3GALT2', 'GALNT10', 'CHST5', 'ALG12', 'PIGA', 'MGAT3', 'MGAT4D', 'GYS1', 'B3GALT2', 'RFNG', 'FUT3', 'B3GAT3', 'HAS2', 'B3GLCT', 'TMTC2', 'GLT6D1', 'B4GALT2', 'DSE', 'MGAT4A', 'UGT1A8', 'COLGALT2', 'GALNT11', 'ST3GAL3', 'ST6GALNAC5', 'C1GALT1', 'GALNT17', 'B3GAT2', 'ST8SIA3', 'CSGALNACT2', 'GALNT12', 'MGAT5', 'PIGM', 'UGT3A2', 'GCNT4', 'B3GNT8', 'GCNT3', 'TMTC3', 'HS3ST3B1', 'C1GALT1C1', 'HS6ST1', 'B3GNT7', 'B3GALT1', 'B3GALT6', 'A4GNT', 'B3GNT3', 'FKRP', 'GAL3ST3', 'FUT5', 'POFUT2', 'B3GALNT1', 'CHST6', 'UGT2A1', 'FUT6', 'UGT2B10', 'GYG1', 'HS6ST3', 'POMT2', 'ST6GALNAC6', 'GCNT2P', 'STT3A', 'B3GNT6', 'B4GALT7', 'EOGT', 'ALG11', 'WSCD1', 'ST8SIA2', 'UGCG', 'FUT8', 'ALG10', 'HS3ST3A1', 'CHST8', 'PIGB', 'ALG8', 'CHSY3', 'B3GAT1', 'B3GNTL1', 'WSCD2', 'CASD1', 'NDST4', 'MGAT4C', 'UGT1A3', 'GALNT4', 'ALG5', 'GBGT1', 'NDST1', 'ST6GALNAC1', 'CHST4', 'ST6GAL1', 'HS3ST4', 'CHST3', 'CHST2', 'CHST9', 'EXT2', 'B4GALNT1', 'MGAT5B', 'EXT1', 'CSGALNACT1', 'PIGW', 'DPY19L2']}
#'B4GALNT2', 'HS6ST2', 'MGAT4EP', 'ST6GALNAC4', 'ALG13', 'ABO', 'STT3B', 'PIGZ', 'UST', 'CHPF', 'ALG9', 'B4GALT4', 'HS2ST1', 'PIGG', 'ALG10B', 'ST6GALNAC3', 'DSEL', 'FUT4', 'GCNT2', 'XYLT1', 'PGAP4', 'GGTA1P', 'GLT8D2', 'UGGT1', 'GALNT14', 'EXTL3', 'POGLUT1', 'B4GAT1', 'MGAT1', 'NDST3', 'POFUT1', 'PYGM', 'ALG3', 'NDST2', 'UGT1A4', 'GAL3ST1', 'XXYLT1', 'UGT1A7', 'GAL3ST2', 'UGT2B7', 'GXYLT2', 'UGT8', 'B4GALNT3', 'CHST12', 'CHST13', 'POGLUT2', 'PIGN', 'FUT9', 'GLT8D1', 'RXYLT1', 'ST6GALNAC2', 'B4GALT5', 'CHST1', 'ALG14', 'GTDC1', 'GALNT5', 'TMTC1', 'ST8SIA5', 'EXTL2', 'GXYLT1', 'B3GNT4', 'B4GALT1', 'HS3ST2', 'PIGV', 'ST3GAL2', 'UGT2B17', 'ST8SIA1', 'B4GALT3', 'FUT1', 'GALNTL5', 'EXTL1', 'UGT2B28', 'PIGL', 'B4GALT6', 'GAL3ST4', 'GALNT13', 'ALG1L2', 'PYGB', 'ST8SIA6', 'DPY19L3', 'MAN1A2', 'GALNT1', 'UGT3A1', 'MGEA5', 'B3GALT4', 'DPY19L1', 'UGT2B4', 'CHST14', 'POMT1', 'DPM1', 'HAS3', 'GALNT16', 'ALG1', 'B3GNT5', 'PIGO', 'MAN1C1', 'ST3GAL4', 'PLOD3', 'FUT11', 'GALNT3', 'GYG2', 'B4GALNT4', 'MGAT4B', 'UGT1A9', 'CMAS', 'CHSY1', 'GALNT8', 'PYGL', 'B3GNT9', 'LARGE1', 'GLT1D1', 'TMTC4', 'MGAT2', 'DPY19L4', 'GLCE', 'CHST10', 'B3GALNT2', 'CHST11', 'UGT1A6', 'CHPF2', 'XYLT2', 'POGLUT3', 'GALNT2', 'ST8SIA4', 'MAN1B1', 'UGT2B15', 'ST3GAL6', 'ALG6', 'POMGNT1', 'UGT2A3', 'COLGALT1', 'UGT2B11', 'MFNG', 'DPAGT1', 'MAN1A1', 'OGT', 'GALNT15', 'HAS1', 'FUT7', 'CHST15', 'GYS2', 'FKTN', 'B3GALT5', 'ALG2', 'CHST7', 'HS3ST6', 'HS3ST1', 'ST3GAL5', 'ST6GAL2', 'LFNG', 'ALG1L', 'A4GALT', 'FUT10', 'B3GNT2', 'GALNT6', 'GALNT7', 'GCNT7', 'HS3ST5', 'POMGNT2', 'LARGE2', 'ST3GAL1', 'GALNT9', 'GALNTL6', 'FUT2', 'UGT1A1', 'GCNT1', 'CERCAM', 'UGT1A5', 'UGGT2', 'POMK', 'GALNT18', 'A3GALT2', 'GALNT10', 'CHST5', 'ALG12', 'PIGA', 'MGAT3', 'MGAT4D', 'GYS1', 'B3GALT2', 'RFNG', 'FUT3', 'B3GAT3', 'HAS2', 'B3GLCT', 'TMTC2', 'GLT6D1', 'B4GALT2', 'DSE', 'MGAT4A', 'UGT1A8', 'COLGALT2', 'GALNT11', 'ST3GAL3', 'ST6GALNAC5', 'C1GALT1', 'GALNT17', 'B3GAT2', 'ST8SIA3', 'CSGALNACT2', 'GALNT12', 'MGAT5', 'PIGM', 'UGT3A2', 'GCNT4', 'B3GNT8', 'GCNT3', 'TMTC3', 'HS3ST3B1', 'C1GALT1C1', 'HS6ST1', 'B3GNT7', 'B3GALT1', 'B3GALT6', 'A4GNT', 'B3GNT3', 'FKRP', 'GAL3ST3', 'FUT5', 'POFUT2', 'B3GALNT1', 'CHST6', 'UGT2A1', 'FUT6', 'UGT2B10', 'GYG1', 'HS6ST3', 'POMT2', 'ST6GALNAC6', 'GCNT2P', 'STT3A', 'B3GNT6', 'B4GALT7', 'EOGT', 'ALG11', 'WSCD1', 'ST8SIA2', 'UGCG', 'FUT8', 'ALG10', 'HS3ST3A1', 'CHST8', 'PIGB', 'ALG8', 'CHSY3', 'B3GAT1', 'B3GNTL1', 'WSCD2', 'CASD1', 'NDST4', 'MGAT4C', 'UGT1A3', 'GALNT4', 'ALG5', 'GBGT1', 'NDST1', 'ST6GALNAC1', 'CHST4', 'ST6GAL1', 'HS3ST4', 'CHST3', 'CHST2', 'CHST9', 'EXT2', 'B4GALNT1', 'MGAT5B', 'EXT1', 'CSGALNACT1', 'PIGW', 'DPY19L2'

gene_set_dict = {
'dpy19l':['DPY19L1', 'DPY19L2', 'DPY19L3', 'DPY19L4'],
'piga':['PGAP4', 'PIGA', 'PIGB', 'PIGM', 'PIGV', 'PIGZ'],
'ugcg':['A4GALT', 'B3GALNT1', 'B3GALT4', 'B3GNT5', 'B4GALNT1', 'B4GALT5', 'B4GALT6', 'UGCG'],
'ugt8':['UGT8'],
'ost':['ALG10', 'ALG10B', 'ALG11', 'ALG12', 'ALG13', 'ALG14', 'ALG2', 'ALG3', 'ALG6', 'ALG8', 'ALG9', 'DPAGT1', 'FUT8', 'MGAT1', 'MGAT2', 'MGAT3', 'MGAT4A', 'MGAT4B', 'MGAT4C', 'MGAT4D', 'MGAT5', 'STT3A', 'UGGT1', 'UGGT2'],
'ogt':['OGT'],
'colgalt':['COLGALT1', 'COLGALT2'],
'eogt':['EOGT'],
'galnt':['B3GNT6', 'C1GALT1', 'C1GALT1C1', 'GALNT1', 'GALNT10', 'GALNT11', 'GALNT12', 'GALNT13', 'GALNT14', 'GALNT15', 'GALNT16', 'GALNT17', 'GALNT18', 'GALNT2', 'GALNT3', 'GALNT4', 'GALNT5', 'GALNT6', 'GALNT7', 'GALNT8', 'GALNT9', 'GCNT1', 'GCNT3', 'GCNT4'],
'pofut1':['LFNG', 'MFNG', 'POFUT1', 'RFNG'],
'pofut2':['B3GLCT', 'POFUT2'],
'poglut':['GXYLT1', 'GXYLT2', 'POGLUT1', 'POGLUT2', 'POGLUT3', 'XXYLT1'],
'pomt':['B3GALNT2', 'B4GAT1', 'FKRP', 'FKTN', 'LARGE1', 'LARGE2', 'MGAT5B', 'POMGNT1', 'POMGNT2', 'POMK', 'POMT1', 'POMT2', 'RXYLT1'],
'tmtc':['TMTC1', 'TMTC2', 'TMTC3', 'TMTC4', 'TMEM260'],
'xylt1/2':['B3GALT6', 'B3GAT3', 'B4GALT7', 'CHPF', 'CHPF2', 'CHSY1', 'CHSY3', 'CSGALNACT1', 'CSGALNACT2', 'EXT1', 'EXT2', 'EXTL1', 'EXTL2', 'EXTL3', 'XYLT1', 'XYLT2']
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


#gene_set  = "MIR892B,MIR873,USH1C,CTDSP2,RAD50,FEM1B,CDK2,CDK3,CDK4,PSME3,CDK5,CDK6,CTDSPL,CDK7,CDKN1A,CDK2AP2,CDKN1B,AKAP8,CDKN1C,CDKN2A,CDKN2B,CCNO,CDKN2C,CDKN2D,CDKN3,NPM2,BTN2A2,NDC80,GPNMB,MAD2L2,TACC3,UBD,CENPE,CENPF,KHDRBS1,NES,PLK2,ARPP19,NEK6,DBF4,CCNI,RCC1,PIM2,UBE2C,TOPBP1,CHEK1,FOXN3,ZWINT,FAM107A,CHEK2,TREX1,CDCA5,ECD,PHB2,PABIR1,CKS1B,CKS2,PRAP1,DCUN1D3,PLK3,PLK5,IQGAP3,CHMP4B,TRIM71,TPRA1,LSM11,HUS1B,NACC2,ATF2,CRY1,MAPK14,CACUL1,E2F7,RAD9B,EME1,SPC24,DTX3L,NEK10,CYP1A1,SASS6,SDE2,DDB1,DDX3X,CDC14C,DLG1,DNA2,DNM2,DUSP1,E2F1,E2F2,E2F3,E2F4,E2F6,EGFR,ARID2,EME2,EIF4E,EIF4EBP1,EIF4G1,AIF1,ENSA,EPS8,ERCC2,AKT1,ERCC3,ERCC6,ESRRB,EZH2,FANCD2,STOX1,CCNY,FGF10,MAPK15,FHL1,ATF5,SIRT2,PAXIP1,MYO16,FOXM1,CLASP2,PHF8,SMC5,FBXL7,PLCB1,KLHL18,CLASP1,UFL1,BRD4,SPDYA,STXBP4,MBLAC1,FBXO7,ZNF324,INTS7,ANAPC15,SIN3A,SYF2,KANK2,HINFP,ANKRD17,GIGYF2,APPL1,TIPRL,FBXO6,FBXO5,FBXO4,LATS2,GFI1,AKAP8L,MTBP,KCNH5,VPS4A,CHMP2A,UBE2S,PRPF19,GLI1,GML,CCDC57,NSMCE2,RGCC,BABAM1,BRD7,GSPT1,TMOD3,ANAPC2,GPR132,RPA4,ANAPC4,DONSON,NOP53,ANXA1,H2AX,APBB1,APBB2,APC,PRMT2,HSPA2,BIRC5,HUS1,HYAL1,ID2,ID4,GEN1,APP,IK,INCENP,INHBA,ITGB1,KCNA5,USP17L2,GPR15LG,NANOGP8,MIR10A,MIR133A1,MIR137,MIR15A,MIR15B,MIR16-1,MIR193A,MIR195,MIR19B1,MIR208A,MIR214,MIR221,MIR222,MIR26A1,MIR29A,MIR29B1,MIR29C,MIR30C2,MAD2L1,MDM2,MECP2,MLF1,MAP3K11,FOXO4,MN1,MNAT1,MOS,MRE11,EIF2AK4,MIR133B,MIR372,MSH2,MUC1,MYC,NBN,ATM,NEUROG1,NFIA,NFIB,NPAT,NPM1,DDR2,ATP2B4,CRNN,ORC1,OVOL1,RRM2B,PBX1,RPS27L,DYNC1LI1,ING4,MRNIP,CDK16,TFDP3,CDK17,CDK18,WAC,DACT1,FZR1,TAOK3,MBTPS2,CRLF3,PPME1,ACTL6B,ANAPC5,ANAPC7,LCMT1,TRIAP1,GTSE1,DTL,ANAPC11,SIRT7,METTL13,CPSF3,UIMC1,MAP3K20,CDK14,ABCB1,PKD1,PKD2,PLCG2,PLK1,PLRG1,PML,POLE,ANLN,ETAA1,ATR,INO80,CCNJ,PAF1,NSUN2,SPDL1,TIPIN,PINX1,USP47,ZWILCH,PPP1R10,CDCA8,PPP2CA,RFWD3,PBRM1,APPL2,PHF10,FBXW7,PPP3CA,PIDD1,PPP6C,AMBRA1,PKIA,CHFR,CDK5RAP2,RIOK2,PCID2,CENPJ,PPP2R2D,KMT2E,PRKDC,RCC2,TEX14,SUSD2,MEPCE,PROX1,TRIM39,TCIM,PSMG2,KNL1,AVEN,BACH1,GJC2,PSME1,PSME2,PTEN,MIR362,SPC25,MIR451A,MIR495,MIR515-1,MIR520A,MIR519D,MIR520H,MIR503,ARID1B,MTA3,HECW2,RPTOR,TAOK1,USP28,CAMSAP3,USP29,USP37,PTPN6,CCAR2,PTPN11,PTPRC,BARD1,RAD1,CTDSP1,RHOU,INIP,BCAT1,RAD9A,RAD17,RAD21,RAD51,RAD51C,RAD51B,RB1,RBBP8,RBL1,RBL2,CCND1,BCL2,RDX,UPF1,DPF2,RFPL1,ACTB,BCL7A,RINT1,MIIP,RPA2,RPL24,RPL26,RPS6,RPS6KB1,RRM1,RRM2,CCL2,BID,CLSPN,BLM,SETMAR,CCNI2,ANAPC1,NABP1,STIL,SIX3,SKP2,CDK15,INTS3,SMARCA2,SMARCA4,STK33,SMARCB1,SMARCC1,DDRGK1,SMARCC2,SMARCD1,SMARCD2,SMARCD3,SMARCE1,SOX2,SPAST,BRCA1,BRCA2,ZFP36L1,ZFP36L2,AURKA,ADAM17,TAF1,TAF2,TAF10,TBX2,MIR638,BUB1,BUB1B,TERT,TFAP4,TFDP1,TP53,TP53BP1,TPD52L1,TPR,TTK,UBE2A,UBE2E2,UBE2L3,WEE1,WNT10B,XPC,XRCC3,ZNF16,ZNF207,CACNB4,ZNF655,NABP2,FBXL15,BRCC3,DDX39B,PAGR1,CDC73,CCNJL,RNASEH2B,FBXO31,KDM8,NEK11,ATAD5,CCNP,JADE1,WDR76,CALM1,CTC1,DBF4B,TTI2,MUS81,CEP63,CDK5RAP3,CALM2,CUL5,CALM3,DPF3,CAMK2A,FAM83D,CDT1,TMEM14B,DPF1,ARID1A,CDC7,CDC45,GFI1B,CASP2,NUF2,PARP9,RHNO1,MAD1L1,USP26,HASPIN,BRIP1,HORMAD1,USP44,ATRIP,ABRAXAS1,DYRK3,DOT1L,BRSK1,CUL4B,CUL4A,CUL3,CUL2,CUL1,KLF11,PPP1R9B,KLHL22,PPM1D,MASTL,ZFYVE19,LSM10,CCNB3,PIAS1,CDC14B,CDC14A,CDK10,THOC5,ACTL6A,CDC23,MBTPS1,CRADD,RAB11A,IER3,CDC16,ZPR1,NAE1,CCNA2,CCNA1,CCNB1,TIMELESS,PHOX2B,CCND2,CCND3,CCNE1,CCNF,ACVR1,CCNG1,CCNG2,CCNH,BRSK2,TM4SF5,TICRR,PKMYT1,ACVR1B,LATS1,CCNB2,CCNE2,CTDP1,ZNF830,SLFN11,CHMP7,ZW10,BUB3,CCNQ,AURKB,CHMP4C,BCL7C,BCL7B,KLF4,TRIP13,TAOK2,ADAMTS1,VPS4B,MACROH2A1,CLOCK,BABAM2,MAD2L1BP,MDC1,TTI1,ESPL1,KNTC1,DLGAP5,CDK1,MELK,CDC5L,TELO2,CDC6,CDC20,KIF14,CDC25A,CDC25B,CDC25C,CDC27,CDC34,THOC1".split(',')
gene_set = []
list(map(gene_set.extend, list(gene_set_dict.values())))

if "genes_of_interest" not in list(adata.var.keys()) or overwrite:
    print("Key 'genes_of_interest' not found, computing")
    adata = NF.define_gene_expressions(adata,gene_set,foldername,filepath,gene_names)
else:
    print("Key 'genes_of_interest' found, continue")


adata_glyco = adata[:, adata.var_names.isin(gene_set)]

if "X_scvi_reconstructed_0_umap" not in list(adata.obsm.keys()) or overwrite:
    print("Key 'X_scvi_reconstructed_0_umap' not found, computing")
    adata = NF.umap_group_genes(adata,filepath)
else:
    print("Key 'X_scvi_reconstructed_0_umap' found, continue")

#Highlight: Density differential expression

# figname = "expression_density_difference"
# NF.differential_gene_expression(adata,gene_set,figpath,figname)
#
#
# figname = "glyco_expression_RAW"
# # # figname = "cell_cycle_expression_RAW"
# NF.plot_avg_expression(adata,"X_raw_umap",gene_set_dict,figpath,figname,color_keys)
# #
# figname = "glyco_expression_RECONSTRUCTED"
# # # figname = "cell_cycle_expression_RECONSTRUCTED"
# NF.plot_avg_expression(adata,"X_scvi_reconstructed_0_umap",gene_set_dict,figpath,figname,color_keys)
#
#
# #Highlight: Leiden clustering
NF.plot_neighbour_leiden_clusters(adata,gene_set,figpath,color_keys,filepath,overwrite)

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
# NF.plot_violin_expression_distribution(adata_glyco,present_gene_set,figpath,figname,"scvi_reconstructed_0")


dict_subsets = {
                "glyco":[f"adata_subset_processed_glyco{filename_suffix}",adata.var_names.isin(gene_set)],
                "top20genes":[f"adata_subset_processed_top20genes{filename_suffix}",adata.var['top20_high_exp_genes']],
                "top200genes":[f"adata_subset_processed_top200genes{filename_suffix}",adata.var['high_exp_genes']]
                }

gene_group_names=["glyco","top20genes","top200genes"]

overwrite=True
if "gene_subset_glyco" not in list(adata.var.keys()) or overwrite:
    for gene_group_name in gene_group_names:
        filename_subset,slice = dict_subsets[gene_group_name]
        print("NMF analysis not found for glyco genes, computing")
        filepath_subset = os.path.join(datapath, f"{filename_subset}.h5ad")
        adata_subset = adata[:, slice]
        present_gene_set = adata_subset.var_names.tolist()
        adata,adata_subset = NF.analysis_variance(adata,adata_subset,present_gene_set,filepath_subset,filepath,gene_group_name)


for gene_group_name in gene_group_names:
    filename_subset,slice = dict_subsets[gene_group_name]
    filepath_subset = os.path.join(datapath, f"{filename_subset}.h5ad")
    print("Reading and plotting file : {}".format(filepath_subset))
    adata_subset = sc.read(filepath_subset)
    NF.plot_nmf(adata,adata_subset,color_keys,figpath,gene_group_name)



#TODO: Leiden clusters for one cell type

exit()
