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
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #/opt/project/cellarium-ml/
    sys.path.insert(1,module_dir)
    # Set the PYTHONPATH environment variable for subprocess to inherit
    env = os.environ.copy()
    env['PYTHONPATH'] = module_dir
else:
    # Set the PYTHONPATH environment variable
    env = os.environ.copy()

from cellarium.ml.core import CellariumPipeline, CellariumModule
import notebooks_functions as NF


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

"""


foldername_dict = { 0: ["pmbc_results",'lightning_logs/version_36/checkpoints/epoch=49-step=3150.ckpt',"../example_configs/scvi_config_pbmc.yaml","../data/pbmc_count.h5ad",['final_annotation', 'batch']],
                    1:  ["cas_50m_homo_sapiens_no_cancer_extract_0",'lightning_logs/version_37/checkpoints/epoch=49-step=1000.ckpt',"../example_configs/scvi_config_cas_50m_homo_sapiens_no_cancer.yaml","../data/cas_50m_homo_sapiens_no_cancer_extract_extract_0.h5ad",['final_annotation', 'batch'] ],
                    2 : ["tucker_human_heart_atlas","lightning_logs/version_44/checkpoints/epoch=29-step=16230.ckpt","../example_configs/scvi_config_tucker_heart_atlas.yaml","../data/tucker_human_heart_atlas.h5ad",["Cluster","batch"]],
                    3 : ["human_heart_atlas","lightning_logs/version_45/checkpoints/epoch=39-step=26640.ckpt","../example_configs/scvi_config_human_heart_atlas.yaml","../data/human_heart_atlas.h5ad",[]], #10 GB
                    4 : ["gut_cell_atlas_normed","","../example_configs/scvi_config_gut_cell_atlas_normed.yaml","../data/gut_cell_atlas_normed.h5ad",[]]
                    }


foldername,checkpoint_file, config_file, adata_file, color_keys = foldername_dict[4]

#NF.scanpy_scvi(adata_file) #too slow to handle
subprocess.call(["/opt/conda/bin/python","../cellarium/ml/cli.py","scvi","fit","-c",config_file],env=env)

exit()

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

# get the location of the dataset
with open(config_file, "r") as file:
    config_dict = yaml.safe_load(file)
data_path = config_dict['data']['dadc']['init_args']['filenames']
print(f'Data is coming from {data_path}')

# get a dataset object
dataset = NF.get_dataset_from_anndata(
    data_path,
    batch_size=128,
    shard_size = None,
    shuffle=False,
    seed=0,
    drop_last=False,
)

filename_suffix = "_test"
figpath = f"figures/{foldername}"
overwrite = False

filepath = f"tmp_data/{foldername}/adata_embedded{filename_suffix}.h5ad"
if not os.path.exists(filepath) or overwrite:
    adata = NF.embed(dataset, pipeline,device= device,filepath=filepath)
else:
    print("Reading file : {}".format(filepath))
    adata = sc.read(filepath)

# Reconstruct de-noised/de-batched data
filepath = f"tmp_data/{foldername}/adata_scvi_reconstructed{filename_suffix}.h5ad"
if not os.path.exists(filepath) or overwrite:
    #adata = None
    #added = False
    for label in range(pipeline[-1].n_batch):
        print("Label: {}".format(label))
        adata_tmp = NF.reconstruct_debatched(dataset, pipeline, transform_to_batch_label=label,layer_key_added=f'scvi_reconstructed_{label}', device=device)
        #if adata is None:
            #adata = NF.reconstruct_debatched(dataset, pipeline, transform_to_batch_label=label,layer_key_added=f'scvi_reconstructed_{label}', device=device)
        #else:
            # reconstruct_debatched(dataset, pipeline, transform_to_batch_label=label, layer_key_added=f'scvi_reconstructed_{label}')
        adata.layers[f'scvi_reconstructed_{label}'] = adata_tmp.layers[f'scvi_reconstructed_{label}']
        break
    adata.write(filepath)
else:
    print("Reading file : {}".format(filepath))
    adata = sc.read(filepath)


filepath = f"tmp_data/{foldername}/adata_scvi_reconstructed_raw_umap{filename_suffix}.h5ad"
if not os.path.exists(filepath) or overwrite:
    adata = NF.plot_raw_data(adata,filepath,figpath,color_keys)
else:
    print("Reading file : {}".format(filepath))
    adata = sc.read(filepath)


filepath = f"tmp_data/{foldername}/adata_scvi_reconstructed_raw_scvi_umaps{filename_suffix}.h5ad"
if not os.path.exists(filepath) or overwrite:
    NF.plot_latent_representation(adata,filepath, figpath,color_keys)
else:
    print("Reading file : {}".format(filepath))
    adata = sc.read(filepath)

adata.layers['raw'] = adata.X.copy()

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

#gene_set  = "MIR892B,MIR873,USH1C,CTDSP2,RAD50,FEM1B,CDK2,CDK3,CDK4,PSME3,CDK5,CDK6,CTDSPL,CDK7,CDKN1A,CDK2AP2,CDKN1B,AKAP8,CDKN1C,CDKN2A,CDKN2B,CCNO,CDKN2C,CDKN2D,CDKN3,NPM2,BTN2A2,NDC80,GPNMB,MAD2L2,TACC3,UBD,CENPE,CENPF,KHDRBS1,NES,PLK2,ARPP19,NEK6,DBF4,CCNI,RCC1,PIM2,UBE2C,TOPBP1,CHEK1,FOXN3,ZWINT,FAM107A,CHEK2,TREX1,CDCA5,ECD,PHB2,PABIR1,CKS1B,CKS2,PRAP1,DCUN1D3,PLK3,PLK5,IQGAP3,CHMP4B,TRIM71,TPRA1,LSM11,HUS1B,NACC2,ATF2,CRY1,MAPK14,CACUL1,E2F7,RAD9B,EME1,SPC24,DTX3L,NEK10,CYP1A1,SASS6,SDE2,DDB1,DDX3X,CDC14C,DLG1,DNA2,DNM2,DUSP1,E2F1,E2F2,E2F3,E2F4,E2F6,EGFR,ARID2,EME2,EIF4E,EIF4EBP1,EIF4G1,AIF1,ENSA,EPS8,ERCC2,AKT1,ERCC3,ERCC6,ESRRB,EZH2,FANCD2,STOX1,CCNY,FGF10,MAPK15,FHL1,ATF5,SIRT2,PAXIP1,MYO16,FOXM1,CLASP2,PHF8,SMC5,FBXL7,PLCB1,KLHL18,CLASP1,UFL1,BRD4,SPDYA,STXBP4,MBLAC1,FBXO7,ZNF324,INTS7,ANAPC15,SIN3A,SYF2,KANK2,HINFP,ANKRD17,GIGYF2,APPL1,TIPRL,FBXO6,FBXO5,FBXO4,LATS2,GFI1,AKAP8L,MTBP,KCNH5,VPS4A,CHMP2A,UBE2S,PRPF19,GLI1,GML,CCDC57,NSMCE2,RGCC,BABAM1,BRD7,GSPT1,TMOD3,ANAPC2,GPR132,RPA4,ANAPC4,DONSON,NOP53,ANXA1,H2AX,APBB1,APBB2,APC,PRMT2,HSPA2,BIRC5,HUS1,HYAL1,ID2,ID4,GEN1,APP,IK,INCENP,INHBA,ITGB1,KCNA5,USP17L2,GPR15LG,NANOGP8,MIR10A,MIR133A1,MIR137,MIR15A,MIR15B,MIR16-1,MIR193A,MIR195,MIR19B1,MIR208A,MIR214,MIR221,MIR222,MIR26A1,MIR29A,MIR29B1,MIR29C,MIR30C2,MAD2L1,MDM2,MECP2,MLF1,MAP3K11,FOXO4,MN1,MNAT1,MOS,MRE11,EIF2AK4,MIR133B,MIR372,MSH2,MUC1,MYC,NBN,ATM,NEUROG1,NFIA,NFIB,NPAT,NPM1,DDR2,ATP2B4,CRNN,ORC1,OVOL1,RRM2B,PBX1,RPS27L,DYNC1LI1,ING4,MRNIP,CDK16,TFDP3,CDK17,CDK18,WAC,DACT1,FZR1,TAOK3,MBTPS2,CRLF3,PPME1,ACTL6B,ANAPC5,ANAPC7,LCMT1,TRIAP1,GTSE1,DTL,ANAPC11,SIRT7,METTL13,CPSF3,UIMC1,MAP3K20,CDK14,ABCB1,PKD1,PKD2,PLCG2,PLK1,PLRG1,PML,POLE,ANLN,ETAA1,ATR,INO80,CCNJ,PAF1,NSUN2,SPDL1,TIPIN,PINX1,USP47,ZWILCH,PPP1R10,CDCA8,PPP2CA,RFWD3,PBRM1,APPL2,PHF10,FBXW7,PPP3CA,PIDD1,PPP6C,AMBRA1,PKIA,CHFR,CDK5RAP2,RIOK2,PCID2,CENPJ,PPP2R2D,KMT2E,PRKDC,RCC2,TEX14,SUSD2,MEPCE,PROX1,TRIM39,TCIM,PSMG2,KNL1,AVEN,BACH1,GJC2,PSME1,PSME2,PTEN,MIR362,SPC25,MIR451A,MIR495,MIR515-1,MIR520A,MIR519D,MIR520H,MIR503,ARID1B,MTA3,HECW2,RPTOR,TAOK1,USP28,CAMSAP3,USP29,USP37,PTPN6,CCAR2,PTPN11,PTPRC,BARD1,RAD1,CTDSP1,RHOU,INIP,BCAT1,RAD9A,RAD17,RAD21,RAD51,RAD51C,RAD51B,RB1,RBBP8,RBL1,RBL2,CCND1,BCL2,RDX,UPF1,DPF2,RFPL1,ACTB,BCL7A,RINT1,MIIP,RPA2,RPL24,RPL26,RPS6,RPS6KB1,RRM1,RRM2,CCL2,BID,CLSPN,BLM,SETMAR,CCNI2,ANAPC1,NABP1,STIL,SIX3,SKP2,CDK15,INTS3,SMARCA2,SMARCA4,STK33,SMARCB1,SMARCC1,DDRGK1,SMARCC2,SMARCD1,SMARCD2,SMARCD3,SMARCE1,SOX2,SPAST,BRCA1,BRCA2,ZFP36L1,ZFP36L2,AURKA,ADAM17,TAF1,TAF2,TAF10,TBX2,MIR638,BUB1,BUB1B,TERT,TFAP4,TFDP1,TP53,TP53BP1,TPD52L1,TPR,TTK,UBE2A,UBE2E2,UBE2L3,WEE1,WNT10B,XPC,XRCC3,ZNF16,ZNF207,CACNB4,ZNF655,NABP2,FBXL15,BRCC3,DDX39B,PAGR1,CDC73,CCNJL,RNASEH2B,FBXO31,KDM8,NEK11,ATAD5,CCNP,JADE1,WDR76,CALM1,CTC1,DBF4B,TTI2,MUS81,CEP63,CDK5RAP3,CALM2,CUL5,CALM3,DPF3,CAMK2A,FAM83D,CDT1,TMEM14B,DPF1,ARID1A,CDC7,CDC45,GFI1B,CASP2,NUF2,PARP9,RHNO1,MAD1L1,USP26,HASPIN,BRIP1,HORMAD1,USP44,ATRIP,ABRAXAS1,DYRK3,DOT1L,BRSK1,CUL4B,CUL4A,CUL3,CUL2,CUL1,KLF11,PPP1R9B,KLHL22,PPM1D,MASTL,ZFYVE19,LSM10,CCNB3,PIAS1,CDC14B,CDC14A,CDK10,THOC5,ACTL6A,CDC23,MBTPS1,CRADD,RAB11A,IER3,CDC16,ZPR1,NAE1,CCNA2,CCNA1,CCNB1,TIMELESS,PHOX2B,CCND2,CCND3,CCNE1,CCNF,ACVR1,CCNG1,CCNG2,CCNH,BRSK2,TM4SF5,TICRR,PKMYT1,ACVR1B,LATS1,CCNB2,CCNE2,CTDP1,ZNF830,SLFN11,CHMP7,ZW10,BUB3,CCNQ,AURKB,CHMP4C,BCL7C,BCL7B,KLF4,TRIP13,TAOK2,ADAMTS1,VPS4B,MACROH2A1,CLOCK,BABAM2,MAD2L1BP,MDC1,TTI1,ESPL1,KNTC1,DLGAP5,CDK1,MELK,CDC5L,TELO2,CDC6,CDC20,KIF14,CDC25A,CDC25B,CDC25C,CDC27,CDC34,THOC1".split(',')
gene_set = []
list(map(gene_set.extend, list(gene_set_dict.values())))

print(adata)

exit()

if foldername in ["pmbc_results"]:
    adata.var['genes_of_interest'] = adata.var_names.isin(gene_set) #23 glycogenes found only adata[:,adata.var_names.isin(gene_set)]
    adata_tmp = adata[:, adata.var_names.isin(gene_set)] #only used for counting
    print(adata_tmp.var_names.tolist())
elif foldername in ["tucker_heart_atlas"]:
    adata.var["genes_of_interest"] = adata.var["gene_names"].isin(gene_set)
    adata_tmp = adata[:, adata.var["gene_names"].isin(gene_set)]
    print(adata_tmp.var["gene_names"].tolist())
elif foldername in ["gut_cell_atlas_normed"]:
    adata.var["genes_of_interest"] = adata.var["gene_ids"].isin(gene_set)
    adata_tmp = adata[:, adata.var["gene_ids"].isin(gene_set)]
    print(adata_tmp.var["gene_ids"].tolist())

else:
    adata.var["genes_of_interest"] = adata.var["gene_name-new"].isin(gene_set)
    adata_tmp = adata[:, adata.var["gene_name-new"].isin(gene_set)]
    print(adata_tmp.var["gene_name-new"].tolist())


exit()






#aggregate umi-count expression values
adata.var['expr'] = np.array(adata.layers['raw'].sum(axis=0)).squeeze()
high_gene_set = adata.var.sort_values(by='expr').index[-50:]
low_gene_set = adata.var.sort_values(by='expr').index[:500]
adata.var['low_exp_genes_of_interest'] = adata.var_names.isin(low_gene_set)
adata.var['high_exp_genes_of_interest'] = adata.var_names.isin(high_gene_set)

high_gene_set = adata.var.sort_values(by='expr').index[-50:]
low_gene_set = adata.var.sort_values(by='expr').index[:500]

#adata = adata[:1000]

filepath = f"tmp_data/{foldername}/adata_scvi_reconstructed_raw_scvi_umaps_layers{filename_suffix}.h5ad"
if not os.path.exists(filepath) or overwrite:
    adata = NF.umap_group_genes(adata,filepath)
else:
    adata = sc.read(filepath)


#adata_normalized= sc.pp.normalize_total(adata, target_sum=1, inplace=False, exclude_highly_expressed=False)

#figname = "glyco_expression_umap.pdf"
# NF.plot_avg_expression(adata,"X_raw_umap",gene_set_dict,figpath,figname)
#figname = "glyco_expression_reconstructed.pdf"
# NF.plot_avg_expression(adata,"X_scvi_reconstructed_0_umap",gene_set_dict,figpath,figname)

NF.plot_neighbour_clusters(adata,gene_set,figpath)

exit()

if config_dict['model']['model']['init_args']['batch_embedded']:
    # show batch embeddings as an image
    batch_key = config_dict['data']['batch_keys']['batch_index_n']['key']
    batch_names = adata.obs[batch_key].cat.categories
    plt.imshow(scvi_model.batch_representation_mean_bd.detach().cpu().numpy())
    plt.xlabel('embedding dimension')
    plt.yticks(ticks=range(len(batch_names)), labels=batch_names)
    plt.ylabel('batch index')
    plt.grid(False)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(f"figures/{foldername}/batch_index{filename_suffix}.pdf")
    plt.clf()
    plt.close()

    # show batch embeddings as a umap
    um = umap.UMAP(n_neighbors=3, metric='cosine').fit_transform(scvi_model.batch_representation_mean_bd.detach().cpu().numpy())
    plt.plot(um[:, 0], um[:, 1], '.')
    for x, y, s in zip(um[:, 0], um[:, 1], batch_names):
        plt.annotate(str(s), (x, y), fontsize=10)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('Batch embeddings')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"figures/{foldername}/batch_embeddings{filename_suffix}.pdf")
    plt.clf()
    plt.close()

else:
    print('Model was run with `batch_embedded: false` in the config under model.model.init_args, so we did not learn an embedding of batch.')