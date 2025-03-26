import os
import argparse
import math
import torch
import pickle
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from braceexpand import braceexpand
from tqdm import tqdm
import multiprocessing as mp

# for flex attention
import torch._dynamo
import torch.multiprocessing as mp 
torch._dynamo.config.suppress_errors = True

sc.set_figure_params(figsize=(4, 4))

from cellarium.ml.utilities.inference.cellarium_gpt_inference import \
    CellariumGPTInferenceContext, \
    GeneNetworkAnalysisBase

DEVICES = [torch.device(f'cuda:{idx}') for idx in range(torch.cuda.device_count())]

ROOT_PATH = "/work/hdd/bbjr/mallina1/data/mb-ml-dev-vm"
TRAIN_ROOT_PATH = "/work/hdd/bbjr/mallina1/data/human_cellariumgpt_v2/extract_files"
CHECKPOINT_PATH = "/work/hdd/bbjr/mallina1/cellarium/models/latest/epoch=5-step=504000.ckpt"
OUTPUT_PATH = "/work/hdd/bbjr/mallina1/data/human_cellariumgpt_v2/activations"

REF_ADATA_PATH = os.path.join(ROOT_PATH, "data", "extract_0.h5ad")
GENE_INFO_PATH = os.path.join(ROOT_PATH, "gene_info", "gene_info.tsv")

def process_arg_on_gpu(arg):
    metadata_prompt_dict = {
        "cell_type": False,
        "tissue": False,
        "disease": False,
        "sex": False,
        "development_stage": False
    }
    (train_fname, ctx, tr_adata, indices, batch_size, output_fp) = arg

    for batch_idx in tqdm(range(0, len(indices), batch_size)):
        obs_index = [indices[batch_idx]]
        if len(indices) > batch_idx + 1:
            obs_index.append(indices[batch_idx+1])
        if len(indices) > batch_idx + 2:
            obs_index.append(indices[batch_idx+2])

        tokens_dict, context_indices = ctx.generate_tokens_from_adata(tr_adata, 
                                                                      obs_index=obs_index, 
                                                                      query_var_names=[],
                                                                      metadata_prompt_masks_dict=metadata_prompt_dict)
        query_cell_type_idx = context_indices['query_cell_type']
        query_disease_idx = context_indices['query_disease']
        query_tissue_idx = context_indices['query_tissue']
        query_sex_idx = context_indices['query_sex']
        query_development_stage_idx = context_indices['query_development_stage']

        with torch.inference_mode():
            hidden_states = ctx.get_embeddings_from_tokens(tokens_dict, context_indices, to_cpu=True)

        # shape: (layers, batch_size, context_length, hidden_dim)
        hidden_states = np.stack(hidden_states)

        if os.path.exists(output_fp):
            curr_data = np.load(output_fp, allow_pickle=True)

            curr_cell_type_activations = curr_data['cell_type_activations']
            curr_disease_activations = curr_data['disease_activations']
            curr_tissue_activations = curr_data['tissue_activations']
            curr_sex_activations = curr_data['sex_activations']
            curr_development_stage_activations = curr_data['development_stage_activations']
            
            curr_metadata = curr_data['metadata'].item()
            
            curr_cell_type_activations = np.concatenate((curr_cell_type_activations, hidden_states[:, :, query_cell_type_idx, :].squeeze()), axis=1)
            curr_disease_activations = np.concatenate((curr_disease_activations, hidden_states[:, :, query_disease_idx, :].squeeze()), axis=1)
            curr_tissue_activations = np.concatenate((curr_tissue_activations, hidden_states[:, :, query_tissue_idx, :].squeeze()), axis=1)
            curr_sex_activations = np.concatenate((curr_sex_activations, hidden_states[:, :, query_sex_idx, :].squeeze()), axis=1)
            curr_development_stage_activations = np.concatenate((curr_development_stage_activations, hidden_states[:, :, query_development_stage_idx, :].squeeze()), axis=1)            
            
            curr_data.close()
        else:
            curr_cell_type_activations = hidden_states[:, :, query_cell_type_idx, :].squeeze()
            curr_disease_activations = hidden_states[:, :, query_disease_idx, :].squeeze()
            curr_tissue_activations = hidden_states[:, :, query_tissue_idx, :].squeeze()
            curr_sex_activations = hidden_states[:, :, query_sex_idx, :].squeeze()
            curr_development_stage_activations = hidden_states[:, :, query_development_stage_idx, :].squeeze()
            curr_metadata = {
                'cell_type_labels': [],
                'disease_labels': [],
                'suspension_type_labels': [],
                'assay_labels': [],
                'sex_labels': []
            }

        for idx, jdx in enumerate(obs_index):
            cell_type_label = tr_adata.obs.iloc[jdx].cell_type
            disease_label = tr_adata.obs.iloc[jdx].disease
            susp_type_label = tr_adata.obs.iloc[jdx].suspension_type
            assay_label = tr_adata.obs.iloc[jdx].assay
            sex_label = tr_adata.obs.iloc[jdx].sex

            curr_metadata['cell_type_labels'].append(cell_type_label)
            curr_metadata['disease_labels'].append(disease_label)
            curr_metadata['suspension_type_labels'].append(susp_type_label)
            curr_metadata['assay_labels'].append(assay_label)
            curr_metadata['sex_labels'].append(sex_label)

        np.savez(output_fp, cell_type_activations=curr_cell_type_activations, 
                            disease_activations=curr_disease_activations,
                            sex_activations=curr_sex_activations,
                            tissue_activations=curr_tissue_activations,
                            development_stage_activations=curr_development_stage_activations,
                            metadata=curr_metadata)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fname', default='extract_0.h5ad', type=str)
    cfg = parser.parse_args()

    mp.set_start_method('spawn')
    contexts = []
    for DEVICE in DEVICES:
        ctx = CellariumGPTInferenceContext(
            cellarium_gpt_ckpt_path=CHECKPOINT_PATH,
            ref_adata_path=REF_ADATA_PATH,
            gene_info_tsv_path=GENE_INFO_PATH,
            device=DEVICE,
            attention_backend="mem_efficient"
        )
        contexts.append(ctx)

    # train_fnames = list(braceexpand('extract_{0..1}.h5ad'))
    # train_fnames = ['extract_0.h5ad']
    train_fnames = [cfg.train_fname]

    batch_size = 3
    for train_fname in train_fnames:
        print(f'Processing {train_fname}...')
        tr_adata = sc.read_h5ad(os.path.join(TRAIN_ROOT_PATH, train_fname))
        total_n = tr_adata.obs.shape[0]

        per_gpu_n = math.floor(total_n / len(DEVICES))
        leftover_n = total_n % len(DEVICES)
        args = []
        start_idx = 0
        for idx in range(len(DEVICES)):
            indices = list(range(start_idx, start_idx + per_gpu_n))
            start_idx += per_gpu_n
            if leftover_n > 0 and idx == len(DEVICES) - 1:
                indices += [x for x in range(start_idx, total_n)]

            output_fp = os.path.join(OUTPUT_PATH, f'activations_{train_fname[:-5]}_part_{idx}.npz')
            arg = (train_fname, contexts[idx], tr_adata, indices, batch_size, output_fp)

            args.append(arg)

        with mp.Pool(len(DEVICES)) as p:
            # results = list(tqdm(p.imap(process_arg_on_gpu, args), total=len(DEVICES)))
            p.map(process_arg_on_gpu, args)

if __name__=='__main__':
    main();
