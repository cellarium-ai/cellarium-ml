#!/usr/bin/env python3
import os
import torch
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
import pickle
import logging
import argparse

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Suppress a specific warning from AnnData
warnings.filterwarnings("ignore", category=UserWarning, message="Transforming to str index.")

# Import required modules from the cellarium package
from cellarium.ml.utilities.inference.cellarium_gpt_inference import (
    CellariumGPTInferenceContext
)
from cellarium.ml.utilities.inference.metadata_benchmarking.calculate_metrics import (
    calculate_metrics_for_prediction_output
)

def propagate_probs_over_ontology(
        class_probs_nk: np.ndarray,
        class_names_k: t.List[str],
        ontology_resource_dict: dict[str, t.Any],
) -> t.Tuple[t.List[str], np.ndarray]:
    assert len(class_names_k) == class_probs_nk.shape[1]
    all_class_names_q = sorted(ontology_resource_dict.keys())
    propagated_class_probs_nq = np.zeros((class_probs_nk.shape[0], len(all_class_names_q)))

    given_class_name_to_idx = {class_name: idx for idx, class_name in enumerate(class_names_k)}    
    for q, class_name in enumerate(all_class_names_q):
        for descendant_name in ontology_resource_dict[class_name]["all_descendants"]:
            if descendant_name in given_class_name_to_idx:
                propagated_class_probs_nq[:, q] += class_probs_nk[:, given_class_name_to_idx[descendant_name]]
    return all_class_names_q, propagated_class_probs_nq

def convert_meta_adata_to_query_obj_for_scoring(
        meta_adata: sc.AnnData,
        metadata_key: str,
        scores_col_name_suffix: str = "class_probs",
        ontology_term_ids_uns_key_name_suffix: str = "ontology_term_ids",
        ontology_term_id_obs_col_name_suffix: str = "ontology_term_id",
    ) -> t.Tuple[t.List[dict], t.List[str]]:
    scores_col_name = f"{metadata_key}_{scores_col_name_suffix}"
    ontology_term_ids_uns_key_name = f"{metadata_key}_{ontology_term_ids_uns_key_name_suffix}"
    ontology_term_id_obs_col_name = f"{metadata_key}_{ontology_term_id_obs_col_name_suffix}"
    assert scores_col_name in meta_adata.obsm
    assert ontology_term_ids_uns_key_name in meta_adata.uns
    assert ontology_term_id_obs_col_name in meta_adata.obs.columns
    query_objs = []
    ground_truth_ontology_term_ids = []
    obs_index = meta_adata.obs.index.values
    for i_cell in range(len(meta_adata)):
        obs_row = meta_adata.obs.iloc[i_cell]
        ground_truth_ontology_term_id = obs_row[ontology_term_id_obs_col_name]
        query_obj = dict()
        query_obj["query_cell_id"] = obs_index[i_cell]
        query_obj["matches"] = []
        for ontology_term_id, score in zip(
                meta_adata.uns[ontology_term_ids_uns_key_name],
                meta_adata.obsm[scores_col_name][i_cell, :]):
            query_obj["matches"].append({
                "ontology_term_id": ontology_term_id,
                "score": score,
            })
        query_objs.append(query_obj)
        ground_truth_ontology_term_ids.append(ground_truth_ontology_term_id)
    return query_objs, ground_truth_ontology_term_ids

def main():
    parser = argparse.ArgumentParser(
        description="Cellarium GPT Metadata Prediction CLI Tool"
    )
    parser.add_argument(
        "--cuda_device_index", type=int, default=0,
        help="CUDA device index to use (default: 0)"
    )
    parser.add_argument(
        "--val_adata_path", type=str,
        default="/home/mehrtash/data/data/extract_0.h5ad",
        help="Validation AnnData path"
    )
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="/home/mehrtash/data/100M_long_run/run_001/lightning_logs/version_3/checkpoints/epoch=5-step=504000.ckpt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--ref_adata_path", type=str,
        default="/home/mehrtash/data/data/extract_0.h5ad",
        help="Path to the reference AnnData file"
    )
    parser.add_argument(
        "--gene_info_path", type=str,
        default="/home/mehrtash/data/gene_info/gene_info.tsv",
        help="Path to the gene info TSV file"
    )
    parser.add_argument(
        "--ontology_resource_path", type=str,
        default="/home/mehrtash/data/data/cellariumgpt_artifacts/ontology",
        help="Path to the ontology resources"
    )
    parser.add_argument(
        "--output_path", type=str,
        default="/home/mehrtash/data/data/cellariumgpt_artifacts/metadata_predictions/100M_long_run_last",
        help="Directory where output results will be saved"
    )
    parser.add_argument(
        "--rng_seed", type=int, default=42,
        help="Random number generator seed (default: 42)"
    )
    parser.add_argument(
        "--n_cells", type=int, default=None,
        help="Number of cells to use (default: None, meaning all)"
    )
    # Updated default for n_genes to match the new notebook and added new argument for gene selection method
    parser.add_argument(
        "--n_genes", type=int, default=None,
        help="Number of genes to use. For 'highly_expressed', selects top genes; for 'random', selects random genes. Set to None to use all."
    )
    parser.add_argument(
        "--gene_selection_method", type=str, default="random",
        help="Gene selection method: 'random' or 'highly_expressed' (default: random)"
    )
    parser.add_argument(
        "--rand_prompt_vars_sublist_path", type=str,
        default="/home/mehrtash/data/data/cellariumgpt_artifacts/autosomal_gene_ids.txt",
        help="Gene list to select random genes from."
    )
    parser.add_argument(
        "--fixed_prompt_vars_sublist_path", type=str,
        default="/home/mehrtash/data/data/cellariumgpt_artifacts/sex_gene_ids.txt",
        help="Gene list to select fixed genes from."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16,
        help="Chunk size for processing (default: 16)"
    )
    parser.add_argument(
        "--metric_style", type=str, default="hop_k_call",
        help="Performance metric style: 'hop_k_sensitivity_precision', 'hop_k_inclusion', or 'hop_k_call' (default: hop_k_call)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    logger.info("Loading ontology resources ...")
    ontology_benchmarking_resource_path_dict = {
        'cell_type': os.path.join(args.ontology_resource_path, 'cl_benchmarking_resource.pkl'),
        'development_stage': os.path.join(args.ontology_resource_path, 'hsapdv_benchmarking_resource.pkl'),
        'disease': os.path.join(args.ontology_resource_path, 'mondo_benchmarking_resource.pkl'),
        'tissue': os.path.join(args.ontology_resource_path, 'uberon_benchmarking_resource.pkl'),
        'sex': os.path.join(args.ontology_resource_path, 'sex_benchmarking_resource.pkl'),
    }

    ontology_propagation_resource_path_dict = {
        'cell_type': os.path.join(args.ontology_resource_path, 'cl_propagation_resource.pkl'),
        'development_stage': os.path.join(args.ontology_resource_path, 'hsapdv_propagation_resource.pkl'),
        'disease': os.path.join(args.ontology_resource_path, 'mondo_propagation_resource.pkl'),
        'tissue': os.path.join(args.ontology_resource_path, 'uberon_propagation_resource.pkl'),
        'sex': os.path.join(args.ontology_resource_path, 'sex_propagation_resource.pkl'),
    }

    # Define number of hops for each metadata key
    n_hops_dict = {
        'cell_type': 3,
        'development_stage': 3,
        'disease': 3,
        'tissue': 3,
        'sex': 0,
    }

    # Load the benchmarking ontology resources
    ontology_benchmarking_resource_dicts = {}
    for meta_key, path in ontology_benchmarking_resource_path_dict.items():
        with open(path, "rb") as f:
            ontology_benchmarking_resource_dicts[meta_key] = pickle.load(f)

    # Load the propagation ontology resources into a separate dictionary
    ontology_propagation_resource_dicts = {}
    for meta_key, path in ontology_propagation_resource_path_dict.items():
        with open(path, "rb") as f:
            ontology_propagation_resource_dicts[meta_key] = pickle.load(f)

    logger.info(f"Loading the model checkpoint from {args.checkpoint_path} ...")
    device = torch.device(f"cuda:{args.cuda_device_index}")

    ctx = CellariumGPTInferenceContext(
        cellarium_gpt_ckpt_path=args.checkpoint_path,
        ref_adata_path=args.ref_adata_path,
        gene_info_tsv_path=args.gene_info_path,
        device=device,
        attention_backend="mem_efficient",
        verbose=False
    )
    logger.info(f"Loading the validation AnnData object from {args.val_adata_path} ...")
    adata = sc.read_h5ad(args.val_adata_path)
    
    if args.n_genes is not None:

        with open(args.fixed_prompt_vars_sublist_path, "r") as f:
            fixed_prompt_var_names_sublist = f.read().splitlines()
        logger.info(f"Starting with {len(fixed_prompt_var_names_sublist)} fixed genes.")

        if args.gene_selection_method == "highly_expressed":
            logger.info(f"In addition, using up to {args.n_genes} highly expressed genes.")
            X_g = np.asarray(adata.X.sum(0)).flatten()
            highly_expressed_gene_indices = np.argsort(X_g)[::-1]
            selected_gene_set = set(fixed_prompt_var_names_sublist)
            target_n_genes = args.n_genes + len(fixed_prompt_var_names_sublist)
            for idx in highly_expressed_gene_indices:
                if len(selected_gene_set) >= target_n_genes:
                    break
                gene_id = adata.var_names[highly_expressed_gene_indices[idx]]
                selected_gene_set.add(gene_id)
            logger.info(f"Selected {len(selected_gene_set)} genes.")
            fixed_prompt_var_names_sublist = list(selected_gene_set)
            rand_prompt_var_names_sublist = []
            n_rand_prompt_vars = 0
            torch_rng = torch.Generator().manual_seed(args.rng_seed)
        
        elif args.gene_selection_method == "random":
            logger.info(f"In addition, using {args.n_genes} random genes (seed = {args.rng_seed}).")
            torch_rng = torch.Generator().manual_seed(args.rng_seed)
            n_rand_prompt_vars = args.n_genes
            with open(args.rand_prompt_vars_sublist_path, "r") as f:
                rand_prompt_var_names_sublist = f.read().splitlines()
        
        else:
            raise ValueError(f"Unknown gene selection method: {args.gene_selection_method}")
    else:
        logger.info(f"Using all genes.")
        n_rand_prompt_vars = None
        rand_prompt_var_names_sublist = None
        fixed_prompt_var_names_sublist = None
        
    if args.n_cells is None:
        logger.info(f"Using all cells.")
    else:
        n_cells = min(args.n_cells, len(adata))
        logger.info(f"Using {n_cells} random cells (seed = {args.rng_seed}).")
        rng = np.random.RandomState(args.rng_seed)
        adata = adata[rng.choice(len(adata), n_cells, replace=False)]

    logger.info(f"Predicting metadata for {len(adata)} cells ...")
    preds = ctx.predict_metadata_chunked(
        adata=adata,
        chunk_size=args.chunk_size,
        n_rand_prompt_vars=n_rand_prompt_vars,
        rand_prompt_var_names_sublist=rand_prompt_var_names_sublist,
        fixed_prompt_var_names_sublist=fixed_prompt_var_names_sublist,
        rng=torch_rng
    )

    logger.info("Inserting predictions into an AnnData object ...")
    meta_adata = sc.AnnData(obs=adata.obs.copy())

    # Updated to include logits and compute class probabilities via np.exp()
    for meta_key, meta_preds in preds.items():
        meta_adata.obsm[meta_key + "_class_logits"] = meta_preds
        meta_adata.obsm[meta_key + "_class_probs"] = np.exp(meta_preds)
        meta_adata.uns[meta_key + "_ontology_term_ids"] = ctx.metadata_ontology_infos[meta_key]["names"]
        meta_adata.uns[meta_key + "_labels"] = ctx.metadata_ontology_infos[meta_key]["labels"]

    for meta_key, ontology_resource_dict in ontology_benchmarking_resource_dicts.items():
        class_probs_nk = meta_adata.obsm[meta_key + "_class_probs"]
        all_class_names_q, propagated_class_probs_nq = propagate_probs_over_ontology(
            class_probs_nk=class_probs_nk,
            class_names_k=meta_adata.uns[meta_key + "_ontology_term_ids"],
            ontology_resource_dict=ontology_resource_dict,
        )
        meta_adata.obsm[meta_key + "_propagated_class_probs"] = propagated_class_probs_nq
        meta_adata.uns[meta_key + "_propagated_ontology_term_ids"] = all_class_names_q
        meta_adata.uns[meta_key + "_propagated_labels"] = list(
            map(ontology_propagation_resource_dicts[meta_key]['ontology_term_id_to_label'].get, all_class_names_q)
        )
    meta_keys = ontology_benchmarking_resource_dicts.keys()

    if args.metric_style in {"hop_k_sensitivity_precision"}:
        logger.info(f"Calculating performance metrics using {args.metric_style} style ...")
        scores_col_name_suffix = "propagated_class_probs"
        ontology_term_ids_uns_key_name_suffix = "propagated_ontology_term_ids"
        ontology_term_id_obs_col_name_suffix: str = "ontology_term_id"
    elif args.metric_style in {"hop_k_inclusion", "hop_k_call"}:
        logger.info(f"Calculating performance metrics using {args.metric_style} style ...")
        scores_col_name_suffix = "class_probs"
        ontology_term_ids_uns_key_name_suffix = "ontology_term_ids"
        ontology_term_id_obs_col_name_suffix: str = "ontology_term_id"
    else:
        raise ValueError(f"Unknown metric style: {args.metric_style}")

    results_dfs = []
    for meta_key in meta_keys:
        logger.info(f"Calculating performance metrics for {meta_key} ...")
        query_objs, ground_truth_ontology_term_ids = convert_meta_adata_to_query_obj_for_scoring(
            meta_adata=meta_adata,
            metadata_key=meta_key,
            scores_col_name_suffix=scores_col_name_suffix,
            ontology_term_ids_uns_key_name_suffix=ontology_term_ids_uns_key_name_suffix,
            ontology_term_id_obs_col_name_suffix=ontology_term_id_obs_col_name_suffix)
        results_df = calculate_metrics_for_prediction_output(
            model_predictions=query_objs,
            ground_truth_ontology_term_ids=ground_truth_ontology_term_ids,
            ontology_resource=ontology_benchmarking_resource_dicts[meta_key],
            num_hops=n_hops_dict[meta_key],
            metric_style=args.metric_style)
        results_df.columns = [
            f"{meta_key}_{col}" if col != "query_cell_id" else col 
            for col in results_df.columns
        ]
        results_dfs.append(results_df)

    final_results_df = pd.concat(results_dfs, axis=1)
    meta_adata.obs.index.name = "query_cell_id"
    meta_adata.obs = pd.concat([meta_adata.obs, final_results_df], axis=1)
    output_prefix = os.path.splitext(os.path.basename(args.val_adata_path))[0]
    meta_adata_output_file_path = os.path.join(
        args.output_path, f"{output_prefix}_metadata_prediction_scores.h5ad"
    )
    logger.info(f"Saving the results to {meta_adata_output_file_path} ...")
    meta_adata.write_h5ad(meta_adata_output_file_path)
    logger.info("Done!")

if __name__ == "__main__":
    main()
