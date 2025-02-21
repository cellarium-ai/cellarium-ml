# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python
import argparse
import os
import pickle

import numpy as np
import scanpy as sc
import torch

from cellarium.ml.utilities.inference.cellarium_gpt_inference import CellariumGPTInferenceContext
from cellarium.ml.utilities.linreg import batch_linear_regression


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command line tool for gene expression range and linear response analysis using CellariumGPT."
    )

    # Global parameters
    parser.add_argument("--cuda_device_index", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/mehrtash/data/100M_long_run/run_001/lightning_logs/version_3/checkpoints/epoch=5-step=504000.ckpt",
        help="Path to CellariumGPT checkpoint",
    )
    parser.add_argument(
        "--ref_adata_path",
        type=str,
        default="/home/mehrtash/data/data/extract_0.h5ad",
        help="Path to reference AnnData file",
    )
    parser.add_argument(
        "--gene_info_tsv_path",
        type=str,
        default="/home/mehrtash/data/gene_info/gene_info.tsv",
        help="Path to gene info TSV",
    )
    parser.add_argument(
        "--validation_adata_path",
        type=str,
        default="/home/mehrtash/data/data/cellariumgpt_artifacts/cell_types_for_validation_filtered.h5ad",
        help="Path to validation AnnData file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/mehrtash/data/data/linear_response/100M_long_run_last_checkpoint",
        help="Directory for output files",
    )
    parser.add_argument(
        "--cell_index", type=int, default=0, help="Index of the cell in the validation AnnData to analyze"
    )

    # Gene expression range determination parameters
    parser.add_argument(
        "--query_chunk_size", type=int, default=1000, help="Chunk size for gene expression query (default: 1000)"
    )
    parser.add_argument("--total_prob_mass", type=float, default=0.5, help="Total probability mass (default: 0.5)")
    parser.add_argument("--max_counts", type=int, default=1000, help="Maximum counts (default: 1000)")
    parser.add_argument("--symmetric_range_pad", type=int, default=1, help="Symmetric range pad (default: 1)")
    parser.add_argument(
        "--max_query_genes",
        type=int,
        default=None,
        help="Maximum number of query genes (default: None, meaning use all)",
    )
    parser.add_argument(
        "--total_mrna_umis",
        type=float,
        default=None,
        help="Total mRNA UMIs (default: None, will be taken from the AnnData)",
    )

    # Linear response analysis parameters
    parser.add_argument(
        "--n_points", type=int, default=5, help="Number of points for the linear response analysis (default: 5)"
    )
    parser.add_argument(
        "--query_chunk_size_linear_response",
        type=int,
        default=64,
        help="Query chunk size for linear response analysis (default: 64)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Load the validation AnnData
    print(f"Loading validation AnnData from {args.validation_adata_path} ...")
    val_adata = sc.read_h5ad(args.validation_adata_path)
    print(f"Total number of cells in validation AnnData: {val_adata.shape[0]}")

    if args.cell_index >= val_adata.shape[0]:
        raise ValueError("cell_index is out of range")

    print(f"Selected cell index: {args.cell_index}")
    val_adata_row = val_adata.obs.iloc[args.cell_index]
    print(val_adata_row)

    # Set device and load the checkpoint
    print("Loading CellariumGPT checkpoint ...")
    device = torch.device(f"cuda:{args.cuda_device_index}")

    ctx = CellariumGPTInferenceContext(
        cellarium_gpt_ckpt_path=args.checkpoint_path,
        ref_adata_path=args.ref_adata_path,
        gene_info_tsv_path=args.gene_info_tsv_path,
        device=device,
        attention_backend="mem_efficient",
        verbose=False,
    )

    all_query_gene_ids = val_adata.var["feature_id"].values
    print(f"Total number of query genes from validation AnnData: {len(all_query_gene_ids)}")

    if args.max_query_genes is not None:
        query_gene_ids = all_query_gene_ids[: args.max_query_genes]
        print(f"Limiting to {args.max_query_genes} first query genes for linear response analysis.")
    else:
        query_gene_ids = all_query_gene_ids
        print(f"Using all {len(all_query_gene_ids)} query genes for linear response analysis.")

    assay = val_adata_row.assay
    suspension_type = val_adata_row.suspension_type
    prompt_metadata_dict = {
        "cell_type": val_adata_row.cell_type,
        "tissue": val_adata_row.tissue,
        "disease": val_adata_row.disease,
        "sex": val_adata_row.sex,
    }

    if args.total_mrna_umis is None:
        total_mrna_umis = float(val_adata_row.total_mrna_umis)
        print(f"Using total_mrna_umis from validation AnnData: {total_mrna_umis:.3f}")
    else:
        total_mrna_umis = args.total_mrna_umis
        print(f"Overriding total_mrna_umis from validation AnnData with {total_mrna_umis:.3f}")

    print("Determining gene expression range ...")
    gex_range_dict = ctx.predict_gene_expression_range_for_metadata(
        assay=assay,
        suspension_type=suspension_type,
        prompt_metadata_dict=prompt_metadata_dict,
        total_mrna_umis=total_mrna_umis,
        query_gene_ids=query_gene_ids,
        query_chunk_size=args.query_chunk_size,
        total_prob_mass=args.total_prob_mass,
        symmetric_range_pad=args.symmetric_range_pad,
        max_counts=args.max_counts,
    )

    print("Generating gene dose response ...")
    dose_response_dict = ctx.generate_gene_dose_response_for_metadata(
        assay=assay,
        suspension_type=suspension_type,
        prompt_metadata_dict=prompt_metadata_dict,
        total_mrna_umis=total_mrna_umis,
        query_gene_ids=query_gene_ids,
        perturb_gene_ids=query_gene_ids,
        x_lo_p=gex_range_dict["x_lo_q"].cpu().numpy(),
        x_hi_p=gex_range_dict["x_hi_q"].cpu().numpy(),
        n_points=args.n_points,
        query_chunk_size=args.query_chunk_size_linear_response,
        max_counts=args.max_counts,
    )

    print("Performing linear regression ...")
    n_query_vars = len(query_gene_ids)
    doses_pi = dose_response_dict["doses_pi"]
    responses_mean_pqi = dose_response_dict["responses_mean_pqi"]

    # Repeat doses_pi along a new axis so that its shape matches the required dimensions
    x_bn = np.repeat(doses_pi[None, :, :], n_query_vars, axis=-3)
    # Transpose responses_mean_pqi to match the expected dimensions
    y_bn = responses_mean_pqi.transpose(1, 0, 2)

    slope_qp, intercept_qp, r_squared_qp = batch_linear_regression(x_bn=x_bn, y_bn=y_bn)

    print("Generating output ...")
    output_dict = {
        # Global parameters
        "checkpoint_path": args.checkpoint_path,
        "validation_adata_path": args.validation_adata_path,
        "cell_index": args.cell_index,
        # Gene expression range determination parameters
        "total_prob_mass": args.total_prob_mass,
        "max_counts": args.max_counts,
        "symmetric_range_pad": args.symmetric_range_pad,
        "max_query_genes": args.max_query_genes,
        # Linear response analysis parameters
        "n_points": args.n_points,
        # Prompt metadata
        "assay": assay,
        "suspension_type": suspension_type,
        "total_mrna_umis": total_mrna_umis,
        "prompt_metadata_dict": prompt_metadata_dict,
        # Expression range outputs
        "x_lo_q": gex_range_dict["x_lo_q"].cpu().numpy(),
        "x_hi_q": gex_range_dict["x_hi_q"].cpu().numpy(),
        "range_q": gex_range_dict["range_q"].cpu().numpy(),
        "gene_logits_qk": gex_range_dict["gene_logits_qk"].cpu().numpy(),
        "gene_logits_mode_q": gex_range_dict["gene_logits_mode_q"].cpu().numpy(),
        "gene_marginal_mean_q": gex_range_dict["gene_marginal_mean_q"].cpu().numpy(),
        "gene_marginal_std_q": gex_range_dict["gene_marginal_std_q"].cpu().numpy(),
        # Dose response outputs
        "query_gene_ids": query_gene_ids,
        "perturb_gene_ids": query_gene_ids,
        "doses_pi": dose_response_dict["doses_pi"],
        "responses_mean_pqi": dose_response_dict["responses_mean_pqi"],
        # Linear regression outputs
        "slope_qp": slope_qp,
        "intercept_qp": intercept_qp,
        "r_squared_qp": r_squared_qp,
    }

    print("Saving output ...")
    output_file_name = os.path.join(args.output_path, f"linear_response_cell_index_{args.cell_index}.pkl")

    with open(output_file_name, "wb") as f:
        pickle.dump(output_dict, f)

    print("Script finished successfully.")


if __name__ == "__main__":
    main()
