# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import numpy as np
import pandas as pd
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.data.fileio import read_h5ad_gcs
from cellarium.ml.hop_scoring import calculate_hop_metrics_for_batch

DEFAULT_CKPT_PATH = (
    "gs://cellarium-file-system-cas-archive/curriculum/lrexp_human_training_split_20241106/models/"
    "SOCAM_lambda_experiment_models/lambda_100_trial_15072026/lightning_logs/corrected_run1/checkpoints/"
    "epoch=5-step=25500.ckpt"
)
DEFAULT_INPUT_H5AD = (
    "gs://cellarium-file-system-cas-archive/curriculum/lrexp_human_validation_split_20241126/extract_files/"
    "extract_1.h5ad"
)
DEFAULT_CO_RESOURCE_PATH = (
    "gs://cellarium-file-system-cas-archive/curriculum/lrexp_human_validation_split_20241126/shared_meta/"
    "dev_benchmarking_june_2024_metadata_benchmarking_resource_schema_5-0.pickle"
)
DEFAULT_OUTPUT_CSV = (
    "gs://cellarium-file-system-cas-archive/curriculum/lrexp_human_validation_split_20241126/"
    "SOCAM_inference_july_26/extract_1_hop_scores.csv"
)


def _replace_ontology_ids(values: np.ndarray) -> np.ndarray:
    return np.char.replace(values.astype(str), ":", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOCAM lambda=100 hop-scoring inference for one h5ad extract.")
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--input-h5ad", default=DEFAULT_INPUT_H5AD)
    parser.add_argument("--co-resource-path", default=DEFAULT_CO_RESOURCE_PATH)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading checkpoint: {args.ckpt_path}")
    module = CellariumModule.load_from_checkpoint(args.ckpt_path, map_location="cpu")
    module.eval()
    model = module.model
    active_cl_names = _replace_ontology_ids(np.asarray(model.active_cl_names))
    print(f"Loaded SOCAM with {len(active_cl_names)} active classes")

    print(f"Loading h5ad: {args.input_h5ad}")
    adata = read_h5ad_gcs(args.input_h5ad)
    print(f"Loaded h5ad shape={adata.shape}")

    co_resource = pd.read_pickle(args.co_resource_path)
    print(f"Loaded hop-scoring resource with {len(co_resource)} terms")

    result_dfs: list[pd.DataFrame] = []
    n_obs = adata.n_obs
    with torch.inference_mode():
        for start in range(0, n_obs, args.batch_size):
            stop = min(start + args.batch_size, n_obs)
            batch_adata = adata[start:stop]
            batch = {
                "x_ng": batch_adata.X,
                "var_names_g": batch_adata.var_names.to_numpy(),
                "total_mrna_umis_n": torch.as_tensor(
                    batch_adata.obs["total_mrna_umis"].to_numpy(),
                    dtype=torch.float32,
                ),
            }
            output = module.module_pipeline.predict(batch)
            metrics_df = calculate_hop_metrics_for_batch(
                query_cell_ids=batch_adata.obs_names.to_numpy().astype(str),
                prediction_scores_nc=output["cell_type_probs_nc"].detach().cpu().numpy(),
                ground_truth_cl_names=_replace_ontology_ids(
                    batch_adata.obs["cell_type_ontology_term_id"].to_numpy(),
                ),
                cell_type_ontology_term_ids=active_cl_names,
                co_resource=co_resource,
                num_hops=4,
            )
            result_dfs.append(metrics_df)
            print(f"Processed cells {start}:{stop}")

    result_df = pd.concat(result_dfs, ignore_index=True)
    result_df.sort_values(by="query_cell_id", inplace=True)
    print(f"Writing {len(result_df)} rows to {args.output_csv}")
    result_df.to_csv(args.output_csv, header=True, index=False)
    print("Done")


if __name__ == "__main__":
    main()
