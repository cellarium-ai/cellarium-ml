#!/bin/python3

"""Run linear_response_network_pipeline.py code on all cell types in the validation data.
$ conda activate vertex
(vertex) $ ./linear_resposne_network_pipeline_runner.py
"""

import os
import tempfile

import anndata

from linear_response_network_pipeline import main as run_pipeline


val_adata_path = "gs://cellarium-scratch/cellariumgpt_artifacts/cell_types_for_validation_filtered.h5ad"

def main():

    with tempfile.TemporaryDirectory() as tmpdir:
        val_adata_tmp_path = os.path.join(tmpdir, os.path.basename(val_adata_path))
        os.system(f"gsutil cp {val_adata_path} {val_adata_tmp_path}")
        adata = anndata.read_h5ad(val_adata_tmp_path)

    for i, celltype in zip(adata.obs_names, adata.obs["cell_type"].values):
        print("=" * 80)
        print(f"Running pipeline for cell type {i} ({celltype})")
        run_pipeline(celltype)


if __name__ == "__main__":
    main()
