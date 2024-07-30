version 1.0

## Copyright Broad Institute, 2024
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task compute_jacobian {
    input {
        File cell_h5ad
        File trained_model_ckpt = "gs://cellarium-ml/curriculum/homo_sap_no_cancer/models/cellarium_gpt/downsample/bs_200_max_prefix_4000_context_4500/lightning_logs/version_0/checkpoints/epoch=0-step=152100.ckpt"
        File? genes_csv
        String cell_type_name
        String file_suffix = "jacobian"

        # software
        String docker_image = "us-central1-docker.pkg.dev/broad-dsde-methods/cellarium-ai/cellarium-ml:noiseprompt"
        String git_hash = "0fc55d9714f37b17172fa6f8caac5062324c2536"

        # Hardware-related inputs
        String hardware_zones = "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
        Int hardware_disk_size_GB = 50
        Int hardware_boot_disk_size_GB = 20
        Int hardware_preemptible_tries = 0
        Int hardware_max_retries = 0  # nvidia driver install failures in cromwell
        Int hardware_cpu_count = 4
        Int hardware_memory_GB = 16
        String hardware_gpu_type = "nvidia-tesla-t4"
        Int hardware_gpu_count = 1
    }

    String output_file = "~{cell_type_name}__~{file_suffix}.csv"
    Boolean use_default_genes = (if defined(genes_csv) then false else true)

    command <<<

        set -e
        echo "Installing cellarium-ml from github"
        echo "pip install --no-cache-dir -U git+https://github.com/cellarium-ai/cellarium-ml.git@~{git_hash}"
        pip install --no-cache-dir -U git+https://github.com/cellarium-ai/cellarium-ml.git@~{git_hash}
        pip install --no-cache-dir -U torch==2.3.1
        pip install --no-cache-dir tqdm scikit-learn scanpy gseapy

        python <<CODE

        from cellarium.ml.downstream.cellarium_utils import get_pretrained_model_as_pipeline, harmonize_anndata_with_model
        from cellarium.ml.downstream.noise_prompting import compute_jacobian
        import anndata
        import torch
        import numpy as np
        import pandas as pd
        import glob
        import time
        import os
        
        # NOTE here https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            print('torch.backends.cuda.mem_efficient_sdp_enabled()')
            print(torch.backends.cuda.mem_efficient_sdp_enabled())
            print('torch.backends.cuda.flash_sdp_enabled()')
            print(torch.backends.cuda.flash_sdp_enabled())
            print('torch.backends.cuda.math_sdp_enabled()')
            print(torch.backends.cuda.math_sdp_enabled())

            # pretrained model
            pipeline = get_pretrained_model_as_pipeline(
                trained_model="~{trained_model_ckpt}",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(pipeline)

            adata = anndata.read_h5ad("~{cell_h5ad}")
            print(adata)
            adata.X = adata.layers["count"].copy()
            print("... harmonizing")
            adata_cell = harmonize_anndata_with_model(adata, pipeline)
            adata_cell.layers["count"] = adata_cell.X.copy()

            print("... determining genes to include in computation")
            if ~{if (use_default_genes) then "True" else "False"}:
                adata_cell.var["jacobian_include"] = adata.var["gpt_include"].copy()
            else:
                included_genes = set(pd.read_csv("~{genes_csv}", header=None, squeeze=True).values)
                adata_cell.var["jacobian_include"] = [g in included_genes for g in adata_cell.var["gene_name"]]
            
            var_inclusion_key = "jacobian_include"
            print(adata_cell.var[var_inclusion_key].value_counts(dropna=False))
            n_genes = adata_cell.var[var_inclusion_key].sum()
            print(f"... {n_genes} genes included")
            print(f"... projected time to completion = {(n_genes / 185)**2.6 / 60} mins")

            print("... computing jacobian")
            t = time.time()
            jacobian_df = compute_jacobian(
                adata_cell,
                pipeline=pipeline,
                var_key_include_genes=var_inclusion_key,
                summarize="mean",
                layer="count",
                var_key_gene_name="gene_name",
            )
            print(f"... done in {(time.time() - t) / 60:.2f} mins")

            jacobian_df.to_csv("~{output_file}", index=True)
            print("Saved ~{output_file}\n")

        CODE

    >>>

    output {
        File output_csv = "~{output_file}"
        String cell_type = "~{cell_type_name}"
    }

    runtime {
        docker: "${docker_image}"
        bootDiskSizeGb: hardware_boot_disk_size_GB
        disks: "local-disk ${hardware_disk_size_GB} HDD"
        memory: "${hardware_memory_GB}G"
        cpu: hardware_cpu_count
        zones: "${hardware_zones}"
        gpuCount: hardware_gpu_count
        gpuType: "${hardware_gpu_type}"
        preemptible: hardware_preemptible_tries
        maxRetries: hardware_max_retries  # can be used in case of a PAPI error code 2 failure to install GPU drivers
    }

    meta {
        author: "Stephen Fleming"
        email: "sfleming@broadinstitute.org"
        description: "Compute the Jacobian of a trained CellariumGPT model at one particular cell"
    }

    parameter_meta {
        cell_h5ad :
            {help: "gsURL path to an h5ad file containing a single cell. must contain 'count' in adata.layers"}
        genes_csv :
            {help: "gsURL path to a csv file containing gene names to include. one column, one gene name per line, no header. if missing, defaults to using genes in adata.var['gpt_include']"}
        cell_type_name :
            {help: "name of the cell type in the h5ad file"}
        file_suffix :
            {help: "string to append to the end of the output filename"}
        hardware_preemptible_tries :
            {help: "choose 0, as there is no checkpointing and this is a long-running task (depending on number of genes)"}
        hardware_gpu_count :
            {help: "choose 1"}
    }
}

workflow jacobian {

    call compute_jacobian

    output {
        File output_csv = compute_jacobian.output_csv
    }
}
