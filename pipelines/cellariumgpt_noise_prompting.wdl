version 1.0

## Copyright Broad Institute, 2024
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task run_noise_prompting {
    input {
        String h5ad_file
        String cell_type_name
        String msigdb_json_file
        String collection
        Float fraction_of_set_to_perturb = 0.5
        Int n_random_splits = 10
        Int n_perturbations = 100
        Float perturbation_scale = 1.0
        Int min_gene_set_length = 10
        Int max_gene_set_length = 200
        Int n_pcs_in_output = 5
        Int gsea_n_perm = 1000
        Int seed = 0
        Int n_pcs = 50
        Int n_ics = 10

        # software
        String docker_image = "us.gcr.io/broad-dsde-methods/cellarium:noise_prompting"
        String? git_hash

        # Hardware-related inputs
        String hardware_zones = "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
        Int hardware_disk_size_GB = 50
        Int hardware_boot_disk_size_GB = 20
        Int hardware_preemptible_tries = 10
        Int hardware_max_retries = 0
        Int hardware_cpu_count = 4
        Int hardware_memory_GB = 16
        String hardware_gpu_type = "nvidia-tesla-t4"
        Int hardware_gpu_count = 0
    }

    Boolean install_from_git = (if defined(git_hash) then true else false)
    String output_file = "noise_prompting_~{cell_type_name}_~{collection}.csv"

    command <<<
        set -e  # fail the workflow if there is an error

        # install a specific commit from github if called for
        if [[ ~{install_from_git} == true ]]; then
            echo "Uninstalling pre-installed cellarium-ml"
            yes | pip uninstall cellarium-ml
            echo "Installing cellarium-ml from github"
            # this more succinct version is broken in some older versions of cellbender
            echo "pip install --no-cache-dir -U git+https://github.com/ellarium-ai/cellarium-ml.git@~{git_hash}"
            git clone -q https://github.com/ellarium-ai/cellarium-ml.git /cromwell_root/cellarium
            cd /cromwell_root/cellarium
            git checkout -q ~{git_hash}
            yes | pip install --no-cache-dir -U -e /cromwell_root/cellarium
            pip list
            cd /cromwell_root
        fi

        pip install tqdm scikit-learn gseapy

        python <<CODE

        from cellarium.ml.downstream.cellarium_utils import get_pretrained_model_as_pipeline, harmonize_anndata_with_model
        from cellarium.ml.downstream.gene_set_utils import GeneSetRecords
        from cellarium.ml.downstream.noise_prompting import noise_prompt_gene_set_collection
        import anndata
        import torch

        print("torch.cuda.is_available()")
        print(torch.cuda.is_available())

        # pretrained model
        pipeline = get_pretrained_model_as_pipeline(device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"pipeline is on device: {pipeline.device}")

        # data
        adata_cell = anndata.read_h5ad("~{h5ad_file}")
        assert "gpt_include" in adata_cell.var.keys(), "adata.var must contain the key 'gpt_include' to indicate which genes to include in the analysis"
        assert "count" in adata_cell.layers.keys(), "adata.layers must contain the key 'count' with the raw count matrix"
        adata = harmonize_anndata_with_model(adata=adata, pipeline=pipeline)
        adata.X = adata.layers["count"].copy()

        # gene sets
        msigdb = GeneSetRecords("~{msigdb_json_file}")
        
        df = noise_prompt_gene_set_collection(
            adata_cell,
            pipeline=pipeline,
            msigdb=msigdb,
            collection="~{collection}",
            fraction_of_set_to_perturb=~{fraction_of_set_to_perturb}, 
            n_random_splits=~{n_random_splits},
            n_perturbations=~{n_perturbations}, 
            perturbation_scale=~{perturbation_scale},
            min_gene_set_length=~{min_gene_set_length},
            max_gene_set_length=~{max_gene_set_length},
            n_pcs_in_output=~{n_pcs_in_output},
            gsea_n_perm=~{gsea_n_perm},
            seed=~{seed},
            n_pcs=~{n_pcs},
            n_ics=~{n_ics},
            save_intermediates_to_tmp_file="~{output_file}",
        )
        df.to_csv("~{output_file}", index=False)
        print("saved final outputs to ~{output_file}")
        print("done")

        CODE

    >>>

    output {
        File output_csv = "~{output_file}"
        String cell_type = "~{cell_type_name}"
        String collection_name = "~{collection}"
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
        description: "Run noise prompting using CellariumGPT: one cell, one gene set collection."
    }

    parameter_meta {
        h5ad_file :
            {help: "gsURL path to an h5ad file containing a single cell. must contain 'gpt_include' in adata.var and 'count' in adata.layers"}
        msigdb_json_file :
            {help: "gsURL path to a json file containing gene set collections in json format. see 'JSON' here https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp"}
        cell_type_name :
            {help: "name of the cell type in the h5ad file"}
        collection :
            {help: "name of the gene set collection in the msigdb_json_file"}
        hardware_gpu_count :
            {help: "choose 0 to use CPU only, otherwise choose 1"}
    }
}