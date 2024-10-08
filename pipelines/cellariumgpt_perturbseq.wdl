version 1.0

import "https://api.firecloud.org/ga4gh/v1/tools/cellarium%3Acellxgene_census_query/versions/15/plain-WDL/descriptor" as util

## Copyright Broad Institute, 2024
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task run_perturbseq {
    input {
        File trained_model_ckpt = "gs://cellarium-ml/curriculum/homo_sap_no_cancer/models/cellarium_gpt/downsample/bs_200_max_prefix_4000_context_4500/lightning_logs/version_0/checkpoints/epoch=0-step=152100.ckpt"
        File h5ad_file = "gs://broad-dsde-methods-sfleming/ReplogleWeissman2022_rpe1_100controlcells_10batches.h5ad"
        File genes_to_perturb_csv_no_header_no_index = "gs://broad-dsde-methods-sfleming/replogle_perturbed_genes.csv"
        File? genes_in_prompt_and_query_csv_no_header_no_index
        File? checkpoint_tar  # optional previous checkpoint (say from a partially failed run)
        Int shard
        Int n_shards
        Int query_total_umis = 0
        Float perturbation_mean_expression_multiplier = 0.0  # this default represents a knockout
        Boolean downsample_n_cells = false  # for testing

        # software
        String docker_image = "us-central1-docker.pkg.dev/broad-dsde-methods/cellarium-ai/cellarium-ml:noiseprompt"
        String? git_hash = "dbf8add1ffc8dc88a3eefb583873abe101b8bbc6"

        # Hardware-related inputs
        String hardware_zones = "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
        Int hardware_disk_size_GB = 50
        Int hardware_boot_disk_size_GB = 20
        Int hardware_preemptible_tries = 10
        Int hardware_max_retries = 1  # nvidia driver install failures in cromwell
        Int hardware_cpu_count = 4
        Int hardware_memory_GB = 15
        String hardware_gpu_type = "nvidia-tesla-t4"
        Int hardware_gpu_count = 1
    }

    Boolean install_from_git = (if defined(git_hash) then true else false)
    Boolean subset_genes = (if defined(genes_in_prompt_and_query_csv_no_header_no_index) then true else false)
    Boolean checkpoint_present = (if defined(checkpoint_tar) then true else false)

    command <<<

        set +e

        # get checkpoint file in order
        if [[ ~{checkpoint_present} == true ]]; then
            mv ~{checkpoint_tar} ckpt.tar
        fi
        # extract the tarball if it is present
        touch ckpt.tar
        tar -xvf ckpt.tar -C .
        echo "ls -lh *.csv"
        ls -lh *.csv

        set -e

        # install a specific commit from github if called for
        if [[ ~{install_from_git} == true ]]; then
            echo "Installing cellarium-ml from github"
            echo "pip install --no-cache-dir -U git+https://github.com/cellarium-ai/cellarium-ml.git@~{git_hash}"
            pip install --no-cache-dir -U git+https://github.com/cellarium-ai/cellarium-ml.git@~{git_hash}
        fi

        pip install tqdm scikit-learn==1.5.1 gseapy==1.1.3 scanpy==1.10.2 numpy==1.26.4
        pip list

        python <<CODE

        import scanpy as sc
        import pandas as pd
        import numpy as np
        import scipy.sparse as sp
        import torch
        import os
        import itertools
        import tqdm

        from cellarium.ml.downstream.noise_prompting import in_silico_perturbation
        from cellarium.ml.downstream.cellarium_utils import get_pretrained_model_as_pipeline

        import warnings
        warnings.filterwarnings("ignore")

        print("torch.cuda.is_available()")
        print(torch.cuda.is_available())

        # pretrained model
        pipeline = get_pretrained_model_as_pipeline(
            trained_model="~{trained_model_ckpt}",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(pipeline)

        # check genes
        allowed_gene_ids_from_model = set(pipeline[0].token_to_id.keys())
        print(f"allowed_gene_ids_from_model: {len(allowed_gene_ids_from_model)}")
            
        # the perturbseq data from the RPE1 cell line
        adata_control = sc.read_h5ad('~{h5ad_file}')
        print(adata_control)

        # subset genes if called for
        if "~{subset_genes}" == "true":
            print("subsetting genes ...")
            prompt_query_gids = pd.read_csv("~{genes_in_prompt_and_query_csv_no_header_no_index}", header=None)[0].tolist()
            if not np.all([gid in allowed_gene_ids_from_model for gid in prompt_query_gids]):
                print("WARNING: Some genes in the supplied genes_in_prompt_and_query_csv_no_header_no_index are not in the trained model")
                print("Dropping these genes from prompt and query gene list:")
                print([gid for gid, allowed in zip(prompt_query_gids, [gid in allowed_gene_ids_from_model for gid in prompt_query_gids]) if not allowed])
            prompt_query_gids = [gid for gid in prompt_query_gids if gid in allowed_gene_ids_from_model]
        else:
            # still subset to the genes in the model
            prompt_query_gids = allowed_gene_ids_from_model
        adata_control = adata_control[:, adata_control.var_names.isin(prompt_query_gids)].copy()
        print(adata_control)

        # figure out all the perturbations ...
        gids = pd.read_csv("~{genes_to_perturb_csv_no_header_no_index}", header=None)[0].tolist()
        measured_gids = set(adata_control.var_names.values)
        if not np.all([gid in allowed_gene_ids_from_model for gid in gids]):
            print([gid for gid, allowed in zip(gids, [gid in allowed_gene_ids_from_model for gid in gids]) if not allowed])
            raise ValueError("Some genes in the supplied genes_to_perturb_csv_no_header_no_index are not in the trained model")
        print(f"genes to perturb: {len(gids)}")
        print(f"total measured_gids: {len(measured_gids)}")

        # ... that we can do
        pert_gids_in_prompt = [gid for gid in gids if gid in measured_gids]
        if len(pert_gids_in_prompt) < len(gids):
            print(f"WARNING: some genes in the perturbation list are not in the prompt genes. dropping:")
            print([gid for gid in gids if gid not in measured_gids])
        gids = pert_gids_in_prompt
        print(f"genes to perturb, among measured genes: {len(gids)}")

        # ... in this shard
        this_shard = ~{shard}
        total_shards = ~{n_shards}
        if total_shards > 1:
            print(f"Zero-indexed shard {this_shard} of {total_shards}")
            sharded_gids = np.array_split(gids, total_shards)
            gids_in_shard = sharded_gids[this_shard]
        else:
            gids_in_shard = gids

        # figure out query_total_umis
        query_total_umis_value = ~{query_total_umis}
        if query_total_umis_value == 0:
            print("query_total_umis is 0, so we will use the median in the control cells")
            query_total_umis_value = np.median(adata_control.obs['total_mrna_umis'])
            print(f"query_total_umis: {query_total_umis_value}")

        # ignore those already computed to obtain final to-compute list of gene IDs
        print(f"gids_in_shard: {len(gids_in_shard)}")
        print("checking for already computed files ...")
        gids = [gid for gid in gids_in_shard if (not os.path.exists(f'perturbseq_lfc_{gid}.csv'))]
        print(f"to compute: {len(gids)}")
        if len(gids) == 0:
            print("nothing left to compute, exiting python")
            exit(0)

        # no perturbation
        print('running control cells to get baseline ...')
        adata_out_nopert = in_silico_perturbation(
            adata_control,
            pipeline=pipeline,
            prompt_gene_inds=torch.arange(adata_control.shape[1]).long(),
            perturbation={},
            measured_count_layer_key='count',
            output_layer_key='perturbed_gpt',
            query_total_umis=query_total_umis_value,
        )
        print('... done with control cells.')

        # the perturbations
        for gid in tqdm.tqdm(gids):

            out_path = f'perturbseq_lfc_{gid}.csv'

            print(f'... working on {gid} =====================================')

            if sp.issparse(adata_control.layers['count']):
                gene_exp_measured_values = np.array(adata_control.layers['count'][:, adata_control.var_names == gid].todense()).flatten()
            else:
                gene_exp_measured_values = adata_control.layers['count'][:, adata_control.var_names == gid].flatten()
            mean_gene_exp_value = gene_exp_measured_values.mean()
            print(f'mean_gene_exp_value: {mean_gene_exp_value}')
            print('adding offset 1 count')
            mean_gene_exp_value = mean_gene_exp_value + 1
            print(f'mean_gene_exp_value: {mean_gene_exp_value}')
            gene_exp_value = ~{perturbation_mean_expression_multiplier} * mean_gene_exp_value
            print(f'target value per cell after perturbation: {gene_exp_value}')

            # exclude cells from the calculation if they are already at the target value
            keeper_cell_logic = (gene_exp_measured_values != gene_exp_value)

            if keeper_cell_logic.sum() > 0:
                adata_out = in_silico_perturbation(
                    adata_control[keeper_cell_logic].copy(),
                    pipeline=pipeline,
                    prompt_gene_inds=torch.arange(adata_control.shape[1]).long(),
                    perturbation={gid: gene_exp_value},
                    measured_count_layer_key='count',
                    output_layer_key='perturbed_gpt',
                    query_total_umis=query_total_umis_value,
                )
                print('perturbation complete')
                lfc_ng = (np.log2(adata_out.layers['perturbed_gpt'] + 1e-10) 
                        - np.log2(adata_out_nopert.layers['perturbed_gpt'][keeper_cell_logic, :] + 1e-10))
            else:
                print('no cells to perturb (all cells target values are same as measured values)')
                lfc_ng = np.zeros((1, adata_out_nopert.shape[1]))

            lfc_df = pd.DataFrame(
                lfc_ng.mean(axis=0),
                index=adata_out_nopert.var['gene_name'],
                columns=[gid],
            ).transpose()

            lfc_df.to_csv(out_path)
            print(f'wrote {out_path}')

            # add to the checkpoint
            os.system(f"tar -rvf ckpt.tar {out_path}")
            print(f'added {out_path} to ckpt.tar')
            
            if ("~{downsample_n_cells}" == "true") and (keeper_cell_logic.sum() >= 10):
                print('systematically downsampling n_cells')
                for n_cells in range(5, len(adata_out), 5):
                    print(f'n_cells: {n_cells}')
                    n_cells_dfs = []
                    for inds in itertools.islice(itertools.combinations(range(len(adata_out)), n_cells), 100):
                        # for each combination of n_cells cells
                        n_cells_df = pd.DataFrame(
                            lfc_ng[inds, :].mean(axis=0),
                            index=adata_out.var['gene_name'],
                            columns=[gid],
                        ).transpose()
                        n_cells_dfs.append(n_cells_df)
                    lfcs_df = pd.concat(n_cells_dfs, axis=0)
                    mean_lfc_df = lfcs_df.mean(axis=0).to_frame().rename(columns={0: gid}).transpose()
                    std_lfc_df = lfcs_df.std(axis=0).to_frame().rename(columns={0: gid}).transpose()
                    out_path_downsample_prefix = f'perturbseq_lfc_{gid}__downsample_{n_cells}__'
                    mean_lfc_df.to_csv(out_path_downsample_prefix + 'mean.csv')
                    std_lfc_df.to_csv(out_path_downsample_prefix + 'std.csv')
                    print(f'wrote {out_path_downsample_prefix}mean.csv and {out_path_downsample_prefix}std.csv')
                    os.system(f"tar -rvf ckpt.tar {out_path_downsample_prefix}mean.csv")
                    os.system(f"tar -rvf ckpt.tar {out_path_downsample_prefix}std.csv")
                    print(f'added {out_path_downsample_prefix}mean.csv and {out_path_downsample_prefix}std.csv to ckpt.tar')

        CODE

        cp ckpt.tar shard_~{shard}_ckpt.tar

    >>>

    output {
        File output_tar = "shard_~{shard}_ckpt.tar"
        Int input_shard = shard
        Int input_n_shards = n_shards
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
        checkpointFile: "ckpt.tar"
        preemptible: hardware_preemptible_tries
        maxRetries: hardware_max_retries  # can be used in case of a PAPI error code 2 failure to install GPU drivers
    }

    meta {
        author: "Stephen Fleming"
        email: "sfleming@broadinstitute.org"
        description: "Run in silico perturb-seq using CellariumGPT."
    }

    parameter_meta {
        h5ad_file :
            {help: "gsURL path to an h5ad file containing a control cells. must have Ensembl IDs as adata.var_names and must contain 'count' in adata.layers and 'gene_name' in adata.var and 'total_mrna_umis' in adata.obs."}
        genes_to_perturb_csv_no_header_no_index :
            {help: "gsURL path to a CSV file containing a list of Ensembl gene IDs to perturb, newline delimited, no header, no index."}
        perturbation_mean_expression_multiplier :
            {help: "perturbed counts per cell = perturbation_mean_expression_multiplier * (mean_control_counts + 1). the value 0.0 represents a knockout."}
        query_total_umis :
            {help: "the total number of UMIs to request for the query (if 0, this will be the median in control cells)"}
        hardware_gpu_count :
            {help: "choose 0 to use CPU only, otherwise choose 1"}
    }
}


task consolidate_outputs {
    input {
        Array[File] output_tars
    }

    command <<<
        set -e

        # empty tar file
        tar -cf combined.tar --files-from /dev/null

        # add each file to the combined tar
        for file in ~{sep=" " output_tars}; do
            tar -Af combined.tar "$file"
        done

        gzip combined.tar
    >>>

    output {
        File output_tarball = "combined.tar.gz"
    }

    runtime {
        docker: "google/cloud-sdk:slim"
        bootDiskSizeGb: 20
        disks: "local-disk 500 HDD"
        memory: "16G"
        cpu: 4
        preemptible: 0
        maxRetries: 2
    }
}


workflow in_silico_perturbseq {
    input {
        Int n_shards = 20
        File? h5ad_file
        File? checkpoint_tar
    }
    Boolean run_query = (if defined(h5ad_file) then false else true)
    Array[Int] shard_counter = range(n_shards)

    if (run_query) {
        call util.query_cellxgene as query_cellxgene
        File? query_h5ad = query_cellxgene.output_h5ad
    }
    File h5ad = select_first([h5ad_file, query_h5ad])

    scatter (shard in shard_counter) {
        call run_perturbseq {
            input:
                h5ad_file = h5ad,
                shard = shard,
                n_shards = n_shards,
                checkpoint_tar = checkpoint_tar,
        }
    }

    call consolidate_outputs {
        input:
            output_tars = run_perturbseq.output_tar,
    }

    output {
        File output_tarball = consolidate_outputs.output_tarball
    }
}
