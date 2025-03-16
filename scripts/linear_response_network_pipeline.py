# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/python3

import tempfile
import sys

from google.auth import default
from google.auth.transport.requests import Request
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)
from kfp import compiler, dsl


@dsl.component(
    packages_to_install=[
        "gcsfs",
        "psutil",
    ],
    base_image="us-central1-docker.pkg.dev/broad-dsde-methods/cellarium-ai/cellarium-ml:gpgnasp",
)
def run_analysis(cell_type: str):

    import os
    import psutil
    import subprocess
    import tempfile

    import anndata
    import numpy as np
    import pandas as pd

    from cellarium.ml.utilities.inference.gene_network_analysis import EmpiricalCorrelationContext, LinearResponseContext
    from cellarium.ml.utilities.inference.gene_set_utils import GeneSetRecords

    # constants ================

    DEVICE = "cpu"
    output_dir = "/gcs/cellarium-scratch/sfleming/empirical_gene_correlation2"

    # linear_response_path_fn = lambda model, metacell_index: (
    #     f"gs://cellarium-scratch/linear_response/{model}/linear_response_cell_index_{metacell_index}.pkl"
    # )
    linear_response_path_fn = lambda model, metacell_index: (
        f"gs://cellarium-scratch/linear_response_compute_optimal/{model}/linear_response_cell_index_{metacell_index}.pkl"
    )

    gene_info_tsv_path = "gs://cellarium-scratch/mb-ml-dev-vm/gene_info/gene_info.tsv"
    val_adata_path = "gs://cellarium-scratch/cellariumgpt_artifacts/cell_types_for_validation_filtered.h5ad"
    empirical_corr_base_path = "gs://cellarium-scratch/sfleming/empirical_gene_correlation2"

    msigdb_path = "gs://cellarium-scratch/sfleming/references/msigdb.v2024.1.Hs.json"
    corum_path = "gs://cellarium-scratch/sfleming/references/corum_humanComplexes.txt"
    pango_path = "gs://cellarium-scratch/sfleming/references/pan_go_annotations.csv"

    # choose cell types and models
    models = ["10M_001_bs1536", "19M_001_bs2048", "30M_001_bs2560", "59M_001_bs3072"]  #, "98M_001_bs4608"]

    # optionally limit to specific gene set collections
    specific_collections = None
    # specific_collections = ["C5:GO:BP"]

    # other constants
    min_tpm = 0.1
    min_r_squared = 0.25
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # =========================


    # helper function
    def process_and_compute_metrics(
        ctx: EmpiricalCorrelationContext | LinearResponseContext,
        gene_records: GeneSetRecords,
        model_name: str,
        metacell_index: int,
        output_dir: str,
    ):
        # process and compute adjacency
        if isinstance(ctx, EmpiricalCorrelationContext):
            response_normalization_strategy = "none"
        elif isinstance(ctx, LinearResponseContext):
            response_normalization_strategy = "log1p"
        else:
            raise ValueError("Invalid context type")
        ctx.reprocess(
            response_normalization_strategy=response_normalization_strategy,
            feature_normalization_strategy="z_score",
            feature_max_value=None,
            query_response_amp_min_pct=None,
            min_prompt_gene_tpm=min_tpm,
            min_query_gene_tpm=min_tpm,
            norm_pseudo_count=0.0,  # not needed for log1p normalization strategy
            query_hv_top_k=None,
        )
        ctx.compute_adjacency_matrix(
            adjacency_strategy="positive_correlation",
            n_neighbors=None,
            beta=1.0,
            self_loop=False,
            scale_by_node_degree=False,
        )

        relevant_collections = (
            specific_collections if (specific_collections is not None) else gene_records.get_collections()
        )

        # # compute knn metrics
        # print("Computing knn metrics", flush=True)
        # for collection in relevant_collections:
        #     best_metrics_df = ctx.gridsearch_optimal_k_neighbors_given_gene_sets(
        #         reference_gene_sets=gene_records.get_gene_set_dict_for_collection(collection),
        #         k_values=k_values,
        #         metric_name="intersection",
        #         gene_naming="symbols",
        #     )
        #     best_metrics_df["precision"] = best_metrics_df["intersection"] / best_metrics_df["k"]
        #     print(f"{collection} mean precision: {best_metrics_df['precision'].mean()}", flush=True)
        #     # write to file
        #     outfile = f"{output_dir}/{model_name}__cell_{metacell_index}__precision_{collection}.csv"
        #     best_metrics_df.to_csv(outfile, index=False)
        #     print(f"Saved to {outfile}", flush=True)

        # compute auc
        print("Computing AUC", flush=True)
        for collection in relevant_collections:
            auc_p, var_names_p, tpr_pn, fpr_pn = ctx.compute_network_adjacency_auc_metric_per_gene(
                reference_gene_sets=gene_records.get_gene_set_dict_for_collection(collection),
                reference_set_exclusion_fraction=0.25,
                min_set_size=3,
                thin_output_to_n=100,
            )
            print(f"{collection} AUC: {auc_p}", flush=True)
            # write to file
            outfile = (
                f"{output_dir}/{cell_type.replace(' ', '_')}__{tissue.replace(' ', '_')}"
                + f"/{model_name}__cell_{metacell_index}__auc_{collection}.npy"
            )
            np.save(
                outfile,
                {
                    "auc_p": auc_p,
                    "var_names_p": var_names_p,
                    "tpr_pn": tpr_pn,
                    "fpr_pn": fpr_pn,
                },
                allow_pickle=True,
            )
            print(f"Saved to {outfile}", flush=True)

    # set env variables to allow pytorch to use all CPUs
    num_physical_cores = psutil.cpu_count(logical=False)
    os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_physical_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_physical_cores)  # Only if using OpenBLAS
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_physical_cores)  # Not critical for PyTorch

    # load gene sets
    with tempfile.TemporaryDirectory() as tmpdir:
        msigdb_path_local = os.path.join(tmpdir, os.path.basename(msigdb_path))
        os.system(f"gsutil cp {msigdb_path} {msigdb_path_local}")
        corum_path_local = os.path.join(tmpdir, os.path.basename(corum_path))
        os.system(f"gsutil cp {corum_path} {corum_path_local}")
        pango_path_local = os.path.join(tmpdir, os.path.basename(pango_path))
        os.system(f"gsutil cp {pango_path} {pango_path_local}")
        
        gene_records = GeneSetRecords(
            msigdb_file=msigdb_path_local,
            corum_file=corum_path_local,
            pan_go_file=pango_path_local,
        )
    gene_records.df = gene_records.df.loc[~gene_records.df.index.str.startswith("Unknown")].copy()  # useless pango
    gene_records._reindex()

    # read the anndata file with the validation cell types
    with tempfile.TemporaryDirectory() as tmpdir:
        val_adata_tmp_path = os.path.join(tmpdir, os.path.basename(val_adata_path))
        os.system(f"gsutil cp {val_adata_path} {val_adata_tmp_path}")
        adata = anndata.read_h5ad(val_adata_tmp_path)

    # go through each cell type
    for metacell_index in adata.obs[adata.obs["cell_type"] == cell_type].index:
        cell_type = adata.obs.loc[metacell_index, "cell_type"]
        tissue = adata.obs.loc[metacell_index, "tissue"]

        # load empirical correlation model
        trained_model_path_str = (
            f"{empirical_corr_base_path}/{cell_type.replace(' ', '_')}__{tissue.replace(' ', '_')}/"
            + "lightning_logs/version_*/checkpoints/epoch=0-step=*.ckpt"
        )
        try:
            result = subprocess.run(
                f"gsutil ls {trained_model_path_str}",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            output_files = [path for path in result.stdout.split("\n") if path]
        except subprocess.CalledProcessError as e:
            print(f"Error running gsutil ls: {e}")
            output = ""
        if not output_files:
            print(f"WARNING: No trained empirical model found for {cell_type}__{tissue}")
            print(f"Skipping {cell_type}__{tissue}" + "*" * 30)
            continue
        trained_model_path = sorted(output_files)[-1]  # the last "version" is assumed to be the latest

        # download the checkpoint and load it
        print(f"Processing empirical__cell_{metacell_index}", flush=True)
        print(adata.obs.iloc[int(metacell_index)])
        empirical_ctx = EmpiricalCorrelationContext.from_model_ckpt(
            ckpt_path=trained_model_path,
            gene_info_tsv_path=gene_info_tsv_path,
            total_mrna_umis=adata.obs.iloc[[metacell_index]]["total_mrna_umis"].mean(),
            device="cpu",
            response_normalization_strategy="none",
            feature_normalization_strategy="z_score",
            min_prompt_gene_tpm=min_tpm,
            min_query_gene_tpm=min_tpm,
        )

        # process and compute metrics and save
        process_and_compute_metrics(
            ctx=empirical_ctx,
            gene_records=gene_records,
            model_name="empirical",
            metacell_index=metacell_index,
            output_dir=output_dir,
        )

        # analyze the linear response data for each cellariumgpt model
        for model in models:
            print(f"Processing {model}__cell_{metacell_index}", flush=True)
            print(adata.obs.iloc[int(metacell_index)])

            linear_response_gsurl = linear_response_path_fn(model, metacell_index)

            # make marginal means and stds into pandas series
            marginal_mean_g = pd.Series(
                empirical_ctx.processed.prompt_marginal_mean_p,
                index=empirical_ctx.processed.prompt_var_names,
            )
            marginal_std_g = pd.Series(
                empirical_ctx.processed.prompt_marginal_std_p,
                index=empirical_ctx.processed.prompt_var_names,
            )

            # load linear response data
            linresponse_ctx = LinearResponseContext.from_linear_response_pkl(
                linear_response_path=linear_response_gsurl,
                adata_obs=adata.obs.iloc[[metacell_index]],
                gene_info_tsv_path=gene_info_tsv_path,
                min_r_squared=min_r_squared,
                marginal_mean_g=marginal_mean_g,
                marginal_std_g=marginal_std_g,
            )

            # process and compute metrics and save
            process_and_compute_metrics(
                ctx=linresponse_ctx,
                gene_records=gene_records,
                model_name=model,
                metacell_index=metacell_index,
                output_dir=output_dir,
            )


def main(cell_type=None):
    """Run the pipeline for a specific cell type.
    If no cell type is provided, it will read from the command line arguments.
    """
    if cell_type is None:
        if len(sys.argv) != 2:
            print("Usage: python linear_response_network_pipeline.py '<cell_type>'")
            sys.exit(1)
        cell_type = sys.argv[1]
    print(cell_type)

    project = "dsp-cell-annotation-service"
    location = "us-central1"
    display_name = f"sfleming_linear_response_network_metrics_{cell_type}"

    # Get default credentials and add required scopes
    credentials, _ = default()
    credentials.refresh(Request())

    # Initialize with proper credentials
    aiplatform.init(
        project=project,
        location=location,
        credentials=credentials,
    )

    custom_training_job = create_custom_training_job_from_component(
        run_analysis,
        display_name=display_name,
        replica_count=1,
        machine_type="n1-standard-16",
        accelerator_type=None,
        accelerator_count=0,
    )

    @dsl.pipeline(name=display_name, description=display_name)
    def pipeline():
        custom_training_job(
            project=project,
            location=location,
            cell_type=cell_type,
        ).set_display_name(display_name)

    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(pipeline_func=pipeline, package_path=f.name)

        job = aiplatform.PipelineJob(
            display_name=display_name,
            template_path=f.name,
        )

        job.submit()


if __name__ == "__main__":
    main()
