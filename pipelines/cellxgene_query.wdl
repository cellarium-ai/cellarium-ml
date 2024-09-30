version 1.0

## Copyright Broad Institute, 2024
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task query_cellxgene {
    input {
        String data_name
        String obs_value_filter
        Int max_cells = 100
        Int min_cells_per_dataset_donor = 20
        Boolean average_expression = false
        String organism = "Homo sapiens"
        String X_name = "raw"

        # software
        String docker_image = "us.gcr.io/broad-dsde-methods/cellxgene_census:1.16.1"

        # Hardware-related inputs
        Int hardware_disk_size_GB = 50
        Int hardware_boot_disk_size_GB = 20
        Int hardware_cpu_count = 4
        Int hardware_memory_GB = 16
    }

    String output_file = "~{data_name}.h5ad"

    command <<<

        set -e

        python <<CODE

        import cellxgene_census
        import anndata
        import numpy as np

        # print info about the census
        with cellxgene_census.open_soma() as census:
            info = census["census_info"]["summary"].read().concat().to_pandas()
            print(info)
            info = census["census_info"]["summary_cell_counts"].read().concat().to_pandas()
            print(info[info['category'] == 'all'])

            # run the query
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="~{organism}",
                X_name="~{X_name}",
                obs_value_filter="""~{obs_value_filter}""",
                obs_column_names=["assay", "cell_type", "tissue", "dataset_id", "donor_id", "tissue_general", "suspension_type", "disease", "sex"],
            )

        # optional removal of cells from donors with fewer than a certain number
        adata.obs["dataset_donor"] = adata.obs["dataset_id"].astype(str) + "_" + adata.obs["donor_id"].astype(str)
        vc = adata.obs["dataset_donor"].value_counts()
        print('combination of dataset and donor:')
        print(vc)
        print('Removing gruops with fewer than ~{min_cells_per_dataset_donor} cells')
        allowed_groups = vc.index[vc >= ~{min_cells_per_dataset_donor}]
        adata = adata[adata.obs["dataset_donor"].isin(allowed_groups)].copy()
        print('Remaining data:')
        print(adata)
        print(adata.obs["dataset_donor"].value_counts())

        # optional averaging
        if "~{average_expression}" == "true":
            print('Averaging expression of the above cells...')
            adata = anndata.AnnData(
                X=adata.X.mean(axis=0),
                obs=adata.obs.iloc[[0]],
                var=adata.var,
            )

        # show adata
        print('Dataset:')
        print(adata)
        print('Limiting to ~{max_cells} cells, randomly shuffling first')
        adata = adata[np.random.permutation(len(adata))][:~{max_cells}].copy()
        print('Final data:')
        print(adata)

        # write the output h5ad file
        output_file = "~{output_file}"
        adata.write_h5ad(output_file)
        print(f"Output written to {output_file}")

        CODE

    >>>

    output {
        File output_h5ad = "~{output_file}"
    }

    runtime {
        docker: "${docker_image}"
        bootDiskSizeGb: hardware_boot_disk_size_GB
        disks: "local-disk ${hardware_disk_size_GB} HDD"
        memory: "${hardware_memory_GB}G"
        cpu: hardware_cpu_count
    }

    meta {
        author: "Stephen Fleming"
        email: "sfleming@broadinstitute.org"
        description: "Query data from cellxgene-census and write to h5ad file"
    }

    parameter_meta {
        data_name :
            {help: "becomes the name of the output file: <data_name>.h5ad"}
        obs_value_filter :
            {help: "the cellxgene-census query, see https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_query_extract.html"}
        max_cells :
            {help: "max allowed number of cells in the h5ad file"}
        min_cells_per_dataset_donor :
            {help: "only include cells if a given (dataset_id, donor_id) combination has at least this many cells matching the query. this helps remove chance spurious labels."}
        average_expression :
            {help: "true to produce a single pseudobulk output with the mean expression of each gene; false to return the single cell data."}
        organism :
            {help: "defaults to 'Homo sapiens', see https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata"}
        X_name :
            {help: "use 'raw' to query raw counts. see https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata"}
    }
}

workflow query_cellxgene_workflow {

    call query_cellxgene

    output {
        File h5ad = query_cellxgene.output_h5ad
    }
}
