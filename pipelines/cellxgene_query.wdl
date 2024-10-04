version 1.0

## Copyright Broad Institute, 2024
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task query_cellxgene {
    input {
        String data_name
        String obs_cell_types  # list of cell types like "['retinal pigment epithelial cell']". bare text = python list of quoted strings
        String obs_value_filter = "(is_primary_data == True) and (suspension_type == 'cell') and (assay == \"10x 3\\' v3\") and (disease == 'normal')"
        Int max_cells = 100  # if -1, return all cells matching query
        Int min_cells_per_dataset_donor = 20
        Boolean average_expression = false
        String organism = "Homo sapiens"
        String X_name = "raw"
        Boolean remove_unmeasured_features = true
        String obs_column_names = '["assay", "cell_type", "tissue", "dataset_id", "donor_id", "tissue_general", "suspension_type", "disease", "sex"]'

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
        import scipy.sparse as sp

        # always include soma_joinid in obs_column_names
        obs_column_names = ~{obs_column_names}  # plopping in this bare text results in this being a list of strings
        obs_column_names = list(set(obs_column_names).union(set(["soma_joinid"])))  # add soma_joinid if not present
        print(f"obs_column_names: {obs_column_names}")

        with cellxgene_census.open_soma() as census:
            # print info about the census
            info = census["census_info"]["summary"].read().concat().to_pandas()
            print(info)
            info = census["census_info"]["summary_cell_counts"].read().concat().to_pandas()
            print(info[info["category"] == "all"])

            # run the obs query, returning only cell metadata
            # Reads SOMADataFrame as a slice
            print("obs_value_filter:")
            print("""~{obs_value_filter}""")
            value_filter_string = """(cell_type in ~{obs_cell_types}) and ~{obs_value_filter}"""
            print("value_filter:")
            print(value_filter_string)
            print("Running the obs query...")
            organism = "~{organism}".lower().replace(" ", "_")
            obs = census["census_data"][organism].obs.read(
                value_filter=value_filter_string,
                column_names=obs_column_names,
            ).concat().to_pandas()  # to pyarrow.Table, to pandas.DataFrame
            print(obs)

        if len(obs) == 0:
            raise ValueError("cellxgene-census found no cells found matching the intitial query")

        # optional removal of cells from donors with fewer than a certain number
        obs["dataset_donor"] = obs["dataset_id"].astype(str) + "::" + obs["donor_id"].astype(str)
        vc = obs["dataset_donor"].value_counts()
        print("Combination of dataset and donor:")
        print(vc)
        print("Removing gruops with fewer than ~{min_cells_per_dataset_donor} cells")
        allowed_groups = vc.index[vc >= ~{min_cells_per_dataset_donor}]
        obs = obs[obs["dataset_donor"].isin(allowed_groups)].copy()
        print("Remaining data:")
        print(obs)
        print(obs["dataset_donor"].value_counts())

        if len(obs) == 0:
            raise ValueError("removal of (dataset_id, donor_id) combinations with fewer than ~{min_cells_per_dataset_donor} cells left no cells")

        # limit to max_cells
        if (~{max_cells} > 0) and (len(obs) > ~{max_cells}):
            print("Limiting to ~{max_cells} cells, randomly shuffling first")
            obs = obs.iloc[np.random.permutation(len(obs))][:~{max_cells}].copy()
            print(obs)
        else:
            print("Not limiting number of cells")
            print("max_cells: ~{max_cells}")
            print(f"len(obs): {len(obs)}")

        # run the count matrix query only for cells required
        requested_soma_joinids = obs["soma_joinid"].unique().tolist()
        print(f"Requesting {len(requested_soma_joinids)} cells")
        if len(requested_soma_joinids) == 0:
            raise ValueError("Not requesting any cells... this error is not anticipated")
        print("Running the count matrix query...")
        with cellxgene_census.open_soma() as census:
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="~{organism}",
                measurement_name="RNA",
                X_name="~{X_name}",
                obs_coords=requested_soma_joinids,
                obs_column_names=obs_column_names,
            )
        print(adata)

        if len(adata) == 0:
            raise ValueError("cellxgene-census returned no cells for the count matrix query... this error is not anticipated")

        # optional averaging
        all_dataset_ids = adata.obs["dataset_id"].unique()
        if "~{average_expression}" == "true":
            print("Averaging expression of the above cells...")
            all_soma_joinids = adata.obs["soma_joinid"].unique().tolist()
            adata = anndata.AnnData(
                X=sp.csr_matrix(adata.X.mean(axis=0)),
                obs=adata.obs.iloc[[0]],
                var=adata.var,
            )
            assert len(adata) == 1, "something went wrong during pseudobulking"
            # keep a record of all the cells included
            adata.obs.loc[adata.obs_names[0], "soma_joinid"] = ",".join([str(id) for id in all_soma_joinids])

        # optional removal of unmeasured features
        if "~{remove_unmeasured_features}" == "true":
            print("Removing unmeasured features...")
            with cellxgene_census.open_soma() as census:
                # https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_dataset_presence.html
                # grab the necessary from cellxgene
                var_df = census["census_data"]["homo_sapiens"].ms["RNA"].var.read().concat().to_pandas()
                datasets_df = census["census_info"]["datasets"].read().concat().to_pandas()
                presence_matrix = cellxgene_census.get_presence_matrix(census, organism="Homo sapiens", measurement_name="RNA")
            # Slice the dataset(s) of interest, and get the joinid(s)
            dataset_joinids = datasets_df.loc[datasets_df["dataset_id"].isin(all_dataset_ids)]["soma_joinid"]
            # Slice the presence matrix by the first dimension, i.e., by dataset
            var_joinids = presence_matrix[dataset_joinids, :].tocoo().col
            # From the feature (var) dataframe, slice out features which have a joinid in the list.
            subset_var_df = var_df.loc[var_df["soma_joinid"].isin(var_joinids)]

            # subset to genes measured
            adata = adata[:, adata.var["feature_id"].isin(subset_var_df["feature_id"].values)].copy()
            print(adata)
            assert len(subset_var_df) == adata.shape[1], "something went wrong subsetting features to those measured"

        # format correctly for cellarium
        adata.var.index = adata.var["feature_id"].copy()
        adata.var["ensembl_id"] = adata.var["feature_id"].copy()
        adata.var["gene_name"] = adata.var["feature_name"].copy()
        adata.layers["count"] = adata.X.copy()
        adata.obs["total_mrna_umis"] = np.array(adata.X.sum(axis=1)).squeeze()

        # show adata     
        print("Final data:")
        print(adata)

        # checks
        assert set(adata.obs["cell_type"].unique().tolist()) == set(~{obs_cell_types}), "cell types are not as expected"

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
        obs_cell_types :
            {help: "the cellxgene-census query will include (cell_type in <obs_cell_types>). e.g., in double quotes: ['retinal pigment epithelial cell']"}
        obs_value_filter :
            {help: "the cellxgene-census query (excluding cell type: above), see https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_query_extract.html"}
        max_cells :
            {help: "max allowed number of cells in the h5ad file; set to -1 to include all cells matching the query"}
        min_cells_per_dataset_donor :
            {help: "only include cells if a given (dataset_id, donor_id) combination has at least this many cells matching the query. this helps remove chance spurious labels."}
        average_expression :
            {help: "true to produce a single pseudobulk output 'cell' with the mean expression of each gene over cells; false to return the single cell data."}
        organism :
            {help: "defaults to 'Homo sapiens', see https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata"}
        X_name :
            {help: "use 'raw' to query raw counts. see https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata"}
        remove_unmeasured_features :
            {help: "true to remove features that were not measured in the dataset (as annotated by cellxgene); false to include all possible features measured in any dataset"}
        obs_column_names :
            {help: "column names to include in obs. soma_joinid will be included regardless of whether you include it here."}
    }
}

workflow query_cellxgene_workflow {

    call query_cellxgene

    output {
        File h5ad = query_cellxgene.output_h5ad
    }
}
