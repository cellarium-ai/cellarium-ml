"""Utility functions for notebook work using cellarium models."""

from cellarium.ml.core import CellariumPipeline, CellariumModule
from cellarium.ml.utilities.data import AnnDataField, densify
from cellarium.ml.core.datamodule import CellariumAnnDataDataModule

import torch
import anndata
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import tempfile
from google.cloud import storage


categorical_model = "gs://cellarium-ml/curriculum/human_10x_gt_8000/models/cellarium_gpt/benchmark/measured_context/bs_200_context_4500_max_prefix_4000/lightning_logs/version_0/checkpoints/epoch=4-step=159600.ckpt"
categorical_model_with_downsampling = "gs://cellarium-ml/curriculum/homo_sap_no_cancer/models/cellarium_gpt/downsample/bs_200_max_prefix_4000_context_4500/lightning_logs/version_0/checkpoints/epoch=0-step=152100.ckpt"
default_model = categorical_model_with_downsampling


def get_pretrained_model_as_pipeline(
    trained_model: str = default_model, 
    transforms: list[torch.nn.Module] = [],
    device: str = "cuda",
) -> CellariumPipeline:

    if trained_model.startswith("gs://"):
        # download the trained model
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = os.path.join(tmpdir, 'model.ckpt')

            client = storage.Client()
            bucket_name = trained_model.split("/")[2]
            blob_name = "/".join(trained_model.split("/")[3:])
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(tmp_file)

            # load the model
            model = CellariumModule.load_from_checkpoint(tmp_file).model
    else:
        model = CellariumModule.load_from_checkpoint(trained_model).model

    # insert the trained model params
    model.to(device)
    model.eval()

    # construct the pipeline
    pipeline = CellariumPipeline(transforms + [model])

    return pipeline


def get_datamodule(
    adata: anndata.AnnData, 
    batch_size: int = 4, 
    num_workers: int = 4, 
    shuffle: bool = False, 
    seed: int = 0, 
    drop_last: bool = False,
) -> CellariumAnnDataDataModule:
    """
    Create a CellariumAnnDataDataModule from an AnnData object.

    Returns:
        CellariumAnnDataDataModule with methods train_dataloader() and predict_dataloader()
    """

    # with tempfile.TemporaryDirectory() as tmpdir:
    #     anndata_filename = os.path.join(tmpdir, "tmp.h5ad")
    #     adata.write(anndata_filename)
    batch_keys={
        "x_ng": AnnDataField(attr="X", convert_fn=densify),
        "var_names_g": AnnDataField(attr="var_names"),
    }
    if "total_mrna_umis_n" in adata.obs.columns:
        batch_keys |= {
            "total_mrna_umis_n": AnnDataField(
                attr="obs", 
                key="total_mrna_umis",
                convert_fn=np.asarray,
            ),
        }
    
    dm = CellariumAnnDataDataModule(
        adata,
        batch_keys=batch_keys,
        batch_size=batch_size,
        # train_size=None,
        # val_size=0.99,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    dm.setup(stage="predict")
    return dm


def harmonize_anndata_with_model(adata: anndata.AnnData, pipeline: CellariumPipeline):
    """
    Use sanitize function to harmonize anndata with the model feature schema.

    Args:
        adata: AnnData object
        pipeline: CellariumPipeline object

    Returns:
        AnnData object with harmonized feature schema
    """

    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    var = adata.var.set_index('ensembl_id').copy()

    # sanitize the data to match the model feature schema
    obs_names = adata.obs_names.copy()
    adata = sanitize(
        adata=adata,
        cas_feature_schema_list=pipeline[-1].gene_categories,
        count_matrix_input="X",
        feature_ids_column_name="ensembl_id",
    )
    adata.obs_names = obs_names
    adata.var['gene_name'] = var['gene_name']
    adata.var['ensembl_id'] = adata.var.index.copy()

    return adata


def _get_adata_var_index_or_by_column(adata: anndata.AnnData, var_column_name: str) -> list[str]:
    if var_column_name == "index":
        return adata.var.index.tolist()

    return adata.var[var_column_name].values.tolist()


def sanitize(
    adata: anndata.AnnData,
    cas_feature_schema_list: list[str],
    feature_ids_column_name: str,
    feature_names_column_name: str | None = None,
    count_matrix_input: str = "X",
) -> anndata.AnnData:
    """
    Cellarium CAS sanitizing script. Returns a new `anndata.AnnData` instance, based on the feature expression
    matrix of the input instance. Extra features get omitted. Missing features get filled with zeros.

    :param adata: Instance to sanitize
    :param cas_feature_schema_list: List of Ensembl feature ids to rely on.
    :param count_matrix_input: Where to obtain a feature expression count matrix from. Choice of: 'X', 'raw.X'
    :param feature_ids_column_name: Column name where to obtain Ensembl feature ids. Default `index`.
    :param feature_names_column_name: Column name where to obtain feature names. If not provided, no feature names
        should be mapped |br|
    `Default`: ``None``
    :return: `anndata.AnnData` instance that is ready to use with CAS
    """
    if feature_ids_column_name != "index" and feature_ids_column_name not in adata.var.columns.values:
        raise ValueError(
            "`feature_ids_column_name` should have a value of either 'index' "
            "or be present as a column in the `adata.var` object."
        )
    if feature_names_column_name not in {"index", None} and feature_names_column_name not in adata.var.columns.values:
        raise ValueError(
            "`feature_ids_name_column_name` should have a value of either 'index' "
            "or be present as a column in the `adata.var` object."
        )

    adata_feature_schema_list = _get_adata_var_index_or_by_column(adata=adata, var_column_name=feature_ids_column_name)
    original_obs_ids = adata.obs.index.values

    cas_feature_schema_set = set(cas_feature_schema_list)
    adata_feature_schema_set = set(adata_feature_schema_list)
    feature_id_intersection = adata_feature_schema_set.intersection(cas_feature_schema_set)

    cas_feature_id_map = {feature_id: index for index, feature_id in enumerate(cas_feature_schema_list)}
    adata_feature_id_map = {feature_id: index for index, feature_id in enumerate(adata_feature_schema_list)}
    feature_id_intersection_cas_indices = list(map(cas_feature_id_map.get, feature_id_intersection))
    feature_id_intersection_adata_indices = list(map(adata_feature_id_map.get, feature_id_intersection))

    n_cells = adata.shape[0]
    n_features = len(cas_feature_schema_list)
    input_matrix = adata.X if count_matrix_input == "X" else adata.raw.X

    # Translate the columns from one matrix to another, convert to COO format to make this efficient.
    col_trans = np.zeros(n_features, dtype=int)
    for i, k in enumerate(feature_id_intersection_cas_indices):
        col_trans[i] = k
    if not sp.issparse(input_matrix):
        input_matrix = sp.csr_matrix(input_matrix)
    vals = input_matrix.tocsc()[:, feature_id_intersection_adata_indices]
    vals = vals.tocoo()
    new_col = col_trans[vals.col]
    result_matrix = sp.coo_matrix((vals.data, (vals.row, new_col)), shape=(n_cells, n_features))
    del col_trans, vals, new_col

    # Create `obs` index
    obs = adata.obs.copy()
    obs.index = original_obs_ids

    var_df_data = {}
    if feature_names_column_name is not None:
        adata_feature_name_list = _get_adata_var_index_or_by_column(
            adata=adata, var_column_name=feature_names_column_name
        )
        gene_id_to_gene_symbol_map = {
            gene_id: gene_symbol for gene_symbol, gene_id in zip(adata_feature_name_list, adata_feature_schema_list)
        }

        cas_feature_name_list = [
            gene_id_to_gene_symbol_map[gene_id] if gene_id in gene_id_to_gene_symbol_map else "N/A"
            for gene_id in cas_feature_schema_list
        ]

        var_df_data["feature_name"] = cas_feature_name_list

    var_df = pd.DataFrame(index=cas_feature_schema_list, data=var_df_data)
    return anndata.AnnData(
        result_matrix.tocsr(),
        obs=obs,
        obsm=adata.obsm,
        var=var_df,
    )


def batch_from_adata(
    adata: anndata.AnnData,
    pipeline: CellariumPipeline,
    gene_inds: torch.LongTensor,
    layer: str,
    query_umis: int = 10000,
):
    """
    Return a generator that yields batches from an AnnData object, ready for use in 
    the forward and predict methods of a Cellarium model.

    Args:
        adata: AnnData object
        pipeline: CellariumPipeline object
        gene_inds: LongTensor of gene indices to use
        layer: layer of the adata object to use
        query_umis: number of total UMIs to use for the query

    Returns:
        Generator that yields batches
    """
    adata.X = adata.layers[layer].copy()
    device = pipeline[-1].transformer.parameters().__next__().device
    gene_inds = torch.unique(gene_inds).sort().values.long().to(device)
    adata_out = adata[:, gene_inds.cpu().numpy()].copy()
    dm = get_datamodule(adata_out)

    # prep tensors
    for batch in dm.predict_dataloader():
        batch["prompt_name_ns"] = np.broadcast_to(batch["var_names_g"].values[None, :], batch["x_ng"].shape)
        batch["prompt_value_ns"] = batch["x_ng"].to(device)
        batch["prompt_total_mrna_umis_n"] = batch.get("total_mrna_umis_n", batch["x_ng"].sum(-1)).to(device)
        batch["prompt_measured_genes_mask_ns"] = torch.ones_like(batch["prompt_value_ns"], dtype=bool)
        batch["query_name_nq"] = batch["prompt_name_ns"]
        batch["query_total_mrna_umis_n"] = torch.ones_like(batch["prompt_total_mrna_umis_n"]) * query_umis
        yield batch
