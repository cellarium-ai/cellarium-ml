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


def get_pretrained_model_as_pipeline(
    trained_model: str = "gs://dsp-cell-annotation-service/cellarium/trained_models/cerebras/lightning_logs/version_0/checkpoints/epoch=2-step=83250.ckpt", 
    transforms: list[torch.nn.Module] = [],
    device: str = "cuda",
) -> CellariumPipeline:

    # download the trained model
    with tempfile.TemporaryDirectory() as tmpdir:

        # download file
        tmp_file = os.path.join(tmpdir, 'model.ckpt')
        os.system(f"gsutil cp {trained_model} {tmp_file}")

        # load the model
        model = CellariumModule.load_from_checkpoint(tmp_file).model

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

    with tempfile.TemporaryDirectory() as tmpdir:
        anndata_filename = os.path.join(tmpdir, "tmp.h5ad")
        adata.write(anndata_filename)
        dm = CellariumAnnDataDataModule(
            anndata_filename,
            shard_size=len(adata),
            max_cache_size=1,
            batch_keys={
                "x_ng": AnnDataField(attr="X", convert_fn=densify),
                "var_names_g": AnnDataField(attr="var_names"),
            },
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    dm.setup()
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
        cas_feature_schema_list=pipeline[-1].var_names_g,
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
