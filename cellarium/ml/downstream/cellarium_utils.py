"""Utility functions for notebook work using cellarium models."""

from cellarium.ml.core import CellariumPipeline, CellariumModule
from cellarium.ml.utilities.data import AnnDataField, densify
from cellarium.ml.core.datamodule import CellariumAnnDataDataModule
from cellarium.cas.data_preparation import sanitize

import torch
import anndata
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
        tmp_file = 'model.ckpt'
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
        anndata_filename = os.path.join(tmpdir, "test_cm.h5ad")
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
        count_matrix_name="X",
        feature_ids_column_name="ensembl_id",
    )
    adata.obs_names = obs_names
    adata.var['gene_name'] = var['gene_name']
    adata.var['ensembl_id'] = adata.var.index.copy()

    return adata
