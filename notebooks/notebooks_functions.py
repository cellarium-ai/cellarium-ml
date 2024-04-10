from cellarium.ml.core import CellariumPipeline, CellariumModule
from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.data.fileio import read_h5ad_file
from cellarium.ml.utilities.data import AnnDataField, densify, categories_to_codes

import torch
import numpy as np
import anndata
import tqdm

import tempfile


class AutosizedDistributedAnnDataCollection(DistributedAnnDataCollection):

    def __init__(self, *args, **kwargs):
        # I'm being lazy here and doing something real ugly
        # I want it to take the shard_size from the first file
        try:
            # this allows super to find the list of filenames
            super().__init__(*args, **kwargs)
        except AssertionError:
            try:
                # this allows super to create the cache
                kwargs.pop("shard_size")
                kwargs = kwargs | {"shard_size": 10000}
                super().__init__(*args, **kwargs)
            except AssertionError:
                pass
            # load first file and cache it
            adata0 = self.cache[self.filenames[0]] = read_h5ad_file(self.filenames[0])
            # pull shard_size from that file
            kwargs.pop("shard_size")
            kwargs = kwargs | {"shard_size": len(adata0)}
            # finally initialize for real
            super().__init__(*args, **kwargs)


def get_dataset_from_anndata(
    adata: anndata.AnnData | str,
    batch_size: int = 128,
    shard_size: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
):
    """
    Get IterableDistributedAnnDataCollectionDataset from an AnnData object or h5ad file specifier.

    Args:
        adata: AnnData object or h5ad file, allowing brace notation for several files.
        batch_size: Batch size.
        shard_size: Shard size.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        IterableDistributedAnnDataCollectionDataset.
    """

    if isinstance(adata, anndata.AnnData):
        tmpfile = tempfile.mkstemp(suffix='.h5ad')
        adata.write(tmpfile[1])
        file = tmpfile[1]
    else:
        file = adata

    dadc = AutosizedDistributedAnnDataCollection(
        file,
        shard_size=shard_size,
        max_cache_size=1,
    )

    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "var_names_g": AnnDataField(attr="var_names"),
            "batch_index_n": AnnDataField(attr="obs", key="batch", convert_fn=categories_to_codes),
        },
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    return dataset

def embed(
    dataset: IterableDistributedAnnDataCollectionDataset,
    pipeline: CellariumPipeline,
    maximum_anndata_files_to_download: int = 5,
    obsm_key_added: str = 'X_scvi',
    device : str = "cpu",
) -> anndata.AnnData:
    """
    Embed the dataset using the pipeline.

    Args:
        dataset: Dataset.
        pipeline: Pipeline.
        maximum_anndata_files_to_download: Maximum number of anndata files to download.

    Returns:
        AnnData with scVI embeddings in adata.obsm[obsm_key_added]
    """

    # get the anndata object
    adatas = [dataset.dadc.adatas[i].adata for i in range(min(maximum_anndata_files_to_download, len(dataset.dadc.adatas)))]
    adata = anndata.concat(adatas, axis=0, merge="same")

    # get the latent space dimension
    latent_space_dim = pipeline[-1].z_encoder.mean_encoder.out_features

    # run the pipeline
    i = 0
    adata.obsm[obsm_key_added] = np.zeros((len(adata), latent_space_dim), dtype=np.float32)
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            batch['x_ng'] = torch.from_numpy(batch['x_ng']).to(device)
            batch['batch_index_n'] = torch.from_numpy(batch['batch_index_n']).to(device)
            out = pipeline.predict(batch)
            z_mean_nk = out['qz'].mean
            adata.obsm[obsm_key_added][i:(i + len(z_mean_nk)), :] = z_mean_nk.cpu().numpy()
            i += len(z_mean_nk)

    return adata