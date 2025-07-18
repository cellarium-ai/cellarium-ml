# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, Sequence, TypedDict
import shutil

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml import CellariumModule, CellariumAnnDataDataModule
from cellarium.ml.models import SingleCellVariationalInference
from cellarium.ml.utilities.data import collate_fn, densify, categories_to_codes, AnnDataField
from tests.common import BoringDatasetSCVI


@pytest.mark.parametrize("n_batch", [2, 4], ids=lambda s: f"n_batch_{s}")
@pytest.mark.parametrize("n_latent_batch", [None, 4], ids=["batch_latent_size_default", "batch_latent_size_4"])
@pytest.mark.parametrize(
    "batch_embedded,batch_representation_sampled,batch_kl_weight",
    [(False, False, 0), (True, False, 0), (True, True, 0), (True, True, 0.1)],
    ids=["batch_not_embedded", "batch_embedded_not_sampled", "batch_embedded_sampled", "batch_embedded_sampled_KL"],
)
def test_load_from_checkpoint_multi_device(
    n_batch: int,
    n_latent_batch: int | None,
    batch_embedded: bool,
    batch_representation_sampled: bool,
    batch_kl_weight: float,
    tmp_path: Path,
):
    n, g = 100, 50
    batch_size = 8  # must be > 1 for BatchNorm
    var_names_g = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCVI(
            data=np.random.poisson(lam=2.0, size=(n, g)),
            batch_index_n=np.random.randint(0, n_batch, size=n),
            var_names=np.array(var_names_g),
        ),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    # model
    if (not batch_embedded) and (n_latent_batch is not None) and (n_latent_batch != n_batch):
        with pytest.raises(ValueError):
            model = SingleCellVariationalInference(
                var_names_g=var_names_g,
                n_batch=n_batch,
                batch_embedded=batch_embedded,
                batch_representation_sampled=batch_representation_sampled,
                n_latent_batch=n_latent_batch,
                batch_kl_weight_max=batch_kl_weight,
                encoder={
                    "hidden_layers": [
                        {
                            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
                        },
                    ],
                    "final_layer": {
                        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                        "init_args": {"batch_to_bias_hidden_layers": []},
                    },
                },
                decoder={
                    "hidden_layers": [
                        {
                            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
                        },
                    ],
                    "final_layer": {
                        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                        "init_args": {"batch_to_bias_hidden_layers": []},
                    },
                    "final_additive_bias": False,
                },
            )

    else:
        model = SingleCellVariationalInference(
            var_names_g=var_names_g,
            n_batch=n_batch,
            batch_embedded=batch_embedded,
            batch_representation_sampled=batch_representation_sampled,
            n_latent_batch=n_latent_batch,
            batch_kl_weight_max=0,
            encoder={
                "hidden_layers": [
                    {
                        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                        "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
                    },
                ],
                "final_layer": {
                    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                    "init_args": {"batch_to_bias_hidden_layers": []},
                },
            },
            decoder={
                "hidden_layers": [
                    {
                        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                        "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
                    },
                ],
                "final_layer": {
                    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                    "init_args": {"batch_to_bias_hidden_layers": []},
                },
                "final_additive_bias": False,
            },
        )

        module = CellariumModule(
            model=model,
            optim_fn=torch.optim.Adam,
            optim_kwargs={"lr": 1e-3},
        )
        # trainer
        strategy: str | DDPStrategy = (
            DDPStrategy(
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            if devices > 1
            else "auto"
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            strategy=strategy,
            devices=devices,
            max_epochs=1,
            default_root_dir=tmp_path,
        )
        # fit
        trainer.fit(module, train_dataloaders=train_loader)

        # run tests only for rank 0
        if trainer.global_rank != 0:
            return

        # load model from checkpoint
        ckpt_path = (
            tmp_path / f"lightning_logs/version_0/checkpoints/epoch=0-step={math.ceil(n / batch_size / devices)}.ckpt"
        )
        assert ckpt_path.is_file()
        loaded_model = CellariumModule.load_from_checkpoint(ckpt_path).model
        assert isinstance(loaded_model, SingleCellVariationalInference)
        assert np.array_equal(model.var_names_g, loaded_model.var_names_g)
        assert hasattr(model.z_encoder.fully_connected.module_list[0].layer, "weight")
        assert hasattr(loaded_model.z_encoder.fully_connected.module_list[0].layer, "weight")
        # check that the weights are the same
        torch.testing.assert_close(
            model.z_encoder.fully_connected.module_list[0].layer.weight,
            loaded_model.z_encoder.fully_connected.module_list[0].layer.weight,
        )


class SCVIKwargs(TypedDict, total=False):
    # this is for mypy
    var_names_g: Sequence[str]
    encoder: dict
    decoder: dict
    n_batch: int
    n_latent: int
    n_continuous_cov: int
    n_cats_per_cov: list[int]
    dropout_rate: float
    dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"]
    log_variational: bool
    gene_likelihood: Literal["zinb", "nb", "poisson"]
    latent_distribution: Literal["normal", "ln"]
    batch_embedded: bool
    batch_representation_sampled: bool
    n_latent_batch: int | None
    batch_kl_weight: float
    use_batch_norm: Literal["encoder", "decoder", "none", "both"]
    use_layer_norm: Literal["encoder", "decoder", "none", "both"]
    use_size_factor_key: bool
    use_observed_lib_size: bool


linear_encoder_kwargs: dict = {
    "hidden_layers": [],
    "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
}

linear_decoder_kwargs: dict = {
    "hidden_layers": [],
    "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
    "final_additive_bias": False,
}

var_names_g = [f"gene_{i}" for i in range(10)]

standard_kwargs: SCVIKwargs = dict(
    var_names_g=var_names_g,
    encoder=linear_encoder_kwargs,
    decoder=linear_decoder_kwargs,
    n_batch=4,
    n_latent=10,
    n_continuous_cov=0,
    n_cats_per_cov=[],
    dispersion="gene",
    log_variational=True,
    gene_likelihood="nb",
    latent_distribution="normal",
    batch_embedded=False,
    batch_representation_sampled=False,
    n_latent_batch=None,
    batch_kl_weight_max=0.0,
    use_batch_norm="both",
    use_layer_norm="none",
)


def test_vae_architectures():
    # linear model, no batch injection at all
    print("linear model, no batch injection at all")
    print(standard_kwargs)
    model = SingleCellVariationalInference(**standard_kwargs)

    # batch injection in encoder but not decoder
    kwargs: SCVIKwargs = copy.deepcopy(standard_kwargs)
    kwargs["encoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
    ]
    print("batch injection in encoder but not decoder")
    print(kwargs)
    model = SingleCellVariationalInference(**kwargs)

    # batch injection in decoder but not encoder
    kwargs2: SCVIKwargs = copy.deepcopy(standard_kwargs)
    kwargs2["decoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
    ]
    print("batch injection in decoder but not encoder")
    print(kwargs2)
    model = SingleCellVariationalInference(**kwargs2)

    # batch injection in both encoder and decoder
    kwargs3: SCVIKwargs = copy.deepcopy(standard_kwargs)
    kwargs3["encoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
    ]
    kwargs3["decoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
    ]
    print("batch injection in both encoder and decoder")
    print(kwargs3)
    model = SingleCellVariationalInference(**kwargs3)

    # batch injection in both encoder and decoder, 2 hidden layers each
    kwargs4: SCVIKwargs = copy.deepcopy(standard_kwargs)
    kwargs4["encoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 16, "batch_to_bias_hidden_layers": []},
        },
    ]
    kwargs4["decoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 16, "batch_to_bias_hidden_layers": []},
        },
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
    ]
    print("batch injection in both encoder and decoder, 2 hidden layers each")
    print(kwargs4)
    model = SingleCellVariationalInference(**kwargs4)

    # batch injection in both encoder and decoder with decoder final_additive_bias
    kwargs5: SCVIKwargs = copy.deepcopy(standard_kwargs)
    kwargs5["encoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
    ]
    kwargs5["decoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
            "final_additive_bias": True,
        },
    ]
    print("batch injection in both encoder and decoder with decoder final additive bias")
    print(kwargs5)
    model = SingleCellVariationalInference(**kwargs5)

    # batch injection in both encoder and decoder with decoder final_additive_bias with batch embedded and sampled
    kwargs6: SCVIKwargs = copy.deepcopy(standard_kwargs)
    kwargs6["batch_embedded"] = True
    kwargs6["batch_representation_sampled"] = True
    kwargs6["n_latent_batch"] = 2
    kwargs6["encoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
        },
    ]
    kwargs6["decoder"]["hidden_layers"] = [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
            "final_additive_bias": True,
        },
    ]
    print(
        "batch injection in both encoder and decoder with decoder final additive bias with batch embedded and sampled"
    )
    print(kwargs6)
    model = SingleCellVariationalInference(**kwargs6)

    # we are merely testing that the models can be instantiated above
    # here put some data through the last model to make sure it runs
    print("putting data through the last model")
    n_batch = 2
    n = 4
    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCVI(
            data=np.random.poisson(lam=2.0, size=(n, len(var_names_g))),
            batch_index_n=np.random.randint(0, n_batch, size=n),
            var_names=np.array(var_names_g),
        ),
        collate_fn=collate_fn,
        batch_size=n,
    )
    for batch in train_loader:
        # encoder
        batch_nb = model.batch_representation_from_batch_index(batch["batch_index_n"])
        print("putting data through encoder")
        latent = model.z_encoder(x_ng=batch["x_ng"].float(), batch_nb=batch_nb, categorical_covariate_np=None)
        z_nk = latent.sample()
        # decoder
        print("putting data through decoder")
        model.decoder(
            z_nk=z_nk,
            batch_nb=batch_nb,
            categorical_covariate_np=None,
            inverse_overdispersion=model.px_r.exp(),
            library_size_n1=batch["x_ng"].float().sum(dim=-1, keepdim=True),
        )


def compute_neighbor_accuracy(
    train_data: anndata.AnnData, 
    test_data: anndata.AnnData,
    latent_obsm_key: str = "X_cellarium",
    labels_obs_key: str = "cell_type",
    metric: Literal["cosine", "euclidean"] = "euclidean",
) -> float:
    """
    Compute the following accuracy metric:
        - for each test sample, find the nearest training sample in latent space
        - compute the accuracy as the fraction of neighbor training samples that have the right label
    """
    from sklearn.neighbors import NearestNeighbors

    # pull data from anndata objects
    train_latent = train_data.obsm[latent_obsm_key]
    test_latent = test_data.obsm[latent_obsm_key]
    train_labels = train_data.obs[labels_obs_key].astype(str).values
    test_labels = test_data.obs[labels_obs_key].astype(str).values

    # compute the nearest neighbors of each test point
    nbrs = NearestNeighbors(n_neighbors=1, metric=metric).fit(train_latent)
    _, indices = nbrs.kneighbors(test_latent)

    # get the labels of the nearest neighbors and compute accuracy
    train_labels = train_labels[indices.flatten()]
    accuracy = np.mean(train_labels == test_labels)

    return accuracy


@pytest.fixture(scope="module")
def testing_anndatas() -> tuple[anndata.AnnData, anndata.AnnData]:
    """
    Get the train and test data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # # data
        # train_data = "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/UBERON_0002115_train.h5ad"
        # test_data = "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/UBERON_0002115_test.h5ad"

        # # download the data
        # train_path = tmpdir / "train.h5ad"
        # test_path = tmpdir / "test.h5ad"
        # import requests

        # response = requests.get(train_data)
        # response.raise_for_status()
        # with open(train_path, "wb") as f:
        #     f.write(response.content)

        # response = requests.get(test_data)
        # response.raise_for_status()
        # with open(test_path, "wb") as f:
        #     f.write(response.content)

        # temp hack because I'm on slow wifi =======================
        train_path = Path("/Users/sfleming/Downloads/UBERON_0002115_train.h5ad")
        test_path = Path("/Users/sfleming/Downloads/UBERON_0002115_test.h5ad")
        shutil.copy(train_path, tmpdir / "train.h5ad")
        shutil.copy(test_path, tmpdir / "test.h5ad")
        # ===========================================================

        # print out the contents of the temp directory
        print(f"tmpdir contents: {os.listdir(tmpdir)}")

        train_data = anndata.read_h5ad(tmpdir / "train.h5ad")
        test_data = anndata.read_h5ad(tmpdir / "test.h5ad")
        return train_data, test_data


# # Lys test case
# n_latent: int = 50
# n_hidden: int = 512
# n_layers: int = 2
# batch_size: int = 1024
# max_epochs: int = 5

# small dataset test case
n_latent: int = 10
n_hidden: int = 128
n_layers: int = 1
batch_size: int = 512
max_epochs: int = 10


# other params
n_epochs_kl_warmup: int = 10
max_z_kl_weight: float = 10.0
batch_key: str = "batch_concat_cellxgene"


@pytest.fixture(scope="module")
def train_scvi_tools_model(
    testing_anndatas,
    latent_obsm_key: str = "X_scvi",
) -> tuple[np.ndarray, np.ndarray, anndata.AnnData, anndata.AnnData]:
    """
    Train a scvi-tools model on the training data and embed both the training and test data.

    Returns:
        - train_data: the training data with the model's latent representation added to obsm[latent_obsm_key]
        - test_data: the test data with the model's latent representation added to obsm[latent_obsm_key]
    """
    
    # retrieve the training and test data from the fixture
    train_data, test_data = testing_anndatas

    # train the scvi-tools model on the training data
    from scvi.model import SCVI

    # set up and train scvi-tools model
    SCVI.setup_anndata(
        train_data,
        batch_key=batch_key,
        categorical_covariate_keys=None,
    )
    model = SCVI(train_data, gene_likelihood="nb", n_latent=n_latent, 
                 n_layers=n_layers, n_hidden=n_hidden, dispersion="gene",
                 deeply_inject_covariates=True)
    model.train(max_epochs=max_epochs, train_size=1, batch_size=batch_size, 
                plan_kwargs={"n_epochs_kl_warmup": n_epochs_kl_warmup, 
                             "max_kl_weight": max_z_kl_weight, 
                             "min_kl_weight": 0.0})
    # embed the training data
    train_latent = model.get_latent_representation(train_data)
    # embed the test data
    SCVI.setup_anndata(test_data)
    test_latent = model.get_latent_representation(test_data)

    # add the latent representation to the obsm of the training and test data
    train_data.obsm[latent_obsm_key] = train_latent
    test_data.obsm[latent_obsm_key] = test_latent

    return train_data, test_data


@pytest.fixture(scope="module")
def train_cellarium_model(
    testing_anndatas,
    latent_obsm_key: str = "X_cellarium",
) -> tuple[anndata.AnnData, anndata.AnnData]:
    """
    Train a Cellarium model on the training data and embed both the training and test data.

    Returns:
        - train_data: the training data with the model's latent representation added to obsm[latent_obsm_key]
        - test_data: the test data with the model's latent representation added to obsm[latent_obsm_key]
    """
    
    # retrieve the training and test data from the fixture
    train_data, test_data = testing_anndatas

    # set up and train Cellarium model
    cellarium_model = SingleCellVariationalInference(
        var_names_g=train_data.var_names.values,
        n_batch=train_data.obs[batch_key].nunique(),
        n_latent=n_latent,
        kl_annealing_start=0.0,
        kl_warmup_epochs=n_epochs_kl_warmup,
        z_kl_weight_max=max_z_kl_weight,
        batch_kl_weight_max=0.0,
        batch_embedded=False,
        batch_representation_sampled=False,
        gene_likelihood="nb",
        dispersion="gene",
        encoder={
            "hidden_layers": [
                {
                    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                    "init_args": {"out_features": n_hidden, "batch_to_bias_hidden_layers": []},
                    "dressing_init_args": {
                        "use_batch_norm": True,
                        "use_layer_norm": False,
                        "dropout_rate": 0.1,
                    },
                },
            ] * n_layers,
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
        },
        decoder={
            "hidden_layers": [
                {
                    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                    "init_args": {"out_features": n_hidden, "batch_to_bias_hidden_layers": []},
                    "dressing_init_args": {
                        "use_batch_norm": True,
                        "use_layer_norm": False,
                        "dropout_rate": 0.0,
                    },
                },
            ] * n_layers,
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            "final_additive_bias": False,
        },
    )
    
    module = CellariumModule(
        model=cellarium_model,
        optim_fn=torch.optim.AdamW,
        optim_kwargs={"lr": 1e-3, "weight_decay": 1e-6, "eps": 0.01},  # trying to match scvi-tools defaults
    )

    # data
    train_datamodule = CellariumAnnDataDataModule(
        dadc=train_data,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "var_names_g": AnnDataField(attr="var_names"),
            "batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes),
            "obs_names_n": AnnDataField(attr="obs_names"),
        },
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    test_datamodule = CellariumAnnDataDataModule(
        dadc=test_data,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "var_names_g": AnnDataField(attr="var_names"),
            "batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes),
            "obs_names_n": AnnDataField(attr="obs_names"),
        },
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    # trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=tempfile.gettempdir(),
    )
    
    # fit
    trainer.fit(module, train_datamodule)
    
    # embed the training data
    train_prediction_output = trainer.predict(module, train_datamodule)
    train_latent = pd.concat(
        [pd.DataFrame(out["x_ng"], index=out["obs_names_n"].astype(str)) for out in train_prediction_output], 
        axis=0,
    )
    
    # embed the test data
    test_prediction_output = trainer.predict(module, test_datamodule)
    test_latent = pd.concat(
        [pd.DataFrame(out["x_ng"], index=out["obs_names_n"].astype(str)) for out in test_prediction_output], 
        axis=0,
    )

    # add the latent representation to the obsm of the training and test data
    # print(train_data.obs_names)
    # print(train_latent.index)
    train_data.obsm[latent_obsm_key] = train_latent.loc[train_data.obs_names].values
    test_data.obsm[latent_obsm_key] = test_latent.loc[test_data.obs_names].values

    return train_data, test_data


@pytest.mark.parametrize("metric", ["euclidean", "cosine"], ids=["euclidean", "cosine"])
@pytest.mark.parametrize("annotation_key", 
                         ["cell_type", "cell_type_coarse_ontology_term_id"], 
                         ids=["celltype", "coarsecelltype"])
def test_latent_accuracy_metric(
    train_scvi_tools_model, 
    train_cellarium_model,
    metric: Literal["euclidean", "cosine"], 
    annotation_key: Literal["cell_type", "cell_type_coarse_ontology_term_id"],
):
    """
    Run the following test:
    - train on real training data (that has author labels)
    - embed the training data (that has author labels)
    - embed held-out test data
    - compute the following accuracy metric:
        - for each test sample, find the nearest training sample in latent space
        - compute the accuracy as the fraction of neighbor training samples that have the right label

    Compare the accuracy metric to that same metric computed via scvi-tools (with some margin of error).
    """
    tolerable_discrepancy = 0.01  # 1% discrepancy is acceptable

    # compute the accuracy metric for scvi-tools
    train_data, test_data = train_scvi_tools_model
    accuracy_scvi_tools = compute_neighbor_accuracy(
        train_data=train_data,
        test_data=test_data,
        latent_obsm_key="X_scvi",
        labels_obs_key=annotation_key,
        metric=metric,
    )
    print(f"scvi-tools accuracy ({metric}): {accuracy_scvi_tools:.4f}")

    # compute the accuracy metric for Cellarium
    train_data, test_data = train_cellarium_model
    accuracy_cellarium = compute_neighbor_accuracy(
        train_data=train_data,
        test_data=test_data,
        latent_obsm_key="X_cellarium",
        labels_obs_key=annotation_key,
        metric=metric,
    )
    print(f"cellarium accuracy ({metric}): {accuracy_cellarium:.4f}")

    assert accuracy_cellarium > accuracy_scvi_tools - tolerable_discrepancy, (
        f"Cellarium ({accuracy_cellarium:.4f}); scvi-tools ({accuracy_scvi_tools:.4f})"
    )
