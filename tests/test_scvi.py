# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math
import os
from pathlib import Path
from typing import Literal, Sequence, TypedDict

import anndata
import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml import CellariumModule
from cellarium.ml.models import SingleCellVariationalInference
from cellarium.ml.models.scvi import EncoderSCVI, DecoderSCVI
from cellarium.ml.utilities.data import collate_fn
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


@pytest.mark.parametrize(
    "use_batch_norm,use_layer_norm",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
    ids=["no_norm", "batch_norm", "layer_norm"],
)
@pytest.mark.parametrize("n_layers", [1, 2], ids=["one_layer", "two_layers"])
@pytest.mark.parametrize("hidden_size", [16, 32], ids=["hidden_16", "hidden_32"])
def test_encoder_matches_scvi_tools(use_batch_norm, use_layer_norm, n_layers, hidden_size):
    try:
        import scvi
        from scvi.nn import Encoder as SCVIEncoder
    except ImportError:
        pytest.skip("scvi-tools is not installed, skipping test")

    # Setup test data
    n, g = 100, 50
    var_names_g = [f"gene_{i}" for i in range(g)]
    X = np.random.poisson(lam=2.0, size=(n, g))
    batch_indices = np.random.randint(0, 2, size=n)

    # Create anndata for scvi-tools
    adata = anndata.AnnData(X)
    adata.var_names = var_names_g
    adata.obs["batch"] = batch_indices
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")

    # set params
    dropout_rate = 0.0
    n_layers = 1  # number of hidden layers in the encoder

    # Initialize scvi-tools encoder
    scvi_encoder = SCVIEncoder(
        n_input=g,
        n_output=10,  # n_latent
        n_cat_list=[2],  # assuming 2 batches for this example
        n_hidden=hidden_size,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        distribution="normal",
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        inject_covariates=False,
    )

    # Initialize Cellarium encoder with matching architecture
    cellarium_encoder = EncoderSCVI(
        in_features=g,
        out_features=10,  # n_latent
        hidden_layers=[
            {
                "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                "init_args": {
                    "out_features": hidden_size,
                    "n_batch": 2,  # assuming 2 batches for this example
                    "batch_to_bias_hidden_layers": [],
                },
                "dressing_init_args": {
                    "use_batch_norm": use_batch_norm,
                    "use_layer_norm": use_layer_norm,
                    "dropout_rate": dropout_rate,
                },
            }
        ]
        + (n_layers - 1)
        * [
            {
                "class_path": "torch.nn.Linear",
                "init_args": {"out_features": hidden_size},
                "dressing_init_args": {
                    "use_batch_norm": use_batch_norm,
                    "use_layer_norm": use_layer_norm,
                    "dropout_rate": dropout_rate,
                },
            }
        ],
        final_layer={"class_path": "torch.nn.Linear", "init_args": {}},
    )

    # Set the same weights manually for both encoders
    with torch.no_grad():
        print("cellarium encoder")
        print(cellarium_encoder)
        print("scvi encoder")
        print(scvi_encoder)

        # Set FC layer weights for layer 1
        assert hasattr(cellarium_encoder.fully_connected.module_list[0].layer, "weight")
        assert hasattr(scvi_encoder.encoder.fc_layers[0][0], "weight")
        assert hasattr(cellarium_encoder.fully_connected.module_list[0].layer, "bias")
        assert hasattr(scvi_encoder.encoder.fc_layers[0][0], "bias")
        cellarium_encoder.fully_connected.module_list[0].layer.weight.copy_(  # type: ignore[operator]
            scvi_encoder.encoder.fc_layers[0][0].weight[:, :g]
        )
        cellarium_encoder.fully_connected.module_list[0].layer.bias.copy_(  # type: ignore[operator]
            scvi_encoder.encoder.fc_layers[0][0].bias[:g]
        )

        # set batch weights for layer 1
        assert hasattr(cellarium_encoder.fully_connected.module_list[0].layer, "bias_decoder")
        assert hasattr(cellarium_encoder.fully_connected.module_list[0].layer.bias_decoder, "module_list")
        assert isinstance(
            cellarium_encoder.fully_connected.module_list[0].layer.bias_decoder.module_list,
            torch.nn.ModuleList,
        )
        assert hasattr(scvi_encoder.encoder.fc_layers[0][0], "weight")
        cellarium_encoder.fully_connected.module_list[0].layer.bias_decoder.module_list[0].weight.copy_(  # type: ignore[operator]
            scvi_encoder.encoder.fc_layers[0][0].weight[:, g:]
        )

        # set FC layer weights for subsequent layers
        for i in range(1, n_layers):
            cellarium_encoder.fully_connected.module_list[i].layer.weight.copy_(  # type: ignore[union-attr, operator]
                scvi_encoder.encoder.fc_layers[i][0].weight
            )
            cellarium_encoder.fully_connected.module_list[i].layer.bias.copy_(  # type: ignore[union-attr, operator]
                scvi_encoder.encoder.fc_layers[i][0].bias
            )

        # Set mean encoder weights
        cellarium_encoder.mean_encoder.weight.copy_(scvi_encoder.mean_encoder.weight)
        cellarium_encoder.mean_encoder.bias.copy_(scvi_encoder.mean_encoder.bias)

        # Set var encoder weights
        cellarium_encoder.var_encoder.weight.copy_(scvi_encoder.var_encoder.weight)
        cellarium_encoder.var_encoder.bias.copy_(scvi_encoder.var_encoder.bias)

        # set batch normalization parameters if they exist
        if hasattr(scvi_encoder, "batch_norm"):
            cellarium_encoder.batch_norm.weight.copy_(scvi_encoder.batch_norm.weight)  # type: ignore[union-attr, operator]
            cellarium_encoder.batch_norm.bias.copy_(scvi_encoder.batch_norm.bias)  # type: ignore[union-attr, operator]
            cellarium_encoder.batch_norm.running_mean.copy_(scvi_encoder.batch_norm.running_mean)  # type: ignore[union-attr, operator]
            cellarium_encoder.batch_norm.running_var.copy_(scvi_encoder.batch_norm.running_var)  # type: ignore[union-attr, operator]

    # Test on same input
    x = torch.FloatTensor(X)
    cellarium_batch_nb = torch.nn.functional.one_hot(
        torch.from_numpy(batch_indices).squeeze().long(),
        num_classes=2,
    ).float()

    # Get outputs
    with torch.no_grad():
        cellarium_dist = cellarium_encoder(x, cellarium_batch_nb, None)
        scvi_out = scvi_encoder(x, cellarium_batch_nb)  # returns (mean, var, sample)

        # Compare means and vars
        print("Comparing Cellarium and scvi-tools encoder outputs")
        print("Cellarium means:", cellarium_dist.loc[:1, :2])
        print("scvi-tools means:", scvi_out[0][:1, :2])
        print("Cellarium scales:", cellarium_dist.scale[:1, :2])
        print("scvi-tools scales:", scvi_out[1][:1, :2])
        torch.testing.assert_close(
            cellarium_dist.loc,
            scvi_out[0],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Encoder means do not match: cellarium {cellarium_dist.loc[:1, :2]} vs scvi {scvi_out[0][:1, :2]}",
        )
        torch.testing.assert_close(
            cellarium_dist.scale.square(),
            scvi_out[1],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Encoder scales do not match: cellarium {cellarium_dist.scale[:1, :2].square()} "
            "vs scvi {scvi_out[1][:1, :2]}",
        )


@pytest.mark.parametrize(
    "use_batch_norm,use_layer_norm",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
    ids=["no_norm", "batch_norm", "layer_norm"],
)
@pytest.mark.parametrize("n_layers", [1, 2], ids=["one_layer", "two_layers"])
@pytest.mark.parametrize("hidden_size", [16, 32], ids=["hidden_16", "hidden_32"])
def test_decoder_mean_matches_scvi_tools(use_batch_norm, use_layer_norm, n_layers, hidden_size):
    try:
        import scvi
        from scvi.nn import DecoderSCVI as SCVIDecoder
    except ImportError:
        pytest.skip("scvi-tools is not installed, skipping test")

    # Setup test data
    n, g = 100, 50
    var_names_g = [f"gene_{i}" for i in range(g)]
    n_latent = 10
    n_batch = 2
    z_nk = torch.randn(n, n_latent)
    batch_indices = np.random.randint(0, n_batch, size=n)
    library_size_n1 = torch.rand(n, 1) + 4

    # Create anndata for scvi-tools
    adata = anndata.AnnData(np.random.poisson(lam=2.0, size=(n, g)))
    adata.var_names = var_names_g
    adata.obs["batch"] = batch_indices
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")

    # Initialize scvi-tools decoder
    scvi_decoder = SCVIDecoder(
        n_input=n_latent,
        n_output=g,
        n_cat_list=[n_batch],
        n_hidden=hidden_size,
        n_layers=n_layers,
        # dropout_rate=0.0,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        inject_covariates=False,
    )

    # Initialize Cellarium decoder with matching architecture
    cellarium_decoder = DecoderSCVI(
        in_features=n_latent,
        out_features=g,
        hidden_layers=[
            {
                "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                "init_args": {
                    "out_features": hidden_size,
                    "n_batch": n_batch,
                    "batch_to_bias_hidden_layers": [],
                },
                "dressing_init_args": {
                    "use_batch_norm": use_batch_norm,
                    "use_layer_norm": use_layer_norm,
                    "dropout_rate": 0.0,
                },
            }
        ]
        + (n_layers - 1)
        * [
            {
                "class_path": "torch.nn.Linear",
                "init_args": {"out_features": hidden_size},
                "dressing_init_args": {
                    "use_batch_norm": use_batch_norm,
                    "use_layer_norm": use_layer_norm,
                    "dropout_rate": 0.0,
                },
            }
        ],
        final_layer={
            "class_path": "torch.nn.Linear",
            "init_args": {},
        },
        n_batch=n_batch,
        dispersion="gene",
        gene_likelihood="nb",
        scale_activation="softmax",
        final_additive_bias=False,
    )

    # Set the same weights manually for both decoders
    with torch.no_grad():
        print("cellarium decoder")
        print(cellarium_decoder)
        print("scvi decoder")
        print(scvi_decoder)

        # Set FC layer weights for layer 1
        assert hasattr(cellarium_decoder.fully_connected.module_list[0].layer, "weight")
        assert hasattr(scvi_decoder.px_decoder.fc_layers[0][0], "weight")
        assert hasattr(cellarium_decoder.fully_connected.module_list[0].layer, "bias")
        assert hasattr(scvi_decoder.px_decoder.fc_layers[0][0], "bias")
        cellarium_decoder.fully_connected.module_list[0].layer.weight.copy_(
            scvi_decoder.px_decoder.fc_layers[0][0].weight[:, :n_latent]
        )
        cellarium_decoder.fully_connected.module_list[0].layer.bias.copy_(
            scvi_decoder.px_decoder.fc_layers[0][0].bias[:hidden_size]
        )

        # set batch weights for layer 1
        assert hasattr(cellarium_decoder.fully_connected.module_list[0].layer, "bias_decoder")
        assert hasattr(cellarium_decoder.fully_connected.module_list[0].layer.bias_decoder, "module_list")
        assert isinstance(
            cellarium_decoder.fully_connected.module_list[0].layer.bias_decoder.module_list,
            torch.nn.ModuleList,
        )
        assert hasattr(scvi_decoder.px_decoder.fc_layers[0][0], "weight")
        cellarium_decoder.fully_connected.module_list[0].layer.bias_decoder.module_list[0].weight.copy_(
            scvi_decoder.px_decoder.fc_layers[0][0].weight[:, n_latent:]
        )

        # set FC layer weights for subsequent layers
        for i in range(1, n_layers):
            cellarium_decoder.fully_connected.module_list[i].layer.weight.copy_(
                scvi_decoder.px_decoder.fc_layers[i][0].weight
            )
            cellarium_decoder.fully_connected.module_list[i].layer.bias.copy_(
                scvi_decoder.px_decoder.fc_layers[i][0].bias
            )

        # Set final layer weights
        cellarium_decoder.normalized_count_decoder.weight.copy_(scvi_decoder.px_scale_decoder[0].weight)
        cellarium_decoder.normalized_count_decoder.bias.copy_(scvi_decoder.px_scale_decoder[0].bias)

        # set batch normalization parameters if they exist
        if hasattr(scvi_decoder, "batch_norm"):
            cellarium_decoder.batch_norm.weight.copy_(scvi_decoder.batch_norm.weight)
            cellarium_decoder.batch_norm.bias.copy_(scvi_decoder.batch_norm.bias)
            cellarium_decoder.batch_norm.running_mean.copy_(scvi_decoder.batch_norm.running_mean)
            cellarium_decoder.batch_norm.running_var.copy_(scvi_decoder.batch_norm.running_var)

    # Test on same input
    batch_nb = torch.nn.functional.one_hot(
        torch.from_numpy(batch_indices).squeeze().long(),
        num_classes=n_batch,
    ).float()

    # Get outputs
    with torch.no_grad():
        # Set a fixed inverse_overdispersion for testing
        inverse_overdispersion = torch.ones(g)
        
        cellarium_dist = cellarium_decoder(
            z_nk=z_nk,
            batch_nb=batch_nb,
            categorical_covariate_np=None,
            inverse_overdispersion=inverse_overdispersion,
            library_size_n1=library_size_n1,
        )
        scvi_output = scvi_decoder(
            'gene',
            z_nk,
            library_size_n1,
            batch_nb,
        )

        # Compare means
        print("Comparing Cellarium and scvi-tools decoder outputs")
        print("Cellarium means:", cellarium_dist.mean[:1, :2])
        print("scvi-tools means:", scvi_output[2][:1, :2])
        torch.testing.assert_close(
            cellarium_dist.mean,
            scvi_output[2],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Decoder means do not match: cellarium {cellarium_dist.mean[:1, :2]} vs scvi {scvi_output[2][:1, :2]}",
        )


@pytest.mark.parametrize("gene_likelihood", ["poisson", "nb"], ids=["poisson", "negative_binomial"])
@pytest.mark.parametrize("n_latent", [5, 10], ids=["latent_5", "latent_10"])
def test_loss_matches_scvi_tools(gene_likelihood, n_latent):
    try:
        import scvi
    except ImportError:
        pytest.skip("scvi-tools is not installed, skipping test")

    # Setup test data
    dispersion = "gene"
    n, g = 100, 50
    n_batch = 2
    var_names_g = [f"gene_{i}" for i in range(g)]
    X = np.random.poisson(lam=2.0, size=(n, g))
    batch_indices = np.random.randint(0, n_batch, size=n)

    # Create anndata for scvi-tools
    adata = anndata.AnnData(X)
    adata.var_names = var_names_g
    adata.obs["batch"] = batch_indices

    # Initialize scvi-tools model
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    scvi_model = scvi.model.SCVI(
        adata,
        n_hidden=32,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood,
        n_layers=1,
        encode_covariates=True,
        dispersion=dispersion,
    )
    print('scvi params')
    print(scvi_model._module_kwargs)
    scvi_model.train(max_epochs=0)  # Setup only, no training

    print('scvi-tools model:')
    print(scvi_model.module.z_encoder)
    print(scvi_model.module.decoder)

    # Initialize Cellarium model with matching architecture
    cellarium_model = SingleCellVariationalInference(
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood,
        dispersion=dispersion,
        encoder={
            "hidden_layers": [
                {
                    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                    "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
                    "dressing_init_args": {
                        "use_batch_norm": True,
                        "use_layer_norm": False,
                        "dropout_rate": 0.1,
                    },
                },
            ],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
        },
        decoder={
            "hidden_layers": [
                {
                    "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
                    "init_args": {"out_features": 32, "batch_to_bias_hidden_layers": []},
                    "dressing_init_args": {
                        "use_batch_norm": True,
                        "use_layer_norm": False,
                        "dropout_rate": 0.0,
                    },
                },
            ],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            "final_additive_bias": False,
        },
    )

    # Set both models to eval mode
    cellarium_model.eval()
    scvi_model.module.eval()

    # Copy weights from scvi-tools to cellarium model
    with torch.no_grad():
        # Copy encoder weights
        cellarium_model.z_encoder.fully_connected.module_list[0].layer.weight.copy_(
            scvi_model.module.z_encoder.encoder.fc_layers[0][0].weight[:, :g]
        )
        cellarium_model.z_encoder.fully_connected.module_list[0].layer.bias.copy_(
            scvi_model.module.z_encoder.encoder.fc_layers[0][0].bias
        )
        cellarium_model.z_encoder.fully_connected.module_list[0].layer.bias_decoder.module_list[0].weight.copy_(
            scvi_model.module.z_encoder.encoder.fc_layers[0][0].weight[:, g:]
        )
        cellarium_model.z_encoder.mean_encoder.weight.copy_(scvi_model.module.z_encoder.mean_encoder.weight)
        cellarium_model.z_encoder.mean_encoder.bias.copy_(scvi_model.module.z_encoder.mean_encoder.bias)
        cellarium_model.z_encoder.var_encoder.weight.copy_(scvi_model.module.z_encoder.var_encoder.weight)
        cellarium_model.z_encoder.var_encoder.bias.copy_(scvi_model.module.z_encoder.var_encoder.bias)

        # Copy decoder weights
        cellarium_model.decoder.fully_connected.module_list[0].layer.weight.copy_(
            scvi_model.module.decoder.px_decoder.fc_layers[0][0].weight[:, :n_latent]
        )
        cellarium_model.decoder.fully_connected.module_list[0].layer.bias.copy_(
            scvi_model.module.decoder.px_decoder.fc_layers[0][0].bias
        )
        cellarium_model.decoder.fully_connected.module_list[0].layer.bias_decoder.module_list[0].weight.copy_(
            scvi_model.module.decoder.px_decoder.fc_layers[0][0].weight[:, n_latent:]
        )
        cellarium_model.decoder.normalized_count_decoder.weight.copy_(
            scvi_model.module.decoder.px_scale_decoder[0].weight
        )
        cellarium_model.decoder.normalized_count_decoder.bias.copy_(
            scvi_model.module.decoder.px_scale_decoder[0].bias
        )

        # Copy dispersion parameters if using negative binomial
        if gene_likelihood == "nb":
            cellarium_model.px_r.copy_(scvi_model.module.px_r)

        # Copy batch norm parameters: they exist
        # if hasattr(scvi_model.module.z_encoder.encoder.fc_layers[0][1], "batch_norm"):
        bn_encoder = cellarium_model.z_encoder.fully_connected.module_list[0].dressing[0]
        bn_encoder.weight.copy_(scvi_model.module.z_encoder.encoder.fc_layers[0][1].weight)
        bn_encoder.bias.copy_(scvi_model.module.z_encoder.encoder.fc_layers[0][1].bias)
        bn_encoder.running_mean.copy_(scvi_model.module.z_encoder.encoder.fc_layers[0][1].running_mean)
        bn_encoder.running_var.copy_(scvi_model.module.z_encoder.encoder.fc_layers[0][1].running_var)

        # if hasattr(scvi_model.module.decoder, "batch_norm"):
        bn_decoder = cellarium_model.decoder.fully_connected.module_list[0].dressing[0]
        bn_decoder.weight.copy_(scvi_model.module.decoder.px_decoder.fc_layers[0][1].weight)
        bn_decoder.bias.copy_(scvi_model.module.decoder.px_decoder.fc_layers[0][1].bias)
        bn_decoder.running_mean.copy_(scvi_model.module.decoder.px_decoder.fc_layers[0][1].running_mean)
        bn_decoder.running_var.copy_(scvi_model.module.decoder.px_decoder.fc_layers[0][1].running_var)

    # Prepare input data
    x = torch.FloatTensor(X)
    batch_nb = torch.nn.functional.one_hot(
        torch.from_numpy(batch_indices).squeeze().long(),
        num_classes=2,
    ).float()

    # Test that the encoders produce the same latent representations
    with torch.no_grad():

        # Cellarium encoder
        cellarium_z_dist = cellarium_model.z_encoder(x_ng=x, batch_nb=batch_nb, categorical_covariate_np=None)
        cellarium_mean_z_nk = cellarium_z_dist.loc

        # scvi-tools encoder
        scvi_encoder_z_dist = scvi_model.module.z_encoder(x, batch_nb)[0]
        scvi_mean_z_nk = scvi_encoder_z_dist.loc

        print("Batch norm states:")
        print("Cellarium running mean:", cellarium_model.z_encoder.fully_connected.module_list[0].dressing[0].running_mean)
        print("scvi-tools running mean:", scvi_model.module.z_encoder.encoder.fc_layers[0][1].running_mean)
        print("Cellarium running var:", cellarium_model.z_encoder.fully_connected.module_list[0].dressing[0].running_var)
        print("scvi-tools running var:", scvi_model.module.z_encoder.encoder.fc_layers[0][1].running_var)

        # assert batch norm states are the same
        torch.testing.assert_close(
            cellarium_model.z_encoder.fully_connected.module_list[0].dressing[0].running_mean,
            scvi_model.module.z_encoder.encoder.fc_layers[0][1].running_mean,
            rtol=1e-5,
            atol=1e-5,
            msg="Batch norm running mean does not match",
        )
        torch.testing.assert_close(
            cellarium_model.z_encoder.fully_connected.module_list[0].dressing[0].running_var,
            scvi_model.module.z_encoder.encoder.fc_layers[0][1].running_var,
            rtol=1e-5,
            atol=1e-5,
            msg="Batch norm running var does not match",
        )

        # Compare latent representations
        print("Comparing latent representations")
        print("Cellarium z_nk:", cellarium_mean_z_nk[:3])
        print("scvi-tools z_nk:", scvi_mean_z_nk[:3])
        torch.testing.assert_close(
            cellarium_mean_z_nk,
            scvi_mean_z_nk,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Latent representations do not match: cellarium {cellarium_mean_z_nk[:3].detach()} "
            f"vs scvi-tools {scvi_mean_z_nk[:3].detach()}",
        )

        # Get scvi-tools data
        train_dl = scvi_model._make_data_loader(
            adata=adata,
            batch_size=len(X),  # Use full dataset to match our cellarium input
            shuffle=False,
        )
        batch = next(iter(train_dl))

        # Get Cellarium loss
        cellarium_loss = cellarium_model(
            x_ng=batch["X"],
            var_names_g=np.array(var_names_g),
            batch_index_n=batch["batch"],
        )

        # get scvi-tools loss
        scvi_inference_tensors = scvi_model.module._get_inference_input(batch)
        scvi_inference_output = scvi_model.module._regular_inference(**scvi_inference_tensors)
        scvi_inference_output["z"] = cellarium_loss["z_nk"]  # cellarium's z_nk to avoid disagreement due to sampling

        print("cellarium sampled z_nk:", cellarium_loss["z_nk"][:3])
        print("scvi-tools sampled z_nk:", scvi_inference_output["z"][:3])

        scvi_generative_tensors = scvi_model.module._get_generative_input(batch, scvi_inference_output)
        scvi_generative_output = scvi_model.module.generative(**scvi_generative_tensors)
        scvi_loss = scvi_model.module.loss(
            tensors=batch,
            inference_outputs=scvi_inference_output,
            generative_outputs=scvi_generative_output,
            kl_weight=1.0,
        )

        # cellarium decoder output distribution
        cellarium_decoder_dist = cellarium_model.generative(
            z_nk=cellarium_loss["z_nk"],
            library_size_n1=x.sum(dim=-1, keepdim=True).log(),  # Use sum as library size
            batch_nb=batch_nb,
            categorical_covariate_np=None,
        )["px"]
        cellarium_mean_x_ng = cellarium_decoder_dist.mean

        # scvi-tools decoder output distribution
        scvi_decoder_dist = scvi_generative_output["px"]
        scvi_mean_x_ng = scvi_decoder_dist.mean

        # Compare decoder outputs
        print("Comparing decoder outputs")
        print("Cellarium mean x_ng:", cellarium_mean_x_ng[:3])
        print("scvi-tools mean x_ng:", scvi_mean_x_ng[:3])
        torch.testing.assert_close(
            cellarium_mean_x_ng,
            scvi_mean_x_ng,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Decoder means do not match: cellarium {cellarium_mean_x_ng[:3, :3].detach()} "
            f"vs scvi-tools {scvi_mean_x_ng[:3, :3].detach()}",
        )

        if gene_likelihood == "nb":
            print("Comparing overdispersion parameters")
            print("Cellarium negbinom variance:", cellarium_decoder_dist.variance[:3])
            print("scvi-tools negbinom variance:", scvi_decoder_dist.variance[:3])
            torch.testing.assert_close(
                cellarium_decoder_dist.variance,
                scvi_decoder_dist.variance,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Overdispersion parameters do not match: cellarium {cellarium_decoder_dist.variance[:3].detach()} "
                f"vs scvi-tools {scvi_decoder_dist.variance[:3].detach()}",
            )

        # Compare KL divergence for z
        print('cellarium KL divergence')
        print(cellarium_loss['kl_divergence_z'])
        print('scvi-tools KL divergence')
        print(scvi_loss.kl_local['kl_divergence_z'])
        torch.testing.assert_close(
            cellarium_loss['kl_divergence_z'],
            scvi_loss.kl_local['kl_divergence_z'],
            rtol=1e-5,
            atol=1e-5,
            msg=f"KL divergence for z does not match: cellarium {cellarium_loss['kl_divergence_z'][:3].detach()} "
            f"vs scvi-tools {scvi_loss.kl_local['kl_divergence_z'][:3].detach()}",
        )

        # Compare reconstruction losses
        print('cellarium reconstruction loss')
        print(cellarium_loss['reconstruction_loss'])
        print('scvi-tools reconstruction loss')
        print(scvi_loss.reconstruction_loss['reconstruction_loss'])
        torch.testing.assert_close(
            cellarium_loss['reconstruction_loss'],
            scvi_loss.reconstruction_loss['reconstruction_loss'],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Reconstruction losses do not match: cellarium {cellarium_loss['reconstruction_loss'][:3].detach()} "
            f"vs scvi-tools {scvi_loss.reconstruction_loss['reconstruction_loss'][:3].detach()}",
        )

        # Compare full losses
        print("Comparing losses:")
        print(f"scvi-tools loss: {scvi_loss.loss}")
        print(f"cellarium loss: {cellarium_loss['loss']}")
        torch.testing.assert_close(
            cellarium_loss['loss'],
            scvi_loss.loss,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Losses do not match: cellarium {cellarium_loss['loss'].detach()} "
            f"vs scvi-tools {scvi_loss.loss.detach()}",
        )
