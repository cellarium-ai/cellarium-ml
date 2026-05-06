# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from cellarium.ml.cli import main

devices = os.environ.get("TEST_DEVICES", "1")

CONFIGS = [
    {
        "model_name": "geneformer",
        "subcommand": "fit",
        "fit": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.Geneformer",
                    "init_args": {
                        "hidden_size": "2",
                        "num_hidden_layers": "1",
                        "num_attention_heads": "1",
                        "intermediate_size": "4",
                        "max_position_embeddings": "2",
                    },
                },
                "optim_fn": "torch.optim.Adam",
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                },
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "1",
            },
        },
    },
    {
        "model_name": "geneformer",
        "subcommand": "predict",
        "predict": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.Geneformer",
                    "init_args": {
                        "hidden_size": "2",
                        "num_hidden_layers": "1",
                        "num_attention_heads": "1",
                        "intermediate_size": "4",
                        "max_position_embeddings": "2",
                    },
                },
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                },
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "1",
                "limit_predict_batches": "1",
            },
            "return_predictions": "false",
        },
    },
    {
        "model_name": "contrastive_mlp",
        "subcommand": "fit",
        "fit": {
            "model": {
                "transforms": [{"class_path": "cellarium.ml.transforms.Duplicate"}],
                "model": {
                    "class_path": "cellarium.ml.models.ContrastiveMLP",
                    "init_args": {
                        "n_obs": "36601",
                        "embed_dim": "4",
                        "hidden_size": [8],
                        "temperature": "1.0",
                    },
                },
                "optim_fn": "torch.optim.Adam",
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                },
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "1",
            },
        },
    },
    {
        "model_name": "contrastive_mlp",
        "subcommand": "predict",
        "predict": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.ContrastiveMLP",
                    "init_args": {
                        "n_obs": "36601",
                        "embed_dim": "4",
                        "hidden_size": [8],
                        "temperature": "1.0",
                    },
                },
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                },
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "1",
                "limit_predict_batches": "1",
            },
            "return_predictions": "false",
        },
    },
    {
        "model_name": "probabilistic_pca",
        "subcommand": "fit",
        "fit": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.ProbabilisticPCA",
                    "init_args": {
                        "n_components": "2",
                        "ppca_flavor": "marginalized",
                    },
                },
                "optim_fn": "torch.optim.Adam",
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                },
                "batch_size": "50",
                "shuffle": "true",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "4",
            },
        },
    },
    {
        "model_name": "onepass_mean_var_std",
        "subcommand": "fit",
        "fit": {
            "model": {
                "transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.NormalizeTotal",
                        "init_args": {"target_count": "10_000"},
                    },
                    "cellarium.ml.transforms.Log1p",
                ],
                "model": "cellarium.ml.models.OnePassMeanVarStd",
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": ["total_mrna_umis"],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                    "total_mrna_umis_n": {
                        "attr": "obs",
                        "key": "total_mrna_umis",
                    },
                },
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
            },
        },
    },
    {
        "model_name": "incremental_pca",
        "subcommand": "fit",
        "fit": {
            "model": {
                "transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.NormalizeTotal",
                        "init_args": {"target_count": "10_000"},
                    },
                    "cellarium.ml.transforms.Log1p",
                ],
                "model": {
                    "class_path": "cellarium.ml.models.IncrementalPCA",
                    "init_args": {"n_components": "50"},
                },
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": ["total_mrna_umis"],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                    "total_mrna_umis_n": {
                        "attr": "obs",
                        "key": "total_mrna_umis",
                    },
                },
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
            },
        },
    },
    {
        "model_name": "incremental_pca",
        "subcommand": "predict",
        "predict": {
            "model": {
                "transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.NormalizeTotal",
                        "init_args": {"target_count": "10_000"},
                    },
                    "cellarium.ml.transforms.Log1p",
                ],
                "model": {
                    "class_path": "cellarium.ml.models.IncrementalPCA",
                    "init_args": {"n_components": "50"},
                },
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": ["total_mrna_umis"],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                    "obs_names_n": {"attr": "obs_names"},
                    "total_mrna_umis_n": {
                        "attr": "obs",
                        "key": "total_mrna_umis",
                    },
                },
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "callbacks": {
                    "class_path": "cellarium.ml.callbacks.PredictionWriter",
                    "init_args": {"output_dir": "./output"},
                },
            },
            "return_predictions": "false",
        },
    },
    {
        "model_name": "logistic_regression",
        "subcommand": "fit",
        "fit": {
            "model": {
                "cpu_transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.Filter",
                        "init_args": {
                            "filter_list": [
                                "ENSG00000187642",
                                "ENSG00000078808",
                                "ENSG00000272106",
                                "ENSG00000162585",
                                "ENSG00000272088",
                                "ENSG00000204624",
                                "ENSG00000162490",
                                "ENSG00000177000",
                                "ENSG00000011021",
                            ]
                        },
                    }
                ],
                "model": "cellarium.ml.models.LogisticRegression",
                "optim_fn": "torch.optim.Adam",
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": ["cell_type"],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {
                        "attr": "var_names",
                    },
                    "y_n": {
                        "attr": "obs",
                        "key": "cell_type",
                        "convert_fn": "cellarium.ml.utilities.data.categories_to_codes",
                    },
                    "y_categories": {
                        "attr": "obs",
                        "key": "cell_type",
                        "convert_fn": "cellarium.ml.utilities.data.get_categories",
                    },
                },
                "batch_size": "50",
                "shuffle": "true",
                "num_workers": "2",
                "val_size": "0.1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "4",
                "val_check_interval": "2",
            },
        },
    },
    {
        "model_name": "hvg_seurat_v3",
        "subcommand": "fit",
        "fit": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.HVGSeuratV3",
                    "init_args": {
                        "n_top_genes": "10",
                    },
                },
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                },
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
            },
        },
    },
    {
        "model_name": "tdigest",
        "subcommand": "fit",
        "fit": {
            "model": {"model": "cellarium.ml.models.TDigest"},
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                },
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
            },
        },
    },
    {
        "model_name": "socam",
        "subcommand": "fit",
        "fit": {
            "model": {
                "cpu_transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.Filter",
                        "init_args": {
                            "filter_list": [
                                "ENSG00000187642",
                                "ENSG00000078808",
                                "ENSG00000272106",
                                "ENSG00000162585",
                                "ENSG00000272088",
                                "ENSG00000204624",
                                "ENSG00000162490",
                                "ENSG00000177000",
                                "ENSG00000011021",
                            ]
                        },
                    }
                ],
                "model": {
                    "class_path": "cellarium.ml.models.SOCAM",
                    "init_args": {
                        "n_obs": "200",
                        "descendant_tensor": torch.eye(89),
                        "cl_names": [
                            "CL:0000003",
                            "CL:0000034",
                            "CL:0000050",
                            "CL:0000057",
                            "CL:0000071",
                            "CL:0000079",
                            "CL:0000082",
                            "CL:0000084",
                            "CL:0000115",
                            "CL:0000126",
                            "CL:0000127",
                            "CL:0000128",
                            "CL:0000129",
                            "CL:0000150",
                            "CL:0000158",
                            "CL:0000160",
                            "CL:0000165",
                            "CL:0000171",
                            "CL:0000212",
                            "CL:0000216",
                            "CL:0000232",
                            "CL:0000235",
                            "CL:0000236",
                            "CL:0000312",
                            "CL:0000498",
                            "CL:0000499",
                            "CL:0000540",
                            "CL:0000556",
                            "CL:0000561",
                            "CL:0000576",
                            "CL:0000584",
                            "CL:0000586",
                            "CL:0000604",
                            "CL:0000623",
                            "CL:0000624",
                            "CL:0000625",
                            "CL:0000669",
                            "CL:0000679",
                            "CL:0000738",
                            "CL:0000740",
                            "CL:0000750",
                            "CL:0000786",
                            "CL:0000789",
                            "CL:0000794",
                            "CL:0000814",
                            "CL:0000815",
                            "CL:0000817",
                            "CL:0000826",
                            "CL:0000827",
                            "CL:0000843",
                            "CL:0000860",
                            "CL:0000875",
                            "CL:0000893",
                            "CL:0000895",
                            "CL:0000896",
                            "CL:0000897",
                            "CL:0000900",
                            "CL:0000913",
                            "CL:0000921",
                            "CL:0000980",
                            "CL:0000986",
                            "CL:0001054",
                            "CL:0001082",
                            "CL:0002063",
                            "CL:0002079",
                            "CL:0002117",
                            "CL:0002132",
                            "CL:0002187",
                            "CL:0002319",
                            "CL:0002340",
                            "CL:0002341",
                            "CL:0002453",
                            "CL:0002553",
                            "CL:0002563",
                            "CL:0002629",
                            "CL:0008019",
                            "CL:0009099",
                            "CL:0011026",
                            "CL:0019028",
                            "CL:1000271",
                            "CL:1000272",
                            "CL:1000334",
                            "CL:1000413",
                            "CL:1000495",
                            "CL:1000692",
                            "CL:1001107",
                            "CL:2000006",
                            "CL:2000059",
                            "CL:extra_not_in_training_data",
                        ],
                        "cl_name_subset": [
                            "CL:0000003",
                            "CL:0000034",
                            "CL:0000050",
                            "CL:0000057",
                            "CL:0000071",
                            "CL:0000079",
                            "CL:0000082",
                            "CL:0000084",
                            "CL:0000115",
                            "CL:0000126",
                            "CL:0000127",
                            "CL:0000128",
                            "CL:0000129",
                            "CL:0000150",
                            "CL:0000158",
                            "CL:0000160",
                            "CL:0000165",
                            "CL:0000171",
                            "CL:0000212",
                            "CL:0000216",
                            "CL:0000232",
                            "CL:0000235",
                            "CL:0000236",
                            "CL:0000312",
                            "CL:0000498",
                            "CL:0000499",
                            "CL:0000540",
                            "CL:0000556",
                            "CL:0000561",
                            "CL:0000576",
                            "CL:0000584",
                            "CL:0000586",
                            "CL:0000604",
                            "CL:0000623",
                            "CL:0000624",
                            "CL:0000625",
                            "CL:0000669",
                            "CL:0000679",
                            "CL:0000738",
                            "CL:0000740",
                            "CL:0000750",
                            "CL:0000786",
                            "CL:0000789",
                            "CL:0000794",
                            "CL:0000814",
                            "CL:0000815",
                            "CL:0000817",
                            "CL:0000826",
                            "CL:0000827",
                            "CL:0000843",
                            "CL:0000860",
                            "CL:0000875",
                            "CL:0000893",
                            "CL:0000895",
                            "CL:0000896",
                            "CL:0000897",
                            "CL:0000900",
                            "CL:0000913",
                            "CL:0000921",
                            "CL:0000980",
                            "CL:0000986",
                            "CL:0001054",
                            "CL:0001082",
                            "CL:0002063",
                            "CL:0002079",
                            "CL:0002117",
                            "CL:0002132",
                            "CL:0002187",
                            "CL:0002319",
                            "CL:0002340",
                            "CL:0002341",
                            "CL:0002453",
                            "CL:0002553",
                            "CL:0002563",
                            "CL:0002629",
                            "CL:0008019",
                            "CL:0009099",
                            "CL:0011026",
                            "CL:0019028",
                            "CL:1000271",
                            "CL:1000272",
                            "CL:1000334",
                            "CL:1000413",
                            "CL:1000495",
                            "CL:1000692",
                            "CL:1001107",
                            "CL:2000006",
                            "CL:2000059",
                        ],
                    },
                },
                "optim_fn": "torch.optim.Adam",
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": ["cell_type_ontology_term_id"],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {
                        "attr": "var_names",
                    },
                    "cl_names_n": {
                        "attr": "obs",
                        "key": "cell_type_ontology_term_id",
                    },
                },
                "batch_size": "50",
                "shuffle": "true",
                "num_workers": "2",
                "val_size": "0",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "4",
                "val_check_interval": "2",
            },
        },
    },
]


@pytest.mark.parametrize(
    "config",
    CONFIGS,
    ids=[config["model_name"] + "-" + config["subcommand"] for config in CONFIGS],  # type: ignore[operator]
)
def test_cpu_multi_device(config: dict[str, Any]):
    if config["subcommand"] == "predict":
        assert config["predict"]["return_predictions"] == "false"
    main(config)


def test_checkpoint_loader(tmp_path: Path) -> None:
    onepass_config = f"""
    model:
      transforms:
        - class_path: cellarium.ml.transforms.NormalizeTotal
          init_args:
            target_count: 10_000
        - cellarium.ml.transforms.Log1p
      model: cellarium.ml.models.OnePassMeanVarStd
    data:
      dadc:
        class_path: cellarium.ml.data.DistributedAnnDataCollection
        init_args:
          filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
          shard_size: 100
          obs_columns_to_validate: [total_mrna_umis]
          max_cache_size: 2
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        var_names_g:
          attr: var_names
        total_mrna_umis_n:
          attr: obs
          key: total_mrna_umis
      batch_size: 50
      num_workers: 2
    trainer:
      accelerator: cpu
      devices: 1
      default_root_dir: {tmp_path}
    """
    with open(tmp_path / "onepass_config.yaml", "w") as f:
        f.write(onepass_config)
    main(["onepass_mean_var_std", "fit", "--config", str(tmp_path / "onepass_config.yaml")])
    ckpt_path = tmp_path / "lightning_logs" / "version_0" / "checkpoints" / "epoch=0-step=4.ckpt"

    lr_config = f"""
    model:
      transforms:
        - class_path: cellarium.ml.transforms.NormalizeTotal
          init_args:
            target_count: 10_000
        - cellarium.ml.transforms.Log1p
        - class_path: cellarium.ml.transforms.ZScore
          init_args:
            mean_g:
              !CheckpointLoader
              file_path: {ckpt_path}
              attr: model.mean_g
              convert_fn: null
            std_g:
              !CheckpointLoader
              file_path: {ckpt_path}
              attr: model.std_g
              convert_fn: null
            var_names_g:
              !CheckpointLoader
              file_path: {ckpt_path}
              attr: model.var_names_g
              convert_fn: null
      model: cellarium.ml.models.LogisticRegression
      optim_fn: torch.optim.Adam
    data:
      dadc:
        class_path: cellarium.ml.data.DistributedAnnDataCollection
        init_args:
          filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
          shard_size: 100
          obs_columns_to_validate: [cell_type, total_mrna_umis]
          max_cache_size: 2
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        y_n:
          attr: obs
          key: cell_type
          convert_fn: cellarium.ml.utilities.data.categories_to_codes
        y_categories:
          attr: obs
          key: cell_type
          convert_fn: cellarium.ml.utilities.data.get_categories
        var_names_g:
          attr: var_names
        total_mrna_umis_n:
          attr: obs
          key: total_mrna_umis
      batch_size: 50
      num_workers: 2
    trainer:
      accelerator: cpu
      devices: 1
      max_steps: 1
    """
    with open(tmp_path / "lr_config.yaml", "w") as f:
        f.write(lr_config)
    main(["logistic_regression", "fit", "--config", str(tmp_path / "lr_config.yaml")])


def test_compute_var_names_g(tmp_path: Path) -> None:
    ipca_config = f"""
    model:
      transforms:
        - class_path: cellarium.ml.transforms.NormalizeTotal
          init_args:
            target_count: 10_000
        - cellarium.ml.transforms.Log1p
      model:
        class_path: cellarium.ml.models.IncrementalPCA
        init_args:
            n_components: 50
    data:
      dadc:
        class_path: cellarium.ml.data.DistributedAnnDataCollection
        init_args:
          filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
          shard_size: 100
          obs_columns_to_validate: [total_mrna_umis]
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        var_names_g:
          attr: var_names
        total_mrna_umis_n:
          attr: obs
          key: total_mrna_umis
      batch_size: 100
    trainer:
      accelerator: cpu
      devices: 1
      default_root_dir: {tmp_path}
    """
    with open(ipca_config_path := str(tmp_path / "ipca_config.yaml"), "w") as f:
        f.write(ipca_config)
    main(["incremental_pca", "fit", "--config", ipca_config_path])
    ckpt_path = tmp_path / "lightning_logs" / "version_0" / "checkpoints" / "epoch=0-step=2.ckpt"

    lr_config = f"""
    model:
      transforms:
        - !CheckpointLoader
          file_path: {ckpt_path}
          attr: null
          convert_fn: null
      model: cellarium.ml.models.LogisticRegression
      optim_fn: torch.optim.Adam
    data:
      dadc:
        class_path: cellarium.ml.data.DistributedAnnDataCollection
        init_args:
          filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
          shard_size: 100
          obs_columns_to_validate: [cell_type, total_mrna_umis]
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        y_n:
          attr: obs
          key: cell_type
          convert_fn: cellarium.ml.utilities.data.categories_to_codes
        y_categories:
          attr: obs
          key: cell_type
          convert_fn: cellarium.ml.utilities.data.get_categories
        var_names_g:
          attr: var_names
        total_mrna_umis_n:
          attr: obs
          key: total_mrna_umis
      batch_size: 100
    trainer:
      accelerator: cpu
      devices: 1
      max_steps: 1
    """
    with open(lr_config_path := str(tmp_path / "lr_config.yaml"), "w") as f:
        f.write(lr_config)
    main(["logistic_regression", "fit", "--config", lr_config_path])

    onepass_config_with_cpu_filter = f"""
    model:
      cpu_transforms:
        - class_path: cellarium.ml.transforms.Filter
          init_args:
            filter_list:
              - ENSG00000187642
              - ENSG00000078808
              - ENSG00000272106
              - ENSG00000162585
              - ENSG00000272088
              - ENSG00000204624
              - ENSG00000162490
              - ENSG00000177000
              - ENSG00000011021
      transforms:
        - class_path: cellarium.ml.transforms.NormalizeTotal
          init_args:
            target_count: 10_000
        - cellarium.ml.transforms.Log1p
      model: cellarium.ml.models.OnePassMeanVarStd
    data:
      dadc:
        class_path: cellarium.ml.data.DistributedAnnDataCollection
        init_args:
          filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
          shard_size: 100
          obs_columns_to_validate: [total_mrna_umis]
          max_cache_size: 2
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        var_names_g:
          attr: var_names
        total_mrna_umis_n:
          attr: obs
          key: total_mrna_umis
      batch_size: 50
      num_workers: 2
    trainer:
      accelerator: cpu
      devices: 1
      default_root_dir: {tmp_path}
    """
    with open(tmp_path / "onepass_config_with_cpu_filter.yaml", "w") as f:
        f.write(onepass_config_with_cpu_filter)
    main(["onepass_mean_var_std", "fit", "--config", str(tmp_path / "onepass_config_with_cpu_filter.yaml")])


def test_return_predictions_userwarning(tmp_path: Path):
    """Ensure a warning is emitted when return_predictions is set to true and the subcommand is predict."""

    # using a pre-parsed config
    config: dict[str, Any] = {
        "model_name": "geneformer",
        "subcommand": "predict",
        "predict": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.Geneformer",
                    "init_args": {
                        "hidden_size": "2",
                        "num_hidden_layers": "1",
                        "num_attention_heads": "1",
                        "intermediate_size": "4",
                        "max_position_embeddings": "2",
                    },
                },
            },
            "data": {
                "dadc": {
                    "class_path": "cellarium.ml.data.DistributedAnnDataCollection",
                    "init_args": {
                        "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                        "shard_size": "100",
                        "max_cache_size": "2",
                        "obs_columns_to_validate": [],
                    },
                },
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                },
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": "1",
                "limit_predict_batches": "1",
            },
            "return_predictions": "true",
        },
    }

    match_str = r"`return_predictions` argument should"

    with pytest.warns(UserWarning, match=match_str):
        main(copy.deepcopy(config))  # running main modifies the config dict

    config["predict"]["return_predictions"] = "false"
    with pytest.warns(UserWarning, match=match_str) as record:
        warnings.warn("we need one warning: " + match_str, UserWarning)
        main(copy.deepcopy(config))
    n = 0
    for r in record:
        assert isinstance(r.message, Warning)
        warning_message = r.message.args[0]
        if match_str in warning_message:
            n += 1
    assert n < 2, "Unexpected UserWarning when running predict with return_predictions=false"

    # using a config file
    config_file_text = f"""
    # lightning.pytorch==2.5.0.post0
    seed_everything: true
    trainer:
      accelerator: cpu
      devices: 1
      max_steps: 1
      limit_predict_batches: 1
      default_root_dir: {tmp_path}
    model:
      cpu_transforms: null
      transforms: null
      model:
        class_path: cellarium.ml.models.Geneformer
        init_args:
          hidden_size: 2
          num_hidden_layers: 1
          num_attention_heads: 1
          intermediate_size: 4
          max_position_embeddings: 2
    data:
      dadc:
        class_path: cellarium.ml.data.DistributedAnnDataCollection
        init_args:
          filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad
          shard_size: 100
          max_cache_size: 2
          obs_columns_to_validate: []
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        var_names_g:
          attr: var_names
      batch_size: 5
      num_workers: 1
    return_predictions: RETURN_PREDICTIONS_VALUE
    ckpt_path: null
    """

    # there should be a warning with return_predictions=true
    with open(config_file_path := tmp_path / "config.yaml", "w") as f:
        f.write(config_file_text.replace("RETURN_PREDICTIONS_VALUE", "true"))

    with pytest.warns(UserWarning, match=match_str):
        main(["geneformer", "predict", "--config", str(config_file_path)])

    # there should be no warning with return_predictions=false
    with open(config_file_path, "w") as f:
        f.write(config_file_text.replace("RETURN_PREDICTIONS_VALUE", "false"))

    with pytest.warns(UserWarning, match=match_str) as record:
        warnings.warn("we need one warning: " + match_str, UserWarning)
        main(["geneformer", "predict", "--config", str(config_file_path)])
    n = 0
    for r in record:
        assert isinstance(r.message, Warning)
        warning_message = r.message.args[0]
        if match_str in warning_message:
            print(warning_message)
            n += 1
    assert n < 2, "Unexpected UserWarning when running predict with return_predictions=false"
