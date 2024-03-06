# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Any

import pytest

from cellarium.ml.cli import main

devices = os.environ.get("TEST_DEVICES", "1")

CONFIGS = [
    {
        "model_name": "cellarium_gpt",
        "subcommand": "fit",
        "fit": {
            "model": {
                "model": {
                    "class_path": "cellarium.ml.models.CellariumGPT",
                    "init_args": {
                        "d_model": 2,
                        "d_ffn": 4,
                        "n_heads": 1,
                        "n_blocks": 1,
                        "dropout": 0.0,
                        "use_bias": False,
                        "n_context": 5,
                        "attn_mult": 1,
                        "input_mult": 1.0,
                        "output_mult": 1.0,
                        "initializer_range": 0.02,
                        "backend": "torch",
                        "log_metrics": True,
                    },
                },
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_steps": 1,
                "log_every_n_steps": 1,
            },
        },
    },
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
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
                "model": "cellarium.ml.models.OnePassMeanVarStd",
                "transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.NormalizeTotal",
                        "init_args": {"target_count": "10_000"},
                    },
                    "cellarium.ml.transforms.Log1p",
                ],
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
                "model": {
                    "class_path": "cellarium.ml.models.IncrementalPCA",
                    "init_args": {"n_components": "50"},
                },
                "transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.NormalizeTotal",
                        "init_args": {"target_count": "10_000"},
                    },
                    "cellarium.ml.transforms.Log1p",
                ],
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
                "model": {
                    "class_path": "cellarium.ml.models.IncrementalPCA",
                    "init_args": {"n_components": "50"},
                },
                "transforms": [
                    {
                        "class_path": "cellarium.ml.transforms.NormalizeTotal",
                        "init_args": {"target_count": "10_000"},
                    },
                    "cellarium.ml.transforms.Log1p",
                ],
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
                "batch_keys": {
                    "x_ng": {
                        "attr": "X",
                        "convert_fn": "cellarium.ml.utilities.data.densify",
                    },
                    "var_names_g": {"attr": "var_names"},
                    "obs_names": {"attr": "obs_names"},
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
                "transforms": [
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
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
        "model_name": "tdigest",
        "subcommand": "fit",
        "fit": {
            "model": {"model": "cellarium.ml.models.TDigest"},
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
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
]


@pytest.mark.parametrize("config", CONFIGS)
def test_cpu_multi_device(config: dict[str, Any]):
    if config["subcommand"] == "predict":
        assert config["predict"]["return_predictions"] == "false"
    main(config)


def test_checkpoint_loader(tmp_path):
    onepass_config = f"""
    model:
      model: cellarium.ml.models.OnePassMeanVarStd
      transforms:
        - class_path: cellarium.ml.transforms.NormalizeTotal
          init_args:
            target_count: 10_000
        - cellarium.ml.transforms.Log1p
    data:
      filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
      shard_size: 100
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
      model: cellarium.ml.models.LogisticRegression
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
              convert_fn: numpy.ndarray.tolist
    data:
      filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
      shard_size: 100
      max_cache_size: 2
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        y_n:
          attr: obs
          key: cell_type
          convert_fn: cellarium.ml.utilities.data.categories_to_codes
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


def test_compute_var_names_g(tmp_path):
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
      filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
      shard_size: 100
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
        - class_path: cellarium.ml.cli.CellariumModuleLoadFromCheckpoint
          init_args:
            checkpoint_path: {ckpt_path}
      model: cellarium.ml.models.LogisticRegression
    data:
      filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{{0..1}}.h5ad
      shard_size: 100
      batch_keys:
        x_ng:
          attr: X
          convert_fn: cellarium.ml.utilities.data.densify
        y_n:
          attr: obs
          key: cell_type
          convert_fn: cellarium.ml.utilities.data.categories_to_codes
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
