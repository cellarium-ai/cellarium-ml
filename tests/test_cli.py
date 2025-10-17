# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import warnings
from pathlib import Path
from typing import Any

import pytest

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
        "model_name": "nmf",
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
                    "class_path": "cellarium.ml.models.NonNegativeMatrixFactorization",
                    "init_args": {
                        "k_values": [10],
                        "r": 10,
                        "algorithm": "mairal",
                        "var_names_g": [
                            "ENSG00000187642",
                            "ENSG00000078808",
                            "ENSG00000272106",
                            "ENSG00000162585",
                            "ENSG00000272088",
                            "ENSG00000204624",
                            "ENSG00000162490",
                            "ENSG00000177000",
                            "ENSG00000011021",
                        ],
                    },
                },
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
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": devices,
                "max_epochs": 1,
                "strategy": {
                    "class_path": "lightning.pytorch.strategies.DDPStrategy",
                    "init_args": {"broadcast_buffers": "true"},
                },
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
