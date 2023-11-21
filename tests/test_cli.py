# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Any

import pytest

from cellarium.ml.cli import main
from tests.common import requires_crick

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
                    "feature_g": {
                        "attr": "var_names",
                        "convert_fn": "cellarium.ml.utilities.data.pandas_to_numpy",
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
                    "feature_g": {
                        "attr": "var_names",
                        "convert_fn": "cellarium.ml.utilities.data.pandas_to_numpy",
                    },
                    "obs_names": {
                        "attr": "obs_names",
                        "convert_fn": "cellarium.ml.utilities.data.pandas_to_numpy",
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
                        "k_components": "2",
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
                    "feature_g": {
                        "attr": "var_names",
                        "convert_fn": "cellarium.ml.utilities.data.pandas_to_numpy",
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
        "model_name": "onepass_mean_var_std",
        "subcommand": "fit",
        "fit": {
            "model": {
                "model": "cellarium.ml.models.OnePassMeanVarStd",
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
                    "feature_g": {
                        "attr": "var_names",
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
                    "init_args": {"k_components": "50"},
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
                    "feature_g": {
                        "attr": "var_names",
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
                    "init_args": {"k_components": "50"},
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
                    "feature_g": {
                        "attr": "var_names",
                    },
                    "obs_names": {
                        "attr": "obs_names",
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
                    "feature_g": {
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
    pytest.param(
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
                        "feature_g": {
                            "attr": "var_names",
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
        marks=requires_crick,
    ),
]


@pytest.mark.parametrize("config", CONFIGS)
def test_cpu_multi_device(config: dict[str, Any]):
    if config["subcommand"] == "predict":
        assert config["predict"]["return_predictions"] == "false"
    main(config)
