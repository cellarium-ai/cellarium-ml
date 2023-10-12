# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import pytest

from cellarium.ml.cli import main

from .common import requires_crick

CONFIGS = [
    {
        "model_name": "geneformer",
        "subcommand": "fit",
        "fit": {
            "model": {
                "module": {
                    "class_path": "cellarium.ml.module.GeneformerFromCLI",
                    "init_args": {"num_hidden_layers": "1", "num_attention_heads": "1"},
                },
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": "1",
                "max_steps": "1",
            },
        },
    },
    {
        "model_name": "geneformer",
        "subcommand": "predict",
        "predict": {
            "model": {
                "module": {
                    "class_path": "cellarium.ml.module.GeneformerFromCLI",
                    "init_args": {"num_hidden_layers": "1", "num_attention_heads": "1"},
                },
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
                "batch_size": "5",
                "num_workers": "1",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": "1",
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
                "module": "cellarium.ml.module.ProbabilisticPCAFromCLI",
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
                "batch_size": "50",
                "shuffle": "true",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": "1",
                "max_steps": "4",
            },
        },
    },
    {
        "model_name": "onepass_mean_var_std",
        "subcommand": "fit",
        "fit": {
            "model": {
                "module": "cellarium.ml.module.OnePassMeanVarStdFromCLI",
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": "1",
            },
        },
    },
    {
        "model_name": "incremental_pca",
        "subcommand": "fit",
        "fit": {
            "model": {
                "module": {
                    "class_path": "cellarium.ml.module.IncrementalPCAFromCLI",
                    "init_args": {"k_components": "50"},
                },
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": "1",
            },
        },
    },
    {
        "model_name": "incremental_pca",
        "subcommand": "predict",
        "predict": {
            "model": {
                "module": {
                    "class_path": "cellarium.ml.module.IncrementalPCAFromCLI",
                    "init_args": {"k_components": "50"},
                },
            },
            "data": {
                "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                "shard_size": "100",
                "max_cache_size": "2",
                "batch_size": "50",
                "num_workers": "2",
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": "1",
                "callbacks": {
                    "class_path": "cellarium.ml.callbacks.PredictionWriter",
                    "init_args": {"output_dir": "./output"},
                },
            },
            "return_predictions": "false",
        },
    },
    pytest.param(
        {
            "model_name": "tdigest",
            "subcommand": "fit",
            "fit": {
                "model": {"module": "cellarium.ml.module.TDigestFromCLI"},
                "data": {
                    "filenames": "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..1}.h5ad",
                    "shard_size": "100",
                    "max_cache_size": "2",
                    "batch_size": "50",
                    "num_workers": "2",
                },
                "trainer": {
                    "accelerator": "cpu",
                    "devices": "1",
                },
            },
        },
        marks=requires_crick,
    ),
]


@pytest.mark.parametrize("config", CONFIGS)
def test_cpu(config: dict[str, Any]):
    if config["subcommand"] == "predict":
        assert config["predict"]["return_predictions"] == "false"
    main(config)
