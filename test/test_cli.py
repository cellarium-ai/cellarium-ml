# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from importlib import import_module

import pytest

from .common import requires_crick

CONFIGS = [
    {
        "cli": "cellarium.ml.cli.geneformer",
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
        "cli": "cellarium.ml.cli.geneformer",
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
        "cli": "cellarium.ml.cli.probabilistic_pca",
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
        "cli": "cellarium.ml.cli.onepass_mean_var_std",
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
        "cli": "cellarium.ml.cli.incremental_pca",
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
        "cli": "cellarium.ml.cli.incremental_pca",
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
            "cli": "cellarium.ml.cli.tdigest",
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
def test_cpu(config: dict[str, str]):
    if config["subcommand"] == "predict":
        assert config["predict"]["return_predictions"] == "false"
    cli = import_module(config.pop("cli"))
    cli.main(config)
