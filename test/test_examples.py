# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from subprocess import check_call

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), "examples")

EXAMPLES = [
    (
        "probabilistic_pca.py fit "
        "--model.module.class_path scvid.module.ProbabilisticPCAFromCLI "
        "--data.filenames "
        "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        "--data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 5000 "
        "--data.shuffle true --data.num_workers 2 "
        "--trainer.accelerator cpu --trainer.devices 1 --trainer.max_steps 4"
    ),
    (
        "onepass_mean_var_std.py fit "
        "--model.module.class_path scvid.module.OnePassMeanVarStdFromCLI "
        "--data.filenames "
        "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        "--data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 5000 "
        "--data.num_workers 2 "
        "--trainer.accelerator cpu --trainer.devices 1"
    ),
    (
        "incremental_pca.py fit "
        "--model.module.class_path scvid.module.IncrementalPCAFromCLI "
        "--model.module.init_args.k_components 50 "
        "--data.filenames "
        "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        "--data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 5000 "
        "--data.num_workers 2 "
        "--trainer.accelerator cpu --trainer.devices 1"
    ),
    (
        "incremental_pca.py predict "
        "--model.module.class_path scvid.module.IncrementalPCAFromCLI "
        "--model.module.init_args.k_components 50 "
        "--data.filenames "
        "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        "--data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 5000 "
        "--data.num_workers 2 "
        "--trainer.accelerator cpu --trainer.devices 1 "
        "--trainer.callbacks scvid.callbacks.PredictionWriter --trainer.callbacks.output_dir ./output"
    ),
    (
        "tdigest.py fit "
        "--model.module.class_path scvid.module.TDigestFromCLI "
        "--data.filenames "
        "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        "--data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 5000 "
        "--data.num_workers 2 "
        "--trainer.accelerator cpu --trainer.devices 1"
    ),
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_cpu(example: str):
    print(f"Running:\npython examples/{example}")
    example_list = example.split()
    filename, args = example_list[0], example_list[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
