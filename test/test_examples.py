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
        "--model.module.class_path scvid.module.ProbabilisticPCAWithDefaults "
        "--data.filenames "
        "'https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad' "
        "--data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 5000 "
        "--data.shuffle true --data.num_workers 2 "
        "--trainer.accelerator cpu --trainer.devices 1 --trainer.max_steps 4"
    ),
    (
        "onepass_mean_var_std.py fit "
        "--model.module.class_path scvid.module.OnePassMeanVarStdWithDefaults "
        "--data.filenames "
        "'https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad' "
        "--data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 5000 "
        "--data.num_workers 2 "
        "--trainer.accelerator cpu --trainer.devices 1"
    ),
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_cpu(example):
    print(f"Running:\npython examples/{example}")
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
