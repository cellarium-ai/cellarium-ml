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
        "probabilistic_pca.py "
        "--filenames https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        "--batch_size 5000 --accelerator cpu --max_steps 4 --strategy ddp"
    ),
    (
        "onepass_mean_var_std.py "
        "--filenames https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        " --batch_size 5000 --accelerator cpu --strategy ddp"
    ),
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_cpu_multi_device(example):
    devices = os.environ.get("TEST_DEVICES", "1")
    example += f" --devices {devices}"
    print(f"Running:\npython examples/{example}")
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
