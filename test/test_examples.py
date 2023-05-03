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
        "--batch_size 5000 --accelerator cpu --max_steps 4 --devices 1"
    ),
    (
        "onepass_mean_var_std.py "
        "--filenames https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..001}.h5ad "
        " --batch_size 5000 --accelerator cpu --devices 1"
    ),
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_cpu(example: str):
    print(f"Running:\npython examples/{example}")
    example_list = example.split()
    filename, args = example_list[0], example_list[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
