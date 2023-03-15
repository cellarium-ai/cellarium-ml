import os
import sys
from subprocess import check_call

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), "examples")

EXAMPLES = [
    "probabilistic_pca.py --num_shards 1 --batch_size 5000 --accelerator cpu --max_steps 2 --strategy ddp",
    "onepass_mean_var_std.py --num_shards 1 --batch_size 5000 --accelerator cpu --strategy ddp",
]


@pytest.mark.skipif("CI" in os.environ, reason="GCS keys are not available")
@pytest.mark.parametrize("example", EXAMPLES)
def test_cpu_multi_device(example):
    devices = os.environ.get("TEST_DEVICES", "1")
    example += f" --devices {devices}"
    print(f"Running:\npython examples/{example}")
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
