# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from cellarium.ml.lr_schedulers import WarmupCosineLR


def _scheduler(num_training_steps: int = 10, warmup_fraction: float = 0.2) -> WarmupCosineLR:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    return WarmupCosineLR(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_fraction=warmup_fraction,
    )


def test_warmup_cosine_lr_warms_up_then_cosine_decays_to_zero():
    scheduler = _scheduler(num_training_steps=10, warmup_fraction=0.2)

    assert scheduler.num_warmup_steps == 2
    assert scheduler._lr_lambda(0) == 0.0
    assert scheduler._lr_lambda(1) == 0.5
    assert scheduler._lr_lambda(2) == 1.0
    assert scheduler._lr_lambda(6) == pytest.approx(0.5)
    assert scheduler._lr_lambda(10) == 0.0
    assert scheduler._lr_lambda(11) == 0.0


def test_warmup_cosine_lr_accepts_explicit_warmup_steps():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    scheduler = WarmupCosineLR(optimizer, num_training_steps=10, num_warmup_steps=3)

    assert scheduler.num_warmup_steps == 3
    assert scheduler._lr_lambda(1) == pytest.approx(1 / 3)
    assert scheduler._lr_lambda(3) == 1.0


def test_warmup_cosine_lr_validates_inputs():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0)

    with pytest.raises(ValueError, match="num_training_steps"):
        WarmupCosineLR(optimizer, num_training_steps=0)
    with pytest.raises(ValueError, match="warmup_fraction"):
        WarmupCosineLR(optimizer, num_training_steps=10, warmup_fraction=1.2)
    with pytest.raises(ValueError, match="num_warmup_steps"):
        WarmupCosineLR(optimizer, num_training_steps=10, num_warmup_steps=11)
