# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from jsonargparse import ArgumentParser
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

# from torchdistx.deferred_init import deferred_init, materialize_module


def _load_state(
    cls: type[pl.LightningModule],
    checkpoint: dict[str, Any],
    strict: bool | None = None,
    **cls_kwargs_new: Any,
) -> pl.LightningModule:
    parser = ArgumentParser()
    parser.add_class_arguments(cls, "model")

    # pass in the values we saved automatically
    if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
        cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
    else:
        cfg = parser.get_defaults()

    # instantiate the model
    # with torch.device("meta"):
    cfg_init = parser.instantiate_classes(cfg)
    obj = cfg_init.model

    # load the state_dict on the model automatically
    assert strict is not None
    keys = obj.load_state_dict(checkpoint["state_dict"], strict=strict)

    if not strict:
        if keys.missing_keys:
            rank_zero_warn(
                f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
            )
        if keys.unexpected_keys:
            rank_zero_warn(
                f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
            )

    return obj
