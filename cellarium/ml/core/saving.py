# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jsonargparse import ArgumentParser, Namespace
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

if TYPE_CHECKING:
    from cellarium.ml.core.module import CellariumModule


def _load_state(
    cls: type[CellariumModule],
    checkpoint: dict[str, Any],
    strict: bool | None = None,
    **cls_kwargs_new: Any,
) -> CellariumModule:
    """
    Re-implementation of :func:`lightning.pytorch.core.saving._load_state` that instantiates the model
    using the configuration saved in the checkpoint.
    """
    ### cellarium.ml - this part is different from the original
    parser = ArgumentParser()
    parser.add_class_arguments(cls, "model")

    cfg = Namespace(model={})

    # pass in the values we saved automatically
    if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
        cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]

    # update cfg with cls_kwargs_new, such that new has higher priority
    for key, value in cls_kwargs_new.items():
        cfg["model"][key] = value

    # instantiate the model
    cfg_init = parser.instantiate_classes(cfg)
    obj = cfg_init.model

    # save the cfg to the :attr:`obj.hparams` to be able to load the model checkpoint
    obj._set_hparams(cfg)
    ### cellarium.ml - end of modification

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
