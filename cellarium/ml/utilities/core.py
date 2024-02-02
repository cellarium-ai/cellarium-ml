# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from functools import singledispatch
from typing import Any

from jsonargparse._util import import_object
from lightning.pytorch.core.mixins import HyperparametersMixin


@singledispatch
def uninitialize_object(value):
    """
    Given an object, return a dictionary containing the class path and the init args.
    """
    return value


@uninitialize_object.register
def _(value: HyperparametersMixin) -> dict[str, Any]:
    if not value.hparams:
        warnings.warn(
            f"Module {value.__class__.__name__} has no hyperparameters. "
            "Consider using `save_hyperparameters` to save the module's hyperparameters.",
            UserWarning,
        )
    return {"class_path": f"{value.__module__}.{value.__class__.__name__}", "init_args": value.hparams}


@singledispatch
def initialize_object(value):
    """
    Given a class path and init args, return an object.
    """
    return value


@initialize_object.register
def _(value: str) -> HyperparametersMixin:
    return initialize_object({"class_path": value, "init_args": {}})


@initialize_object.register
def _(value: dict) -> HyperparametersMixin:
    return import_object(value["class_path"])(**value["init_args"])
