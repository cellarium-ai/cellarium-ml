# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math
from collections.abc import Callable
from typing import Any

import torch

from cellarium.ml.utilities.testing import assert_nonnegative, assert_positive


def copy_module(
    module: torch.nn.Module, self_device: torch.device, copy_device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Return an original module on ``self_device`` and its copy on ``copy_device``.
    If the module is on meta device then it is moved to ``self_device`` efficiently using ``to_empty`` method.

    Args:
        module:
            The module to copy.
        self_device:
            The device to send the original module to.
        copy_device:
            The device to copy the module to.

    Returns:
        A tuple of the original module and its copy.
    """
    module_copy = copy.deepcopy(module)
    if any(param.device.type == "meta" for param in module.parameters()) or any(
        buffer.device.type == "meta" for buffer in module.buffers()
    ):
        module.to_empty(device=self_device)
        module_copy.to_empty(device=copy_device)
    else:
        module.to(device=self_device)
        module_copy.to(device=copy_device)
    return module, module_copy


def train_val_split(n_samples: int, train_size: float | int | None, val_size: float | int | None) -> tuple[int, int]:
    """
    Validate the train and validation sizes and return the number of samples for each.

    Args:
        n_samples:
            The number of samples in the dataset.
        train_size:
            Size of the train split. If :class:`float`, should be between ``0.0`` and ``1.0`` and represent
            the proportion of the dataset to include in the train split. If :class:`int`, represents
            the absolute number of train samples. If ``None``, the value is automatically set to the complement
            of the ``val_size``.
        val_size:
            Size of the validation split. If :class:`float`, should be between ``0.0`` and ``1.0`` and represent
            the proportion of the dataset to include in the validation split. If :class:`int`, represents
            the absolute number of validation samples. If ``None``, the value is set to the complement of
            the ``train_size``. If ``train_size`` is also ``None``, it will be set to ``0``.

    Returns:
        A tuple of the number of samples for training and validation.
    """
    if train_size is None and val_size is None:
        n_val = 0
        n_train = n_samples

    elif train_size is None:
        if isinstance(val_size, int):
            assert_nonnegative("val_size", val_size)
            n_val = val_size
        elif isinstance(val_size, float):
            if val_size < 0.0 or val_size >= 1.0:
                raise ValueError(f"If validation size is a float it should be 0.0 <= val_size < 1.0. Got {val_size}")
            n_val = math.ceil(n_samples * val_size)
        n_train = n_samples - n_val

    elif val_size is None:
        if isinstance(train_size, int):
            assert_positive("train_size", train_size)
            n_train = train_size
        elif isinstance(train_size, float):
            if train_size <= 0.0 or train_size > 1.0:
                raise ValueError(f"If train size is a float it should be 0.0 < train_size <= 1.0. Got {train_size}")
            n_train = math.ceil(n_samples * train_size)
        n_val = n_samples - n_train

    else:
        if isinstance(train_size, int):
            n_train = train_size
        elif isinstance(train_size, float):
            n_train = math.ceil(n_samples * train_size)

        if isinstance(val_size, int):
            n_val = val_size
        elif isinstance(val_size, float):
            n_val = math.ceil(n_samples * val_size)

    if n_train + n_val > n_samples:
        raise ValueError(
            f"Size of train and validation splits ({n_train + n_val}) is greater than "
            f"the number of samples ({n_samples})"
        )

    return n_train, n_val


def call_func_with_batch(
    func: Callable,
    batch: dict[str, Any],
) -> Any:
    """
    Call a function with a batch dictionary. If the function is a method of a :class:`CellariumModule`, the function
    is called with the batch dictionary as its only argument. Otherwise, the function is called with the keys from
    the batch dictionary that are present in its annotations. If the function has a ``kwargs`` annotation, all keys from
    the batch dictionary are passed to the function.

    Args:
        func:
            The function to call.
        batch:
            The batch dictionary.
    """
    from cellarium.ml import CellariumModule

    if isinstance(getattr(func, "__self__", None), CellariumModule):
        # in case module is a CellariumModule, e.g. PCA checkpoint is used as a transform
        return func(batch)

    ann = func.__annotations__
    input_keys = {key for key in ann if key != "return" and key in batch}
    # allow all keys to be passed to the function
    if "kwargs" in ann:
        input_keys |= batch.keys()
    return func(**{key: batch[key] for key in input_keys})
