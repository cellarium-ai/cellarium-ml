# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import torch


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
    module_copy = copy.deepcopy(module).to(copy_device)
    if any(param.device.type == "meta" for param in module.parameters()) or any(
        buffer.device.type == "meta" for buffer in module.buffers()
    ):
        module.to_empty(device=self_device)
    else:
        module.to(device=self_device)
    return module, module_copy
